# File: src/ingest_v2/scripts/ingest_one.py
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY
# ── pipeline imports (already in your repo) ─────────────────────────────────────
from src.ingest_v2.configs.settings import settings_v2
from src.ingest_v2.entities.postprocess import postprocess_aai_entities
from src.ingest_v2.pipelines.build_children import build_children_from_raw
from src.ingest_v2.pipelines.build_parents import build_parent
from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
from src.ingest_v2.router.cache import load as router_cache_load, save as router_cache_save
from src.ingest_v2.router.enrich_parent import enrich_parent_router_fields_async
from src.ingest_v2.speakers.resolve import resolve_speakers
from src.ingest_v2.transcripts.normalize import normalize_to_sentences
from src.ingest_v2.utils.progress import progress

# Optional: Pinecone client for purge-first
try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None  # type: ignore


def setup_logger():
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s - ingest_one - %(levelname)s - %(message)s",
    )

# --- add near the top (after imports) -----------------------------------------
def _resolve_pinecone_target() -> tuple[str, str]:
    """Return (index_name, namespace) after applying env overrides."""
    from src.ingest_v2.configs.settings import settings_v2
    idx = os.getenv("PINECONE_INDEX_NAME", settings_v2.PINECONE_INDEX_NAME)
    ns  = os.getenv("PINECONE_NAMESPACE", settings_v2.NAMESPACE_VIDEOS)
    return idx, ns


def _fast_json_load(path: Path) -> Any:
    data = path.read_bytes()
    try:
        import orjson  # type: ignore
        return orjson.loads(data)
    except Exception:
        return json.loads(data.decode("utf-8"))


def _ms_to_s(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x) / 1000.0
    except Exception:
        return None


def _is_trivial_raw_segment(seg: Dict[str, Any]) -> bool:
    no_text = not (seg.get("text") or "").strip()
    no_words = not seg.get("words")
    start_none = seg.get("start") is None
    end_none = seg.get("end") is None
    return no_text and no_words and start_none and end_none


def _convert_assemblyai_json_to_raw(obj: Any) -> Dict[str, Any]:
    segs_in = None
    if isinstance(obj, dict):
        if "segments" in obj and isinstance(obj["segments"], list):
            segs_in = obj["segments"]
        elif "utterances" in obj and isinstance(obj["utterances"], list):
            segs_in = []
            for utt in obj["utterances"]:
                segs_in.append({
                    "start": utt.get("start"),
                    "end": utt.get("end"),
                    "speaker": utt.get("speaker") or utt.get("speaker_label"),
                    "text": utt.get("text"),
                    "words": utt.get("words") or [],
                })
        elif "words" in obj or "text" in obj or obj.get("status") == "completed":
            words = obj.get("words")
            text = obj.get("text")
            if isinstance(words, list) and words:
                first_word = words[0] if isinstance(words[0], dict) else {}
                last_word = words[-1] if isinstance(words[-1], dict) else {}
                segs_in = [{
                    "start": obj.get("audio_start_from", first_word.get("start")),
                    "end": obj.get("audio_end_at", last_word.get("end")),
                    "speaker": obj.get("speaker") or obj.get("speaker_label"),
                    "text": text,
                    "words": words,
                }]
            elif isinstance(text, str) and text.strip():
                segs_in = [{
                    "start": obj.get("audio_start_from"),
                    "end": obj.get("audio_end_at"),
                    "speaker": obj.get("speaker") or obj.get("speaker_label"),
                    "text": text,
                    "words": [],
                }]
            else:
                segs_in = []
    elif isinstance(obj, list):
        segs_in = obj

    if not isinstance(segs_in, list):
        raise ValueError("Expected top-level list or dict with 'segments'/'utterances'")

    out_segments: List[Dict[str, Any]] = []
    for seg in segs_in:
        if _is_trivial_raw_segment(seg):
            continue

        words_in = seg.get("words")
        out_words = None
        if isinstance(words_in, list) and words_in:
            out_words = []
            for w in words_in:
                wtext = (w.get("text") or "").strip()
                if not wtext:
                    continue
                out_words.append({
                    "text": wtext,
                    "start": _ms_to_s(w.get("start")),
                    "end": _ms_to_s(w.get("end")),
                    "speaker": w.get("speaker") or seg.get("speaker") or "S1",
                })

        start_s = _ms_to_s(seg.get("start"))
        end_s = _ms_to_s(seg.get("end"))
        text = (seg.get("text") or "").strip()
        if start_s is None and end_s is None and not text and not out_words:
            continue

        out_segments.append({
            "start": start_s,
            "end": end_s,
            "speaker": seg.get("speaker") or seg.get("speaker_label") or "S1",
            "text": text,
            **({"words": out_words} if out_words is not None else {}),
        })

    return {"segments": out_segments}


_DATE_ID_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<id>[A-Za-z0-9_-]{11})_")
_MULTI_US_RE = re.compile(r"_{1,3}([A-Za-z0-9_-]{11})_")
_TOKEN_11_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def _extract_video_id_from_path(path: Path) -> Optional[str]:
    pieces = [
        path.name, path.stem, path.as_posix(),
        path.parent.name, getattr(path.parent, "stem", path.parent.name),
        path.parent.parent.name if path.parent and path.parent.parent else "",
    ]
    for s in pieces:
        m = _DATE_ID_RE.search(s)
        if m:
            return m.group("id")
    for s in pieces:
        m = _MULTI_US_RE.search(s)
        if m:
            return m.group(1)
    for s in pieces:
        m = _TOKEN_11_RE.search(s)
        if m:
            return m.group(0)
    return None


def _extract_title_from_dir(path: Path) -> str:
    leaf = path.parent.name.strip()
    if leaf and "_diarized_content" not in leaf:
        return leaf
    return path.stem.replace("_diarized_content", "")


def _extract_channel_from_path(path: Path) -> Optional[str]:
    maybe = path.parent.parent.name
    return maybe if maybe.startswith("@") else None


def _extract_date_prefix(title_or_dir: str) -> Optional[str]:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\b", title_or_dir)
    return m.group(1) if m else None


def _guess_audio_path(json_path: Path) -> Optional[Path]:
    stem = json_path.stem.replace("_diarized_content", "")
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.with_name(f"{stem}{ext}")
        if cand.exists():
            return cand
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _entities_sidecar_path(json_path: Path) -> Path:
    stem = json_path.stem.replace("_diarized_content", "")
    return json_path.with_name(f"{stem}_entities.json")


def _load_entities_for_json_path(content_path: Path, obj: Any) -> List[Dict[str, Any]]:
    def _norm(payload) -> List[Dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
            payload = payload["entities"]
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return payload
            if payload and isinstance(payload[0], str):
                return [{"text": s, "entity_type": "custom"} for s in payload if isinstance(s, str)]
            return []
        return []

    sidecar = _entities_sidecar_path(content_path)
    try:
        if sidecar.exists():
            raw = _fast_json_load(sidecar)
            ents = _norm(raw)
            logging.info(
                "[entities] using sidecar file=%s for content=%s (items=%d)",
                sidecar.name, content_path.name, len(ents)
            )
            # optional peek:
            if ents:
                logging.info("[entities] sample (sidecar): %s", [e.get("text") for e in ents[:12]])
            return ents
    except Exception as e:
        logging.warning("[entities] sidecar read failed file=%s: %s", sidecar.name, e)

    top_payload = obj.get("entities") if isinstance(obj, dict) else None
    ents = _norm(top_payload)
    if ents:
        logging.info(
            "[entities] using top-level 'entities' in %s (no sidecar or empty) (items=%d)",
            content_path.name, len(ents)
        )
        logging.info("[entities] sample (top-level): %s", [e.get("text") for e in ents[:12]])
    else:
        logging.info(
            "[entities] no entities for %s (checked sidecar=%s)",
            content_path.name, sidecar.name
        )
    return ents


def _find_json_for_video(root: Path, video_id: str) -> Path:
    candidates = []
    for p in root.rglob("*_diarized_content.json"):
        vid = _extract_video_id_from_path(p)
        if vid == video_id:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No '*_diarized_content.json' found under {root} for video_id={video_id}")
    candidates.sort(key=lambda p: (len(p.as_posix()), p.as_posix()))
    return candidates[0]


def purge_parent_vectors(video_id: str, *, index_name: str, namespace: str) -> None:
    if Pinecone is None:
        logging.warning("[purge] pinecone client not installed; skip purge")
        return
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logging.warning("[purge] PINECONE_API_KEY not set; skip purge")
        return
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)
    logging.info("[purge] deleting vectors where parent_id=%s in namespace=%s", video_id, namespace)
    idx.delete(namespace=namespace, filter={"parent_id": {"$eq": video_id}})
    logging.info("[purge] delete requested")


def ingest_one(video_id: str, root: Path, *, purge_first: bool, force_enrich: bool, skip_speakers: bool) -> int:
    progress.set_parents_total(1)

    json_path = _find_json_for_video(root, video_id)
    logging.info("[locate] found %s", json_path)

    obj = _fast_json_load(json_path)
    raw_norm = _convert_assemblyai_json_to_raw(obj)
    segs = raw_norm.get("segments") or []
    if not segs:
        logging.error("[ingest] no usable segments in %s", json_path)
        return 2

    duration_s = max((float(s.get("end") or 0.0) for s in segs), default=0.0)
    title = _extract_title_from_dir(json_path)
    channel_name = _extract_channel_from_path(json_path)
    published_at = _extract_date_prefix(title)

    aai_entities_raw = _load_entities_for_json_path(json_path, obj)
    cleaned_entities = postprocess_aai_entities(aai_entities_raw)

    meta: Dict[str, Any] = {
        "video_id": video_id,
        "title": title,
        "description": "",
        "channel_name": channel_name,
        "speaker_primary": None,
        "published_at": published_at,
        "duration_s": duration_s,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": None,
        "language": "en",
        "entities": cleaned_entities,
        "chapters": None,
        "document_type": "youtube_video",
        "source": "youtube",
    }
    raw_for_segmenter = {
        "segments": segs,
        "caption_lines": [],
        "diarization": [],
        "entities": aai_entities_raw,
    }

    if skip_speakers:
        logging.info("[speakers] skipping speaker resolution (--skip-speakers)")
    else:
        try:
            audio_hint = _guess_audio_path(json_path)
            spk = resolve_speakers(meta, raw_for_segmenter, audio_hint_path=audio_hint)
            if spk.get("speaker_map"):
                meta["speaker_map"] = spk["speaker_map"]
            if spk.get("speaker_primary"):
                meta["speaker_primary"] = spk["speaker_primary"]
        except Exception as e:
            logging.warning("[speakers] resolve failed: %s", e)

    try:
        enrich = None
        if not force_enrich:
            enrich = router_cache_load(video_id)
        if enrich is None:
            sents = normalize_to_sentences(raw_for_segmenter)
            enrich = asyncio.run(enrich_parent_router_fields_async(meta, sents))
            try:
                router_cache_save(video_id, enrich)
            except Exception as e:
                logging.warning("[router/cache] save failed: %s", e)
        meta.update({
            "description": enrich.get("description", meta.get("description", "")),
            "topic_summary": enrich.get("topic_summary"),
            "router_tags": enrich.get("router_tags"),
            "aliases": enrich.get("aliases"),
            "canonical_entities": enrich.get("canonical_entities"),
            "is_explainer": enrich.get("is_explainer"),
            "router_boost": enrich.get("router_boost"),
        })
    except Exception as e:
        logging.warning("[router] enrich failed (continuing with defaults): %s", e)

    parent = build_parent({
        "video_id": meta["video_id"],
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "channel_name": meta.get("channel_name"),
        "speaker_primary": meta.get("speaker_primary"),
        "published_at": meta.get("published_at"),
        "duration_s": meta.get("duration_s", 0.0),
        "url": meta.get("url"),
        "thumbnail_url": meta.get("thumbnail_url"),
        "language": meta.get("language", "en"),
        "entities": meta.get("entities", []),
        "chapters": meta.get("chapters"),
        "document_type": "youtube_video",
        "source": "youtube",
        "router_tags": meta.get("router_tags"),
        "aliases": meta.get("aliases"),
        "canonical_entities": meta.get("canonical_entities"),
        "is_explainer": meta.get("is_explainer"),
        "router_boost": meta.get("router_boost"),
        "topic_summary": meta.get("topic_summary"),
        "speaker_map": meta.get("speaker_map"),
    }).dict()

    # --- inside ingest_one(...) just after you build `parent` and before purge/upsert
    index_name, namespace = _resolve_pinecone_target()
    logging.info(
        "[pinecone] target index=%s namespace=%r (env PINECONE_INDEX_NAME=%r, PINECONE_NAMESPACE=%r; defaults idx=%s ns=%r)",
        index_name, namespace,
        os.getenv("PINECONE_INDEX_NAME"), os.getenv("PINECONE_NAMESPACE"),
        settings_v2.PINECONE_INDEX_NAME, settings_v2.NAMESPACE_VIDEOS,
    )

    # (optional) sanity peek: print current namespace keys & count if client available
    try:
        if Pinecone is not None and os.getenv("PINECONE_API_KEY"):
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            idx = pc.Index(index_name)
            stats = idx.describe_index_stats()
            ns_stats = (stats or {}).get("namespaces", {})
            ninfo = ns_stats.get(namespace, {}) or {}
            logging.info("[pinecone] namespaces present: %s", ", ".join(sorted(ns_stats.keys()) or ["<none>"]))
            logging.info("[pinecone] namespace=%r vector_count=%s", namespace, ninfo.get("vector_count", "unknown"))
    except Exception as e:
        logging.warning("[pinecone] stats probe failed: %s", e)

    if purge_first:
        try:
            purge_parent_vectors(
                video_id,
                index_name=os.getenv("PINECONE_INDEX_NAME", settings_v2.PINECONE_INDEX_NAME),
                namespace=os.getenv("PINECONE_NAMESPACE", settings_v2.NAMESPACE_VIDEOS),
            )
        except Exception as e:
            logging.warning("[purge] failed (continuing): %s", e)

    children = build_children_from_raw(parent, raw_for_segmenter)
    if not children:
        logging.info("[ingest] no child segments emitted (text too short or filters)")
        return 3

    stats = upsert_children(children)
    logging.info(
        "[done] video=%s children=%d embed=%.2fs upsert=%.2fs embed_reqs=%d pinecone_batches=%d",
        video_id, len(children), stats["t_embed"], stats["t_upsert"], stats["embed_reqs"], stats["pinecone_batches"]
    )
    return 0


def main(
    video_id: Optional[str] = None,
    root: Optional[str | Path] = None,
    *,
    purge_first: bool = False,
    force_enrich: bool = False,
    skip_speakers: bool = False,
) -> int:
    """
    If video_id and root are provided, runs directly (IDE-friendly).
    Otherwise, falls back to CLI flags for backwards compatibility.
    """
    setup_logger()

    if video_id is not None and root is not None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            logging.error("root does not exist: %s", root_path)
            return 1
        return ingest_one(
            video_id=video_id.strip(),
            root=root_path,
            purge_first=purge_first,
            force_enrich=force_enrich,
            skip_speakers=skip_speakers,
        )

    # CLI fallback
    ap = argparse.ArgumentParser(description="Ingest exactly one YouTube video from local diarized JSON.")
    ap.add_argument("--video-id", required=True, help="11-char YouTube ID (e.g., Bsrh7o_DnGc)")
    ap.add_argument("--root", required=True, help="Root directory containing *_diarized_content.json trees")
    ap.add_argument("--purge-first", action="store_true", help="Delete existing vectors before upsert")
    ap.add_argument("--force-enrich", action="store_true", help="Ignore router cache and re-enrich via LLM")
    ap.add_argument("--skip-speakers", action="store_true", help="Skip cross-video speaker resolution")
    args = ap.parse_args()

    root_path = Path(args.root).expanduser().resolve()
    if not root_path.exists():
        logging.error("root does not exist: %s", root_path)
        return 1

    try:
        return ingest_one(
            video_id=args.video_id.strip(),
            root=root_path,
            purge_first=bool(args.purge_first),
            force_enrich=bool(args.force_enrich),
            skip_speakers=bool(args.skip_speakers),
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 2
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        return 99


if __name__ == "__main__":
    # ── EDIT THESE TWO LINES for IDE run ────────────────────────────────────────
    VIDEO_ID = "Bsrh7o_DnGc"
    ROOT = YOUTUBE_VIDEO_DIRECTORY

    # Optional toggles:
    PURGE_FIRST = True
    FORCE_ENRICH = False
    SKIP_SPEAKERS = False

    sys.exit(
        main(
            video_id=VIDEO_ID,
            root=ROOT,
            purge_first=PURGE_FIRST,
            force_enrich=FORCE_ENRICH,
            skip_speakers=SKIP_SPEAKERS,
        )
    )
