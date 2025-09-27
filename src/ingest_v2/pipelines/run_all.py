import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ingest_v2.configs.settings import settings_v2
from src.ingest_v2.pipelines.build_children import build_children_from_raw
from src.ingest_v2.pipelines.build_parents import run_build_parents
from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
from src.ingest_v2.router.cache import load as router_cache_load
from src.ingest_v2.router.cache import save as router_cache_save
from src.ingest_v2.router.enrich_parent import (
    enrich_parent_router_fields_async,
)
from src.ingest_v2.speakers.resolve import resolve_speakers
from src.ingest_v2.transcripts.normalize import normalize_to_sentences
from src.ingest_v2.utils.logging import setup_logger

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY

# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────

def _snip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")

def _ascii(s: str, max_len: int = 400) -> str:
    s = _snip(s or "", max_len)
    try:
        return s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return s

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _ms_to_s(x: Any) -> Optional[float]:
    val = _safe_float(x)
    return None if val is None else (val / 1000.0)

# ────────────────────────────────────────────────────────────────────────────────
# Detect trivial/empty rows
# ────────────────────────────────────────────────────────────────────────────────

def _is_trivial_raw_segment(seg: Dict[str, Any]) -> bool:
    no_text = not (seg.get("text") or "").strip()
    no_words = not seg.get("words")
    start_none = seg.get("start") is None
    end_none = seg.get("end") is None
    return no_text and no_words and start_none and end_none

def _looks_like_single_empty_file(obj: Any) -> bool:
    try:
        if not isinstance(obj, list) or len(obj) != 1:
            return False
        seg = obj[0]
        return (
            isinstance(seg, dict)
            and (seg.get("text") in ("", None))
            and seg.get("start") is None
            and seg.get("end") is None
            and isinstance(seg.get("words"), list)
            and len(seg["words"]) == 0
        )
    except Exception:
        return False

# ────────────────────────────────────────────────────────────────────────────────
# AssemblyAI → v2 raw normalizer
# ────────────────────────────────────────────────────────────────────────────────

def _convert_assemblyai_json_to_raw(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "segments" in obj:
        segs_in = obj["segments"]
    elif isinstance(obj, list):
        segs_in = obj
    else:
        raise ValueError("Expected top-level list or dict with 'segments'")

    out_segments: List[Dict[str, Any]] = []
    for seg in segs_in:
        if _is_trivial_raw_segment(seg):
            continue

        words_in = seg.get("words")
        out_words = None
        if isinstance(words_in, list) and words_in:
            out_words = [
                {
                    "text": (w.get("text") or "").strip(),
                    "start": _ms_to_s(w.get("start")),
                    "end": _ms_to_s(w.get("end")),
                    "speaker": w.get("speaker") or seg.get("speaker") or "S1",
                }
                for w in words_in
                if (w.get("text") or "").strip()
            ]

        start_s = _ms_to_s(seg.get("start"))
        end_s = _ms_to_s(seg.get("end"))

        text = (seg.get("text") or "").strip()
        if start_s is None and end_s is None and not text and not out_words:
            continue

        out_segments.append(
            {
                "start": start_s,
                "end": end_s,
                "speaker": seg.get("speaker") or seg.get("speaker_label") or "S1",
                "text": text,
                **({"words": out_words} if out_words is not None else {}),
            }
        )

    return {"segments": out_segments}

# ────────────────────────────────────────────────────────────────────────────────
# Robust YouTube ID extraction
# ────────────────────────────────────────────────────────────────────────────────
# We expect <YYYY-MM-DD>_<VIDEOID>_<TITLE>, and IDs can contain underscores/hyphens.
# We handle: single/double/triple underscores, anywhere in the component; if all else
# fails, we pick the first 11-char token.
_DATE_ID_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<id>[A-Za-z0-9_-]{11})_")
_MULTI_US_RE = re.compile(r"_{1,3}([A-Za-z0-9_-]{11})_")
_TOKEN_11_RE = re.compile(r"[A-Za-z0-9_-]{11}")

def _extract_video_id_from_path(path: Path) -> Optional[str]:
    pieces = [
        path.name, path.stem, path.as_posix(),
        path.parent.name, getattr(path.parent, "stem", path.parent.name),
        path.parent.parent.name if path.parent and path.parent.parent else "",
    ]
    # 1) <date>_<id>_
    for s in pieces:
        m = _DATE_ID_RE.search(s)
        if m:
            return m.group("id")
    # 2) _{1..3}<id>_
    for s in pieces:
        m = _MULTI_US_RE.search(s)
        if m:
            return m.group(1)
    # 3) first 11-char token
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

# ────────────────────────────────────────────────────────────────────────────────
# Iterator
# ────────────────────────────────────────────────────────────────────────────────

def _iter_youtube_assets_from_fs(
    root_dir: Path, prune_empty: bool = False
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
    """
    Yield (meta, raw_for_segmenter, json_path).
    """
    for p in root_dir.rglob("*_diarized_content.json"):
        try:
            raw_text = p.read_text(encoding="utf-8")
            obj = json.loads(raw_text)
        except Exception as e:
            logging.warning(f"[v2] skip unreadable JSON: {p} ({_ascii(str(e))})")
            continue

        if prune_empty and _looks_like_single_empty_file(obj):
            try:
                p.unlink(missing_ok=True)
                logging.info(f"[v2] pruned empty AssemblyAI file: {p}")
            except Exception as e:
                logging.warning(f"[v2] failed to prune {p}: {_ascii(str(e))}")
            continue

        try:
            raw_norm = _convert_assemblyai_json_to_raw(obj)
        except Exception as e:
            logging.warning(f"[v2] skip malformed AssemblyAI JSON: {p} ({_ascii(str(e))})")
            continue

        segs = raw_norm.get("segments", [])
        if not segs:
            logging.info(f"[v2] no non-trivial segments after normalization: {p}")
            continue

        ends: List[float] = []
        for s in segs:
            e = s.get("end")
            if isinstance(e, (int, float)):
                ends.append(float(e))
        duration_s = max(ends) if ends else 0.0

        video_id = _extract_video_id_from_path(p)
        if not video_id:
            logging.warning(f"[v2] SKIP: could not find YouTube video_id in path={p}")
            continue

        title = _extract_title_from_dir(p)
        channel_name = _extract_channel_from_path(p)
        published_at = _extract_date_prefix(title)

        meta = {
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
            "entities": [],
            "chapters": None,
            "document_type": "youtube_video",
            "source": "youtube",
        }

        raw_for_segmenter = {
            "segments": segs,
            "caption_lines": [],
            "diarization": [],
        }

        yield meta, raw_for_segmenter, p

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

async def _enrich_async(meta: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sentences = normalize_to_sentences(raw)
    except Exception:
        sentences = []
    return await enrich_parent_router_fields_async(meta, sentences)

def main():
    setup_logger("ingest_v2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    ap.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    ap.add_argument("--root", default=YOUTUBE_VIDEO_DIRECTORY, help="Root dir with diarized JSONs")
    ap.add_argument("--prune-empty", action="store_true", help="Delete obviously empty AssemblyAI files")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("ROUTER_GEN_CONCURRENCY", "6")))
    args = ap.parse_args()

    if args.doc_type != "youtube_video":
        logging.info("[v2] 'stream' not wired yet in run_all; nothing to do.")
        return

    root = Path(args.root).expanduser().resolve()
    logging.info(f"[v2] scanning for AssemblyAI JSON under: {root} (prune_empty={args.prune_empty})")

    metas_raw_paths = list(_iter_youtube_assets_from_fs(root, prune_empty=args.prune_empty))
    if not metas_raw_paths:
        logging.info("[v2] no youtube assets found; exiting.")
        return

    # Resolve speakers Tier1+Tier2 in-process (fast)
    metas_raw_paths2 = []
    for (meta, raw, json_path) in metas_raw_paths:
        # Optional audio search next to JSON (same stem .mp3/.wav)
        audio_path = _guess_audio_path(json_path)
        spk = resolve_speakers(meta, raw, audio_hint_path=audio_path)
        if spk.get("speaker_map"):
            meta["speaker_map"] = spk["speaker_map"]
        if spk.get("speaker_primary"):
            meta["speaker_primary"] = spk["speaker_primary"]
        metas_raw_paths2.append((meta, raw, json_path))

    # Enrich via OpenAI with cache (async batching)
    enriched_metas: List[Dict[str, Any]] = []
    enriched = 0
    cached_hits = 0
    failed_enrich = 0

    to_enrich: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for (meta, raw, _p) in metas_raw_paths2:
        pid = meta["video_id"]
        enrich = None
        try:
            enrich = router_cache_load(pid)
        except Exception as e:
            logging.warning(f"[v2/router/cache] read failed for {pid}: {_ascii(str(e))}")

        if enrich is not None:
            cached_hits += 1
            n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
            logging.info(
                "[router/cache] vid=%s boost=%s tags=%s desc_preview=%r topic_preview=%r",
                pid,
                enrich.get("router_boost"),
                (enrich.get("router_tags") or [])[:6],
                _ascii(_snip(enrich.get("description") or "", n)),
                _ascii(_snip(enrich.get("topic_summary") or "", max(80, n // 2))),
            )
            merged = dict(meta)
            merged.update({
                "description": enrich.get("description", merged.get("description", "")),
                "topic_summary": enrich.get("topic_summary"),
                "router_tags": enrich.get("router_tags"),
                "aliases": enrich.get("aliases"),
                "canonical_entities": enrich.get("canonical_entities"),
                "is_explainer": enrich.get("is_explainer"),
                "router_boost": enrich.get("router_boost"),
            })
            enriched_metas.append(merged)
        else:
            to_enrich.append((meta, raw))

    if to_enrich:
        async def _batch_enrich(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], concurrency: int):
            sem = asyncio.Semaphore(concurrency)
            async def _one(meta: Dict[str, Any], raw: Dict[str, Any]):
                pid = meta["video_id"]
                try:
                    async with sem:
                        enrich = await _enrich_async(meta, raw)
                    try:
                        router_cache_save(pid, enrich)
                    except Exception as e:
                        logging.warning(f"[v2/router/cache] save failed for {pid}: {_ascii(str(e))}")
                    return meta, enrich, None
                except Exception as e:
                    return meta, None, e
            tasks = [asyncio.create_task(_one(m,r)) for (m,r) in pairs]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_batch_enrich(to_enrich, args.concurrency))
        for meta, enrich, err in results:
            pid = meta["video_id"]
            if err or enrich is None:
                logging.warning(f"[v2/router] enrichment failed for {pid}: {_ascii(str(err))}")
                failed_enrich += 1
                enrich = {
                    "description": meta.get("description") or "",
                    "topic_summary": "",
                    "router_tags": [],
                    "aliases": [],
                    "canonical_entities": [],
                    "is_explainer": False,
                    "router_boost": 1.0,
                }
            else:
                enriched += 1

            merged = dict(meta)
            merged.update({
                "description": enrich.get("description", merged.get("description", "")),
                "topic_summary": enrich.get("topic_summary"),
                "router_tags": enrich.get("router_tags"),
                "aliases": enrich.get("aliases"),
                "canonical_entities": enrich.get("canonical_entities"),
                "is_explainer": enrich.get("is_explainer"),
                "router_boost": enrich.get("router_boost"),
            })
            enriched_metas.append(merged)

    logging.info(
        f"[v2/router] total={len(enriched_metas)} cached_hits={cached_hits} "
        f"fresh_enriched={enriched} failed={failed_enrich}"
    )

    # Build parents (note: we pass through speaker fields)
    parents = run_build_parents(metas=[{
        "video_id": m["video_id"],
        "title": m.get("title", ""),
        "description": m.get("description", ""),
        "channel_name": m.get("channel_name"),
        "speaker_primary": m.get("speaker_primary"),
        "published_at": m.get("published_at"),
        "duration_s": m.get("duration_s", 0.0),
        "url": m.get("url"),
        "thumbnail_url": m.get("thumbnail_url"),
        "language": m.get("language", "en"),
        "entities": m.get("entities", []),
        "chapters": m.get("chapters"),
        "document_type": "youtube_video",
        "source": "youtube",
        "router_tags": m.get("router_tags"),
        "aliases": m.get("aliases"),
        "canonical_entities": m.get("canonical_entities"),
        "is_explainer": m.get("is_explainer"),
        "router_boost": m.get("router_boost"),
        "topic_summary": m.get("topic_summary"),
        # speakers
        "speaker_map": m.get("speaker_map"),
    } for m in enriched_metas])

    parents_map = {p.parent_id: p.dict() for p in parents}

    total_children = 0
    missing_parents = 0
    skipped_empty = 0

    for (meta, raw, _path) in metas_raw_paths2:
        pid = meta["video_id"]
        parent = parents_map.get(pid)
        if not parent:
            logging.warning(f"[v2] missing parent for video_id={pid}; skipping")
            missing_parents += 1
            continue

        # You can modify build_children_from_raw to use parent["speaker_map"] to relabel speakers if desired.
        children = build_children_from_raw(parent, raw)
        if not children:
            logging.info(f"[v2] no children emitted for {pid} ({_ascii(meta.get('title') or '')})")
            skipped_empty += 1
            continue

        upsert_children(children)
        total_children += len(children)

    logging.info(
        f"[ingest_v2] finished upserting {total_children} child segments "
        f"from {len(parents)} parents "
        f"(missing_parents={missing_parents}, skipped_empty={skipped_empty}, "
        f"router_cached={cached_hits}, router_enriched={enriched}, router_failed={failed_enrich})."
    )

def _guess_audio_path(json_path: Path) -> Optional[Path]:
    # Look for a sibling .mp3/.wav using the same stem
    stem = json_path.stem.replace("_diarized_content", "")
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.with_name(f"{stem}{ext}")
        if cand.exists():
            return cand
    # or parent dir with same basename
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

if __name__ == "__main__":
    main()
