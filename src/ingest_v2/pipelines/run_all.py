# src/ingest_v2/pipelines/run_all.py

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
    enrich_parent_router_fields,
    enrich_parent_router_fields_async,
)
from src.ingest_v2.transcripts.normalize import normalize_to_sentences
from src.ingest_v2.utils.logging import setup_logger

# Reuse your existing constant so we don't duplicate paths.
from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY


# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────

def _snip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")


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
# Helpers: detect trivial/empty AssemblyAI rows
# ────────────────────────────────────────────────────────────────────────────────

def _is_trivial_raw_segment(seg: Dict[str, Any]) -> bool:
    """
    A segment is trivial if it has no text, no words, and no usable start/end.
    Matches the bad file shape:
      [{"text": "", "start": null, "end": null, "words": []}]
    """
    no_text = not (seg.get("text") or "").strip()
    no_words = not seg.get("words")
    start_none = seg.get("start") is None
    end_none = seg.get("end") is None
    return no_text and no_words and start_none and end_none


def _looks_like_single_empty_file(obj: Any) -> bool:
    """Detect the exact 'single empty row' file you showed."""
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
# Accepts top-level list OR {"segments":[...]} and converts ms→s
# ────────────────────────────────────────────────────────────────────────────────

def _convert_assemblyai_json_to_raw(obj: Any) -> Dict[str, Any]:
    """
    Convert AssemblyAI diarized JSON (ms timestamps) to v2 raw format:
      {"segments":[{"start":s,"end":s,"speaker":"S1","text":"...", "words":[...]}]}
    - Filters trivial/empty segments
    - Converts ms→s when start/end are numeric
    """
    if isinstance(obj, dict) and "segments" in obj:
        segs_in = obj["segments"]
    elif isinstance(obj, list):
        segs_in = obj
    else:
        raise ValueError("Expected top-level list or dict with 'segments'")

    out_segments: List[Dict[str, Any]] = []
    for seg in segs_in:
        # Drop trivial rows up front
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
        # If both times are None and no content, skip
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
# Helpers: path → (video_id, title, channel, published_at)
# Assumes folder shape:
# .../diarized_youtube_content_YYYY-MM-DD/@ChannelName/<DATE>_<VIDEOID>_<TITLE>/<same>_diarized_content.json
# ────────────────────────────────────────────────────────────────────────────────

_YTID_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


def _extract_video_id_from_path(path: Path) -> Optional[str]:
    """
    Best-effort extraction of YouTube video_id (11 chars) from filename or directory.
    """
    candidates: List[str] = []
    parts = [path.stem, path.name, path.parent.name]
    for p in parts:
        for m in _YTID_RE.finditer(p):
            candidates.append(m.group(1))
    return candidates[0] if candidates else None


def _extract_title_from_dir(path: Path) -> str:
    """Use the leaf directory name as title if available, else filename without suffix."""
    leaf = path.parent.name.strip()
    if leaf and "_diarized_content" not in leaf:
        return leaf
    return path.stem.replace("_diarized_content", "")


def _extract_channel_from_path(path: Path) -> Optional[str]:
    # parent-of-parent is usually "@Channel"
    maybe = path.parent.parent.name
    return maybe if maybe.startswith("@") else None


def _extract_date_prefix(title_or_dir: str) -> Optional[str]:
    # e.g. "2025-08-04_-pjDPU6q2es_Title" → "2025-08-04"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\b", title_or_dir)
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────────────────
# Iterator over assets in YOUTUBE_VIDEO_DIRECTORY
# ────────────────────────────────────────────────────────────────────────────────

def _iter_youtube_assets_from_fs(
    root_dir: Path, prune_empty: bool = False
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Walk for *_diarized_content.json, normalize, and yield (meta, raw_for_segmenter).
    If prune_empty=True, delete files that are clearly empty/ill-formed.
    """
    for p in root_dir.rglob("*_diarized_content.json"):
        try:
            raw_text = p.read_text(encoding="utf-8")
            obj = json.loads(raw_text)
        except Exception as e:
            logging.warning(f"[v2] skip unreadable JSON: {p} ({e})")
            continue

        # Optional pruning for files that are exactly the bad shape shown
        if prune_empty and _looks_like_single_empty_file(obj):
            try:
                p.unlink(missing_ok=True)
                logging.info(f"[v2] pruned empty AssemblyAI file: {p}")
            except Exception as e:
                logging.warning(f"[v2] failed to prune {p}: {e}")
            continue

        try:
            raw_norm = _convert_assemblyai_json_to_raw(obj)
        except Exception as e:
            logging.warning(f"[v2] skip malformed AssemblyAI JSON: {p} ({e})")
            continue

        segs = raw_norm.get("segments", [])
        if not segs:
            logging.info(f"[v2] no non-trivial segments after normalization: {p}")
            continue

        # Compute duration as the max valid 'end'
        ends: List[float] = []
        for s in segs:
            e = s.get("end")
            if isinstance(e, (int, float)):
                ends.append(float(e))
            # tolerate None / missing

        duration_s = max(ends) if ends else 0.0

        video_id = _extract_video_id_from_path(p) or os.path.basename(p).split("_")[0]
        title = _extract_title_from_dir(p)
        channel_name = _extract_channel_from_path(p)
        published_at = _extract_date_prefix(title)  # yyyy-mm-dd if present

        meta = {
            "video_id": video_id,
            "title": title,
            "description": "",  # will be enriched by router step
            "channel_name": channel_name,
            "speaker_primary": None,
            "published_at": published_at,  # yyyy-mm-dd or None
            "duration_s": duration_s,
            "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
            "thumbnail_url": None,
            "language": "en",
            "entities": [],
            "chapters": None,
            "document_type": "youtube_video",
            "source": "youtube",
        }

        # shape expected by build_children_from_raw()
        raw_for_segmenter = {
            "segments": segs,  # sentence normalization uses words if present
            "caption_lines": [],  # optional
            "diarization": [],  # optional
        }

        yield meta, raw_for_segmenter


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    setup_logger("ingest_v2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    ap.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    ap.add_argument("--root", default=YOUTUBE_VIDEO_DIRECTORY, help="Root dir with diarized JSONs")
    ap.add_argument("--prune-empty", action="store_true", help="Delete obviously empty AssemblyAI files")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("ROUTER_GEN_CONCURRENCY", "6")),
        help="Max concurrent OpenAI enrich calls for cache misses",
    )
    args = ap.parse_args()

    if args.doc_type != "youtube_video":
        logging.info("[v2] 'stream' not wired yet in run_all; nothing to do.")
        return

    root = Path(args.root).expanduser().resolve()
    logging.info(f"[v2] scanning for AssemblyAI JSON under: {root} (prune_empty={args.prune_empty})")

    metas_raw = list(_iter_youtube_assets_from_fs(root, prune_empty=args.prune_empty))
    if not metas_raw:
        logging.info("[v2] no youtube assets found; exiting.")
        return

    # ── Enrich parents with router fields using sidecar cache
    enriched_metas: List[Dict[str, Any]] = []
    enriched = 0
    cached_hits = 0
    failed_enrich = 0

    # Split into cache hits and misses
    to_enrich: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (meta, raw)
    for (meta, raw) in metas_raw:
        pid = meta["video_id"]
        enrich = None
        try:
            enrich = router_cache_load(pid)
        except Exception as e:
            logging.warning(f"[v2/router/cache] read failed for {pid}: {e}")

        if enrich is not None:
            cached_hits += 1

            # Preview logging for cache hits
            n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
            logging.info(
                "[router/cache] vid=%s boost=%s tags=%s desc_preview=%r topic_preview=%r",
                pid,
                enrich.get("router_boost"),
                (enrich.get("router_tags") or [])[:6],
                _snip(enrich.get("description") or "", n),
                _snip(enrich.get("topic_summary") or "", max(80, n // 2)),
            )

            merged = dict(meta)
            merged.update(
                {
                    "description": enrich.get("description", merged.get("description", "")),
                    "topic_summary": enrich.get("topic_summary"),
                    "router_tags": enrich.get("router_tags"),
                    "aliases": enrich.get("aliases"),
                    "canonical_entities": enrich.get("canonical_entities"),
                    "is_explainer": enrich.get("is_explainer"),
                    "router_boost": enrich.get("router_boost"),
                }
            )
            enriched_metas.append(merged)
        else:
            to_enrich.append((meta, raw))

    # Concurrently enrich cache misses
    if to_enrich:
        async def _batch_enrich(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], concurrency: int):
            sem = asyncio.Semaphore(concurrency)

            async def _one(meta: Dict[str, Any], raw: Dict[str, Any]):
                pid = meta["video_id"]
                try:
                    try:
                        sentences = normalize_to_sentences(raw)
                    except Exception:
                        sentences = []
                    async with sem:
                        enrich = await enrich_parent_router_fields_async(meta, sentences)
                    try:
                        router_cache_save(pid, enrich)
                    except Exception as e:
                        logging.warning(f"[v2/router/cache] save failed for {pid}: {e}")
                    return meta, enrich, None
                except Exception as e:
                    return meta, None, e

            tasks = [asyncio.create_task(_one(meta, raw)) for (meta, raw) in pairs]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_batch_enrich(to_enrich, args.concurrency))

        for meta, enrich, err in results:
            pid = meta["video_id"]
            if err or enrich is None:
                logging.warning(f"[v2/router] enrichment failed for {pid}: {err}")
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
            merged.update(
                {
                    "description": enrich.get("description", merged.get("description", "")),
                    "topic_summary": enrich.get("topic_summary"),
                    "router_tags": enrich.get("router_tags"),
                    "aliases": enrich.get("aliases"),
                    "canonical_entities": enrich.get("canonical_entities"),
                    "is_explainer": enrich.get("is_explainer"),
                    "router_boost": enrich.get("router_boost"),
                }
            )
            enriched_metas.append(merged)

    logging.info(
        f"[v2/router] total={len(enriched_metas)} cached_hits={cached_hits} "
        f"fresh_enriched={enriched} failed={failed_enrich}"
    )

    # Build parents (one per asset) with enriched metadata
    parents = run_build_parents(
        metas=[
            {
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
                # Router fields (ParentNode schema has them)
                "router_tags": m.get("router_tags"),
                "aliases": m.get("aliases"),
                "canonical_entities": m.get("canonical_entities"),
                "is_explainer": m.get("is_explainer"),
                "router_boost": m.get("router_boost"),
                "topic_summary": m.get("topic_summary"),
            }
            for m in enriched_metas
        ]
    )

    parents_map = {p.parent_id: p.dict() for p in parents}

    # Build children & upsert
    total_children = 0
    missing_parents = 0
    skipped_empty = 0

    for (meta, raw) in metas_raw:
        pid = meta["video_id"]
        parent = parents_map.get(pid)
        if not parent:
            logging.warning(f"[v2] missing parent for video_id={pid}; skipping")
            missing_parents += 1
            continue

        children = build_children_from_raw(parent, raw)
        if not children:
            logging.info(f"[v2] no children emitted for {pid} ({meta.get('title')})")
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


if __name__ == "__main__":
    main()
