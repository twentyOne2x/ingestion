from __future__ import annotations

import concurrent.futures
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import assemblyai
from .entities import (
    entities_cache_key,
    fast_json_load,
    load_entities_for_json_path,
    postprocess_entities_with_cache,
)
from .extractors import (
    extract_channel_from_path,
    extract_date_prefix,
    extract_title_from_path,
    extract_video_id_from_path,
)
from .text import ascii_safe
from src.utils.global_thread_guard import get_global_thread_limiter


def _iter_paths(root_dir: Path) -> List[Path]:
    return list(root_dir.rglob("*_diarized_content.json"))


def iter_youtube_assets_from_fs(
    root_dir: Path,
    prune_empty: bool = False,
    allowed_channels: Optional[Iterable[str]] = None,
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
    """
    Yield (meta, raw_for_segmenter, json_path) for each AssemblyAI diarized JSON.
    """
    paths = _iter_paths(root_dir)
    if not paths:
        return []

    channel_filter: Optional[set[str]] = None
    if allowed_channels:
        channel_filter = {ch.strip() for ch in allowed_channels if ch and ch.strip()}
        if channel_filter:
            paths = [
                path for path in paths
                if extract_channel_from_path(path) in channel_filter
            ]

    total = len(paths)
    if total == 0:
        logging.info("[v2/entities] no AssemblyAI JSON files matched the requested channels.")
        return []

    logging.info(
        "[v2/entities] scanning %d JSON file(s)%s",
        total,
        f" across {len(channel_filter)} channel(s)" if channel_filter else "",
    )

    cpu = os.cpu_count() or 4
    workers_env = os.getenv("ENTITIES_WORKERS", "").strip()
    workers = int(workers_env) if workers_env.isdigit() and int(workers_env) > 0 else max(1, math.floor(0.7 * cpu))

    def _process_one(path: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
        try:
            obj = fast_json_load(path)
        except Exception:
            logging.warning("[v2] skip unreadable JSON: %s", path)
            return None

        if prune_empty and assemblyai.looks_like_single_empty_file(obj):
            try:
                path.unlink(missing_ok=True)
                logging.info("[v2] pruned empty AssemblyAI file: %s", path)
            except Exception as exc:
                logging.warning("[v2] failed to prune %s: %s", path, ascii_safe(str(exc)))
            return None

        try:
            raw_norm = assemblyai.convert_assemblyai_json_to_raw(obj)
        except Exception as exc:
            logging.warning("[v2] skip malformed AssemblyAI JSON: %s (%s)", path, ascii_safe(str(exc)))
            return None

        segments = raw_norm.get("segments", [])
        if not segments:
            logging.info("[v2] no non-trivial segments after normalization: %s", path)
            return None

        ends = [float(segment.get("end")) for segment in segments if isinstance(segment.get("end"), (int, float))]
        duration_s = max(ends) if ends else 0.0

        video_id = extract_video_id_from_path(path)
        if not video_id:
            logging.warning("[v2] SKIP: could not find YouTube video_id in path=%s", path)
            return None

        title = extract_title_from_path(path)
        channel_name = extract_channel_from_path(path)
        published_at = extract_date_prefix(title)

        aai_entities_raw = load_entities_for_json_path(path, obj)

        try:
            if os.getenv("ENTITIES_FAST", "0").lower() in ("1", "true", "yes", "y"):
                cap = int(os.getenv("ENTITIES_MAX_RAW", "400"))
                if cap > 0 and len(aai_entities_raw) > cap:
                    aai_entities_raw = aai_entities_raw[:cap]
        except Exception:
            pass

        cache_key = entities_cache_key(path, obj)
        cleaned_entities = postprocess_entities_with_cache(aai_entities_raw, cache_key)
        logging.debug("[v2/entities] vid=%s cleaned=%d sample=%s", video_id, len(cleaned_entities), cleaned_entities[:8])

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
            "entities": cleaned_entities,
            "chapters": None,
            "document_type": "youtube_video",
            "source": "youtube",
        }
        raw_for_segmenter = {
            "segments": segments,
            "caption_lines": [],
            "diarization": [],
            "entities": aai_entities_raw,
        }
        return meta, raw_for_segmenter, path

    results: List[Tuple[Dict[str, Any], Dict[str, Any], Path]] = []
    limiter = get_global_thread_limiter()
    with limiter.claim(workers, label="assets-entities"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, path) for path in paths]
            processed = 0
            log_every = max(1, total // 20)  # ~5% increments
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                if item is not None:
                    results.append(item)
                processed += 1
                if processed % log_every == 0 or processed == total:
                    remaining = total - processed
                    logging.info(
                        "[v2/entities] progress %d/%d (%.1f%%) remaining=%d",
                        processed,
                        total,
                        (processed / total) * 100.0,
                        remaining,
                    )
    return results
