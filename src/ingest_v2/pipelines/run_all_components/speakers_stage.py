from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.ingest_v2.speakers.name_filters import filter_to_people, normalize_alias
from src.ingest_v2.speakers.resolve import resolve_speakers
from src.utils.global_thread_guard import get_global_thread_limiter

from .prioritize import Asset, filter_speaker_map_people
from .text import ascii_safe


def guess_audio_path(json_path: Path) -> Optional[Path]:
    stem = json_path.stem.replace("_diarized_content", "")
    for ext in (".mp3", ".wav", ".m4a"):
        candidate = json_path.with_name(f"{stem}{ext}")
        if candidate.exists():
            return candidate
    for ext in (".mp3", ".wav", ".m4a"):
        candidate = json_path.parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _resolve_single(item: Asset) -> Tuple[Asset, float]:
    meta, raw, json_path = item
    start = time.perf_counter()
    audio_path = guess_audio_path(json_path)
    try:
        speakers = resolve_speakers(meta, raw, audio_hint_path=audio_path)
    except Exception as exc:
        logging.warning("[v2/speakers] resolve failed vid=%s: %s", meta.get("video_id"), ascii_safe(str(exc)))
        speakers = {}
    elapsed = time.perf_counter() - start

    if speakers.get("speaker_map"):
        meta["speaker_map"] = speakers["speaker_map"]
    if speakers.get("speaker_primary"):
        meta["speaker_primary"] = speakers["speaker_primary"]

    if meta.get("speaker_primary"):
        meta["speaker_primary"] = normalize_alias(meta["speaker_primary"])
    primary = meta.get("speaker_primary")

    speaker_map = meta.get("speaker_map") or {}
    keep = {primary} if primary else set()
    speaker_map = filter_speaker_map_people(speaker_map, keep_keys=keep, host_names=[primary or ""])
    meta["speaker_map"] = speaker_map

    video_id = meta.get("video_id")
    host_info = speaker_map.get(primary or "", {}) if primary else {}
    host_name = (host_info.get("name") or "").strip() or "Host"

    guest_names = []
    for label, info in speaker_map.items():
        if label == primary:
            continue
        name = (info.get("name") or "").strip()
        if name:
            guest_names.append(normalize_alias(name))
    guest_names = filter_to_people(guest_names, host_names=[primary or ""])

    guests = []
    for label, info in speaker_map.items():
        name = (info.get("name") or "").strip()
        if not name:
            continue
        if normalize_alias(name) in guest_names:
            guests.append(f"{name}({info.get('confidence', 0.0):.2f},{info.get('source', '')})")

    logging.info("[v2/speakers] vid=%s resolved in %.2fs host=%r guests=%s", video_id, elapsed, host_name, guests or "[]")
    return (meta, raw, json_path), elapsed


def resolve_speakers_for_assets(
    assets: Iterable[Asset],
    workers: Optional[int] = None,
) -> Tuple[List[Asset], float]:
    asset_list = list(assets)
    if not asset_list:
        return [], 0.0

    cpu = os.cpu_count() or 4
    pool_workers = workers or max(1, math.floor(0.7 * cpu))
    logging.info("[v2/speakers] resolving across %d assets with %d workers", len(asset_list), pool_workers)

    resolved_assets: List[Asset] = []
    total_time = 0.0
    limiter = get_global_thread_limiter()
    with limiter.claim(pool_workers, label="speakers-stage"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_workers) as pool:
            futures = [pool.submit(_resolve_single, item) for item in asset_list]
            for future in concurrent.futures.as_completed(futures):
                resolved, elapsed = future.result()
                resolved_assets.append(resolved)
                total_time += elapsed

    return resolved_assets, total_time
