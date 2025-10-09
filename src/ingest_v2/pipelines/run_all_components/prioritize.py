from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.ingest_v2.speakers.name_filters import looks_like_person


Meta = Dict[str, object]
Raw = Dict[str, object]
Asset = Tuple[Meta, Raw, Path]


def prioritize_assets(
    assets: List[Asset],
    deprioritize_channels: Optional[List[str]] = None,
    push_to_end_regexes: Optional[List[str]] = None,
) -> List[Asset]:
    deprio_set = set(deprioritize_channels or [])
    regex_objs = [re.compile(pattern) for pattern in (push_to_end_regexes or [])]

    def _is_deprioritized(channel: Optional[str]) -> bool:
        name = (channel or "").strip()
        if name in deprio_set:
            return True
        return any(pattern.search(name) for pattern in regex_objs)

    def _sort_key(item: Asset):
        meta, _raw, _path = item
        channel = meta.get("channel_name")
        date = meta.get("published_at") or "0000-00-00"
        video_id = meta.get("video_id") or ""
        return (_is_deprioritized(channel), date, video_id)

    return sorted(assets, key=_sort_key)


def filter_speaker_map_people(
    speaker_map: Dict[str, Dict[str, object]],
    keep_keys: set[str],
    host_names=(),
) -> Dict[str, Dict[str, object]]:
    filtered: Dict[str, Dict[str, object]] = {}
    for key, info in (speaker_map or {}).items():
        if key in keep_keys:
            filtered[key] = info
            continue
        name = (info.get("name") or "").strip()
        if name and looks_like_person(name, host_names=host_names):
            filtered[key] = info
    return filtered
