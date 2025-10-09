from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


_DATE_ID_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<id>[A-Za-z0-9_-]{11})_")
_MULTI_US_RE = re.compile(r"_{1,3}([A-Za-z0-9_-]{11})_")
_TOKEN_11_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_video_id_from_path(path: Path) -> Optional[str]:
    fragments = [
        path.name,
        path.stem,
        path.as_posix(),
        path.parent.name,
        getattr(path.parent, "stem", path.parent.name),
        path.parent.parent.name if path.parent and path.parent.parent else "",
    ]
    for fragment in fragments:
        match = _DATE_ID_RE.search(fragment)
        if match:
            return match.group("id")
    for fragment in fragments:
        match = _MULTI_US_RE.search(fragment)
        if match:
            return match.group(1)
    for fragment in fragments:
        match = _TOKEN_11_RE.search(fragment)
        if match:
            return match.group(0)
    return None


def extract_title_from_path(path: Path) -> str:
    leaf = path.parent.name.strip()
    if leaf and "_diarized_content" not in leaf:
        return leaf
    return path.stem.replace("_diarized_content", "")


def extract_channel_from_path(path: Path) -> Optional[str]:
    return path.parent.parent.name


def extract_date_prefix(title_or_dir: str) -> Optional[str]:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})\b", title_or_dir)
    return match.group(1) if match else None
