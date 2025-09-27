from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile
import os

from ..configs.settings import settings_v2

def _cache_dir() -> Path:
    p = Path(settings_v2.ROUTER_CACHE_DIR).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def cache_path(parent_id: str) -> Path:
    # one file per parent
    return _cache_dir() / f"{parent_id}.json"

def load(parent_id: str) -> Optional[Dict[str, Any]]:
    p = cache_path(parent_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save(parent_id: str, data: Dict[str, Any]) -> Path:
    p = cache_path(parent_id)
    tmp_dir = p.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # atomic-ish write
    with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    os.replace(tmp, p)
    return p
