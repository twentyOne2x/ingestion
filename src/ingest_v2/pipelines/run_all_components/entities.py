from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from src.ingest_v2.configs.settings import settings_v2


def fast_json_load(path: Path) -> Any:
    """
    Prefer orjson for speed; fall back to the stdlib json loader.
    """
    data = path.read_bytes()
    try:
        import orjson  # type: ignore

        return orjson.loads(data)
    except Exception:
        return json.loads(data.decode("utf-8"))


def entities_cache_dir() -> Path:
    cache_path = Path(settings_v2.ENTITIES_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def entities_sidecar_path(json_path: Path) -> Path:
    stem = json_path.stem.replace("_diarized_content", "")
    return json_path.with_name(f"{stem}_entities.json")


def entities_cache_key(json_path: Path, obj: Any) -> str:
    """
    Stable cache key based on sidecar (if present) or main JSON mtime/size.
    Include FAST flag so caches don’t cross-pollute modes.
    """
    fast_flag = "fast" if os.getenv("ENTITIES_FAST", "0").lower() in ("1", "true", "yes", "y") else "full"
    sidecar = entities_sidecar_path(json_path)
    if sidecar.exists():
        st = sidecar.stat()
        base = f"sc::{sidecar.as_posix()}::{st.st_mtime_ns}::{st.st_size}::{fast_flag}"
    else:
        st = json_path.stat()
        count = 0
        if isinstance(obj, dict) and isinstance(obj.get("entities"), list):
            count = len(obj["entities"])
        base = f"tl::{json_path.as_posix()}::{st.st_mtime_ns}::{st.st_size}::n{count}::{fast_flag}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def postprocess_entities_with_cache(raw_entities: List[Dict[str, Any]], cache_key: str) -> List[str]:
    cache_file = entities_cache_dir() / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return fast_json_load(cache_file)
        except Exception:
            pass

    from src.ingest_v2.entities.postprocess import postprocess_aai_entities

    cleaned = postprocess_aai_entities(raw_entities)
    try:
        cache_file.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logging.debug("[entities/cache] failed to write cache %s: %s", cache_file, exc)
    return cleaned


def load_entities_for_json_path(content_path: Path, obj: Any) -> List[Dict[str, Any]]:
    """
    Prefer a sibling '*_entities.json'. Fallback to top-level obj['entities'].
    Accepts:
      - list[dict{text, entity_type, ...}] (AAI style)
      - list[str]                           (wrapped into dicts)
      - dict with key 'entities'            (the list itself)
    """

    def _normalize(payload) -> List[Dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
            payload = payload["entities"]
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return payload
            if payload and isinstance(payload[0], str):
                return [{"text": text, "entity_type": "custom"} for text in payload if isinstance(text, str)]
            return []
        return []

    sidecar = entities_sidecar_path(content_path)
    try:
        if sidecar.exists():
            raw = fast_json_load(sidecar)
            entities = _normalize(raw)
            logging.info("[v2/entities] using sidecar for %s (%d items)", content_path.name, len(entities))
            return entities
    except Exception as exc:
        logging.warning("[v2/entities] sidecar read failed %s: %s", sidecar, exc)

    top_payload = obj.get("entities") if isinstance(obj, dict) else None
    entities = _normalize(top_payload)
    if entities:
        logging.info("[v2/entities] using top-level 'entities' for %s (%d items)", content_path.name, len(entities))
    else:
        logging.info("[v2/entities] no entities for %s", content_path.name)
    return entities
