from typing import Dict, Any, Optional
import json
from pathlib import Path
from ..utils.ids import sha1_hex
from ..schemas.parent import ParentNode
from ..configs.settings import settings_v2

def _safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default

def build_parent_from_metadata(meta: Dict[str, Any]) -> ParentNode:
    raw_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    source_hash = sha1_hex(raw_bytes)
    parent = ParentNode(
        parent_id=meta["video_id"],
        document_type="youtube_video",
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        channel_name=meta.get("channel_name"),
        speaker_primary=_safe_get(meta, "speaker_primary"),
        published_at=_safe_get(meta, "published_at"),
        start_ts=None,
        end_ts=None,
        duration_s=float(meta.get("duration_s", meta.get("duration", 0)) or 0),
        url=meta["url"],
        thumbnail_url=meta.get("thumbnail_url"),
        language=meta.get("language", "en"),
        entities=meta.get("entities", []),
        chapters=meta.get("chapters"),
        rights=settings_v2.RIGHTS_DEFAULT,
        ingest_version=2,
        source="youtube",
        source_hash=source_hash,
        router_tags=meta.get("router_tags"),
        aliases=meta.get("aliases"),
        canonical_entities=meta.get("canonical_entities"),
        is_explainer=meta.get("is_explainer"),
        router_boost=meta.get("router_boost"),
        topic_summary=meta.get("topic_summary"),
    )
    return parent

def load_existing_youtube_raw(parent_id: str) -> Optional[Path]:
    p = Path("transcripts/raw") / f"{parent_id}_raw.json"
    return p if p.exists() else None
