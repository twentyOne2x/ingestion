from pydantic import BaseModel, Field, HttpUrl
from typing import Any, Dict, List, Optional, Literal

DocType = Literal["youtube_video", "twitch_vod", "pumpfun_clip", "stream", "media"]

class Chapter(BaseModel):
    title: str
    start_s: float = Field(ge=0)

class ParentNode(BaseModel):
    node_type: Literal["parent"] = "parent"
    parent_id: str
    document_type: DocType
    title: str
    description: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    speaker_primary: Optional[str] = None
    speaker_names: Optional[List[str]] = None
    speaker_map: Optional[Dict[str, Dict[str, Any]]] = None
    published_at: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    duration_s: float = 0
    url: HttpUrl
    thumbnail_url: Optional[HttpUrl] = None
    language: Optional[str] = "en"
    entities: List[str] = []
    chapters: Optional[List[Chapter]] = None
    rights: str = "public_reference_only"
    ingest_version: int = 2
    source: Literal["youtube", "pumpfun", "twitch"] = "youtube"
    source_hash: str
    # add under other fields in ParentNode
    router_tags: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    canonical_entities: Optional[List[str]] = None
    is_explainer: Optional[bool] = None
    router_boost: Optional[float] = None
    topic_summary: Optional[str] = None

    class Config:
        extra = "ignore"
