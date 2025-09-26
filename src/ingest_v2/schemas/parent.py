from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video", "stream"]

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
    speaker_primary: Optional[str] = None
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
    source: Literal["youtube", "pumpfun"] = "youtube"
    source_hash: str

    class Config:
        extra = "ignore"
