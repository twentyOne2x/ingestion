from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video", "stream"]

class ChildNode(BaseModel):
    node_type: Literal["child"] = "child"
    segment_id: str
    parent_id: str
    document_type: DocType
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    start_hms: str
    end_hms: str
    clip_url: Optional[HttpUrl] = None
    speaker: Optional[str] = None
    entities: List[str] = []
    chapter: Optional[str] = None
    language: Optional[str] = "en"
    confidence_asr: Optional[float] = Field(default=None, ge=0, le=1)
    has_music: Optional[bool] = False
    flags: List[str] = []
    rights: str = "public_reference_only"
    ingest_version: int = 2
    source_hash: str

    class Config:
        extra = "ignore"
