import os
from dataclasses import dataclass

@dataclass(frozen=True)
class SettingsV2:
    SEGMENT_MIN_S: float = float(os.getenv("SEGMENT_MIN_S", 15))
    SEGMENT_MAX_S: float = float(os.getenv("SEGMENT_MAX_S", 60))
    SEGMENT_STRIDE_S: float = float(os.getenv("SEGMENT_STRIDE_S", 5))
    SEGMENT_PAD_S: float = float(os.getenv("SEGMENT_PAD_S", 1.5))
    MIN_TEXT_CHARS: int = int(os.getenv("MIN_TEXT_CHARS", 160))

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1536"))
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")  # openai|sentence-transformers

    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "icmfyi")
    NAMESPACE_VIDEOS: str = os.getenv("PINECONE_NAMESPACE_VIDEOS", "videos")
    NAMESPACE_STREAMS: str = os.getenv("PINECONE_NAMESPACE_STREAMS", "streams")

    MAX_METADATA_BYTES: int = int(os.getenv("MAX_METADATA_BYTES", 12000))
    RIGHTS_DEFAULT: str = os.getenv("RIGHTS_DEFAULT", "public_reference_only")

    BACKFILL_DAYS: int = int(os.getenv("BACKFILL_DAYS", 60))

settings_v2 = SettingsV2()
