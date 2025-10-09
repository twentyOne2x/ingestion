import os
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PIPELINE_STORAGE = _REPO_ROOT / "pipeline_storage_v2"

@dataclass(frozen=True)
class SettingsV2:
    SEGMENT_MIN_S: float = float(os.getenv("SEGMENT_MIN_S", 15))
    SEGMENT_MAX_S: float = float(os.getenv("SEGMENT_MAX_S", 60))
    SEGMENT_STRIDE_S: float = float(os.getenv("SEGMENT_STRIDE_S", 5))
    SEGMENT_PAD_S: float = float(os.getenv("SEGMENT_PAD_S", 1.5))
    MIN_TEXT_CHARS: int = int(os.getenv("MIN_TEXT_CHARS", 160))

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "3072"))
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")  # openai|sentence-transformers

    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    NAMESPACE_VIDEOS: str = os.getenv("PINECONE_NAMESPACE_VIDEOS", "videos")
    NAMESPACE_STREAMS: str = os.getenv("PINECONE_NAMESPACE_STREAMS", "streams")

    MAX_METADATA_BYTES: int = int(os.getenv("MAX_METADATA_BYTES", 12000))
    RIGHTS_DEFAULT: str = os.getenv("RIGHTS_DEFAULT", "public_reference_only")

    BACKFILL_DAYS: int = int(os.getenv("BACKFILL_DAYS", 60))
    SEGMENT_TOLERANCE_S: float = float(os.getenv("SEGMENT_TOLERANCE_S", 0.75))
    SEGMENT_OVERLAP_S: float = float(os.getenv("SEGMENT_OVERLAP_S", 3.0))
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Router enrichment
    ROUTER_GEN_MODEL: str = os.getenv("ROUTER_GEN_MODEL", "gpt-4o-mini")
    ROUTER_GEN_MAX_SENTENCES: int = int(os.getenv("ROUTER_GEN_MAX_SENTENCES", "60"))
    ROUTER_GEN_RETRIES: int = int(os.getenv("ROUTER_GEN_RETRIES", "3"))

    PIPELINE_STORAGE_ROOT: str = os.getenv("PIPELINE_STORAGE_ROOT") or str(_DEFAULT_PIPELINE_STORAGE)
    ENTITIES_CACHE_DIR: str = os.getenv("ENTITIES_CACHE_DIR") or str(Path(PIPELINE_STORAGE_ROOT) / "entities_cache")
    # Where to store enrichment sidecars (kept outside Parent docs)
    ROUTER_CACHE_DIR: str = os.getenv("ROUTER_CACHE_DIR") or str(Path(PIPELINE_STORAGE_ROOT) / "router_cache")
    SPEAKER_EMBED_CACHE_DIR: str = os.getenv("SPEAKER_EMBED_CACHE_DIR") or str(Path(PIPELINE_STORAGE_ROOT) / "speaker_embeds")
    SPEAKER_MAP_DIR: str = os.getenv("SPEAKER_MAP_DIR") or str(Path(PIPELINE_STORAGE_ROOT) / "speaker_maps")
    VOICE_LIBRARY_DIR: str = os.getenv("VOICE_LIBRARY_DIR") or str(Path(PIPELINE_STORAGE_ROOT) / "voices")
    # Embedding + upsert knobs
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_CONCURRENCY: int = int(os.getenv("EMBED_CONCURRENCY", "2"))
    PINECONE_UPSERT_BATCH: int = int(os.getenv("PINECONE_UPSERT_BATCH", "500"))


settings_v2 = SettingsV2()
