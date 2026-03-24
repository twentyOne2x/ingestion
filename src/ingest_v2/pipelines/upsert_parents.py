# src/ingest_v2/pipelines/upsert_parents.py
from __future__ import annotations
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Dict, Any, List
from pinecone import Pinecone

from ..configs.settings import settings_v2
from ..utils.ids import parent_point_uuid
from ..utils.vector_store import (
    qdrant_collection_name,
    upsert_qdrant_vectors,
    vector_store_backend,
)

def _detect_dim(idx) -> int:
    stats = idx.describe_index_stats() or {}
    dim = stats.get("dimension")
    if dim:
        return int(dim)
    for ns in (stats.get("namespaces") or {}).values():
        d = ns.get("dimension")
        if d:
            return int(d)
    return int(os.getenv("EMBED_DIM", "3072"))

def _json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items() if x is not None}
    return str(v)  # fallback (e.g., HttpUrl, datetime, Enum, pydantic types)

@lru_cache(maxsize=1)
def _channel_handle_to_id() -> Dict[str, str]:
    """
    Best-effort local enrichment: map `@handle` -> `UC...` channel_id.

    This keeps metadata consistent across:
    - diarized dataset ingests (which often store channel as @handle), and
    - yt-dlp caption ingests (which know channel_id but may not know handle).

    Configure by mounting the mapping JSON and setting:
      YT_CHANNEL_MAPPING=/path/to/channel_handle_to_id_mapping.json
    """
    path = (os.getenv("YT_CHANNEL_MAPPING") or os.getenv("CHANNEL_HANDLE_TO_ID_MAPPING") or "").strip()
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(errors="replace"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        hk = str(k).strip()
        cid = str(v).strip()
        if not hk or not cid:
            continue
        out[hk] = cid
    return out

# src/ingest_v2/pipelines/upsert_parents.py
def upsert_parents(parents: Iterable[Dict[str, Any]]):
    ns = os.getenv("PINECONE_NAMESPACE", "videos")
    backend = vector_store_backend()
    dim = settings_v2.EMBED_DIM

    vecs = []
    for p in parents:
        pid = str(p.get("parent_id") or p.get("video_id"))
        point_id = parent_point_uuid(pid) if backend == "qdrant" else pid

        channel_name = p.get("channel_name")
        channel_handle = p.get("channel_handle")
        if not channel_handle and isinstance(channel_name, str) and channel_name.strip().startswith("@"):
            channel_handle = channel_name.strip()
        channel_id = p.get("channel_id")
        if not channel_id and isinstance(channel_handle, str) and channel_handle.strip():
            channel_id = _channel_handle_to_id().get(channel_handle.strip())

        speaker_map = p.get("speaker_map")
        speaker_names = p.get("speaker_names")
        if (not speaker_names) and isinstance(speaker_map, dict):
            try:
                speaker_names = [
                    str(info.get("name")).strip()
                    for info in speaker_map.values()
                    if isinstance(info, dict) and str(info.get("name") or "").strip()
                ]
            except Exception:
                speaker_names = None

        md = {
            "parent_id": pid,
            "title": p.get("title"),
            "description": p.get("description"),
            "channel_name": channel_name,
            "channel_handle": channel_handle,
            "channel_id": channel_id,
            "speaker_primary": p.get("speaker_primary"),
            "speaker_names": speaker_names,
            "speaker_map": speaker_map,
            "published_at": p.get("published_at"),
            "duration_s": p.get("duration_s"),
            "url": p.get("url"),  # may be HttpUrl; _json_safe will str() it
            "thumbnail_url": p.get("thumbnail_url"),
            "document_type": p.get("document_type", "youtube_video"),
            "node_type": "parent",
            "source": p.get("source"),
            "ingest_lane": p.get("ingest_lane"),
            "transcript_provider": p.get("transcript_provider"),
            "transcript_state": p.get("transcript_state"),
            "entities": p.get("entities"),
            "router_tags": p.get("router_tags"),
            "aliases": p.get("aliases"),
            "canonical_entities": p.get("canonical_entities"),
            "is_explainer": p.get("is_explainer"),
            "router_boost": p.get("router_boost"),
            "topic_summary": p.get("topic_summary"),
        }

        # Epsilon non-zero to satisfy Pinecone
        vals = [0.0] * dim
        vals[0] = 1e-6

        vecs.append({
            "id": point_id,
            "values": vals,
            "metadata": _json_safe(md),
        })

    if not vecs:
        return

    if backend == "qdrant":
        collection = qdrant_collection_name(ns)
        upsert_qdrant_vectors(
            collection_name=collection,
            vectors=vecs,
            dimension=dim,
            batch_size=100,
        )
        return

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2"))
    dim = _detect_dim(idx)
    if dim != settings_v2.EMBED_DIM:
        # Keep vector sizes aligned with existing index when Pinecone dictates it.
        vecs = [
            {
                **item,
                "values": [1e-6] + [0.0] * (dim - 1),
            }
            for item in vecs
        ]
    for i in range(0, len(vecs), 100):
        idx.upsert(vectors=vecs[i : i + 100], namespace=ns)
