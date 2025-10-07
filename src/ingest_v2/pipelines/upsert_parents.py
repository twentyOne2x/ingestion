# src/ingest_v2/pipelines/upsert_parents.py
from __future__ import annotations
import os
from typing import Iterable, Dict, Any, List
from pinecone import Pinecone

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

# src/ingest_v2/pipelines/upsert_parents.py
def upsert_parents(parents: Iterable[Dict[str, Any]]):
    pc  = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2"))
    ns  = os.getenv("PINECONE_NAMESPACE", "videos")
    dim = _detect_dim(idx)

    vecs = []
    for p in parents:
        pid = str(p.get("parent_id") or p.get("video_id"))

        md = {
            "parent_id": pid,
            "title": p.get("title"),
            "channel_name": p.get("channel_name"),
            "published_at": p.get("published_at"),
            "url": p.get("url"),  # may be HttpUrl; _json_safe will str() it
            "document_type": p.get("document_type", "youtube_video"),
            "node_type": "parent",
            "router_tags": p.get("router_tags"),
            "aliases": p.get("aliases"),
            "canonical_entities": p.get("canonical_entities"),
            "is_explainer": p.get("is_explainer"),
            "router_boost": p.get("router_boost"),
            "topic_summary": p.get("topic_summary"),
            "speaker_primary": p.get("speaker_primary"),
        }

        # Epsilon non-zero to satisfy Pinecone
        vals = [0.0] * dim
        vals[0] = 1e-6

        vecs.append({
            "id": pid,
            "values": vals,
            "metadata": _json_safe(md),
        })

    for i in range(0, len(vecs), 100):
        idx.upsert(vectors=vecs[i:i+100], namespace=ns)
