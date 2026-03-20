from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


LOG = logging.getLogger(__name__)


def vector_store_backend() -> str:
    return (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()


def qdrant_collection_name(namespace: str | None = None) -> str:
    index_name = (os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2") or "icmfyi-v2").strip()
    ns = (namespace or os.getenv("PINECONE_NAMESPACE", "videos") or "videos").strip()
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    return template.format(index=index_name, namespace=ns)


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    timeout_s = float(os.getenv("QDRANT_TIMEOUT_S", "120") or 120)
    return QdrantClient(url=url, api_key=api_key, timeout=timeout_s)


def ensure_qdrant_collection(collection_name: str, dimension: int) -> None:
    client = qdrant_client()
    exists = False
    try:
        exists = bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        # Older client fallback
        names = {item.name for item in client.get_collections().collections}
        exists = collection_name in names

    if exists:
        return

    LOG.info("[qdrant] creating collection=%s dim=%d", collection_name, dimension)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
    )
    for field_name, schema in (
        ("parent_id", qm.PayloadSchemaType.KEYWORD),
        ("channel_id", qm.PayloadSchemaType.KEYWORD),
        ("channel_name", qm.PayloadSchemaType.KEYWORD),
        ("published_at", qm.PayloadSchemaType.DATETIME),
        ("document_type", qm.PayloadSchemaType.KEYWORD),
        ("node_type", qm.PayloadSchemaType.KEYWORD),
        ("source", qm.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            # Payload indexes are best-effort for local mode.
            pass


def upsert_qdrant_vectors(
    *,
    collection_name: str,
    vectors: List[Dict[str, Any]],
    dimension: int,
    batch_size: int = 100,
) -> int:
    if not vectors:
        return 0

    ensure_qdrant_collection(collection_name=collection_name, dimension=dimension)
    client = qdrant_client()

    def _is_payload_too_large(exc: Exception) -> bool:
        # Qdrant REST has a default ~32MB JSON payload limit. The qdrant-client surfaces this as an
        # UnexpectedResponse(400) with "Payload error: JSON payload (...) is larger than allowed".
        msg = (str(exc) or "").lower()
        return ("json payload" in msg and "larger than allowed" in msg) or ("payload error" in msg and "json payload" in msg)

    def _points_from(chunk: List[Dict[str, Any]]) -> List[qm.PointStruct]:
        return [
            qm.PointStruct(
                id=str(item["id"]),
                vector=item["values"],
                payload=dict(item.get("metadata") or {}),
            )
            for item in chunk
        ]

    def _upsert_chunk_with_split(chunk: List[Dict[str, Any]]) -> int:
        """
        Upsert a chunk, splitting on Qdrant "payload too large" errors.

        This keeps local ingestion resilient even when transcripts produce large point batches
        (e.g. long videos with many segments).
        """
        if not chunk:
            return 0

        sent = 0
        stack: List[List[Dict[str, Any]]] = [chunk]
        while stack:
            sub = stack.pop()
            try:
                client.upsert(collection_name=collection_name, points=_points_from(sub), wait=False)
                sent += 1
            except Exception as exc:
                if len(sub) <= 1 or not _is_payload_too_large(exc):
                    raise
                mid = max(1, len(sub) // 2)
                LOG.warning(
                    "[qdrant] upsert payload too large; splitting batch size=%d -> %d + %d",
                    len(sub),
                    mid,
                    len(sub) - mid,
                )
                stack.append(sub[mid:])
                stack.append(sub[:mid])
        return sent

    batches = 0
    step = max(1, int(batch_size))
    for start in range(0, len(vectors), step):
        chunk = vectors[start : start + step]
        batches += _upsert_chunk_with_split(chunk)
    return batches


def fetch_qdrant_payloads(
    *,
    collection_name: str,
    ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    wanted = [str(i) for i in ids if i]
    if not wanted:
        return {}
    client = qdrant_client()
    rows = client.retrieve(
        collection_name=collection_name,
        ids=wanted,
        with_payload=True,
        with_vectors=False,
    )
    return {str(row.id): dict(row.payload or {}) for row in rows}
