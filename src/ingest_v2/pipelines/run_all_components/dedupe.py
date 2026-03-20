from __future__ import annotations

import logging
import os
from typing import Set


def get_ingested_parent_ids(index_name: str, namespace: str) -> Set[str]:
    """
    Query Pinecone for existing child vectors and extract unique parent_ids.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    from pinecone import Pinecone

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)

    stats = index.describe_index_stats()
    dimension = None
    for ns_stats in stats.get("namespaces", {}).values():
        if "dimension" in ns_stats:
            dimension = ns_stats["dimension"]
            break

    if dimension is None:
        for candidate in (3072, 1536, 768):
            try:
                index.query(
                    namespace=namespace,
                    filter={"node_type": "child"},
                    top_k=1,
                    include_metadata=True,
                    vector=[0.0] * candidate,
                )
                dimension = candidate
                logging.info("[dedupe] detected dimension=%d", dimension)
                break
            except Exception:
                continue

    if dimension is None:
        raise RuntimeError("Could not determine index dimension")

    logging.info("[dedupe] querying Pinecone for existing parent_ids...")
    result = index.query(
        namespace=namespace,
        filter={"node_type": "child"},
        top_k=10000,
        include_metadata=True,
        vector=[0.0] * dimension,
    )

    parent_ids = {
        match["metadata"]["parent_id"]
        for match in result.get("matches", [])
        if match.get("metadata", {}).get("parent_id")
    }

    logging.info("[dedupe] found %d unique parent_ids already in Pinecone", len(parent_ids))
    return parent_ids


def get_ingested_parent_ids_qdrant(namespace: str) -> Set[str]:
    """
    Qdrant equivalent of Pinecone dedupe:
      - Scroll child nodes
      - Collect unique parent_id payloads

    This is intentionally child-based (not parent-based) so that a prior run that
    only wrote parent stubs but crashed before children will still be re-processed.
    """
    from qdrant_client.http import models as qm

    from ...utils.vector_store import qdrant_client, qdrant_collection_name

    collection = qdrant_collection_name(namespace)
    client = qdrant_client()

    # If the collection does not exist yet, nothing is ingested.
    try:
        exists = bool(client.collection_exists(collection_name=collection))
    except Exception:
        try:
            names = {item.name for item in client.get_collections().collections}
            exists = collection in names
        except Exception:
            exists = False

    if not exists:
        logging.info("[dedupe/qdrant] collection does not exist: %s", collection)
        return set()

    logging.info("[dedupe/qdrant] scrolling existing child nodes for parent_ids (collection=%s)...", collection)
    parent_ids: Set[str] = set()
    offset = None
    scanned = 0

    flt = qm.Filter(
        must=[
            qm.FieldCondition(
                key="node_type",
                match=qm.MatchValue(value="child"),
            )
        ]
    )

    scroll_limit = int(os.getenv("QDRANT_SCROLL_LIMIT", "256") or 256)

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=scroll_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        scanned += len(points)
        for p in points:
            payload = dict(getattr(p, "payload", None) or {})
            pid = payload.get("parent_id")
            if pid:
                parent_ids.add(str(pid))

        offset = next_offset
        if next_offset is None:
            break

    logging.info("[dedupe/qdrant] scanned=%d child_points, found=%d unique parent_ids", scanned, len(parent_ids))
    return parent_ids
