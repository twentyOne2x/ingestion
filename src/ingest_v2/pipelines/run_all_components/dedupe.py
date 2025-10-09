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
