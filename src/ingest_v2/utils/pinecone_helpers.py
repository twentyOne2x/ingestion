# In src/ingest_v2/utils/pinecone_helpers.py (new file)

from pinecone import Pinecone
import os
import logging
from typing import Set


# Add to src/ingest_v2/utils/pinecone_helpers.py

def get_ingested_parent_ids(index_name: str, namespace: str) -> Set[str]:
    """
    Query child vectors and extract unique parent_ids.
    Works efficiently when you have <10k parents (even if >10k children).
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)

    # Detect dimension (reuse your delete script logic)
    stats = idx.describe_index_stats()
    dimension = None
    for ns_stats in stats.get('namespaces', {}).values():
        if 'dimension' in ns_stats:
            dimension = ns_stats['dimension']
            break

    if dimension is None:
        for try_dim in [3072, 1536, 768]:
            try:
                idx.query(namespace=namespace, filter={"node_type": "child"},
                          top_k=1, include_metadata=True, vector=[0.0] * try_dim)
                dimension = try_dim
                break
            except:
                continue

    if dimension is None:
        raise RuntimeError("Could not determine dimension")

    logging.info("[pinecone] Fetching existing parent_ids from child vectors...")

    # Query up to 10k child vectors (covers way more than 1739 parents)
    result = idx.query(
        namespace=namespace,
        filter={"node_type": "child"},
        top_k=10000,
        include_metadata=True,
        vector=[0.0] * dimension
    )

    # Extract unique parent_ids
    parent_ids = {
        match['metadata']['parent_id']
        for match in result.get('matches', [])
        if match.get('metadata', {}).get('parent_id')
    }

    logging.info(f"[pinecone] Found {len(parent_ids)} unique parent_ids already ingested")
    return parent_ids