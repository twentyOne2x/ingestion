from __future__ import annotations
import os
import logging
import time
from typing import Optional, List
from tqdm import tqdm
from pinecone import Pinecone

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Entity canonicalization map (from your config)
ENT_CANON_MAP = {
    "solana": "Solana",
    "sol": "SOL",
    "soul": "SOL",
    "$sol": "SOL",
    "$soul": "SOL",
    "anza labs": "Anza",
    "firedancer": "Firedancer",
}
ENT_CANON_MAP = {k.lower(): v for k, v in ENT_CANON_MAP.items()}


def canon_entity(s: str) -> str:
    """Canonicalize a single entity."""
    key = s.strip().lower()
    return ENT_CANON_MAP.get(key, s)


def canon_entities(ents: List[str]) -> List[str]:
    """Canonicalize a list of entities."""
    return sorted({canon_entity(e) for e in ents if e and e.strip()})


def backfill_fix_entities(
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        batch_size: int = 100,
        dry_run: bool = True
):
    """
    Fix entity metadata in existing Pinecone vectors.

    Args:
        namespace: Namespace to target (uses PINECONE_NAMESPACE env var or 'videos' if not provided)
        index_name: Index name (uses PINECONE_INDEX_NAME env var if not provided)
        batch_size: Number of vectors to process per batch
        dry_run: If True, only shows what would be changed without updating
    """
    logger.info("=" * 60)
    logger.info("Starting entity canonicalization backfill")
    logger.info(f"Parameters: namespace={namespace}, dry_run={dry_run}")

    # Get credentials
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("Index name not provided and PINECONE_INDEX_NAME not set")

    namespace = namespace or os.getenv("PINECONE_NAMESPACE") or "videos"
    logger.info(f"Using index: {index_name}, namespace: {namespace}")

    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)

    # Get dimension
    stats = idx.describe_index_stats()
    dimension = None
    for ns_stats in stats.get('namespaces', {}).values():
        if 'dimension' in ns_stats:
            dimension = ns_stats['dimension']
            break

    if dimension is None:
        # Try common dimensions
        for try_dim in [3072, 1536, 768]:
            try:
                idx.query(namespace=namespace, top_k=1, vector=[0.0] * try_dim, include_metadata=False)
                dimension = try_dim
                break
            except Exception:
                continue

    if dimension is None:
        raise RuntimeError("Could not determine index dimension")

    logger.info(f"Index dimension: {dimension}")

    # Target entities that need fixing
    bad_entities = ["Soul", "$SOUL", "soul", "$soul", "sol", "solana", "anza labs"]

    total_updated = 0
    total_checked = 0

    for bad_entity in bad_entities:
        logger.info(f"\nSearching for vectors with entity: '{bad_entity}'")

        # Query for vectors with this entity
        filter_dict = {"entities": {"$in": [bad_entity]}}

        try:
            results = idx.query(
                namespace=namespace,
                filter=filter_dict,
                top_k=10000,  # Max allowed
                include_metadata=True,
                vector=[0.0] * dimension
            )

            matches = results.get('matches', [])
            logger.info(f"Found {len(matches)} vectors with entity '{bad_entity}'")

            if not matches:
                continue

            # Process in batches
            for i in tqdm(range(0, len(matches), batch_size), desc=f"Processing '{bad_entity}'"):
                batch = matches[i:i + batch_size]

                for match in batch:
                    vec_id = match['id']
                    meta = match.get('metadata', {})
                    ents = meta.get('entities', [])

                    if not ents:
                        continue

                    total_checked += 1
                    fixed = canon_entities(ents)

                    if fixed != ents:
                        logger.debug(f"ID {vec_id}: {ents} → {fixed}")

                        if not dry_run:
                            meta['entities'] = fixed
                            idx.update(id=vec_id, set_metadata=meta, namespace=namespace)

                        total_updated += 1

                # Rate limiting
                if not dry_run:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing entity '{bad_entity}': {e}")
            continue

    logger.info("=" * 60)
    logger.info(f"Backfill complete!")
    logger.info(f"Vectors checked: {total_checked}")
    logger.info(f"Vectors {'would be ' if dry_run else ''}updated: {total_updated}")
    if dry_run:
        logger.info("This was a DRY RUN - no actual updates were made")
    logger.info("=" * 60)

    return {
        "checked": total_checked,
        "updated": total_updated,
        "dry_run": dry_run
    }


if __name__ == "__main__":
    # Preview what would change
    logger.info("Running in DRY RUN mode...")
    result = backfill_fix_entities(dry_run=True)
    print(f"\nResult: {result}\n")

    # Uncomment to actually update:
    # logger.info("Running ACTUAL update...")
    # result = backfill_fix_entities(dry_run=False)
    # print(f"\nResult: {result}\n")