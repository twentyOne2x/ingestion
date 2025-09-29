from __future__ import annotations
import os
import logging
import time
from typing import Optional, Dict, Any

# Optional: load .env if present
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_summaries(
        parent_id: Optional[str] = None,
        *,
        index: Optional[str] = None,
        namespace: Optional[str] = None,
        dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Delete Pinecone vectors with metadata node_type='summary'.
    - If parent_id is provided, targets only that parent; otherwise ALL summaries in the namespace.
    - Uses env vars when args are omitted:
        PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE (default 'videos').
    - Returns a dict describing what it did (or would do).
    """
    logger.info("=" * 60)
    logger.info("Starting delete_summaries operation")
    logger.info(f"Parameters: parent_id={parent_id}, index={index}, namespace={namespace}, dry_run={dry_run}")

    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not set")
        raise RuntimeError("PINECONE_API_KEY not set (export it or put it in a .env).")
    logger.debug("PINECONE_API_KEY found")

    # Get index name
    index = index or os.getenv("PINECONE_INDEX_NAME")
    if not index:
        logger.error("Index not provided and PINECONE_INDEX_NAME not set")
        raise RuntimeError("Index not provided and PINECONE_INDEX_NAME not set.")
    logger.info(f"Using index: {index}")

    # Get namespace
    namespace = namespace or os.getenv("PINECONE_NAMESPACE") or "videos"
    logger.info(f"Using namespace: {namespace}")

    # Build filter
    filt = {"node_type": "summary"}
    if parent_id:
        filt["parent_id"] = parent_id
        logger.info(f"Targeting specific parent_id: {parent_id}")
    else:
        logger.warning("NO parent_id specified - will target ALL summaries in namespace!")

    logger.info(f"Filter to be applied: {filt}")

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No actual deletion will occur")
        logger.info("=" * 60)
        result = {"dry_run": True, "index": index, "namespace": namespace, "filter": filt}
        logger.info(f"Dry run result: {result}")
        return result

    try:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        idx = pc.Index(index)
        logger.info("Successfully connected to Pinecone")

        # Get index stats before deletion
        logger.info("Fetching index statistics before deletion...")
        stats_before = idx.describe_index_stats()
        namespace_stats = stats_before.get('namespaces', {}).get(namespace, {})
        total_vectors_before = namespace_stats.get('vector_count', 0)
        logger.info(f"Namespace '{namespace}' has {total_vectors_before} total vectors before deletion")

        # Get the correct dimension from index stats
        dimension = None
        for ns_name, ns_stats in stats_before.get('namespaces', {}).items():
            if 'dimension' in ns_stats:
                dimension = ns_stats['dimension']
                break

        # If dimension not in stats, try common dimensions
        if dimension is None:
            # Try common dimensions in order
            for try_dim in [3072, 1536, 768, 512, 256, 128, 10]:
                try:
                    test_query = idx.query(
                        namespace=namespace,
                        filter=filt,
                        top_k=1,
                        include_metadata=False,
                        vector=[0.0] * try_dim
                    )
                    dimension = try_dim
                    logger.info(f"Detected index dimension: {dimension}")
                    break
                except Exception:
                    continue

            if dimension is None:
                logger.error("Could not determine index dimension")
                raise RuntimeError("Could not determine index dimension")
        else:
            logger.info(f"Index dimension from stats: {dimension}")

        # Query to count matching vectors (up to 10000)
        logger.info("Querying for vectors that match the filter...")
        query_result = idx.query(
            namespace=namespace,
            filter=filt,
            top_k=10000,  # Max allowed
            include_metadata=False,
            vector=[0.0] * dimension  # Dummy vector with correct dimension
        )
        vectors_found = len(query_result.get('matches', []))
        logger.info(f"Found {vectors_found} vectors matching filter (up to 10000 shown)")

        if vectors_found == 0:
            logger.warning("No vectors found matching the filter - nothing to delete")
            result = {
                "dry_run": False,
                "index": index,
                "namespace": namespace,
                "filter": filt,
                "vectors_deleted": 0,
                "total_vectors_before": total_vectors_before,
                "total_vectors_after": total_vectors_before
            }
            return result

        # List a sample of IDs that will be deleted (for audit trail)
        if vectors_found > 0:
            sample_ids = [match['id'] for match in query_result.get('matches', [])[:5]]
            logger.info(f"Sample of vector IDs to be deleted: {sample_ids}")
            if vectors_found > 5:
                logger.info(f"... and {vectors_found - 5} more vectors")

        # Perform the deletion
        logger.warning("=" * 60)
        logger.warning(f"EXECUTING DELETE operation with filter: {filt}")
        logger.warning(f"This will delete approximately {vectors_found} vectors")
        logger.warning("=" * 60)

        start_time = time.time()
        delete_response = idx.delete(namespace=namespace, filter=filt)
        elapsed_time = time.time() - start_time

        logger.info(f"Delete operation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Delete response from Pinecone: {delete_response}")

        # Wait a moment for consistency
        time.sleep(2)

        # Verify deletion
        logger.info("Verifying deletion...")
        verify_result = idx.query(
            namespace=namespace,
            filter=filt,
            top_k=1,
            include_metadata=False,
            vector=[0.0] * dimension  # Use the correct dimension
        )
        remaining = len(verify_result.get('matches', []))

        # Get index stats after deletion
        stats_after = idx.describe_index_stats()
        namespace_stats_after = stats_after.get('namespaces', {}).get(namespace, {})
        total_vectors_after = namespace_stats_after.get('vector_count', 0)

        vectors_actually_deleted = total_vectors_before - total_vectors_after

        if remaining == 0:
            logger.info("✓ Verification successful: All matching vectors have been deleted")
        else:
            logger.warning(f"⚠ Verification found {remaining} vectors still matching filter")

        logger.info(f"Total vectors in namespace before: {total_vectors_before}")
        logger.info(f"Total vectors in namespace after: {total_vectors_after}")
        logger.info(f"Approximate vectors deleted: {vectors_actually_deleted}")

        result = {
            "dry_run": False,
            "index": index,
            "namespace": namespace,
            "filter": filt,
            "vectors_found_before": vectors_found,
            "vectors_remaining_after": remaining,
            "total_vectors_before": total_vectors_before,
            "total_vectors_after": total_vectors_after,
            "approximate_deleted": vectors_actually_deleted,
            "delete_response": delete_response,
            "elapsed_seconds": elapsed_time
        }

        logger.info("=" * 60)
        logger.info(f"Operation completed: {result}")
        logger.info("=" * 60)
        return result

    except Exception as e:
        logger.error(f"Error during deletion: {type(e).__name__}: {str(e)}")
        raise


def delete_all_node_summaries(
        *,
        index: Optional[str] = None,
        namespace: Optional[str] = None,
        confirm: bool = False,
) -> Dict[str, Any]:
    """
    Delete ALL summary nodes in the namespace (no parent_id filter).

    Args:
        index: Pinecone index name (uses PINECONE_INDEX_NAME env var if not provided)
        namespace: Namespace to target (uses PINECONE_NAMESPACE env var or 'videos' if not provided)
        confirm: Must be True to actually perform deletion (safety check)

    Returns:
        Dict with operation results
    """
    logger.warning("=" * 60)
    logger.warning("DANGEROUS OPERATION: Delete ALL summary nodes")
    logger.warning("=" * 60)

    if not confirm:
        logger.error("Safety check failed: confirm=False")
        logger.info("To delete all summaries, call with confirm=True")
        return {"error": "Operation cancelled - set confirm=True to proceed"}

    logger.warning("Confirmation received - proceeding with deletion of ALL summaries")
    return delete_summaries(
        parent_id=None,  # No parent filter = delete all summaries
        index=index,
        namespace=namespace,
        dry_run=False
    )


if __name__ == "__main__":
    # Set log level from environment variable if provided
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Example 1: Preview deletion for specific parent (dry run)
    # logger.info("Example 1: Preview mode for specific parent")
    # result = delete_summaries(parent_id="dQw4w9WgXcQ", dry_run=True)
    # print(f"\nResult: {result}\n")

    # Example 2: Delete for specific parent (uncomment to run)
    logger.info("Example 2: Delete summaries for specific parent")
    result = delete_summaries(parent_id="C6v9unSN0FU", dry_run=False)
    print(f"\nResult: {result}\n")

    # Example 3: Delete ALL summaries (uncomment and set confirm=True to run)
    # logger.info("Example 3: Delete ALL summaries")
    # result = delete_all_node_summaries(confirm=True)
    # print(f"\nResult: {result}\n")