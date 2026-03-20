import uuid
from hashlib import sha1

NAMESPACE = uuid.NAMESPACE_URL

def parent_point_uuid(parent_id: str) -> str:
    # Qdrant point IDs must be an unsigned int or a UUID; YouTube IDs are not.
    # Keep the original parent_id in payload, but use a deterministic UUID as the point ID.
    return str(uuid.uuid5(NAMESPACE, f"parent:{parent_id}"))

def segment_uuid(parent_id: str, start_s: float, end_s: float) -> str:
    key = f"{parent_id}:{start_s:.3f}:{end_s:.3f}"
    return str(uuid.uuid5(NAMESPACE, key))

def summary_segment_uuid(parent_id: str, start_s: float, end_s: float) -> str:
    """
    Summary nodes live in the same Qdrant collection as child chunks and use `segment_id`
    as the point ID. Use a distinct UUID namespace/key to avoid collisions with child
    segment UUIDs (which are keyed only by parent_id + start/end).
    """
    key = f"summary:{parent_id}:{start_s:.3f}:{end_s:.3f}"
    return str(uuid.uuid5(NAMESPACE, key))

def sha1_hex(payload: bytes) -> str:
    return sha1(payload).hexdigest()
