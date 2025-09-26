import uuid
from hashlib import sha1

NAMESPACE = uuid.NAMESPACE_URL

def segment_uuid(parent_id: str, start_s: float, end_s: float) -> str:
    key = f"{parent_id}:{start_s:.3f}:{end_s:.3f}"
    return str(uuid.uuid5(NAMESPACE, key))

def sha1_hex(payload: bytes) -> str:
    return sha1(payload).hexdigest()
