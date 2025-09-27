import os
import math
import logging
from typing import Any, Dict, Iterable, List

from pinecone import Pinecone, ServerlessSpec
from ..configs.settings import settings_v2
from .backoff import expo_backoff
from .batching import chunked

# Pinecone only allows str | number | bool | list[str] for metadata values.
_ALLOWED_SCALARS = (str, int, float, bool)


def _coerce_list_str(x: Any) -> List[str] | None:
    if not isinstance(x, list):
        return None
    out: List[str] = []
    for v in x:
        if v is None:
            continue
        if isinstance(v, _ALLOWED_SCALARS):
            out.append(str(v))
        else:
            try:
                out.append(str(v))
            except Exception:
                continue
    return out


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only Pinecone-valid metadata types, drop None and bulky/nested fields.
    """
    clean: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue

        # Never ship these to Pinecone (too big / nested / not useful for filtering)
        if k in {"chapters", "caption_lines", "diarization", "words"}:
            continue

        # Drop NaN/Inf
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue

        if isinstance(v, _ALLOWED_SCALARS):
            clean[k] = v
        else:
            as_list = _coerce_list_str(v)
            if as_list is not None:
                clean[k] = as_list
            # else drop

    # v2 tag
    clean.setdefault("ingest_version", 2)

    # If chapter made it and is falsy, remove it (Pinecone forbids null)
    if "chapter" in clean and (clean["chapter"] is None or clean["chapter"] == ""):
        clean.pop("chapter", None)

    return clean


def trim_metadata_utf8(meta: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
    """
    Ensure metadata JSON length in bytes <= max_bytes. We only trim the largest
    string-ish fields (text/description/title) if needed.
    """
    import json

    def size(d: Dict[str, Any]) -> int:
        return len(json.dumps(d, ensure_ascii=False).encode("utf-8"))

    if size(meta) <= max_bytes:
        return meta

    m = dict(meta)
    # Heuristic: trim in this order
    candidates = ["text", "description", "title"]
    for key in candidates:
        v = m.get(key)
        if isinstance(v, str) and len(v) > 0:
            # binary search trim to fit
            lo, hi = 0, len(v)
            best = v
            while lo <= hi:
                mid = (lo + hi) // 2
                m[key] = v[:mid]
                if size(m) <= max_bytes:
                    best = m[key]
                    lo = mid + 1
                else:
                    hi = mid - 1
            # final set with ellipsis if we actually trimmed
            if best != v:
                m[key] = best + "…"
            if size(m) <= max_bytes:
                return m

    # If still too big, drop any remaining oversized fields heuristically
    for k in list(m.keys()):
        if size(m) <= max_bytes:
            break
        if isinstance(m[k], str) and len(m[k]) > 0 and k not in {"parent_id", "segment_id"}:
            m.pop(k, None)

    return m


def get_index(index_name: str, dimension: int):
    """
    Return a Pinecone Index object. Auto-creates a serverless index if missing.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)

    # New SDK returns dict with "indexes"
    names = [x["name"] for x in pc.list_indexes().get("indexes", [])]
    if index_name not in names:
        logging.info(f"[pinecone] creating index {index_name} dim={dimension} metric=cosine region={settings_v2.PINECONE_REGION}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings_v2.PINECONE_REGION),
        )

    return pc.Index(index_name)


def upsert_vectors(index, namespace: str, vectors: List[Dict[str, Any]], batch_size: int = 100):
    """
    vectors: [{'id': str, 'values': list[float], 'metadata': dict}, ...]
    Sanitizes metadata and retries on transient failures.
    """
    if not vectors:
        return

    # sanitize + trim per record
    def _prep(v: Dict[str, Any]) -> Dict[str, Any]:
        md = sanitize_metadata(v.get("metadata") or {})
        md = trim_metadata_utf8(md, settings_v2.MAX_METADATA_BYTES)
        return {"id": v["id"], "values": v["values"], "metadata": md}

    attempt = 0
    for batch in chunked(vectors, batch_size):
        prepped = [_prep(v) for v in batch]

        while True:
            try:
                index.upsert(vectors=prepped, namespace=namespace)
                break
            except Exception as e:
                attempt += 1
                logging.warning(f"[pinecone] upsert retry {attempt} ns={namespace} batch={len(prepped)} err={e}")
                expo_backoff(attempt)
