from __future__ import annotations
import os
import time
import json
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import requests

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None  # type: ignore


def _pc_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")
    if Pinecone is None:
        raise RuntimeError("pinecone-client is not installed. pip install pinecone-client==3.*")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def list_ids(namespace: Optional[str], batch_limit: int = 1000) -> Iterable[str]:
    """
    Prefer REST /vectors/list via PINECONE_HOST if provided.
    Falls back to raising unless VECTOR_ID_MANIFEST is supplied to the CLI.
    """
    host = os.getenv("PINECONE_HOST", "").strip()
    if host:
        yield from _list_ids_via_rest(host, namespace, batch_limit=batch_limit)
        return
    raise RuntimeError(
        "No PINECONE_HOST provided for REST listing. Either set PINECONE_HOST or supply --ids-file."
    )


def _list_ids_via_rest(host: str, namespace: Optional[str], batch_limit: int = 1000) -> Iterable[str]:
    """
    Calls POST {host}/vectors/list with pagination.
    See Pinecone v2 REST API. Requires PINECONE_API_KEY.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required")

    url = f"{host.rstrip('/')}/vectors/list"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    token = None

    while True:
        payload = {
            "limit": int(batch_limit),
            **({"namespace": namespace} if namespace else {}),
            **({"paginationToken": token} if token else {}),
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"list failed: HTTP {resp.status_code} {resp.text[:300]}")

        js = resp.json()
        for v in js.get("vectors", []):
            vid = v.get("id")
            if vid:
                yield vid

        token = js.get("pagination") and js["pagination"].get("next")
        if not token:
            break


def fetch_metadata(ids: List[str], namespace: Optional[str]) -> Dict[str, Dict]:
    """
    Fetch vector metadata for a batch of IDs.
    Returns { id: metadata_dict }
    """
    if not ids:
        return {}

    idx = _pc_index()
    out = idx.fetch(ids=ids, namespace=namespace, include_values=False)
    result = {}
    vectors = (out or {}).get("vectors", {})
    for k, v in vectors.items():
        md = v.get("metadata") or {}
        result[k] = md
    return result


def update_metadata(updates: List[Tuple[str, Dict]], namespace: Optional[str]) -> None:
    """
    Apply partial metadata updates (set_metadata). updates is list of (id, dict_to_set).
    """
    idx = _pc_index()
    for vid, meta in updates:
        idx.update(id=vid, namespace=namespace, set_metadata=meta)
        # Gentle pacing to avoid throttling
        time.sleep(0.001)
