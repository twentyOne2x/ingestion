#!/usr/bin/env python3
# fetch_parent_hardcoded.py
from __future__ import annotations
import os, json, sys
from typing import Dict, Any

# ── dotenv (optional, but recommended) ─────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv
    # Load in this order: project root .env, then local overrides if present
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:
    pass  # if python-dotenv isn't installed, env vars must already be set

# ── Pinecone ───────────────────────────────────────────────────────────────────
from pinecone import Pinecone

# ---- HARD-CODED TARGETS (edit if needed) -------------------------------------
VIDEO_ID   = "O8emx084UIg"   # from your logs
INDEX_NAME = "icmfyi-v2"
NAMESPACE  = "videos"
# ------------------------------------------------------------------------------

def _as_dict(obj):
    if hasattr(obj, "to_dict"): return obj.to_dict()
    if hasattr(obj, "__dict__"): return obj.__dict__
    return obj if isinstance(obj, dict) else {}

def _fetch(idx, pid: str, ns: str):
    try:
        res = idx.fetch(ids=[pid], namespace=ns)
        vectors = _as_dict(res).get("vectors", {}) or {}
        return vectors.get(pid)
    except Exception as e:
        print(f"[error] fetch failed for id={pid!r} ns={ns!r}: {e}")
        return None

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        sys.exit("Set PINECONE_API_KEY (dotenv can provide it).")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(INDEX_NAME)

    print(f"[pinecone] index={INDEX_NAME} namespace={NAMESPACE!r} id={VIDEO_ID!r}")

    candidates = [VIDEO_ID, f"parent:{VIDEO_ID}", f"video:{VIDEO_ID}"]

    hit_id = None
    md: Dict[str, Any] = {}
    for cand in candidates:
        row = _fetch(idx, cand, NAMESPACE)
        if row:
            hit_id = cand
            meta = getattr(row, "metadata", None) or _as_dict(row).get("metadata") or {}
            md = {
                "parent_id": cand,
                "parent_title": meta.get("title"),
                "parent_channel_name": meta.get("channel_name"),
                "parent_published_at": meta.get("published_at") or meta.get("published_date"),
                "parent_url": meta.get("url"),
                "_raw_metadata_keys": sorted(list(meta.keys())) if meta else [],
            }
            break

    if hit_id:
        print("\n=== FOUND PARENT ===")
        print(json.dumps(md, indent=2, ensure_ascii=False))
    else:
        print("\n=== NOT FOUND ===")
        print("No parent/vector with that id (or common prefixes) in this index/namespace.")
        print("Checked:", ", ".join(candidates))

if __name__ == "__main__":
    main()
