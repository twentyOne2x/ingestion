#!/usr/bin/env python3
"""Backfill Pump.fun clip metadata in Pinecone child vectors.

The script scans child vectors whose router tags/SKUs indicate Pump.fun origin
and updates the metadata to align with the new ingestion format:

    channel_name -> "<token name> (Pumpfun)"
    channel_handle -> None
    pumpfun_coin_symbol, pumpfun_coin_name, pumpfun_room, pumpfun_clip_id, ...

Usage example:

    python -m src.ingest_v2.scripts.backfill_pumpfun_metadata \
        --index-name icmfyi-v2 \
        --namespace videos \
        --dry-run

`--dry-run` logs intended updates without touching Pinecone. Drop it once
you've reviewed the planned changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from pinecone import Pinecone

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:
    pass

LOG = logging.getLogger("backfill_pumpfun_metadata")


def _safe_get(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def _iter_paginated(index, namespace: str, limit: int = 99) -> Iterator[dict]:
    limit = max(1, min(int(limit or 99), 99))
    token = None
    while True:
        res = index.list_paginated(namespace=namespace, limit=limit, pagination_token=token)
        vectors = res.get("vectors", []) if isinstance(res, dict) else []
        for vec in vectors:
            yield vec
        pagination = res.get("pagination") if isinstance(res, dict) else None
        token = _safe_get(pagination, "next")
        if not token:
            break


def _fetch(index, namespace: str, ids: List[str], batch: int = 200) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        res = index.fetch(namespace=namespace, ids=chunk) or {}
        vectors = res.get("vectors", {}) if isinstance(res, dict) else {}
        out.update(vectors)
    return out


def _str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_display(channel_name: Optional[str], symbol: Optional[str], room: Optional[str], fallback: str) -> str:
    display = channel_name or symbol or room or fallback
    if not display:
        display = fallback
    return f"{display} (Pumpfun)"


@dataclass
class UpdatePlan:
    vector_ids: List[str]
    updates: Dict[str, Dict[str, object]]
    skipped: List[str]


def plan_updates(vectors: Dict[str, dict]) -> UpdatePlan:
    updates: Dict[str, Dict[str, object]] = {}
    skipped: List[str] = []

    for vid, payload in vectors.items():
        metadata = payload.get("metadata") or {}
        router_tags = metadata.get("router_tags") or []
        if "pumpfun" not in router_tags:
            skipped.append(vid)
            continue

        coin_payload = metadata.get("pumpfun_coin") or {}
        clip_payload = metadata.get("pumpfun_clip") or {}

        symbol = _str(coin_payload.get("symbol")) or _str(metadata.get("pumpfun_coin_symbol"))
        name = _str(coin_payload.get("name")) or _str(metadata.get("pumpfun_coin_name"))
        room = _str(clip_payload.get("roomName")) or _str(metadata.get("pumpfun_room"))
        clip_id = _str(clip_payload.get("clipId")) or _str(metadata.get("pumpfun_clip_id"))
        start_time = _str(clip_payload.get("startTime")) or _str(metadata.get("pumpfun_start_time"))

        channel_display = _build_display(name, symbol, room, fallback=vid)

        updates[vid] = {
            "channel_name": channel_display,
            "channel_handle": None,
            "pumpfun_coin_symbol": symbol,
            "pumpfun_coin_name": name,
            "pumpfun_room": room,
            "pumpfun_clip_id": clip_id,
            "pumpfun_start_time": start_time,
        }

    return UpdatePlan(vector_ids=list(updates.keys()), updates=updates, skipped=skipped)


def apply_updates(index, namespace: str, plan: UpdatePlan, dry_run: bool, batch_size: int = 100):
    if dry_run:
        for vid, meta in list(plan.updates.items())[:10]:
            LOG.info("[dry-run] %s → %s", vid, json.dumps({k: v for k, v in meta.items() if v}, ensure_ascii=False))
        LOG.info("Dry run complete. %d vectors would be updated.", len(plan.vector_ids))
        return

    payloads = []
    for vid, meta in plan.updates.items():
        filtered = {k: v for k, v in meta.items() if v is not None}
        payloads.append({"id": vid, "metadata": filtered})

    for i in range(0, len(payloads), batch_size):
        chunk = payloads[i : i + batch_size]
        LOG.info("Upserting %d pumpfun vectors (%d/%d)", len(chunk), min(len(payloads), i + batch_size), len(payloads))
        index.update(namespace=namespace, vectors=chunk)


def backfill(index_name: str, namespace: str, dry_run: bool, limit: Optional[int], batch_size: int):
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = client.Index(index_name)

    ids: List[str] = []
    for vec in _iter_paginated(index, namespace=namespace):
        vid = _safe_get(vec, "id")
        md = _safe_get(vec, "metadata") or {}
        tags = _safe_get(md, "router_tags") or []
        if "pumpfun" in tags:
            ids.append(vid)
        if limit and len(ids) >= limit:
            break

    LOG.info("Collected %d pumpfun vector IDs in namespace=%s", len(ids), namespace)
    vectors = _fetch(index, namespace=namespace, ids=ids)
    plan = plan_updates(vectors)
    LOG.info("Prepared updates for %d vectors (%d skipped)", len(plan.vector_ids), len(plan.skipped))
    apply_updates(index, namespace=namespace, plan=plan, dry_run=dry_run, batch_size=batch_size)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Pump.fun metadata in Pinecone child nodes")
    parser.add_argument("--index-name", required=False, default=os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2"))
    parser.add_argument("--namespace", required=False, default=os.getenv("PINECONE_NAMESPACE", "videos"))
    parser.add_argument("--dry-run", action="store_true", help="Log updates without writing to Pinecone")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of vectors to process")
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    if "PINECONE_API_KEY" not in os.environ:
        LOG.error("PINECONE_API_KEY env var is required")
        return 1

    backfill(
        index_name=args.index_name,
        namespace=args.namespace,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
