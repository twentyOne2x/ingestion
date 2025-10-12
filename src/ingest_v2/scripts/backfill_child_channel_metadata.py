#!/usr/bin/env python3
"""
Backfill Pinecone child vectors with parent metadata.

Each child/summary vector gets `channel_name`, `channel_id`, `video_id`, and
`published_at` so we can filter by channel or surface release information in
retrieval queries. The script scans parent vectors (node_type "parent") and then
updates child vectors grouped under each parent.

Usage example:

    python -m src.ingest_v2.scripts.backfill_child_channel_metadata \
        --namespace bnb,videos \
        --index-name icmfyi-v2 \
        --dry-run

`--dry-run` logs intended updates without sending them. Pass `--channel-map` to
provide a JSON mapping from channel names/handles to canonical IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from pinecone import Pinecone

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:
    pass

LOG = logging.getLogger("backfill_child_channel_metadata")


def _safe_getattr(obj, name: str, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


def _list_paginated_ids(index, namespace: str, limit: int = 99) -> Iterator[str]:
    """Yield vector IDs via Pinecone pagination."""
    limit = max(1, min(int(limit or 99), 99))
    token = None
    scanned = 0
    while True:
        res = index.list_paginated(namespace=namespace, limit=limit, pagination_token=token)
        vectors = _safe_getattr(res, "vectors") or (res.get("vectors", []) if isinstance(res, dict) else [])
        for vec in vectors:
            vid = _safe_getattr(vec, "id") or (vec.get("id") if isinstance(vec, dict) else None)
            if vid:
                yield vid
                scanned += 1
        if scanned and scanned % (limit * 50) == 0:
            LOG.info("Listed ~%d vector IDs so far...", scanned)
        pagination = _safe_getattr(res, "pagination") or (res.get("pagination") if isinstance(res, dict) else None)
        next_token = _safe_getattr(pagination, "next") or (pagination.get("next") if isinstance(pagination, dict) else None)
        if not next_token:
            break
        token = next_token


def _fetch_metadata_chunked(index, namespace: str, ids: Iterable[str], fetch_batch: int = 200) -> Dict[str, Dict]:
    ids = list(ids)
    out: Dict[str, Dict] = {}
    total = len(ids)
    if not total:
        return out

    start = time.time()
    max_per = max(1, fetch_batch)
    for i in range(0, total, max_per):
        chunk = ids[i : i + max_per]
        res = index.fetch(ids=chunk, namespace=namespace) or {}
        vectors = _safe_getattr(res, "vectors") or (res.get("vectors", {}) if isinstance(res, dict) else {})
        for vid, vec in (vectors or {}).items():
            md = _safe_getattr(vec, "metadata") or (vec.get("metadata") if isinstance(vec, dict) else {}) or {}
            out[vid] = md
        if total >= 1000 and (i // max_per) % 25 == 0:
            LOG.info(
                "Metadata fetch progress: %d/%d (%.1f%%)",
                min(total, i + max_per),
                total,
                (min(total, i + max_per) / max(1, total)) * 100.0,
            )
    LOG.info("Fetched metadata for %d vectors in %.2fs", total, time.time() - start)
    return out


def _detect_dimension(index, namespace: str) -> int:
    try:
        stats = index.describe_index_stats()
        ns_meta = (stats.get("namespaces") if isinstance(stats, dict) else {}) or {}
        ns_info = ns_meta.get(namespace, {}) or {}
        dim = ns_info.get("dimension")
        if dim:
            return int(dim)
    except Exception as exc:
        LOG.debug("describe_index_stats failed (%s); probing dims", exc)

    for dim in (3072, 1536, 1024, 768):
        try:
            index.query(namespace=namespace, top_k=1, vector=[0.0] * dim, include_metadata=False)
            return dim
        except Exception:
            continue
    raise RuntimeError("Could not detect index dimension; specify EMBED_DIM env var?")


def _normalize_channel_id(
    channel_name: Optional[str],
    channel_id: Optional[str],
    mapping: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    name = (channel_name or "").strip()
    cid = (channel_id or "").strip()
    if cid:
        return name or None, cid

    if not mapping:
        return (name, name) if name.startswith("@") else (name, None)

    if name in mapping:
        return name, str(mapping[name])

    variants = {name.lower(), name.upper(), name.lstrip("@"), name.lower().lstrip("@")}
    for key in variants:
        if key in mapping:
            return name, str(mapping[key])
    return name, name if name.startswith("@") else None


def _load_channel_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Channel map file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Channel map must be JSON object mapping names/handles to IDs.")
    mapping: Dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        mapping[str(key).strip()] = str(value).strip()
    return mapping


def _metadata_from_match(match) -> Tuple[str, Dict]:
    vid = match.get("id") if isinstance(match, dict) else _safe_getattr(match, "id")
    md = match.get("metadata", {}) if isinstance(match, dict) else (_safe_getattr(match, "metadata") or {})
    return vid, md or {}


def _chunk(iterable, size: int):
    seq = list(iterable)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _collect_parents_via_listing(
    index,
    namespace: str,
    limit_parents: Optional[int],
    channel_map: Dict[str, str],
) -> Dict[str, Dict[str, Optional[str]]]:
    ids = list(_list_paginated_ids(index, namespace))
    if not ids:
        LOG.warning("list_paginated returned zero IDs for namespace=%s", namespace)
        return {}

    if limit_parents:
        ids = ids[:limit_parents]
    LOG.info("Fetched %d vector ID(s) via pagination; loading metadata...", len(ids))

    meta_by_id = _fetch_metadata_chunked(index, namespace, ids)
    parent_info: Dict[str, Dict[str, Optional[str]]] = {}
    for vec_id, md in meta_by_id.items():
        node_type = str(md.get("node_type") or "").lower()
        if node_type != "parent":
            continue
        parent_id = md.get("parent_id") or md.get("video_id") or vec_id
        if not parent_id:
            continue
        channel_name, channel_id = _normalize_channel_id(md.get("channel_name"), md.get("channel_id"), channel_map)
        parent_info[str(parent_id)] = {
            "parent_vector_id": vec_id,
            "channel_name": channel_name,
            "channel_id": channel_id,
            "video_id": str(md.get("video_id") or parent_id),
            "published_at": md.get("published_at") or md.get("published_date"),
            "published_date": md.get("published_date"),
        }
        if limit_parents and len(parent_info) >= limit_parents:
            break

    LOG.info("Collected %d parent vector(s) via pagination for namespace=%s", len(parent_info), namespace)
    return parent_info


def _collect_parent_matches(
    index,
    namespace: str,
    dim: int,
    limit_parents: Optional[int],
    channel_map: Dict[str, str],
) -> Dict[str, Dict[str, Optional[str]]]:
    filter_parent = {"node_type": {"$eq": "parent"}}
    top_k = min(limit_parents or 10_000, 10_000)
    LOG.info("Querying namespace=%s for up to %d parent vectors...", namespace, top_k)
    try:
        res = index.query(
            namespace=namespace,
            top_k=top_k,
            vector=[0.0] * dim,
            include_metadata=True,
            filter=filter_parent,
        )
    except Exception as exc:
        LOG.error("Parent query failed for namespace %s: %s", namespace, exc)
        return _collect_parents_via_listing(index, namespace, limit_parents, channel_map)

    matches = res.get("matches", []) if isinstance(res, dict) else (_safe_getattr(res, "matches") or [])
    if not matches:
        LOG.info("Parent query returned zero matches; falling back to pagination listing.")
        return _collect_parents_via_listing(index, namespace, limit_parents, channel_map)

    parent_info: Dict[str, Dict[str, Optional[str]]] = {}
    for match in matches:
        vid, md = _metadata_from_match(match)
        if not vid:
            continue
        parent_id = md.get("parent_id") or md.get("video_id") or vid
        if not parent_id:
            continue
        channel_name, channel_id = _normalize_channel_id(md.get("channel_name"), md.get("channel_id"), channel_map)
        parent_info[str(parent_id)] = {
            "parent_vector_id": vid,
            "channel_name": channel_name,
            "channel_id": channel_id,
            "video_id": str(md.get("video_id") or parent_id),
            "published_at": md.get("published_at") or md.get("published_date"),
            "published_date": md.get("published_date"),
        }
        if limit_parents and len(parent_info) >= limit_parents:
            break

    if not limit_parents and len(parent_info) >= top_k:
        LOG.warning("Reached top_k=%d parents; collecting remaining via pagination listing.", top_k)
        extra = _collect_parents_via_listing(index, namespace, limit_parents, channel_map)
        parent_info.update(extra)

    LOG.info("Collected %d parent vector(s) for namespace=%s", len(parent_info), namespace)
    return parent_info


def backfill(
    index,
    index_name: str,
    namespace: str,
    channel_map: Dict[str, str],
    parents: Optional[Iterable[str]] = None,
    limit_parents: Optional[int] = None,
    only_missing: bool = False,
    dry_run: bool = False,
    child_batch_size: int = 200,
    parent_batch_size: int = 25,
) -> None:
    dim = _detect_dimension(index, namespace)
    LOG.info("Using index=%s namespace=%s (dim=%d)", index_name, namespace, dim)

    if parents:
        LOG.info("Fetching metadata for %d specified parent(s)...", len(parents))
        meta = index.fetch(ids=list(parents), namespace=namespace) or {}
        vectors = _safe_getattr(meta, "vectors") or (meta.get("vectors", {}) if isinstance(meta, dict) else {})
        parent_info: Dict[str, Dict[str, Optional[str]]] = {}
        for vec_id, vec in (vectors or {}).items():
            md = _safe_getattr(vec, "metadata") or (vec.get("metadata") if isinstance(vec, dict) else {}) or {}
            channel_name, channel_id = _normalize_channel_id(md.get("channel_name"), md.get("channel_id"), channel_map)
            parent_id = md.get("parent_id") or md.get("video_id") or vec_id
            parent_info[str(parent_id)] = {
                "parent_vector_id": vec_id,
                "channel_name": channel_name,
                "channel_id": channel_id,
                "video_id": str(md.get("video_id") or parent_id),
                "published_at": md.get("published_at") or md.get("published_date"),
                "published_date": md.get("published_date"),
            }
    else:
        parent_info = _collect_parent_matches(index, namespace, dim, limit_parents, channel_map)

    if not parent_info:
        LOG.warning("No parent vectors discovered; nothing to do.")
        return

    LOG.info("Prepared channel metadata for %d parent(s)", len(parent_info))
    missing_id_channels = defaultdict(int)

    total_children_scanned = 0
    total_updates = 0
    max_children = 10_000  # Pinecone query requires a top_k; use generous cap.

    items = list(parent_info.items())
    parent_batches = max(1, parent_batch_size)

    for batch_index in range(0, len(items), parent_batches):
        batch = items[batch_index : batch_index + parent_batches]
        LOG.info("Processing parent batch %d/%d (size=%d)", batch_index // parent_batches + 1,
                 (len(items) + parent_batches - 1) // parent_batches, len(batch))

        for parent_id, info in batch:
            ch_name = info.get("channel_name")
            ch_id = info.get("channel_id")
            if not ch_name:
                LOG.debug("Skipping parent %s (no channel_name)", parent_id)
                continue
            if not ch_id:
                missing_id_channels[ch_name] += 1

            video_id = info.get("video_id") or parent_id
            published_at = info.get("published_at")
            published_date = info.get("published_date")

            filt = {
                "parent_id": {"$eq": parent_id},
                "node_type": {"$in": ["child", "summary"]},
            }
            try:
                res = index.query(
                    namespace=namespace,
                    top_k=max_children,
                    vector=[0.0] * dim,
                    include_metadata=True,
                    filter=filt,
                )
            except Exception as exc:
                LOG.error("Query failed for parent %s: %s", parent_id, exc)
                continue

            matches = res.get("matches", []) if isinstance(res, dict) else (_safe_getattr(res, "matches") or [])
            if not matches:
                continue

            for chunk_idx, chunk in enumerate(_chunk(matches, max(1, child_batch_size)), 1):
                for match in chunk:
                    child_id, md = _metadata_from_match(match)
                    if not child_id:
                        continue
                    total_children_scanned += 1
                    patch: Dict[str, str] = {}

                    target_fields = {
                        "channel_name": ch_name,
                        "video_id": str(video_id) if video_id else None,
                    }
                    if ch_id:
                        target_fields["channel_id"] = ch_id
                    if published_at:
                        target_fields["published_at"] = published_at
                    if published_date:
                        target_fields["published_date"] = published_date

                    for key, desired in target_fields.items():
                        if desired is None:
                            continue
                        current_value = md.get(key)
                        if current_value == desired:
                            continue
                        if only_missing and current_value:
                            continue
                        patch[key] = desired

                    if not patch:
                        continue

                    total_updates += 1
                    if dry_run:
                        LOG.info("[dry-run] child=%s parent=%s patch=%s", child_id, parent_id, patch)
                    else:
                        index.update(id=child_id, namespace=namespace, set_metadata=patch)
                        LOG.info("[update] child=%s parent=%s patch=%s", child_id, parent_id, patch)

                LOG.info(
                    "Parent %s child batch %d processed (batch_size=%d) cumulative_children=%d updates=%d",
                    parent_id,
                    chunk_idx,
                    len(chunk),
                    total_children_scanned,
                    total_updates,
                )

        LOG.info(
            "Batch complete: parents_processed=%d/%d children_scanned=%d updates=%d",
            min(batch_index + len(batch), len(items)),
            len(items),
            total_children_scanned,
            total_updates,
        )

    if missing_id_channels:
        LOG.warning(
            "Missing channel_id for %d channel(s): %s",
            len(missing_id_channels),
            list(missing_id_channels.keys())[:10],
        )

    LOG.info(
        "Finished namespace=%s. Parents processed=%d, children scanned=%d, updates%s=%d",
        namespace,
        len(parent_info),
        total_children_scanned,
        " (dry-run)" if dry_run else "",
        total_updates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill child vectors with channel metadata.")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2"), help="Pinecone index name.")
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "videos"),
        help="Comma-separated namespace list (processed with 'videos' first if present).",
    )
    parser.add_argument("--channel-map", help="Optional JSON file mapping channel names/handles to canonical IDs.")
    parser.add_argument("--parents", help="Comma-separated parent_ids to restrict processing.")
    parser.add_argument("--limit-parents", type=int, help="Maximum number of parents to process." )
    parser.add_argument("--max-children", type=int, default=10_000, help="Top-K limit when querying children per parent.")
    parser.add_argument("--only-missing", action="store_true", help="Only update when metadata is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Log intended updates without writing to Pinecone.")
    parser.add_argument("--parent-batch-size", type=int, default=25, help="Number of parents to process per batch (default 25).")
    parser.add_argument("--child-batch-size", type=int, default=200, help="Number of child updates to apply before logging progress (default 200).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = "%(asctime)s - backfill - %(levelname)s - %(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    # File handler for persistent logs
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "backfill_child_channel_metadata.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    LOG.info("File logging enabled at %s", log_path.resolve())

    raw_namespaces = [ns.strip() for ns in str(args.namespace or "").split(",") if ns and ns.strip()]
    if not raw_namespaces:
        raw_namespaces = ["videos"]
    unique_order = list(dict.fromkeys(raw_namespaces))
    namespaces = sorted(unique_order, key=lambda ns: (0 if ns == "videos" else 1, unique_order.index(ns)))

    channel_map = _load_channel_map(args.channel_map)
    if channel_map:
        LOG.info("Loaded %d channel ID mappings from %s", len(channel_map), args.channel_map)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index_name)

    parents = None
    if args.parents:
        parents = [pid.strip() for pid in args.parents.split(",") if pid.strip()]

    for ns in namespaces:
        LOG.info("=== Processing namespace: %s ===", ns)
        backfill(
            index=index,
            index_name=args.index_name,
            namespace=ns,
            channel_map=channel_map,
            parents=parents,
            limit_parents=args.limit_parents,
            only_missing=args.only_missing,
            dry_run=args.dry_run,
            child_batch_size=max(1, args.child_batch_size),
            parent_batch_size=max(1, args.parent_batch_size),
        )


if __name__ == "__main__":
    main()
