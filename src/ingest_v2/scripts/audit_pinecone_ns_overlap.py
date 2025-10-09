#!/usr/bin/env python3
# audit_pinecone_ns_overlap.py
from __future__ import annotations

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional
from pinecone import Pinecone
from src.ingest_v2.configs.settings import settings_v2

log = logging.getLogger("ns_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - ns_audit - %(levelname)s - %(message)s")

# dotenv loading (optional)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:
    pass


def _list_paginated_ids(index, namespace: str, limit: int = 99) -> Iterable[str]:
    lim = max(1, min(int(limit or 99), 99))
    token = None
    pages = 0
    while True:
        res = index.list_paginated(namespace=namespace, limit=lim, pagination_token=token)
        pages += 1
        vectors = getattr(res, "vectors", None) or (res.get("vectors", []) if isinstance(res, dict) else [])
        for v in vectors or []:
            vid = getattr(v, "id", None) or (v.get("id") if isinstance(v, dict) else None)
            if vid:
                yield vid
        pagination = getattr(res, "pagination", None) or (res.get("pagination") if isinstance(res, dict) else None)
        token = getattr(pagination, "next", None) or (pagination.get("next") if isinstance(pagination, dict) else None)
        if not token:
            break
        if pages % 50 == 0:
            log.info("[list] ns=%r pages=%d (~%d ids)", namespace, pages, lim * pages)


def _fetch_metadata_chunked(index, namespace: str, ids: List[str], start_max_per: int = 200) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    i, max_per = 0, start_max_per
    t0 = time.time()
    while i < len(ids):
        chunk = ids[i:i + max_per]
        res = index.fetch(ids=chunk, namespace=namespace) or {}
        vectors = getattr(res, "vectors", None) or (res.get("vectors", {}) if isinstance(res, dict) else {})
        for vid, v in (vectors or {}).items():
            md = getattr(v, "metadata", None) or (v.get("metadata") if isinstance(v, dict) else {}) or {}
            out[vid] = md
        i += len(chunk)
        if i % 5000 == 0:
            log.info("[fetch] ns=%r %d/%d (%.1f%%) in %.1fs", namespace, i, len(ids), 100.0 * i / max(1, len(ids)), time.time() - t0)
    return out


def _field(md: Dict[str, Any], key: str, default=None):
    v = md.get(key, default)
    return v if v is not None else default


def _parent_id_from_md(md: Dict[str, Any]) -> Optional[str]:
    pid = _field(md, "parent_id", None)
    if not pid:
        pid = _field(md, "video_id", None)
    if pid is None:
        return None
    return str(pid)


def _summarize_ns(index, ns: str, list_limit: int = 99) -> Dict[str, Any]:
    ids = list(_list_paginated_ids(index, ns, limit=list_limit))
    log.info("[ns] %r total_ids=%d", ns, len(ids))

    md_by_id = _fetch_metadata_chunked(index, ns, ids, start_max_per=200)

    parent_counts: Dict[str, int] = {}
    sample_meta: Dict[str, Dict[str, Any]] = {}
    for vid, md in md_by_id.items():
        pid = _parent_id_from_md(md)
        if not pid:
            continue
        parent_counts[pid] = parent_counts.get(pid, 0) + 1
        if pid not in sample_meta:
            sample_meta[pid] = {
                "title": _field(md, "title", None),
                "channel_name": _field(md, "channel_name", None),
                "published_at": _field(md, "published_at", _field(md, "published_date", None)),
                "document_type": _field(md, "document_type", None),
            }

    total_vectors = len(ids)
    distinct_parents = len(parent_counts)
    return {
        "namespace": ns,
        "total_vectors": total_vectors,
        "distinct_parents": distinct_parents,
        "parent_counts": parent_counts,     # {parent_id: child_count}
        "parent_samples": sample_meta,      # minimal display meta per parent
    }


def _print_summary(ns_stats: List[Dict[str, Any]]) -> None:
    headers = ["namespace", "total_vectors", "distinct_parents"]
    colw = {h: max(len(h), max((len(str(s[h])) for s in ns_stats), default=0)) for h in headers}
    line = " | ".join(h.ljust(colw[h]) for h in headers)
    sep = "-+-".join("-" * colw[h] for h in headers)
    print("\n=== Namespace Summary ===")
    print(line)
    print(sep)
    for s in ns_stats:
        print(" | ".join(str(s[h]).ljust(colw[h]) for h in headers))
    print("=========================\n")


def _print_duplicates_report(dups: List[Tuple[str, int, int]], samples_a: Dict[str, Dict[str, Any]], samples_b: Dict[str, Dict[str, Any]], ns_a: str, ns_b: str, topn: int = 40) -> None:
    print(f"=== Duplicate parent_ids across namespaces: {ns_a!r} ∩ {ns_b!r} ===")
    print("parent_id | count_a | count_b | title [channel] | date")
    print("----------+---------+---------+------------------+------")
    for pid, ca, cb in dups[:topn]:
        meta = samples_a.get(pid) or samples_b.get(pid) or {}
        title = meta.get("title") or "(untitled)"
        chan = meta.get("channel_name") or "N/A"
        date = meta.get("published_at") or "N/A"
        print(f"{pid} | {ca} | {cb} | {title} [{chan}] | {date}")
    print("=============================================\n")


def main():
    ap = argparse.ArgumentParser(description="Audit Pinecone namespaces for vector/parent counts and cross-namespace duplicates.")
    ap.add_argument("--index", default=None, help="Pinecone index name (default env PINECONE_INDEX_NAME)")
    ap.add_argument("--namespaces", default=",videos", help="Comma-separated list. Default scans empty-string and 'videos' (',videos').")
    default_report_dir = Path(settings_v2.PIPELINE_STORAGE_ROOT) / "audit_reports"
    ap.add_argument("--report-dir", default=str(default_report_dir), help="Where to write JSON report")
    ap.add_argument("--topn", type=int, default=100, help="How many duplicate parents to print")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is required")
    idx_name = args.index or os.getenv("PINECONE_INDEX_NAME")
    if not idx_name:
        raise SystemExit("Provide --index or set PINECONE_INDEX_NAME")

    # order: empty namespace first, then videos by default
    ns_list = [ns.strip() for ns in args.namespaces.split(",")]

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(idx_name)

    ns_stats: List[Dict[str, Any]] = []
    for ns in ns_list:
        stats = _summarize_ns(idx, ns)
        ns_stats.append(stats)

    _print_summary(ns_stats)

    # If at least two namespaces, compute duplicates between first two
    dup_report: Dict[str, Any] = {}
    if len(ns_stats) >= 2:
        a, b = ns_stats[0], ns_stats[1]
        pa, pb = a["parent_counts"], b["parent_counts"]
        inter = sorted(set(pa.keys()) & set(pb.keys()))
        dup_rows = [(pid, pa.get(pid, 0), pb.get(pid, 0)) for pid in inter]
        dup_rows.sort(key=lambda t: (t[1] + t[2]), reverse=True)

        _print_duplicates_report(dup_rows, a["parent_samples"], b["parent_samples"], a["namespace"], b["namespace"], topn=args.topn)

        dup_report = {
            "namespace_a": a["namespace"],
            "namespace_b": b["namespace"],
            "duplicate_parent_ids_count": len(inter),
            "duplicates": [
                {
                    "parent_id": pid,
                    "count_ns_a": ca,
                    "count_ns_b": cb,
                    "meta": (a["parent_samples"].get(pid) or b["parent_samples"].get(pid) or {}),
                }
                for (pid, ca, cb) in dup_rows
            ],
        }

    # Write JSON report
    outdir = Path(args.report_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jp = outdir / f"ns_audit_{idx_name}_{'_'.join([ns if ns else 'EMPTY' for ns in ns_list])}_{ts}.json"
    payload = {
        "index": idx_name,
        "namespaces": ns_list,
        "summary": [
            {
                "namespace": s["namespace"],
                "total_vectors": s["total_vectors"],
                "distinct_parents": s["distinct_parents"],
            }
            for s in ns_stats
        ],
        "detail_per_namespace": {
            (s["namespace"] if s["namespace"] != "" else "EMPTY"): {
                "parent_counts": s["parent_counts"],
                "parent_samples": s["parent_samples"],
            }
            for s in ns_stats
        },
        "duplicates": dup_report,
        "generated_at": ts,
    }
    with jp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Wrote report %s", jp)


if __name__ == "__main__":
    main()
