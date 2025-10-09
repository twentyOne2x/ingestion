#!/usr/bin/env python3
"""
Duplicate/overlap checker for a Pinecone namespace.

Finds:
  1) Exact-duplicate segment_ids
  2) Content duplicates by source_hash
  3) Content duplicates by TEXT-ONLY hash (tolerates window shifts)
  4) Near-duplicate time overlaps within the same parent_id (IoU-based)

Focuses on reporting; does NOT delete. Outputs:
  - duplicate_summary.txt
  - duplicates_detailed.json
  - duplicates.csv
  - time_overlaps.txt

CLI:
  --index <name>                Pinecone index (or PINECONE_INDEX_NAME)
  --namespace <ns>              Namespace (or PINECONE_NAMESPACE, default 'videos')
  --output-dir <dir>            Output directory base
  --parents "id1,id2,..."       Limit analysis to these parent_ids
  --limit <N>                   Max vectors to scan (for debugging)
  --batch-size <N>              Pagination size for list_paginated (default 10000)
  --fetch-chunk <N>             Initial fetch IDs per request (default 200)
  --min-overlap-sec <f>         Minimum seconds of overlap to consider (default env DUP_MIN_OVERLAP_SEC or 2.0)
  --min-overlap-pct <f>         Minimum % of the shorter window (default env DUP_MIN_OVERLAP_PCT or 15)
  --expected-overlap-sec <f>    Ignore overlaps at/under this (pipeline pad/stride) (default env EXPECTED_PIPELINE_OVERLAP_S or 3.0)

Env (optional):
  PINECONE_API_KEY
  PINECONE_INDEX_NAME
  PINECONE_NAMESPACE
  DUP_MIN_OVERLAP_SEC
  DUP_MIN_OVERLAP_PCT
  EXPECTED_PIPELINE_OVERLAP_S
  PINECONE_FETCH_IDS_PER_REQ
"""

from __future__ import annotations
import os
import logging
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Iterable
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha1

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

from pinecone import Pinecone
import pandas as pd
from src.ingest_v2.configs.settings import settings_v2

# ───────── logging ─────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - dupcheck - %(levelname)s - %(message)s')
log = logging.getLogger("dupcheck")

# ───────── helpers ─────────
def _backoff(attempt: int, base: float = 0.5, cap: float = 20.0):
    delay = min(cap, base * (2 ** max(0, attempt)) + 0.5)
    time.sleep(delay)

def _as_dict(obj: Any) -> Dict[str, Any]:
    try:
        return obj.__dict__
    except Exception:
        return obj if isinstance(obj, dict) else {}

def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).lower()

def _text_hash(s: str) -> str:
    return sha1(_norm_text(s).encode("utf-8")).hexdigest()

def _overlap_stats(a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[float,float,float]:
    s1,e1 = a; s2,e2 = b
    inter = max(0.0, min(e1,e2) - max(s1,s2))
    if inter <= 0:
        return 0.0, 0.0, 0.0
    len1 = max(1e-9, e1 - s1)
    len2 = max(1e-9, e2 - s2)
    pct_of_shorter = inter / min(len1, len2) * 100.0
    union = len1 + len2 - inter
    iou = inter / max(1e-9, union)
    return inter, pct_of_shorter, iou

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

# ───────── data containers ─────────
@dataclass
class DuplicateReport:
    exact_by_segment: Dict[str, List[str]]
    dup_by_source_hash: Dict[str, List[str]]
    dup_by_text_hash: Dict[str, List[str]]
    time_overlaps: Dict[str, List[Dict[str, Any]]]
    summary: Dict[str, Any]

# ───────── pinecone I/O ─────────
def _list_paginated_ids(index, namespace: str, limit: int = 10000) -> Iterable[str]:
    token = None
    while True:
        res = index.list_paginated(namespace=namespace, limit=limit, pagination_token=token)
        # v3 shape
        vectors = getattr(res, "vectors", None)
        if vectors is None:
            # dict-like fallback
            vectors = _as_dict(res).get("vectors", [])
        for v in vectors or []:
            vid = getattr(v, "id", None) or _as_dict(v).get("id")
            if vid:
                yield vid
        # next token
        next_tok = None
        pagination = getattr(res, "pagination", None) or _as_dict(res).get("pagination")
        if pagination:
            next_tok = getattr(pagination, "next", None) or _as_dict(pagination).get("next")
        if not next_tok:
            break
        token = next_tok

def _fetch_metadata_chunked(index, namespace: str, ids: List[str], start_max_per: int = 200) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    i, max_per = 0, max(1, int(os.getenv("PINECONE_FETCH_IDS_PER_REQ", str(start_max_per))))
    while i < len(ids):
        chunk = ids[i:i+max_per]
        attempt = 0
        while True:
            try:
                res = index.fetch(ids=chunk, namespace=namespace, include_values=False) or {}
                vectors = getattr(res, "vectors", None) or res.get("vectors", {})
                # v3: dict[str, Vector]; v2: dict[str, dict]
                for vid, v in (vectors or {}).items():
                    md = getattr(v, "metadata", None)
                    if md is None and isinstance(v, dict):
                        md = v.get("metadata", {})
                    out[vid] = md or {}
                i += len(chunk)
                break
            except Exception as e:
                msg = f"{getattr(e,'status',None)} {str(e)}"
                if "414" in msg or "Request-URI Too Large" in msg or "413" in msg or "Payload Too Large" in msg:
                    if max_per == 1:
                        log.error("Cannot shrink fetch chunk further; failing at id=%s", chunk[0])
                        raise
                    old = max_per
                    max_per = max(1, max_per // 2)
                    log.warning("Fetch payload too large; reducing ids per request %d → %d", old, max_per)
                    chunk = ids[i:i+max_per]
                    continue
                attempt += 1
                log.warning("Fetch failed (attempt %d, size=%d): %s", attempt, len(chunk), e)
                _backoff(attempt)
    return out

# ───────── analysis ─────────
def analyze_streaming(
    all_ids_iter: Iterable[str],
    fetcher,
    *,
    parents_filter: Optional[set] = None,
    min_overlap_sec: float = 2.0,
    min_overlap_pct: float = 15.0,
    expected_overlap_sec: float = 3.0,
    fetch_chunk: int = 200,
) -> DuplicateReport:
    by_segment: Dict[str, List[str]] = defaultdict(list)
    by_source: Dict[str, List[str]] = defaultdict(list)
    by_text: Dict[str, List[str]] = defaultdict(list)
    per_parent: Dict[str, List[Tuple[str,float,float]]] = defaultdict(list)

    scanned = 0
    ids_buf: List[str] = []
    def _flush():
        nonlocal scanned
        if not ids_buf: return
        md_by_id = fetcher(ids_buf)
        for vid, md in md_by_id.items():
            scanned += 1
            # optional filter by parent
            parent_id = md.get("parent_id")
            if parents_filter is not None and parent_id not in parents_filter:
                continue

            # exact duplicates by segment_id
            seg_id = md.get("segment_id")
            if seg_id:
                by_segment[seg_id].append(vid)

            # content duplicates by source_hash
            sh = md.get("source_hash")
            if sh:
                by_source[str(sh)].append(vid)

            # text-only hash
            txt = md.get("text") or ""
            if txt:
                by_text[_text_hash(txt)].append(vid)

            # time overlap candidates (skip summaries)
            if md.get("node_type") != "summary":
                s = _safe_float(md.get("start_s")); e = _safe_float(md.get("end_s"))
                if parent_id and s is not None and e is not None and e > s:
                    per_parent[parent_id].append((vid, s, e))
        ids_buf.clear()

    for vid in all_ids_iter:
        ids_buf.append(vid)
        if len(ids_buf) >= fetch_chunk:
            _flush()
    _flush()

    # collapse groups with >1
    exact_dups = {k:v for k,v in by_segment.items() if len(v) > 1}
    source_dups = {k:v for k,v in by_source.items() if len(v) > 1}
    text_dups = {k:v for k,v in by_text.items() if len(v) > 1}

    # compute overlaps (IoU & pct-of-shorter), ignoring tiny expected overlaps
    overlaps: Dict[str, List[Dict[str, Any]]] = {}
    for pid, spans in per_parent.items():
        if len(spans) < 2:
            continue
        spans.sort(key=lambda x: x[1])  # by start
        active: List[Tuple[str,float,float]] = []
        hits: List[Dict[str, Any]] = []
        for vid, s, e in spans:
            # pop fully ended actives
            active = [a for a in active if a[2] > s]
            for avid, as_, ae in active:
                inter, pct_short, iou = _overlap_stats((s,e),(as_,ae))
                if inter > max(min_overlap_sec, expected_overlap_sec) and pct_short >= min_overlap_pct:
                    hits.append({
                        "ids": (avid, vid),
                        "windows": ((as_, ae), (s, e)),
                        "overlap_seconds": inter,
                        "overlap_percent_of_shorter": pct_short,
                        "iou": iou,
                    })
            active.append((vid, s, e))
        if hits:
            overlaps[pid] = hits

    summary = {
        "scanned_vectors": scanned,
        "unique_segment_ids": len(by_segment),
        "duplicate_segment_id_groups": len(exact_dups),
        "vectors_in_duplicate_segment_ids": sum(len(v) for v in exact_dups.values()),
        "unique_source_hashes": len(by_source),
        "duplicate_source_hash_groups": len(source_dups),
        "vectors_in_duplicate_source_hashes": sum(len(v) for v in source_dups.values()),
        "unique_text_hashes": len(by_text),
        "duplicate_text_hash_groups": len(text_dups),
        "vectors_in_duplicate_text_hashes": sum(len(v) for v in text_dups.values()),
        "parents_with_overlaps": len(overlaps),
        "total_overlapping_pairs": sum(len(v) for v in overlaps.values()),
        "overlap_thresholds": {
            "min_overlap_sec": min_overlap_sec,
            "min_overlap_pct_of_shorter": min_overlap_pct,
            "expected_overlap_sec_ignored": expected_overlap_sec,
        },
    }
    return DuplicateReport(
        exact_by_segment=exact_dups,
        dup_by_source_hash=source_dups,
        dup_by_text_hash=text_dups,
        time_overlaps=overlaps,
        summary=summary,
    )

# ───────── outputs ─────────
def _write_reports(report: DuplicateReport, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) summary text
    sp = outdir / "duplicate_summary.txt"
    with sp.open("w", encoding="utf-8") as f:
        f.write("="*64 + "\nDUPLICATE ANALYSIS SUMMARY\n" + "="*64 + "\n\n")
        for k,v in report.summary.items():
            f.write(f"{k}: {v}\n")

        def _dump_group(title: str, groups: Dict[str, List[str]], key_fmt=lambda k: k):
            f.write("\n" + "-"*64 + f"\n{title}\n" + "-"*64 + "\n")
            if not groups:
                f.write("None.\n")
                return
            for k, ids in list(sorted(groups.items(), key=lambda kv: -len(kv[1])))[:20]:
                f.write(f"key: {key_fmt(k)}  (count={len(ids)})\n  sample: {', '.join(ids[:6])}\n")

        _dump_group("Exact duplicates by segment_id", report.exact_by_segment)
        _dump_group("Content duplicates by source_hash", report.dup_by_source_hash, key_fmt=lambda k: f"{k[:16]}…")
        _dump_group("Content duplicates by TEXT hash", report.dup_by_text_hash, key_fmt=lambda k: f"{k[:16]}…")

        f.write("\n" + "-"*64 + "\nTime overlaps (per parent)\n" + "-"*64 + "\n")
        if not report.time_overlaps:
            f.write("None.\n")
        else:
            for pid, hits in list(sorted(report.time_overlaps.items(), key=lambda kv: -len(kv[1])))[:50]:
                f.write(f"parent_id={pid} overlapping_pairs={len(hits)}\n")
                for h in hits[:3]:
                    (s1,e1),(s2,e2)=h["windows"]
                    f.write(f"  {h['ids'][0]} [{s1:.2f}-{e1:.2f}] vs {h['ids'][1]} [{s2:.2f}-{e2:.2f}] "
                            f"→ {h['overlap_seconds']:.2f}s, {h['overlap_percent_of_shorter']:.1f}% (IoU={h['iou']:.2f})\n")
                if len(hits) > 3:
                    f.write(f"  ... {len(hits)-3} more\n")

    log.info("Wrote %s", sp)

    # 2) JSON (full)
    jp = outdir / "duplicates_detailed.json"
    with jp.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": report.summary,
            "exact_by_segment": report.exact_by_segment,
            "dup_by_source_hash": report.dup_by_source_hash,
            "dup_by_text_hash": report.dup_by_text_hash,
            "time_overlaps": report.time_overlaps,
        }, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", jp)

    # 3) CSV (flat)
    rows: List[Dict[str, Any]] = []
    for k, ids in report.exact_by_segment.items():
        for vid in ids:
            rows.append({"duplicate_type": "segment_id", "group_key": k, "vector_id": vid, "group_size": len(ids)})
    for k, ids in report.dup_by_source_hash.items():
        for vid in ids:
            rows.append({"duplicate_type": "source_hash", "group_key": k, "vector_id": vid, "group_size": len(ids)})
    for k, ids in report.dup_by_text_hash.items():
        for vid in ids:
            rows.append({"duplicate_type": "text_hash", "group_key": k, "vector_id": vid, "group_size": len(ids)})

    if rows:
        df = pd.DataFrame(rows)
        cp = outdir / "duplicates.csv"
        df.to_csv(cp, index=False)
        log.info("Wrote %s (%d rows)", cp, len(rows))

    # 4) time overlaps text (detail)
    if report.time_overlaps:
        op = outdir / "time_overlaps.txt"
        with op.open("w", encoding="utf-8") as f:
            f.write("="*64 + "\nTIME OVERLAPS (IoU / pct-of-shorter)\n" + "="*64 + "\n\n")
            for pid, hits in sorted(report.time_overlaps.items()):
                f.write(f"Parent {pid} — {len(hits)} overlapping pairs\n")
                for h in hits:
                    (s1,e1),(s2,e2) = h["windows"]
                    f.write(f"  {h['ids'][0]} [{s1:.2f},{e1:.2f}] vs {h['ids'][1]} [{s2:.2f},{e2:.2f}]  "
                            f"overlap={h['overlap_seconds']:.2f}s  pctShorter={h['overlap_percent_of_shorter']:.1f}%  IoU={h['iou']:.2f}\n")
                f.write("\n")
        log.info("Wrote %s", op)

# ───────── main ─────────
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Detect duplicate/overlapping vectors in a Pinecone namespace (report only)")
    ap.add_argument("--index", default=None, help="Pinecone index name (or PINECONE_INDEX_NAME)")
    ap.add_argument("--namespace", default=None, help="Namespace (or PINECONE_NAMESPACE, default 'videos')")
    default_outdir = Path(settings_v2.PIPELINE_STORAGE_ROOT) / "duplicate_reports"
    ap.add_argument("--output-dir", default=str(default_outdir), help="Output directory base")
    ap.add_argument("--parents", default=None, help="Comma-separated parent_ids to include (optional)")
    ap.add_argument("--limit", type=int, default=0, help="Max vectors to scan (0 = no limit)")
    ap.add_argument("--batch-size", type=int, default=10000, help="IDs per list_paginated page")
    ap.add_argument("--fetch-chunk", type=int, default=200, help="Initial ids per fetch() call (auto-shrinks on 413/414)")
    ap.add_argument("--min-overlap-sec", type=float, default=float(os.getenv("DUP_MIN_OVERLAP_SEC", "2.0")))
    ap.add_argument("--min-overlap-pct", type=float, default=float(os.getenv("DUP_MIN_OVERLAP_PCT", "15.0")))
    ap.add_argument("--expected-overlap-sec", type=float, default=float(os.getenv("EXPECTED_PIPELINE_OVERLAP_S", "3.0")))
    args = ap.parse_args()

    index_name = args.index or os.getenv("PINECONE_INDEX_NAME")
    namespace = args.namespace or os.getenv("PINECONE_NAMESPACE", "videos")
    if not index_name:
        log.error("Index name required (--index or PINECONE_INDEX_NAME)")
        return 1
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        log.error("PINECONE_API_KEY not set")
        return 1

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)

    # stats
    try:
        stats = idx.describe_index_stats()
        ns_stats = (stats.get("namespaces", {}) if isinstance(stats, dict) else {}).get(namespace, {}) or {}
        total = ns_stats.get("vector_count", "unknown")
        log.info("Index '%s' namespace '%s' vectors=%s", index_name, namespace, total)
    except Exception as e:
        log.warning("describe_index_stats failed (non-fatal): %s", e)

    # parents filter
    parents_filter = None
    if args.parents:
        parents_filter = {x.strip() for x in args.parents.split(",") if x.strip()}
        log.info("Filtering to %d parent_id(s)", len(parents_filter))

    # iterate ids (with optional cap)
    def _ids_iter():
        n = 0
        for vid in _list_paginated_ids(idx, namespace=namespace, limit=args.batch_size):
            yield vid
            n += 1
            if args.limit and n >= args.limit:
                break

    fetcher = lambda chunk_ids: _fetch_metadata_chunked(idx, namespace, chunk_ids, start_max_per=args.fetch_chunk)

    t0 = time.time()
    report = analyze_streaming(
        all_ids_iter=_ids_iter(),
        fetcher=fetcher,
        parents_filter=parents_filter,
        min_overlap_sec=args.min_overlap_sec,
        min_overlap_pct=args.min_overlap_pct,
        expected_overlap_sec=args.expected_overlap_sec,
        fetch_chunk=args.fetch_chunk,
    )
    dt = time.time() - t0

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir) / f"{index_name}_{namespace}_{ts}"
    _write_reports(report, outdir)

    print("\n" + "="*64)
    print("DUPLICATE ANALYSIS COMPLETE")
    print("="*64)
    for k,v in report.summary.items():
        print(f"{k}: {v}")
    print(f"\nReports saved to: {outdir} (elapsed {dt:.2f}s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
