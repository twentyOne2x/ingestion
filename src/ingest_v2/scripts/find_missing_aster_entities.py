#!/usr/bin/env python3
# inspect_aster_entities_ci.py (fast/loggy, dual-namespace + summary)
from __future__ import annotations
import os, json, time, logging, argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterable
from pathlib import Path
from pinecone import Pinecone

log = logging.getLogger("aster_inspect_ci")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - aster_inspect_ci - %(levelname)s - %(message)s")

DEFAULT_NEEDLE = "aster"
NEEDLE_TOK = lambda s: (s or "").strip().lower()

# dotenv loading
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:
    pass


def _needle_variants(needle: str) -> Tuple[List[str], List[str]]:
    n = NEEDLE_TOK(needle)
    return list({n, n.upper(), n.title()}), list({f"${n}", f"${n.upper()}", f"${n.title()}"})

def _detect_dimension(idx, namespace: str) -> int:
    try:
        stats = idx.describe_index_stats()
        ns = (stats.get("namespaces", {}) if isinstance(stats, dict) else {}).get(namespace, {}) or {}
        dim = ns.get("dimension")
        if dim: return int(dim)
    except Exception:
        pass
    for d in (3072, 1536, 1024, 768):
        try:
            idx.query(namespace=namespace, top_k=1, vector=[0.0]*d, include_metadata=False)
            return d
        except Exception:
            continue
    raise RuntimeError("Could not determine index dimension")

def _list_paginated_ids(index, namespace: str, limit: int = 99) -> Iterable[str]:
    lim = max(1, min(int(limit or 99), 99))
    token = None
    pages = 0
    while True:
        pages += 1
        res = index.list_paginated(namespace=namespace, limit=lim, pagination_token=token)
        vectors = getattr(res, "vectors", None) or (res.get("vectors", []) if isinstance(res, dict) else [])
        if pages % 50 == 0:
            log.info("list_paginated pages=%d (+~%d ids)", pages, lim*50)
        for v in vectors or []:
            vid = getattr(v, "id", None) or (v.get("id") if isinstance(v, dict) else None)
            if vid: yield vid
        pagination = getattr(res, "pagination", None) or (res.get("pagination") if isinstance(res, dict) else None)
        next_tok = getattr(pagination, "next", None) or (pagination.get("next") if isinstance(pagination, dict) else None)
        if not next_tok: break
        token = next_tok

def _fetch_metadata_chunked(index, namespace: str, ids: List[str], start_max_per: int = 200) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    i, max_per = 0, start_max_per
    t0 = time.time()
    while i < len(ids):
        chunk = ids[i:i+max_per]
        res = index.fetch(ids=chunk, namespace=namespace) or {}
        vectors = getattr(res, "vectors", None) or (res.get("vectors", {}) if isinstance(res, dict) else {})
        for vid, v in (vectors or {}).items():
            md = getattr(v, "metadata", None) or (v.get("metadata") if isinstance(v, dict) else {}) or {}
            out[vid] = md
        i += len(chunk)
        if i % 5000 == 0:
            log.info("fetch_meta progress: %d/%d (%.1f%%) in %.1fs", i, len(ids), 100.0*i/max(1,len(ids)), time.time()-t0)
    return out

def _or_filters_any_of(pairs: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    ors = []
    for field, values in pairs:
        values = [v for v in values if v]
        if values: ors.append({field: {"$in": values}})
    if not ors: return {}
    return {"$or": ors} if len(ors) > 1 else ors[0]

def _filter_by_parent_or_video(pid: str) -> dict:
    return {"$or": [{"parent_id": {"$eq": pid}}, {"video_id": {"$eq": pid}}]}

def _field(md: Dict[str, Any], key: str, default=None):
    v = md.get(key, default)
    return v if v is not None else default

@dataclass
class ParentHit:
    vector_id: str
    parent_id: str
    title: Optional[str]
    channel_name: Optional[str]
    published_at: Optional[str]
    router_tags: Optional[List[str]]
    canonical_entities: Optional[List[str]]

@dataclass
class ChildHit:
    vector_id: str
    segment_id: Optional[str]
    start_hms: Optional[str]
    end_hms: Optional[str]
    has_needle_entity: bool
    has_needle_text: bool
    entities: Optional[List[str]]
    canonical_entities: Optional[List[str]]
    clip_url: Optional[str]
    speaker: Optional[str]
    score: Optional[float]

def probe_known_parents(idx, ns: str, dim: int, ids: List[str]) -> List[ParentHit]:
    hits: List[ParentHit] = []
    for pid in ids:
        f = _filter_by_parent_or_video(pid)
        res = idx.query(namespace=ns, top_k=10000, vector=[0.0]*dim, include_metadata=True, filter=f)
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []) or []
        if matches:
            md0 = matches[0].get("metadata", {}) if isinstance(matches[0], dict) else getattr(matches[0], "metadata", {}) or {}
            hits.append(ParentHit(
                vector_id=pid, parent_id=pid,
                title=_field(md0,"title",None), channel_name=_field(md0,"channel_name",None),
                published_at=_field(md0,"published_at",_field(md0,"published_date",None)),
                router_tags=_field(md0,"router_tags",None), canonical_entities=_field(md0,"canonical_entities",None),
            ))
            log.info("probe: FOUND children for parent/video_id=%s count=%d", pid, len(matches))
        else:
            log.info("probe: no children for parent/video_id=%s", pid)
    return hits

def find_parents_by_entities(idx, ns: str, dim: int, needle: str, limit_parents: int) -> List[ParentHit]:
    ents, cents = _needle_variants(needle)
    f = _or_filters_any_of([("canonical_entities", cents), ("router_tags", ents), ("aliases", ents)])
    if not f: return []
    res = idx.query(namespace=ns, top_k=min(limit_parents, 100), vector=[0.0]*dim, include_metadata=True, filter=f)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []) or []
    out: List[ParentHit] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        out.append(ParentHit(
            vector_id=m.get("id") if isinstance(m, dict) else getattr(m, "id", ""),
            parent_id=_field(md, "parent_id", _field(md, "video_id", "")),
            title=_field(md, "title", None),
            channel_name=_field(md, "channel_name", None),
            published_at=_field(md, "published_at", _field(md, "published_date", None)),
            router_tags=_field(md, "router_tags", None),
            canonical_entities=_field(md, "canonical_entities", None),
        ))
    log.info("entity/tag search → %d parents", len(out))
    return out

def find_parents_by_year_buckets(idx, ns: str, dim: int, year_start: int, year_end: int, topk_per_bucket: int = 10000) -> List[ParentHit]:
    out: Dict[str, ParentHit] = {}
    for fld in ("published_at","published_date"):
        for year in range(year_start, year_end+1):
            f = {"node_type":{"$eq":"parent"}, fld:{"$gte":f"{year}-01-01","$lte":f"{year}-12-31"}}
            try:
                res = idx.query(namespace=ns, top_k=topk_per_bucket, vector=[0.0]*dim, include_metadata=True, filter=f)
            except Exception as e:
                log.warning("year bucket query failed for %s: %s", fld, e)
                continue
            matches = res.get("matches", []) if isinstance(res, dict) else getattr(res,"matches",[]) or []
            log.info("year %d (%s) → %d parent candidates", year, fld, len(matches))
            for m in matches:
                md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m,"metadata",{}) or {}
                pid = _field(md, "parent_id", _field(md, "video_id", "")) or ""
                if not pid: continue
                if pid not in out:
                    out[pid] = ParentHit(
                        vector_id=m.get("id") if isinstance(m, dict) else getattr(m,"id",""),
                        parent_id=pid, title=_field(md,"title",None),
                        channel_name=_field(md,"channel_name",None),
                        published_at=_field(md,"published_at",_field(md,"published_date",None)),
                        router_tags=_field(md,"router_tags",None),
                        canonical_entities=_field(md,"canonical_entities",None),
                    )
    log.info("year-bucket sweep total parents=%d", len(out))
    return list(out.values())

def find_parents_by_title_fallback(idx, ns: str, limit_scan: int, needle: str) -> List[ParentHit]:
    log.info("fallback: scanning titles/descriptions (cap=%d ids)...", limit_scan)
    ids, scanned = [], 0
    for vid in _list_paginated_ids(idx, namespace=ns, limit=99):
        ids.append(vid); scanned += 1
        if scanned % 5000 == 0: log.info("fallback: listed %d ids...", scanned)
        if limit_scan and len(ids) >= limit_scan: break
    md_by_id = _fetch_metadata_chunked(idx, ns, ids, start_max_per=200)
    n = NEEDLE_TOK(needle)
    parents: Dict[str, ParentHit] = {}
    for i,(vid, md) in enumerate(md_by_id.items(),1):
        if i % 10000 == 0: log.info("fallback: inspected %d/%d metas...", i, len(md_by_id))
        title = _field(md,"title","") or ""; desc = _field(md,"description","") or ""; summ = _field(md,"topic_summary","") or ""
        if n not in f"{title} {desc} {summ}".lower(): continue
        pid = str(_field(md, "parent_id", _field(md, "video_id", "")) or "")
        if not pid:
            continue
        if pid not in parents:
            parents[pid] = ParentHit(vid, pid, title or None, _field(md,"channel_name",None),
                                     _field(md,"published_at",_field(md,"published_date",None)),
                                     _field(md,"router_tags",None), _field(md,"canonical_entities",None))
    log.info("fallback: matched parents=%d", len(parents))
    return list(parents.values())

def fetch_children_for_parent(idx, ns: str, dim: int, parent_id: str, needle: str, limit_children: int) -> List[ChildHit]:
    res = idx.query(namespace=ns, top_k=min(limit_children, 10000), vector=[0.0]*dim, include_metadata=True, filter=_filter_by_parent_or_video(parent_id))
    ents, cents = _needle_variants(needle)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res,"matches",[]) or []
    out: List[ChildHit] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m,"metadata",{}) or {}
        e = _field(md,"entities",[]) or []; ce = _field(md,"canonical_entities",[]) or []; txt = _field(md,"text","") or ""
        has_entity = bool({x.lower() for x in e}&{x.lower() for x in ents}) or bool({x.lower() for x in ce}&{x.lower() for x in cents})
        has_text   = NEEDLE_TOK(needle) in txt.lower()
        out.append(ChildHit(
            vector_id=m.get("id") if isinstance(m, dict) else getattr(m,"id",""),
            segment_id=_field(md,"segment_id",None),
            start_hms=_field(md,"start_hms",None), end_hms=_field(md,"end_hms",None),
            has_needle_entity=has_entity, has_needle_text=has_text,
            entities=e, canonical_entities=ce,
            clip_url=_field(md,"clip_url",_field(md,"url",None)), speaker=_field(md,"speaker",None),
            score=(m.get("score") if isinstance(m, dict) else getattr(m,"score",None)),
        ))
    return out

def summarize_children(children: List[ChildHit]) -> Dict[str, Any]:
    with_ent = [c for c in children if c.has_needle_entity]
    without = [c for c in children if not c.has_needle_entity]
    text_hits = [c for c in without if c.has_needle_text]
    return {
        "total_children": len(children), "with_entity": len(with_ent), "without_entity": len(without),
        "missing_fraction": round((len(without)/max(1,len(children))),3),
        "text_hits_without_entity": len(text_hits),
        "examples_missing": [{
            "vector_id": c.vector_id, "segment_id": c.segment_id, "time": f"{c.start_hms or ''}-{c.end_hms or ''}",
            "clip_url": c.clip_url, "speaker": c.speaker, "entities": c.entities,
            "canonical_entities": c.canonical_entities, "text_has_needle": c.has_needle_text,
        } for c in without[:8]],
    }

def run_namespace(index, ns: str, dim: int, parents: Optional[List[str]],
                  outdir: Optional[str], limit_parents: int, limit_children: int, needle: str, scan_cap: int,
                  probe_ids: Optional[List[str]], year_start: int, year_end: int,
                  idx_name: str) -> Dict[str, Any]:
    log.info("Using index=%s namespace=%r dim=%d", idx_name, ns, dim)

    parent_hits: List[ParentHit] = []
    if probe_ids:
        log.info("probing explicit parent/video ids: %s", ",".join(probe_ids))
        parent_hits.extend(probe_known_parents(index, ns, dim, probe_ids))

    if not parents and not parent_hits:
        parent_hits = find_parents_by_entities(index, ns, dim, needle, limit_parents=limit_parents)

    if not parents and not parent_hits:
        parent_hits = find_parents_by_year_buckets(index, ns, dim, year_start, year_end, topk_per_bucket=10000)

    if not parents and not parent_hits:
        parent_hits = find_parents_by_title_fallback(index, ns, limit_scan=scan_cap, needle=needle)

    if parents:
        parent_hits = [ParentHit(vector_id=f"(unknown for {pid})", parent_id=pid, title=None, channel_name=None,
                                 published_at=None, router_tags=None, canonical_entities=None) for pid in parents]

    log.info("Parent candidates total=%d", len(parent_hits))
    results: Dict[str, Any] = {"needle": needle, "namespace": ns, "parents": []}
    for i, ph in enumerate(parent_hits, 1):
        t0 = time.time()
        kids = fetch_children_for_parent(index, ns, dim, ph.parent_id, needle, limit_children=limit_children)
        summary = summarize_children(kids)
        log.info("Parent[%d/%d] %s | %s (%s) — children=%d, with_entity=%d, missing=%d, text_hits_wo_entity=%d (%.2fs)",
                 i, len(parent_hits), ph.parent_id, ph.title or "(untitled)", ph.channel_name or "N/A",
                 summary["total_children"], summary["with_entity"], summary["without_entity"],
                 summary["text_hits_without_entity"], time.time()-t0)
        results["parents"].append({"parent": asdict(ph), "children_summary": summary})

    if outdir:
        p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_ns = ns if ns != "" else "EMPTY"
        jp = p / f"entities_{NEEDLE_TOK(needle)}_{idx_name}_{safe_ns}_{ts}.json"
        with jp.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info("Wrote %s", jp)

    print(f"\n==== Entity Coverage Report (needle='{needle}', namespace='{ns}') ====")
    for item in results["parents"]:
        ph = item["parent"]; sm = item["children_summary"]
        print(f"- {ph['parent_id']} | {(ph.get('title') or '(untitled)')} [{(ph.get('channel_name') or 'N/A')}]")
        print(f"  children: {sm['total_children']} | with entity: {sm['with_entity']} | missing: {sm['without_entity']} (frac={sm['missing_fraction']}) | text_hits_wo_entity: {sm['text_hits_without_entity']}")
        if sm["examples_missing"]:
            print("  examples missing entities:")
            for ex in sm["examples_missing"]:
                print(f"    • seg={ex['segment_id']} time={ex['time']} url={ex['clip_url']} ents={ex['entities']} cents={ex['canonical_entities']} text_has_needle={ex['text_has_needle']}")
    print("===================================================\n")
    return results

def _summarize_namespace_results(results: Dict[str, Any]) -> Dict[str, Any]:
    parents = results.get("parents", [])
    ns = results.get("namespace", "")
    num_parents = len(parents)
    total_children = sum(p["children_summary"]["total_children"] for p in parents)
    with_entity = sum(p["children_summary"]["with_entity"] for p in parents)
    without_entity = sum(p["children_summary"]["without_entity"] for p in parents)
    text_hits_wo = sum(p["children_summary"]["text_hits_without_entity"] for p in parents)
    miss_frac = round((without_entity / max(1, total_children)), 3)
    return {
        "namespace": ns,
        "parents": num_parents,
        "children": total_children,
        "with_entity": with_entity,
        "without_entity": without_entity,
        "missing_fraction": miss_frac,
        "text_hits_wo_entity": text_hits_wo,
    }

def _print_summary_table(ns_summaries: List[Dict[str, Any]]) -> None:
    headers = ["namespace", "parents", "children", "with_entity", "without_entity", "missing_fraction", "text_hits_wo_entity"]
    colw = {h: max(len(h), max((len(str(r[h])) for r in ns_summaries), default=0)) for h in headers}
    line = " | ".join(h.ljust(colw[h]) for h in headers)
    sep = "-+-".join("-"*colw[h] for h in headers)
    print("=== Namespace Summary ===")
    print(line)
    print(sep)
    for r in ns_summaries:
        print(" | ".join(str(r[h]).ljust(colw[h]) for h in headers))
    print("=========================\n")

def main():
    ap = argparse.ArgumentParser(description="Inspect entity metadata for a needle across one or more namespaces (case-insensitive, fast, read-only).")
    ap.add_argument("--index", default=None)
    ap.add_argument("--namespace", default=None,
                    help="Namespace or comma-separated list. To search 'videos' then empty string, pass: \"videos,\" (note trailing comma).")
    ap.add_argument("--parents", default=None, help="Comma-separated parent_id(s)")
    ap.add_argument("--probe-ids", default=None, help="Comma-separated parent/video ids to probe first")
    ap.add_argument("--needle", default=DEFAULT_NEEDLE)
    ap.add_argument("--outdir", default="pipeline_storage_v2/entity_reports")
    ap.add_argument("--limit-parents", type=int, default=200)
    ap.add_argument("--limit-children", type=int, default=5000)
    ap.add_argument("--scan-cap", type=int, default=50000)
    ap.add_argument("--year-start", type=int, default=2022)
    ap.add_argument("--year-end", type=int, default=2026)
    args = ap.parse_args()

    parents = [x.strip() for x in args.parents.split(",")] if args.parents else None
    probe_ids = [x.strip() for x in args.probe_ids.split(",")] if args.probe_ids else None

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key: raise SystemExit("PINECONE_API_KEY is required")
    idx_name = args.index or os.getenv("PINECONE_INDEX_NAME")
    if not idx_name: raise SystemExit("Provide --index or set PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(idx_name)

    # Namespace list: default to ["videos", ""] per request if not provided.
    if args.namespace is None:
        ns_list = ["videos", ""]
    else:
        # preserve empty entries: a trailing comma yields an empty string
        ns_list = [part.strip() for part in args.namespace.split(",")]

    # Precompute a dimension per namespace, falling back to probe loop if needed per namespace
    ns_dims: Dict[str, int] = {}
    for ns in ns_list:
        ns_dims[ns] = _detect_dimension(idx, ns)

    per_ns_results: List[Dict[str, Any]] = []
    for ns in ns_list:
        res = run_namespace(
            index=idx, ns=ns, dim=ns_dims[ns], parents=parents,
            outdir=args.outdir, limit_parents=args.limit_parents, limit_children=args.limit_children,
            needle=args.needle, scan_cap=args.scan_cap, probe_ids=probe_ids,
            year_start=args.year_start, year_end=args.year_end, idx_name=idx_name
        )
        per_ns_results.append(res)

    # Summary table across namespaces
    summaries = [_summarize_namespace_results(r) for r in per_ns_results]
    _print_summary_table(summaries)

if __name__ == "__main__":
    main()
