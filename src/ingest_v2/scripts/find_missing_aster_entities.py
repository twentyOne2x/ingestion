#!/usr/bin/env python3
"""
inspect_aster_entities.py
Read-only Pinecone inspector for Aster-related content.

What it does
------------
1) Finds PARENT vectors whose metadata suggests they are about Aster:
   - canonical_entities contains "$ASTER"
   - OR router_tags/aliases contain "Aster" (exact string match)
2) For each parent, fetches CHILD vectors (same parent_id) and reports:
   - which child nodes have Aster in entities/canonical_entities
   - which child nodes are missing Aster entities (likely your issue)
3) Emits a concise console report and optional CSV/JSON dumps.

Notes
-----
- Read-only. No updates.
- Uses only metadata filters; no embeddings needed except for Pinecone's
  `query()` shape. The code auto-detects index dimension.
- If your store doesn't keep parent vectors in Pinecone, use --parents to
  force known parent_id(s) and it will inspect only the children.

Env:
  PINECONE_API_KEY     (required)
  PINECONE_INDEX_NAME  (or pass --index)
  PINECONE_NAMESPACE   (default 'videos' if not provided)

CLI examples:
  python inspect_aster_entities.py --namespace streams
  python inspect_aster_entities.py --parents o_DnGc   # inspect a known parent_id
"""

from __future__ import annotations
import os, sys, json, time, logging, argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

from pinecone import Pinecone

log = logging.getLogger("aster_inspect")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - aster_inspect - %(levelname)s - %(message)s")

# ---------- config / canon ----------
ASTER_TERMS_ARRAY = [
    "$ASTER",  # canonical (router-level)
    "Aster", "aster", "ASTER",
]
ASTER_CANONICALS = ["$ASTER"]  # fields like canonical_entities usually hold tickers w/ $

# Some stores place topic tags on parents only; children should still have entities/canonical_entities.
TARGET_FIELDS_PARENTS = [
    ("canonical_entities", ASTER_CANONICALS),
    ("router_tags", ["Aster"]),
    ("aliases", ["Aster"]),  # harmless if none
]
TARGET_FIELDS_CHILDREN = [
    ("canonical_entities", ASTER_CANONICALS),
    ("entities", ["Aster", "aster", "ASTER"]),
]

# ---------- helpers ----------
def _detect_dimension(idx, namespace: str) -> int:
    """Find index dimension via describe_index_stats or trial query."""
    try:
        stats = idx.describe_index_stats()
        ns = (stats.get("namespaces", {}) if isinstance(stats, dict) else {}).get(namespace, {}) or {}
        dim = ns.get("dimension")
        if dim:
            return int(dim)
    except Exception as e:
        log.debug("describe_index_stats failed (non-fatal): %s", e)

    for try_dim in (3072, 1536, 1024, 768):
        try:
            idx.query(namespace=namespace, top_k=1, vector=[0.0]*try_dim, include_metadata=False)
            return try_dim
        except Exception:
            continue
    raise RuntimeError("Could not determine index dimension; set a known dimension or ensure namespace has vectors.")

def _or_filters_any_of(pairs: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    """Builds a Pinecone filter that matches (field in values) for any pair, OR'ed together."""
    ors = []
    for field, values in pairs:
        if not values:
            continue
        ors.append({field: {"$in": values}})
    if not ors:
        return {}
    return {"$or": ors} if len(ors) > 1 else ors[0]

def _filter_by_parent(parent_id: str) -> Dict[str, Any]:
    return {"parent_id": {"$eq": parent_id}}

def _field(md: Dict[str, Any], key: str, default=None):
    v = md.get(key, default)
    return v if v is not None else default

# ---------- dataclasses ----------
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
    has_aster_entity: bool
    entities: Optional[List[str]]
    canonical_entities: Optional[List[str]]
    clip_url: Optional[str]
    speaker: Optional[str]
    score: Optional[float]

# ---------- core ----------
def find_aster_parents(idx, namespace: str, dim: int, limit_parents: int = 1000) -> List[ParentHit]:
    f = _or_filters_any_of(TARGET_FIELDS_PARENTS)
    if not f:
        return []
    res = idx.query(
        namespace=namespace,
        top_k=min(limit_parents, 10000),
        vector=[0.0]*dim,
        include_metadata=True,
        filter=f,
    )
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []) or []
    out: List[ParentHit] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        if (md.get("node_type") or "parent") != "parent":
            # Many pipelines store parents as 'parent'; if absent, still treat as a parent candidate
            pass
        out.append(ParentHit(
            vector_id=m.get("id") if isinstance(m, dict) else getattr(m, "id", ""),
            parent_id=_field(md, "parent_id", _field(md, "video_id", "")),
            title=_field(md, "title", None),
            channel_name=_field(md, "channel_name", None),
            published_at=_field(md, "published_at", _field(md, "published_date", None)),
            router_tags=_field(md, "router_tags", None),
            canonical_entities=_field(md, "canonical_entities", None),
        ))
    return out

def fetch_children_for_parent(idx, namespace: str, dim: int, parent_id: str, limit_children: int = 5000) -> List[ChildHit]:
    res = idx.query(
        namespace=namespace,
        top_k=min(limit_children, 10000),
        vector=[0.0]*dim,
        include_metadata=True,
        filter=_filter_by_parent(parent_id),
    )
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []) or []
    out: List[ChildHit] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        ents = _field(md, "entities", []) or []
        cents = _field(md, "canonical_entities", []) or []
        has = (set(ents) & set(ASTER_TERMS_ARRAY)) or (set(cents) & set(ASTER_CANONICALS))
        out.append(ChildHit(
            vector_id=m.get("id") if isinstance(m, dict) else getattr(m, "id", ""),
            segment_id=_field(md, "segment_id", None),
            start_hms=_field(md, "start_hms", None),
            end_hms=_field(md, "end_hms", None),
            has_aster_entity=bool(has),
            entities=ents,
            canonical_entities=cents,
            clip_url=_field(md, "clip_url", _field(md, "url", None)),
            speaker=_field(md, "speaker", None),
            score=(m.get("score") if isinstance(m, dict) else getattr(m, "score", None)),
        ))
    return out

def summarize_children(children: List[ChildHit]) -> Dict[str, Any]:
    with_aster = [c for c in children if c.has_aster_entity]
    without_aster = [c for c in children if not c.has_aster_entity]
    return {
        "total_children": len(children),
        "with_aster_entity": len(with_aster),
        "without_aster_entity": len(without_aster),
        "missing_fraction": round((len(without_aster) / max(1, len(children))), 3),
        "examples_missing": [
            {
                "vector_id": c.vector_id,
                "segment_id": c.segment_id,
                "time": f"{c.start_hms or ''}-{c.end_hms or ''}",
                "clip_url": c.clip_url,
                "speaker": c.speaker,
                "entities": c.entities,
                "canonical_entities": c.canonical_entities,
            }
            for c in without_aster[:8]
        ],
    }

def run(namespace: Optional[str], index_name: Optional[str], parents: Optional[List[str]], outdir: Optional[str], limit_parents: int, limit_children: int):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is required")

    ns = namespace or os.getenv("PINECONE_NAMESPACE") or "videos"
    idx_name = index_name or os.getenv("PINECONE_INDEX_NAME")
    if not idx_name:
        raise SystemExit("Provide --index or set PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(idx_name)
    dim = _detect_dimension(idx, ns)
    log.info("Using index=%s namespace=%s dim=%d", idx_name, ns, dim)

    # Step 1: parent discovery (unless explicit list given)
    if parents:
        parent_hits = []
        for pid in parents:
            parent_hits.append(ParentHit(vector_id=f"(unknown for {pid})", parent_id=pid,
                                         title=None, channel_name=None, published_at=None,
                                         router_tags=None, canonical_entities=None))
    else:
        parent_hits = find_aster_parents(idx, ns, dim, limit_parents=limit_parents)
        log.info("Found %d parent candidates for Aster", len(parent_hits))

    results: Dict[str, Any] = {"parents": []}
    for ph in parent_hits:
        kids = fetch_children_for_parent(idx, ns, dim, ph.parent_id, limit_children=limit_children)
        summary = summarize_children(kids)
        log.info("Parent %s | %s (%s) — children=%d, with_aster=%d, missing=%d",
                 ph.parent_id, ph.title or "(untitled)", ph.channel_name or "N/A",
                 summary["total_children"], summary["with_aster_entity"], summary["total_children"] - summary["with_aster_entity"])
        results["parents"].append({
            "parent": asdict(ph),
            "children_summary": summary,
        })

    # Optional dumps
    if outdir:
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        jp = p / f"aster_entities_{idx_name}_{ns}_{ts}.json"
        with jp.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info("Wrote %s", jp)

    # Pretty print minimal console summary
    print("\n==== Aster Entity Coverage Report ====")
    for item in results["parents"]:
        ph = item["parent"]
        sm = item["children_summary"]
        title = ph.get("title") or "(untitled)"
        chan = ph.get("channel_name") or "N/A"
        print(f"- {ph['parent_id']} | {title} [{chan}]")
        print(f"  children: {sm['total_children']} | with Aster entity: {sm['with_aster_entity']} | missing: {sm['without_aster_entity']} (frac={sm['missing_fraction']})")
        if sm["examples_missing"]:
            print("  examples missing entities:")
            for ex in sm["examples_missing"]:
                print(f"    • seg={ex['segment_id']} time={ex['time']} url={ex['clip_url']} ents={ex['entities']} cents={ex['canonical_entities']}")
    print("======================================\n")

def main():
    ap = argparse.ArgumentParser(description="Inspect Aster-related entity metadata in Pinecone (read-only).")
    ap.add_argument("--index", default=None, help="Pinecone index name (or PINECONE_INDEX_NAME)")
    ap.add_argument("--namespace", default=None, help="Namespace (or PINECONE_NAMESPACE, default 'videos')")
    ap.add_argument("--parents", default=None, help="Comma-separated parent_id(s) to inspect directly (skips parent discovery)")
    ap.add_argument("--outdir", default="pipeline_storage_v2/entity_reports", help="Optional output dir for JSON report")
    ap.add_argument("--limit-parents", type=int, default=200, help="Max parents to pull via discovery")
    ap.add_argument("--limit-children", type=int, default=5000, help="Max children per parent")
    args = ap.parse_args()

    parents = [x.strip() for x in args.parents.split(",")] if args.parents else None
    run(args.namespace, args.index, parents, args.outdir, args.limit_parents, args.limit_children)

if __name__ == "__main__":
    main()
