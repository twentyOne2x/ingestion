from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .config import load_target_fields
from .pinecone_helpers import list_ids, fetch_metadata, update_metadata


def _chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _read_ids_from_file(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _ensure_outdir(base_dir: Optional[str]) -> Path:
    if base_dir:
        outdir = Path(base_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("pipeline_storage_v2/migrations/metadata_patch") / ts
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _missing_for_node(meta: Dict, targets: Dict[str, object]) -> Dict[str, object]:
    missing = {}
    for k, default in targets.items():
        if k not in meta or meta.get(k) is None:
            missing[k] = default
    return missing


def cmd_plan(args) -> int:
    ns = os.getenv("PINECONE_NAMESPACE", None)
    batch = int(os.getenv("PATCH_BATCH_SIZE", "200"))
    targets = load_target_fields()
    outdir = _ensure_outdir(os.getenv("PATCH_OUTPUT_DIR"))

    # IDs: from REST list or from user-provided file
    ids: List[str]
    if args.ids_file:
        ids = _read_ids_from_file(Path(args.ids_file))
    else:
        ids = list(list_ids(namespace=ns, batch_limit=1000))

    if not ids:
        print("No vector IDs found.")
        return 0

    print(f"[plan] scanning {len(ids)} vectors in namespace={ns or '(default)'} for fields: {list(targets.keys())}")

    rows = []
    plan_path = outdir / "plan.jsonl"
    csv_path = outdir / "report.csv"

    with plan_path.open("w", encoding="utf-8") as f:
        for chunk in tqdm(list(_chunked(ids, batch)), desc="fetch+scan", unit="chunk"):
            md_by_id = fetch_metadata(chunk, ns)
            for vid in chunk:
                meta = md_by_id.get(vid, {})
                missing = _missing_for_node(meta, targets)
                if missing:
                    # plan entry
                    f.write(json.dumps({"id": vid, "namespace": ns, "set_metadata": missing}, ensure_ascii=False) + "\n")
                    rows.append({
                        "id": vid,
                        "namespace": ns or "",
                        "missing_keys": ",".join(sorted(missing.keys())),
                    })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[plan] wrote plan: {plan_path}")
    print(f"[plan] wrote CSV : {csv_path}")
    return 0


def cmd_apply(args) -> int:
    ns = os.getenv("PINECONE_NAMESPACE", None)
    conc = int(os.getenv("PATCH_CONCURRENCY", "8"))
    dry = os.getenv("DRY_RUN", "0") in ("1", "true", "True")

    plan_file = Path(args.plan_file)
    if not plan_file.exists():
        print(f"Plan file not found: {plan_file}")
        return 1

    # Load plan
    updates: List[Tuple[str, Dict]] = []
    with plan_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            vid = obj["id"]
            payload = obj.get("set_metadata") or {}
            if payload:
                updates.append((vid, payload))

    if not updates:
        print("Nothing to update. (Plan is empty.)")
        return 0

    print(f"[apply] will update {len(updates)} vectors in namespace={ns or '(default)'}"
          f"{' [DRY RUN]' if dry else ''}")

    if dry:
        # Show first few examples
        for vid, meta in updates[:10]:
            print(f"  id={vid} set_metadata={json.dumps(meta, ensure_ascii=False)}")
        print("Dry-run: no writes performed.")
        return 0

    # Simple threaded writer (pinecone-python handles pooling)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(update_metadata, [(vid, meta)], ns) for (vid, meta) in updates]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="apply", unit="upd"):
            pass

    print("[apply] done.")
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [patch] - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(prog="metadata_patch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_plan = sub.add_parser("plan", help="Plan missing-field updates and write JSONL+CSV")
    ap_plan.add_argument("--ids-file", help="Optional file with vector IDs (one per line) if PINECONE_HOST not set")
    ap_plan.set_defaults(func=cmd_plan)

    ap_apply = sub.add_parser("apply", help="Apply updates from a plan JSONL")
    ap_apply.add_argument("--plan-file", required=True, help="Path to plan.jsonl produced by 'plan'")
    ap_apply.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
