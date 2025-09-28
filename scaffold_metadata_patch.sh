#!/usr/bin/env bash
set -euo pipefail

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Where to place things
PKG_DIR="src/ingest_v2/migrations/metadata_patch"
RUNNER_SCRIPT="scripts/run_metadata_patch.sh"

echo -e "${YELLOW}Starting metadata patch bootstrap...${NC}"

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p "$PKG_DIR"
mkdir -p "$(dirname "$RUNNER_SCRIPT")"

# -------------------------
# Create __init__.py
# -------------------------
echo -e "${GREEN}Creating $PKG_DIR/__init__.py${NC}"
cat > "$PKG_DIR/__init__.py" << 'EOF'
"""Metadata patch module for Pinecone vector updates."""
EOF

# -------------------------
# Create README.md
# -------------------------
echo -e "${GREEN}Creating $PKG_DIR/README.md${NC}"
cat > "$PKG_DIR/README.md" << 'EOF'
# Metadata Patch (plan/apply)

This module lets you **add new metadata fields** to existing vectors in your index without re-indexing:
1) **plan**: enumerate vectors, fetch metadata, detect missing fields, write a plan (JSONL) + CSV report.
2) **apply**: read the plan and do partial metadata updates for only the missing keys.

## Environment

Required:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- One of:
  - `PINECONE_HOST` for REST listing (e.g., `https://<index>-<proj>.svc.<region>.pinecone.io`)
  - OR `VECTOR_ID_MANIFEST=/path/to/ids.txt` (one vector ID per line)

Optional:
- `PINECONE_NAMESPACE` (default: empty namespace)
- `PATCH_FIELDS` (comma-separated fields to enforce; else defaults from config.py)
- `PATCH_DEFAULT_<FIELDNAME>` to override defaults per field (see config.py)
- `PATCH_BATCH_SIZE` (default 200)
- `PATCH_CONCURRENCY` (default 8)
- `PATCH_OUTPUT_DIR` (default: pipeline_storage_v2/migrations/metadata_patch/<timestamp>)
- `DRY_RUN=1` (apply mode only — print planned updates, do not write)

## Quickstart

1) Plan:
   ```bash
   python -m src.ingest_v2.migrations.metadata_patch.cli plan
   ```

2) Apply:
   ```bash
   python -m src.ingest_v2.migrations.metadata_patch.cli apply \
     --plan-file pipeline_storage_v2/migrations/metadata_patch/<timestamp>/plan.jsonl
   ```

If your SDK cannot list IDs and you don't have PINECONE_HOST, create a simple manifest (IDs only):
```bash
cut -d, -f1 pipeline_storage_v2/some_export.csv > pipeline_storage_v2/vector_ids.txt
export VECTOR_ID_MANIFEST=pipeline_storage_v2/vector_ids.txt
python -m src.ingest_v2.migrations.metadata_patch.cli plan
```

## Notes
- Updates are partial (set_metadata), so they won't overwrite other fields.
- The planner only writes keys that are missing on each node.
- You can safely re-run plan/apply; already-conforming nodes are skipped.
EOF

# -------------------------
# Create config.py
# -------------------------
echo -e "${GREEN}Creating $PKG_DIR/config.py${NC}"
cat > "$PKG_DIR/config.py" << 'PYTHON_EOF'
from __future__ import annotations
import os
from typing import Dict, Any

# Default fields to enforce if PATCH_FIELDS env isn't set.
# You can change these defaults freely; they're only used when a key is missing.
DEFAULT_FIELDS: Dict[str, Any] = {
    "router_tags": [],
    "aliases": [],
    "canonical_entities": [],
    "is_explainer": False,
    "router_boost": 1.0,
    "topic_summary": "",
    # Example child-safe defaults (uncomment if you decide to enforce these too)
    # "rights": "public_reference_only",
    # "ingest_version": 2,
}


def load_target_fields() -> Dict[str, Any]:
    """
    Build the target field->default map from env and DEFAULT_FIELDS.
    You can override any default via env: PATCH_DEFAULT_<UPPERCASE_FIELD>
    Example: PATCH_DEFAULT_ROUTER_BOOST=1.25
    Lists should be JSON-like strings e.g. '["a","b"]' or '[]'
    """
    import json

    fields_env = os.getenv("PATCH_FIELDS", "").strip()
    if fields_env:
        keys = [k.strip() for k in fields_env.split(",") if k.strip()]
        # Only keep specified keys; fall back to DEFAULT_FIELDS for defaults
        base = {k: DEFAULT_FIELDS.get(k, None) for k in keys}
    else:
        base = dict(DEFAULT_FIELDS)

    out: Dict[str, Any] = {}
    for k, v in base.items():
        override = os.getenv(f"PATCH_DEFAULT_{k.upper()}")
        if override is None:
            out[k] = v
        else:
            # Try JSON first, then fall back to raw strings/numbers/bools
            try:
                out[k] = json.loads(override)
            except Exception:
                # Best-effort cast for common types
                if override.lower() in ("true", "false"):
                    out[k] = (override.lower() == "true")
                else:
                    try:
                        out[k] = float(override) if "." in override else int(override)
                    except Exception:
                        out[k] = override
    return out
PYTHON_EOF

# -------------------------
# Create pinecone_helpers.py
# -------------------------
echo -e "${GREEN}Creating $PKG_DIR/pinecone_helpers.py${NC}"
cat > "$PKG_DIR/pinecone_helpers.py" << 'PYTHON_EOF'
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
PYTHON_EOF

# -------------------------
# Create cli.py
# -------------------------
echo -e "${GREEN}Creating $PKG_DIR/cli.py${NC}"
cat > "$PKG_DIR/cli.py" << 'PYTHON_EOF'
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
PYTHON_EOF

# -------------------------
# Create the runner script
# -------------------------
echo -e "${GREEN}Creating $RUNNER_SCRIPT${NC}"
cat > "$RUNNER_SCRIPT" << 'BASH_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# export PINECONE_API_KEY=...
# export PINECONE_INDEX_NAME=...
# export PINECONE_NAMESPACE=""  # or "youtube" etc.
# export PINECONE_HOST="https://<index>-<proj>.svc.<region>.pinecone.io"
# # OR: export VECTOR_ID_MANIFEST=path/to/ids.txt and pass --ids-file

# # Fields to enforce (optional; else defaults from config.py)
# export PATCH_FIELDS="router_tags,aliases,canonical_entities,is_explainer,router_boost,topic_summary"

# # Optional overrides for defaults:
# # export PATCH_DEFAULT_ROUTER_BOOST="1.25"
# # export PATCH_DEFAULT_ROUTER_TAGS='["solana","conference"]'

# # 1) Plan
# python -m src.ingest_v2.migrations.metadata_patch.cli plan

# # 2) Apply (dry-run first)
# export DRY_RUN=1
# python -m src.ingest_v2.migrations.metadata_patch.cli apply --plan-file pipeline_storage_v2/migrations/metadata_patch/*/plan.jsonl

# # 3) Real apply
# unset DRY_RUN
# python -m src.ingest_v2.migrations.metadata_patch.cli apply --plan-file pipeline_storage_v2/migrations/metadata_patch/*/plan.jsonl

exec "$@"
BASH_EOF

# Make the runner script executable
chmod +x "$RUNNER_SCRIPT"

# -------------------------
# Verify all files were created
# -------------------------
echo -e "${YELLOW}Verifying created files...${NC}"

FILES_TO_CHECK=(
    "$PKG_DIR/__init__.py"
    "$PKG_DIR/README.md"
    "$PKG_DIR/config.py"
    "$PKG_DIR/pinecone_helpers.py"
    "$PKG_DIR/cli.py"
    "$RUNNER_SCRIPT"
)

ALL_GOOD=true
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file"
        ALL_GOOD=false
    fi
done

# -------------------------
# Final summary
# -------------------------
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✅ Scaffolding complete!${NC}"
else
    echo -e "${RED}⚠️  Some files failed to create${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Files created:${NC}"
echo "  - $PKG_DIR/__init__.py"
echo "  - $PKG_DIR/README.md"
echo "  - $PKG_DIR/config.py"
echo "  - $PKG_DIR/pinecone_helpers.py"
echo "  - $PKG_DIR/cli.py"
echo "  - $RUNNER_SCRIPT (executable)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1) Install dependencies:"
echo "     pip install 'pinecone-client==3.*' pandas tqdm requests"
echo ""
echo "  2) Set environment variables:"
echo "     export PINECONE_API_KEY='your-api-key'"
echo "     export PINECONE_INDEX_NAME='your-index-name'"
echo "     export PINECONE_HOST='https://<index>-<proj>.svc.<region>.pinecone.io'"
echo ""
echo "  3) Run the planner:"
echo "     python -m src.ingest_v2.migrations.metadata_patch.cli plan"
echo ""
echo "  4) Apply the updates:"
echo "     python -m src.ingest_v2.migrations.metadata_patch.cli apply --plan-file <path-to-plan.jsonl>"
echo ""
echo -e "${GREEN}Script completed successfully!${NC}"
PYTHON_EOF