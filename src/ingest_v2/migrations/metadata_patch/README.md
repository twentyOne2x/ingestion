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
