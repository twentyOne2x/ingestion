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
