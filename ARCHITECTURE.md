# Architecture

## Overview
`icmfyi/ingestion` transforms upstream content into normalized, indexed artifacts for retrieval systems.

## Main Components
- Pipeline orchestration in `src/ingest_v2/pipelines/` coordinates staged execution.
- Source adapters in `src/ingest_v2/sources/` pull and normalize upstream content.
- Schema and validation modules in `src/ingest_v2/schemas/` and `src/ingest_v2/validators/` enforce contracts.
- Service containers in `services/` support diarization/indexing and auxiliary processing.

## Data and Control Flow
1. Acquire source metadata and transcript/content payloads.
2. Normalize into parent/child entities and validate runtime constraints.
3. Enrich, dedupe, and segment for downstream retrieval use.
4. Upsert derived artifacts into vector and metadata stores.

## Ops Notes
- Keep namespace/config updates aligned with `src/ingest_v2/configs/`.
- Run `python3 scripts/knowledge_check.py` after docs or workflow edits.
