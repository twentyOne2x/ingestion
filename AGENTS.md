# AGENTS Map

Short navigation map for agent workflows in this repo.

## Product Surface
- `src/ingest_v2/`: ingestion pipelines, schemas, sources, validators, and utilities.
- `services/`: service containers and deployable helpers.
- `tests/` and `src/ingest_v2/tests/`: end-to-end and component test coverage.

## Knowledge Base
- `docs/index.md`: docs entrypoint.
- `docs/core-beliefs.md`: operating principles.
- `ARCHITECTURE.md`: boundaries and data flow.
- `docs/plans/`: active/completed plans and debt tracker.

## Answer Then Act
- Answer the user's question directly.
- If the truthful answer to a sufficiency, completeness, or quality question is "no" and the concrete fix is inferable from the repo and thread context, implement the fix instead of stopping at analysis.
- If the truthful answer is "yes", report that and stop.
- If a real blocker remains, report the blocker clearly.

## Verification
- Run `python3 scripts/knowledge_check.py` before opening or updating a PR.
