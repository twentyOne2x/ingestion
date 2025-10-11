# Diarization Output Ingestion – Implementation Plan

## Goal
Wire the `diarization-ready` events produced by the YouTube media pipeline into the ingestion
repo so that newly diarized transcripts are indexed automatically for the configured namespaces
and channels.

## Proposed Approach
1. **Trigger**: Use a Pub/Sub push subscription on the `diarization-ready` topic (one per namespace)
   so Cloud Run receives the payload as it becomes available.
2. **Service**: Create a FastAPI-based Cloud Run service (e.g. `diarization-indexer`) that:
   - Verifies the Pub/Sub signature.
   - Downloads the diarized JSON + entities JSON from GCS.
   - Normalises the payload into our internal ingestion schema.
   - Writes the data to `pipeline_storage_v2` and enqueues the parent/child upsert routines already present in `run_all.py`.
3. **Configuration**: Honour the existing namespace/channel filters (from `namespaces.json`) so only authorised channels are ingested.
4. **Observability**: Emit structured logs and, on failure, nack the Pub/Sub delivery (Cloud Run will retry).

We prefer Pub/Sub push over polling the `youtube_diarized/` prefix because it eliminates lag and keeps
compute costs low. Polling remains a fallback if we encounter delivery issues.

## Task Checklist
- [x] Draft ingestion plan (this document).
- [x] Add contract tests describing the expected `diarization-ready` payload and GCS layout.
- [x] Implement Pub/Sub request validation helper.
- [x] Create ingestion service entrypoint (FastAPI handler + business logic stub).
- [x] Add download/normalisation layer with namespace/channel gating.
- [x] Wire ingestion logic to existing `run_all` components (or lightweight wrapper).
- [x] Provide CLI harness for local testing with recorded payloads.
- [x] Add deployment artefacts (Dockerfile + Makefile target).
- [x] Verify end-to-end locally (TDD cycle: tests → code → tests).
- [x] Deploy to Cloud Run & smoke-test with staged Pub/Sub message.

Each task will follow a test-driven loop: write the failing test, implement the minimal code, then
green the test suite before ticking the box above.

## Test Strategy
| Scope                         | Tooling                                  |
|------------------------------ |------------------------------------------|
| Pub/Sub request contract      | pytest unit tests (schema validation)    |
| GCS download normalisation    | pytest + moto/`gcsfs` fixtures            |
| Namespace/channel filtering   | pytest parametrised tests                 |
| End-to-end service flow       | pytest (TestClient) + recorded payloads   |

## Open Questions
1. Should ingestion re-run speaker resolution or trust the diarization output entirely?
2. Do we write directly to Pinecone inside this service or enqueue jobs for the existing batch runner?
3. How do we deduplicate replays (Pub/Sub at-least-once) — rely on segment IDs or maintain an idempotency store?

These will be addressed while implementing the tasks above; any decisions will be recorded here.

## Deployment Notes
- 2025-10-11: Deployed `diarization-indexer` (revision `00008-4qt`) to `us-central1` and wired the
  `diarization-indexer-videos` push subscription (`aud=https://diarization-indexer-406386298457.us-central1.run.app`).
  Smoke tests replayed the `aH7mT0NiheU` payload via Pub/Sub and observed repeated 204 responses with no
  ingestion errors.
- Service account `diarization-indexer@just-skyline-474622-e1.iam.gserviceaccount.com` now has
  `storage.objectViewer`, `secretmanager.secretAccessor`, and `run.invoker` to satisfy GCS reads and token auth.
- Runtime depends on `google-cloud-storage`; fallback to shelling out to `gsutil` is no longer required in Cloud Run.
