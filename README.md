# Ingestion Platform Overview

This repository now powers the ingestion stack behind our Pinecone-backed search
experience.

The durable authenticated Linux request/worker contract for public Twitch,
Pump.fun, X, and YouTube ingestion is documented in
[`docs/public_platform_ingestion.md`](docs/public_platform_ingestion.md).

It contains:

- Cloud-run ingestion services that respond to diarization events and upsert
  parent/child vectors into Pinecone.
- Batch pipelines for backfills, data hygiene, and metadata enrichment.
- Developer tooling for replaying events, running ingestion locally, and
  validating Pinecone state.

The original project started life as a tutorial for building generic RAG
applications (see [Appendix A](#appendix-a---original-readme-snapshot)). This
README captures the current expectations for users and contributors working on
the ingestion system.

---

## What You Get Out Of The Box

- **Ingestion pipelines (`src/ingest_v2`)**
  - `pipelines/run_all.py` – batch ingest AssemblyAI JSONs from disk with
    dedupe, routing enrichment, and Pinecone upserts.
  - `scripts/ingest_one.py` – targeted reingest for a single YouTube video.
  - `cloud/diarization_indexer` – FastAPI service deployed on Cloud Run that
    consumes Pub/Sub `diarization-ready` events.
- **Metadata maintenance**
  - `scripts/backfill_child_channel_metadata.py` extends child vectors with
    parent channel/title/date metadata.
  - Additional scripts audit namespace overlap, patch entities, or reconcile
    routers.
- **Configs and schema**
  - Namespace definitions in `src/ingest_v2/configs/namespaces.json`.
  - Pydantic models for parent/child payloads under `src/ingest_v2/schemas`.
- **CI-friendly utilities**
  - `scripts/run_diarization_ingest.py` to replay events from JSON.
  - Test suite (`tests/`) covering diarization ingestion, segmentation, and
    validators.

---

## Quick Start

> Requires Python 3.10+, `pip`, and access to Pinecone/YouTube/OpenAI secrets.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # create if missing and populate secrets
export PYTHONPATH=$PYTHONPATH:$PWD
```

Populate `.env` (or export variables) with at least:

```bash
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=icmfyi-v2
PINECONE_NAMESPACE=videos
YOUTUBE_API_KEY=...
OPENAI_API_KEY=...
```

Run a batch ingest against local AssemblyAI JSONs:

```bash
python -m src.ingest_v2.pipelines.run_all \
  --root /path/to/assemblyai/json \
  --include-channels @SomeChannel \
  --namespace videos
```

Replay a diarization event:

```bash
python scripts/run_diarization_ingest.py \
  --namespace videos \
  --event-file ./sample_events/diarization_ready.json
```

Backfill metadata for existing child nodes:

```bash
python -m src.ingest_v2.scripts.backfill_child_channel_metadata \
  --namespace videos \
  --index-name icmfyi-v2 \
  --dry-run
```

---

## Architecture & Operational Expectations

### Event-driven ingestion (Cloud Run)

- Service: `diarization-indexer` (FastAPI) in `src/ingest_v2/cloud/diarization_indexer`.
- Trigger: Pub/Sub push on topic `diarization-ready` with payload
  `DiarizationReadyEvent` (`schemas.py`).
- Flow:
  1. Verify bearer token (`pubsub.py`).
  2. Load channel allow list for namespace (`service.py`).
  3. Fetch YouTube metadata (`youtube.py`).
  4. Fetch diarized JSON + optional entities (`gcs.py`).
  5. Build parent and child payloads (`ingest.py`, `build_parents`, `build_children`).
  6. Upsert into Pinecone (`upsert_parents`, `upsert_children`).
- Env vars sourced from GCP secrets (see `gcloud run services describe` output for the latest revision).

### Batch ingestion

- `run_all.py` orchestrates:
  - Asset discovery (`iter_youtube_assets_from_fs`).
  - Deduping against Pinecone (`get_ingested_parent_ids`).
  - Speaker resolution, router enrichment, and progress tracking.
  - Parent upsert followed by child embedding/upsert.
- Use `--skip-dedupe` when repairing missing parents with existing children.

### Targeted reingest

- `ingest_one.py` locates diarization artifacts in a local directory, optionally
  re-runs speaker resolution, enriches router fields, and pushes parent/child
  vectors.
- Supports optional purge-before-upsert logic per video.

### Metadata hygiene

- `backfill_child_channel_metadata.py` now adds `channel_name`, `channel_id`,
  `video_id`, and `published_at` to child vectors.
- Similar scripts exist for entity canonicalisation and namespace audits.

---

## Configuration Reference

| Setting | Source | Notes |
| ------- | ------ | ----- |
| `YT_NAMESPACE_CONFIG` | env / secret | Path or inline JSON describing channel allow lists. |
| `PINECONE_*` | env / secret | Index name, namespace, API key, environment, etc. |
| `YOUTUBE_API_KEY` | env / secret | Needed for video metadata lookups. |
| `OPENAI_API_KEY` | env / secret | Provider for router enrichment embeddings. |
| `EMBED_MODEL`, `EMBED_PROVIDER`, `EMBED_DIM` | env / secret | Configures embedding backend for child vectors. |
| `PUBSUB_VERIFY_SIGNATURE` | env | Toggle signature verification in dev/test. |
| `PIPELINE_STORAGE_ROOT` | env | Scratch space for intermediate files (Cloud Run). |

Namespace-specific settings live in `src/ingest_v2/configs/namespaces.json` and
can be overridden per environment.

### Settled x402 work consumer

The paid-work consumer is a separate process and PostgreSQL login. Apply the
Alembic migrations and the payment service schema first, then apply
`sql/001_payment_worker_security_definer.sql` as the database owner. Create the
fixed security principal `icmfyi_payment_worker` as
`NOSUPERUSER NOBYPASSRLS NOINHERIT`, revoke its
schema/table defaults, and grant only:

- `CONNECT` on the application database and `USAGE` on schema `public`;
- `EXECUTE` on `icmfyi_claim_settled_paid_work()`,
  `icmfyi_ack_settled_paid_work(uuid,uuid,text)`, and the bounded
  `icmfyi_fail_settled_paid_work(...)` retry/dead-letter recorder;
- `SELECT` on `channel_quotes`, `quote_videos`, `checkout_sessions`,
  `channel_packs`, `pack_batches`, `channel_orders`, `payment_receipts`,
  `tenants`, `user_accounts`, `source_channels`,
  `tenant_channel_entitlements`, `ingestion_jobs`, and
  `ingestion_requests`;
- `INSERT` on `checkout_sessions`, `channel_packs`, `pack_batches`,
  `pack_videos`, `channel_orders`, `payment_receipts`, `entitlements`,
  `source_channels`, `tenant_channel_entitlements`, `ingestion_jobs`, and
  `ingestion_requests`;
- `UPDATE` on `checkout_sessions`, `channel_packs`, `pack_batches`,
  `channel_orders`, and `ingestion_jobs`; and
- `USAGE` only on `pack_videos_id_seq`.

Do not rename or alias this login: its exact name is bound into the commerce
RLS policy, and `CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_ROLE` must remain
`icmfyi_payment_worker`. Do not grant direct access to `payment_work_intents`,
`payment_settlements`, or `payment_work_outbox`, role membership, or
`BYPASSRLS`. Configure the worker with its own
`CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_URL`, then run:

```bash
python -m src.ingest_v2.cloud.diarization_indexer.paid_work_worker
```

One short database transaction claims a settled row, derives and validates the
work from authoritative facts, creates or reconciles exactly one order and
receipt, durably queues one canonical `public_item_ingestion` request per billed
video, and only then acknowledges the outbox row. It performs no provider, HTTP,
or filesystem work. Invalid rows commit bounded backoff or dead-letter state, so
one tenant cannot head-of-line block later settlements.

The ordinary public-ingestion worker owns acquisition, transcription, canonical
publication, pack reconciliation, and hash-verified exports after ACK. A new
purchase for already retained clip-ready canonical media requeues the same
globally deduplicated ingestion job and takes the canonical-ready DB/filesystem
path; it does not repeat download or transcription provider effects. Terminal
job failure (including an expired final lease) advances the exact paid pack and
order to `failed` rather than leaving user-visible work queued forever.

The commerce-principal migration reconciles pre-existing commerce rows as whole
undirected lineage components before enabling RLS. A component containing an
ACP job bridge becomes `acp_internal`; every other pre-migration component is
retained in `system_internal` quarantine with null gateway principals. System
quarantine is operator/readiness-owned history: it is not gateway-visible,
tenant-payable, or eligible for x402 projection. Missing parents, malformed JSON
edges, checkout line-item/quote disagreement, detached ACP bridges, or mixed
explicit ownership abort migration/startup instead of splitting a lifecycle
across authorization realms. Reconciliation includes quote expansion pack IDs,
both checkout JSON projections, and ACP request/delivery commerce IDs in addition
to relational quote/checkout/pack/batch/video/order/receipt/entitlement edges.
SQLite validates the whole legacy graph before its first non-transactional DDL
change so an invalid database can be repaired and the migration retried. A
downgrade refuses to discard gateway or otherwise non-reconstructible ownership,
and refuses to invalidate retained `blocked_public_age_gate` source-video rows.
Because the one-time reconciliation holds the complete graph in memory and
updates legacy rows individually, operators must run it in a bounded maintenance
window with commerce writers stopped.

---

## Developer Workflow

1. **Environment** – Use the provided `requirements.txt`; running `pre-commit`
   is encouraged.
2. **Tests** – Execute `pytest tests/` (you can focus on
   `tests/test_diarization_ingest_logic.py` for ingestion regressions).
3. **Local event replay** – Use `scripts/run_diarization_ingest.py` with file
   URIs (`file://...`) for `mp3_uri` and `diarized_uri`.
4. **Backfill/repair** – Run scripts in `src/ingest_v2/scripts`. Many accept
   `--dry-run` and `--parents` to scope their effect.
5. **Cloud Run deployment** – Build the container defined in
   `us-central1-docker.pkg.dev/just-skyline-474622-e1/ingestion/diarization-indexer`.
   Deployment is currently managed via `gcloud run deploy`.

---

## Troubleshooting & Observability

- **Cloud Run logs** – `gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="diarization-indexer"' --limit 50`.
- **Pub/Sub DLQs** – None configured; failed pushes will retry with exponential
  backoff. Monitor Cloud Run metrics for `5xx`.
- **Pinecone verification** – Use `src/ingest_v2/scripts/find_parent_node.py` to
  confirm parent metadata or `backfill_child_channel_metadata.py --dry-run` to
  inspect child updates.
- **Router cache** – Stored under `pipeline_storage_v2/router_cache`. Clearing
  it forces re-enrichment.

---

## Contributing

1. Create a feature branch.
2. Run `pytest` and relevant scripts in `--dry-run` mode.
3. Submit PR with context covering Pinecone impact and backfill needs.
4. Update this README if the ingestion surface changes (new env vars,
   services, or workflows).

---

## Appendix A – Original README Snapshot

The first commit contained a tutorial titled **“LLM Applications”** which
focused on building a generic RAG system with Ray and Anyscale. Core elements:

- Links to Anyscale blog posts, notebooks, and Ray documentation.
- Instructions for launching GPU-enabled Anyscale workspaces (`g3.8xlarge`,
  `default_cluster_env_2.6.2_py39`).
- Steps for downloading sample docs and walking through the `rag.ipynb`
  notebook.
- Guidance on configuring OpenAI and Anyscale endpoint credentials, installing
  requirements, and setting up pre-commit hooks.
- Marketing-oriented callouts (Ray Summit, Anyscale Endpoints pricing).

The repository has since been repurposed into the ingestion platform described
above; the notebook-driven tutorial no longer exists here.

---

## Appendix B – Then vs Now

| Area | First Commit | Current State |
| ---- | ------------ | ------------- |
| Primary goal | Teach readers how to build RAG apps with Ray/Anyscale. | Operate production ingestion pipelines for YouTube-derived content. |
| Runtime | Notebooks and tutorials targeting GPU clusters. | Cloud Run services, batch scripts, Pinecone integration. |
| Data flow | Manual walkthrough of data loading and chunking. | Automated event-driven ingestion plus repair/backfill tooling. |
| Dependencies | Ray ecosystem, Anyscale endpoints. | Pinecone, YouTube Data API, AssemblyAI JSON, OpenAI embeddings. |
| Docs | Marketing-style README with links and setup tips. | This operational handbook for users and developers. |
