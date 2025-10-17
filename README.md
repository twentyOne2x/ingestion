# Ingestion Platform Overview

This repository now powers the ingestion stack behind our Pinecone-backed search
experience. It contains:

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
