# Linux production foundation

The channel service runs migration-managed PostgreSQL in production. Set
`CHANNEL_SERVICE_ENV=production`, provide an explicit PostgreSQL
`CHANNEL_SERVICE_DATABASE_URL`, and configure
`CHANNEL_SERVICE_INTERNAL_SHARED_SECRET` with at least 32 characters. Set
`CHANNEL_SERVICE_CANONICAL_NAMESPACE` to the one allowed production vector
namespace; namespace-bearing write routes reject every other value. Run
`alembic upgrade head` before starting any API, scheduler, or worker process.
Application startup verifies the exact Alembic revision and never creates or
alters PostgreSQL tables.

Vector writers also require an immutable embedding identity in production:
`EMBED_PROVIDER`, `EMBED_MODEL`, positive `EMBED_DIM`, and
`EMBED_MODEL_REVISION` as the exact 40-character lowercase model commit. The
health receipt reports provider, model, revision, and dimension. Local
SentenceTransformer embeddings always pass the pinned revision and normalize
vectors for cosine search.

The authenticated app gateway is the sole external entry point for internal
channel-service routes. It must strip caller-supplied ICMFYI identity headers,
authenticate the user, and then set all three headers on its private upstream
request:

- `x-icmfyi-internal-secret`
- `x-icmfyi-user-id`
- `x-icmfyi-tenant-id`

User and tenant IDs are typed, non-interchangeable scopes: `usr_` plus 64
lowercase hexadecimal characters and `ten_` plus 64 lowercase hexadecimal
characters. The service upserts trusted gateway principals and membership
before tenant API writes so all foreign keys are valid.

The shared secret authenticates the gateway hop. Tenant-scoped handlers must
derive user and tenant identity with `forwarded_internal_identity()` and must
never accept those identities from a body or query parameter. `/healthz`, the
separately verified Pub/Sub push endpoint, the public ACP offerings catalog, and
ACP job routes protected by `ACP_SHARED_SECRET` are the only production
middleware exemptions.

Canonical ingestion work is globally deduplicated by source and pipeline
version while each tenant retains its own request and idempotency record.
Workers claim bounded leases, PostgreSQL claims use `FOR UPDATE SKIP LOCKED`,
expired leases are reclaimable, and a stale owner cannot complete work after
its lease expires. External provider submissions must first reserve an
`ingestion_effects` idempotency row in the same durable database.

Production `/index/youtube` defaults to `clip_ready=true`. Each video is
serialized by one globally deduplicated `youtube_hot_media` ingestion job with
a one-hour reclaimable lease and a `youtube_ytdlp/public_video_download`
provider-effect reservation. Downloads have a bounded runtime and size, resume
from a deterministic per-video staging directory, and publish through a
no-overwrite hard link into the SHA-256 CAS below
`CHANNEL_SERVICE_HOT_MEDIA_ROOT` (production: `/data/hot-media`). Only the
dedicated acquirer mounts that volume read-write; the API and clip service mount
it read-only. The first clip-ready POST queues the durable job and returns HTTP
202 plus `/v1/ingestion-jobs/{job_id}`. After polling reports `ready=true`, the
caller repeats the idempotent POST; the API revalidates the retained object and
publishes transcript, entitlement, canonical media facts, and vectors without
another video download. A durable acquisition receipt prevents completed bytes
from being downloaded again. SHA-256, size, containment, and an ffprobe video
stream are revalidated before canonical `clip_ready` facts are committed. If no
requested item succeeds and none remains pending, the endpoint returns HTTP 502
with `ok=false` rather than claiming a successful batch.

PostgreSQL row-level security is enabled and forced on tenant channel
entitlements, ingestion requests, and tenant exports. Tenant API transactions
set `app.tenant_id` locally; without that transaction-local setting these rows
are invisible and unwritable.

## Tenant SQLite export

`POST /v1/tenant-exports` accepts only an idempotency key; tenant and user
identity come exclusively from authenticated gateway headers. It writes a
durable `tenant_exports` row and a deterministic `tenant-sqlite-v1` database.
The database contains only active channels reachable through that tenant's
active entitlements, their active canonical videos, the current active
transcript revision and segments, and digest-only media references. It includes
FTS5 over transcript text. Stable ordering and source timestamps produce
repeatable bytes; SQLite and manifest files are SHA-256 named and installed
read-only. Artifact downloads re-check tenant RLS visibility and SHA-256.

Canonical clip media resolves through `source_videos.id` (the client media ID),
an active `video_media_refs` row with role `source_video` or `proxy`, its
`media_objects` digest, and one active `hot_local` `media_locations` row.
Storage Box evidence uses the same digest with backend `storagebox`; it does not
claim local clip readiness.

## Archive catalog loader

The local loader accepts the exact heterogeneous
`icmfyi.archive-catalog-import.v1` JSONL contract and its exact
`<sha256>  <filename>` sidecar:

```bash
python scripts/load_archive_catalog.py catalog.jsonl catalog.jsonl.sha256 \
  --expect-jsonl-sha256 <64-lowercase-hex> \
  --receipt-dir /data/exports/archive-import-receipts
```

It validates contract, source, item, media-variant, and aggregate-only inventory
records before idempotently upserting canonical source/video/media evidence.
Acquisition states remain `pending_discovery`, `partial_only`, or
`retained_remote_verified`; `clip_candidate` is separate, and an import that
asserts `clip_ready=true` is rejected. Aggregate Twitch inventory never creates
fabricated item IDs. The immutable receipt reports exact input hashes, effect
counts, and content-addressed readback digests.
