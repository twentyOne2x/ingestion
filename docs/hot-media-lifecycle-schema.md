# Hot-media lifecycle schema

Ingestion owns the canonical PostgreSQL media schema. Alembic revision
`20260826_0006` extends the existing `media_objects`, `media_locations`,
`source_videos`, and `video_media_refs` graph with three durable operational
tables:

- `hot_media_custody_manifests` records exact manifest and custody receipts;
- `hot_media_custody_items` binds each manifest item to its existing canonical
  hot and Storage Box `media_locations` rows and records the exact appliance
  bind-mount path used for byte custody; and
- `hot_media_rehydration_attempts` binds each deterministic attempt path to one
  exact custody-manifest generation and records the verification receipt that
  permits hot readiness to return.

The canonical `media_locations.location_key` remains the ingestion-observed
path under `/data/hot-media`. The lifecycle item also binds that key to the
same file as observed by the host under `${ICMFYI_DATA_ROOT}/hot-media`
(`appliance_hot_path` / `final_appliance_path`). This explicit mount-pair
binding lets the root operator manipulate bytes without rewriting or creating
a parallel canonical media location.

Every custody manifest carries a unique generation identity. Rehydration
attempt identity includes the manifest SHA-256, so the same request and media
digest reconcile to one attempt within a generation but receive a new attempt
after a later eviction.

These rows describe the custody of globally deduplicated media objects. They
contain no tenant-owned payload or entitlement and therefore are not a tenant
RLS surface. They are owner-only instead: the migration revokes PostgreSQL
`PUBLIC` and every already-present known application runtime role, while the
appliance's idempotent role provisioning repeats the revocation after creating
roles. No public, ingestion, clip, or payment runtime receives table access.

## Migration compatibility

On an empty database, `alembic upgrade head` creates the canonical media schema
through the existing revision chain and then creates the lifecycle tables. On a
retained database at `20260825_0005`, the upgrade is additive: existing media,
location, source-video, and reference rows are neither rewritten nor
backfilled. Custody rows begin only when the host publishes and registers a
verified receipt. Repeating `alembic upgrade head` is an Alembic no-op.

Downgrade to `20260825_0005` is supported only while all three lifecycle tables
are empty. It preserves every canonical media row and removes the empty
extension. Once any custody or rehydration evidence exists, downgrade fails
before DDL so a rollback cannot silently destroy durable custody state. Older
application runtimes remain compatible with the additive tables and may ignore
them; the safe rollback is therefore to disable the host lifecycle while
retaining revision `20260826_0006`.

The Linux appliance does not carry a second copy of this schema. Its
`ingestion-migrate` Compose service runs this Alembic history, the release
manifest binds the exact revision bytes and ingestion commit/tree, and the
owner-only hot-media state adapter verifies the tables before any host byte
operation. Activation and a one-object Storage Box canary remain separate host
operator steps.
