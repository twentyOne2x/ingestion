# Linux production foundation

The channel service runs migration-managed PostgreSQL in production. Set
`CHANNEL_SERVICE_ENV=production`, provide an explicit PostgreSQL
`CHANNEL_SERVICE_DATABASE_URL`, and configure
`CHANNEL_SERVICE_INTERNAL_SHARED_SECRET` with at least 32 characters. Run
`alembic upgrade head` before starting any API, scheduler, or worker process.
Application startup verifies the exact Alembic revision and never creates or
alters PostgreSQL tables.

The authenticated app gateway is the sole external entry point for internal
channel-service routes. It must strip caller-supplied ICMFYI identity headers,
authenticate the user, and then set all three headers on its private upstream
request:

- `x-icmfyi-internal-secret`
- `x-icmfyi-user-id`
- `x-icmfyi-tenant-id`

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
