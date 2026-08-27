# Ingestion release convergence

## Definition of Working

- Working target: converge exact public-ingestion head
  `ef448f87312122b3d86e63624bb4c84cebfb92cb` and exact hot-media lifecycle
  head `d246cdd83c5cbf85a4105931a8495d0c14c630e5` without losing either history.
- Operator path: generic YouTube/Twitch/X/Pump.fun ingestion publishes canonical
  PostgreSQL media and Qdrant transcript vectors, while Alembic owns the additive
  hot-media custody and generation-bound rehydration schema.
- Done means: both exact heads are parents of the convergence merge; focused
  PostgreSQL upgrade/downgrade, publication, Qdrant, idempotency, and lifecycle
  tests pass; the full practical ingestion suite and knowledge/static gates pass;
  exact migration and appliance-manifest bindings are recorded; and a task-owned
  draft PR is open.
- Not done: provider execution, secret use, Q/BX41 contact, appliance activation,
  deployment, production migration, or merging either source PR.
- Required tier: local source/unit/integration proof plus disposable local
  PostgreSQL when available. No hosted or production claim.

## Ancestry and conflict audit

- Both source heads descend from common base
  `03f7bfb29429e394129fa6ad4676ed7dfb81e06a`.
- PR #5 changes 19 paths and PR #6 changes 8 paths; the path intersection is
  empty.
- The semantic seam is the canonical media graph: PR #5 reads and publishes
  existing `media_objects`, `media_locations`, `source_videos`, and
  `video_media_refs`; PR #6 adds owner-only lifecycle tables referencing that
  graph and advances the Alembic head without changing those publication paths.
- Migration `20260826_0006` is additive over `20260825_0005`. Empty lifecycle
  state may downgrade; retained lifecycle evidence refuses destructive downgrade.

## Verification

1. Prove the Alembic chain on SQLite and disposable PostgreSQL: empty upgrade,
   retained-media upgrade, repeated head application, empty downgrade/re-upgrade,
   owner-only ACLs, and durable-state downgrade refusal.
2. Run focused generic public ingestion, hardened YouTube, canonical Qdrant,
   idempotency, paid-effect replay, canonical media, and lifecycle suites.
3. Run the full practical suite across `tests/` and `src/ingest_v2/tests/`.
4. Run `python3 scripts/knowledge_check.py`, Python compilation, Ruff error checks,
   `git diff --check`, source-blob comparison, and migration hash readback.

## Appliance manifest binding

The release integrator must bind all of these immutable values from the final
convergence tree:

- ingestion Git commit and tree;
- Alembic head `20260826_0006` and parent `20260825_0005`;
- migration path `alembic/versions/20260826_0006_hot_media_lifecycle.py`;
- migration Git blob and SHA-256;
- requirements file Git blob and SHA-256;
- diarization-indexer Dockerfile Git blob and SHA-256.

Resolved source bindings before the task receipt commit:

- Alembic versions tree: `57f9595ad7d62146c54018d221f755dbf6a7123b`;
- lifecycle migration blob: `ff5e4c1b5a64c34db9fc1d866fdd59a400e046e3`;
- lifecycle migration SHA-256:
  `69a92b62266e9970177a76a2a121725cb7b73a87e7e48a96ffa9e6200c3e8aed`;
- requirements blob: `80725f833488d94179b08b7444b3a3ef467b9e43`;
- requirements SHA-256:
  `69cfdfd93d5fa0948ef4a4a86d589bad47cb2bc5b2968a2b98af2cf311fcf868`;
- diarization-indexer Dockerfile blob:
  `c22b9f6e2cd1049eeb746ab5c35a20f7491b085f`;
- diarization-indexer Dockerfile SHA-256:
  `f30911090f331634d95161e93f3ff9138755018ffd2c1f4dccef8e158849922d`.

## Test receipts

- Focused public-platform, canonical Qdrant, canonical-media, paid-effect replay,
  YouTube-hardening, channel-service, diarization, and lifecycle suite:
  `119 passed, 10 skipped` in 71.95 seconds.
- Full practical repository suite across `tests/` and `src/ingest_v2/tests/`:
  `158 passed, 17 skipped` in 167.70 seconds.
- Disposable PostgreSQL lifecycle, migration/RLS, and paid-work integration
  files: `11 passed, 7 skipped` in 24.79 seconds. Every lifecycle and
  migration/RLS case ran; the seven skips are the paid-work cases that require
  the separately provisioned payment-service schema and dedicated worker URL.
  Their non-provider retry/idempotency behavior remains covered by the focused
  and full in-repository suites.
- `python3 scripts/knowledge_check.py`, in-memory syntax compilation and Ruff
  error-level checks over every changed Python file, `git diff --check`, and
  source-head blob comparison pass.

## Bounded PostgreSQL gate

After the capacity fence was explicitly lifted, the real PostgreSQL gate ran in
task-owned container `icmfyi-convergence-postgres-20260827` from
`postgres:17-alpine` at digest
`sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73`
(115,030,665 bytes). It bound only a random loopback port and used a 1 GiB-capped
tmpfs for `/var/lib/postgresql/data`; Docker reported no mounts or volumes.

`/System/Volumes/Data` stayed above the 31,457,280 KiB floor before and after
pull, start, test, stop, and image removal. The lowest recorded post-step value
was 46,042,768 KiB. The terminal readback is zero task containers, zero task
volumes, and no remaining task-pulled image. No volume prune or dependency
overlay was used.

## Exit criteria

- [x] Exact ancestry, diff coverage, and semantic seam are proven.
- [x] Both exact source heads are integrated with no textual conflict.
- [x] Focused SQLite migration and behavioral suites pass.
- [x] Disposable PostgreSQL ACL and upgrade/downgrade suite passes under the
  bounded-container capacity contract.
- [x] Full practical suite and knowledge/static/scope gates pass.
- [x] Branch `codex/ingestion-convergence-20260827` is pushed and draft PR #7
  targets `codex/linux-appliance-20260825`.

## Rollback

Revert the convergence merge and receipt commits. Downgrade the lifecycle
migration only while its three tables are empty; otherwise leave revision
`20260826_0006` installed and disable the lifecycle consumer. This task performs
no external provider, appliance, storage, or production mutation.
