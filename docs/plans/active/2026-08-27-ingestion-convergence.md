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

## Capacity-fenced gate

The supervisor prohibited Docker/PostgreSQL containers and large dependency or
build artifacts while local capacity was constrained. The existing interpreter
therefore ran all non-container tests with provider/database credentials removed,
bytecode disabled, and pytest cache disabled. The real PostgreSQL ACL and
upgrade/downgrade test remains skipped until that fence is explicitly lifted.
No task-owned container, image, PostgreSQL service, or dependency overlay was
created.

## Exit criteria

- [x] Exact ancestry, diff coverage, and semantic seam are proven.
- [x] Both exact source heads are integrated with no textual conflict.
- [x] Focused SQLite migration and behavioral suites pass.
- [ ] Disposable PostgreSQL ACL and upgrade/downgrade suite passes after the
  capacity fence is lifted.
- [x] Full practical suite and knowledge/static/scope gates pass.
- [ ] Branch is pushed and a draft convergence PR is open.

## Rollback

Revert the convergence merge and receipt commits. Downgrade the lifecycle
migration only while its three tables are empty; otherwise leave revision
`20260826_0006` installed and disable the lifecycle consumer. This task performs
no external provider, appliance, storage, or production mutation.
