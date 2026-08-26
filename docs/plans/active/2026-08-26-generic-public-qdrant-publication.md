# Generic public ingestion Qdrant publication

## Definition of Working

- Working target: generic public item ingestion for Twitch, X/Twitter, Pump.fun,
  and generic YouTube publishes the canonical transcript vectors and verifies
  their authoritative Qdrant readback before the job may become terminal-success.
- Operator path: the existing PostgreSQL-backed public item worker, including a
  restart or retry after canonical PostgreSQL publication has already committed.
- Done means: the pre-fix PostgreSQL-success/Qdrant-missing regression fails, the
  fixed worker repairs or publishes the canonical vectors, verifies Qdrant, and
  reaches success without repeating acquisition or transcription effects.
- Not done: PostgreSQL-only canonical rows, a Qdrant write without readback,
  provider replay, acquisition configuration changes, deployment, or contact
  with Q.
- Required tier: focused local integration/source proof plus the repository PR
  gate and a task-owned draft PR. No hosted-runtime or production claim.

## Goal

Make canonical Qdrant publication a required, idempotent completion boundary for
generic public item ingestion.

## Scope

- In scope:
  - Generic Twitch, X/Twitter, Pump.fun, and YouTube item jobs.
  - Canonical transcript vector publication and authoritative Qdrant readback.
  - Restart-safe repair when PostgreSQL publication already exists.
  - Focused regression coverage and public-ingestion documentation.
- Out of scope:
  - Provider or transcription configuration.
  - New acquisition behavior or paid-provider effects.
  - Q access, production deployment, secrets, and unrelated ingestion routes.

## Steps

1. Reproduce PostgreSQL canonical success with missing Qdrant vectors.
2. Add the smallest idempotent vector publication/readback boundary before job
   completion.
3. Prove retry does not repeat acquisition or transcription effects.
4. Run the focused and affected tests and `python3 scripts/knowledge_check.py`.
5. Commit, push the task-owned branch, and open a draft PR.

## Risks

- Risk: retry repeats a provider or paid transcription effect.
  - Mitigation: reuse existing durable acquisition/transcription effect and
    canonical transcript rows; regression-test call counts across restart.
- Risk: an acknowledged Qdrant write is mistaken for durable publication.
  - Mitigation: require an authoritative point readback matching the canonical
    identity before terminal success.
- Risk: generic publication diverges from the existing canonical vector shape.
  - Mitigation: reuse the established child-building/upsert path and its stable
    identifiers instead of defining a second vector contract.

## Verification

- Commands:
  - Focused regression test selected from `tests/test_public_platform_ingestion.py`.
  - Affected public-ingestion and canonical-media tests.
  - `python3 scripts/knowledge_check.py`.
- Expected result:
  - Missing Qdrant publication prevents success and is repaired on retry.
  - Canonical points are read back before success.
  - Acquisition and transcription effect call counts remain unchanged on retry.

## Exit Criteria

- [x] Focused regression fails on `03f7bfb29429e394129fa6ad4676ed7dfb81e06a`.
- [x] Canonical Qdrant publication and authoritative readback gate terminal success.
- [x] Restart/retry does not duplicate provider or paid effects.
- [x] Affected tests and the knowledge check pass.
- [x] Task-owned commit is pushed and a draft PR is open.

## Rollback

Revert the task commit. The change adds no migration, deployment, acquisition
configuration, or external side effect; existing PostgreSQL canonical rows remain
the recovery input for a later safe publication attempt.
