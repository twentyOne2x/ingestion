# Public ingestion integration

## Definition of Working

- Working target: combine canonical Qdrant publication head
  `daebb8dcaf5b345c623ef9c5a23188dc7bfbe4b8` and YouTube acquisition hardening
  head `4beae90ce6a5d3bd49d1b0010b1274f4804ed401` from exact base
  `03f7bfb29429e394129fa6ad4676ed7dfb81e06a`.
- Operator path: generic public YouTube acquisition uses the shared hardened
  yt-dlp options and the public item worker requires canonical Qdrant transcript
  publication/readback before terminal success. Twitch, X/Twitter, and Pump.fun
  retain the same Qdrant completion boundary.
- Done means: both immutable heads are parents of the integration, combined
  focused tests and the repository practical suite pass, the knowledge check
  passes, the base-to-head diff contains only both source slices plus this plan,
  and a task-owned draft PR is open.
- Not done: merging either source PR, provider execution, cookie/token access,
  contact with Q, deployment, acquisition configuration activation, or a hosted
  runtime claim.
- Required tier: local source/unit/integration proof with embedded Qdrant only.

## Canary

The smallest combined canary captures hardened generic YouTube yt-dlp options,
then proves that a failed canonical Qdrant readback leaves the item job retryable
and does not repeat acquisition or transcription effects.

## Integration Evidence

- The two source diffs contain nine paths each and have no shared paths.
- The integration uses an explicit octopus merge with the exact base and both
  supplied heads as parents; no manual conflict resolution is required.

## Verification

1. Run the focused YouTube option, public-ingestion, canonical-vector, canonical
   media, worker, and diarization suites covering both behavioral contracts.
2. Run the practical repository suite with `pytest` across `tests/` and
   `src/ingest_v2/tests/`; optional external PostgreSQL cases may skip when no
   test DSN is configured.
3. Run `python3 scripts/knowledge_check.py`, Python compilation, Ruff error
   checks, and `git diff --check`.
4. Compare the final base-to-head paths and patch IDs with both source heads to
   detect accidental integration scope.

## Exit Criteria

- [x] Clean integration worktree starts at the exact requested base.
- [x] Both exact source heads are integrated without textual conflict.
- [x] Combined focused and practical repository suites pass.
- [x] Knowledge, static, and final-diff scope checks pass.
- [ ] Integration branch is pushed and a draft PR is open against
  `codex/linux-appliance-20260825`.

## Rollback

Revert the integration merge and plan receipt commits. This integration performs
no migration, provider call, secret access, deployment, or production mutation.
