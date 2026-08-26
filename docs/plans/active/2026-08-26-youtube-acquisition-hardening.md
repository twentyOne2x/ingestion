# YouTube Acquisition Hardening

## Definition of Working

- Working target: ingestion commit `03f7bfb29429e394129fa6ad4676ed7dfb81e06a`.
- Operator path: generic public YouTube acquisition and `/index/youtube` yt-dlp calls.
- Done means: both paths use one hardened option builder with cookie references,
  bundled EJS, the pinned same-host Deno runtime, bounded sleeps, extractor
  arguments, an MP4-compatible highest-quality media policy, and an optional
  fail-closed same-host PO-token provider.
- Not done: Q contact, deployment, secret access, live YouTube/provider calls,
  an HTTP PO-token server, or appliance runtime activation.
- Required tier: focused unit regressions, affected tests, and
  `python3 scripts/knowledge_check.py` from clean task-owned worktrees.

## Canary

The first runnable canary captures the yt-dlp options used by generic public
YouTube media acquisition and retained `/index/youtube` media acquisition. It
must fail on the starting commit because the generic path omits the hardened
cookie, pacing, and extractor settings.

## Scope

- In scope: shared option construction, pinned dependencies, image/runtime
  contract, focused tests, appliance environment wiring if required.
- Out of scope: live downloads, cookies or token values, deployment, Q, and any
  provider or account mutation.

## Verification

- Ingestion: focused YouTube, canonical media, public ingestion, service,
  contract, logic, channel logic, and production-foundation tests.
- Appliance: compose, release-binding, and host-preparation contract tests.
- `python3 scripts/knowledge_check.py` in ingestion.

## Exit Criteria

- [x] Pre-fix option drift is reproduced by the focused regression.
- [x] All YouTube yt-dlp call sites use the shared builder.
- [x] Provider enablement fails closed without logging secret bytes.
- [x] Focused and repository knowledge gates pass.
- [ ] Exact commits, trees, pushes, and draft PRs are read back.
