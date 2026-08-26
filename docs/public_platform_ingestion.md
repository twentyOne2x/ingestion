# Public multi-platform ingestion foundation

The authenticated app gateway can queue one canonical request with:

```http
POST /v1/ingest
Idempotency-Key: exact-printable-ascii-key
X-ICMFYI-User-ID: usr_...
X-ICMFYI-Tenant-ID: ten_...
X-ICMFYI-Internal-Secret: ...
Content-Type: application/json

{
  "platform": "twitch",
  "target_kind": "channel",
  "target": "https://www.twitch.tv/cented",
  "max_items": 25,
  "clip_ready": true,
  "transcription_mode": "local_cpu",
  "language": "en"
}
```

`Idempotency-Key` is required and preserved exactly. Replaying the same tenant,
key, target, and canonical request hash returns the same durable request/job. Reusing
the key with a different canonical request returns HTTP 409. Tenant and user identities are accepted only from
the trusted gateway headers; request JSON cannot widen tenant scope.

The global job is shared by equivalent tenant requests, while the request rows,
channel entitlements, query results, and exports remain tenant scoped. Workers claim
PostgreSQL jobs with expiring leases and heartbeats. Provider calls are reserved in
`ingestion_effects` before execution. Channel discovery creates idempotent per-item
child jobs; it never substitutes a text file or local directory queue.

## Exact support ceiling

- Twitch supports public channel VOD discovery and exact public `/videos/<id>`
  items through `yt-dlp`. It does not use subscriber-only, authenticated, deleted, or
  lifetime-complete surfaces.
- Pump.fun support means only the public room/coin clip API at
  `livestream-api.pump.fun/clips/<room>` and its allowlisted clip playlists. A room is
  an exact Solana coin mint. This is not a general Pumpfun-topic archive and does not
  infer Pumpfun coverage from Twitch, YouTube, or X titles.
- X support means a bounded public syndication-profile window. Both the numeric user
  ID and handle must match, and an exact item request requires the author's numeric
  ID. The current adapter enumerates directly authored video/gif media in that
  window. It is explicitly not lifetime-complete and does not use cookies, private
  sessions, GraphQL credentials, follows, likes, or other social actions.
- YouTube remains supported by the same generic contract in addition to the legacy
  endpoint.

All successful item ingestion retains the exact verified source video in the hot
content-addressed store and publishes a canonical media reference. This remains true
when `clip_ready=false`: the flag means the caller did not require immediate clip use,
not that provenance media may become an untracked CAS orphan.

Terminal item success also requires canonical transcript vectors in Qdrant. After the
canonical PostgreSQL media and transcript revision commit, the worker builds
deterministic child points linked to that media and revision, publishes them to the
configured canonical namespace with a completed-write acknowledgement, and retrieves
the exact points with their payloads and vectors. The readback must match canonical
identity, transcript source hashes, stored text, embedding provider/model/revision,
and the configured vector dimension. Its digest and point counts are retained in the
job's `qdrant_publication` result.

A missing or mismatched point keeps the job retryable. Replay first reads Qdrant and
upserts only absent or mismatched deterministic points. It reuses the already-succeeded
public acquisition and transcription effect rows plus the retained canonical media and
transcript, so a Qdrant repair does not repeat a provider download or paid
transcription submission. Twitch uses `twitch_vod`, Pump.fun uses `pumpfun_clip`, X
uses `media`, and generic YouTube uses `youtube_video` payload identity within the same
canonical namespace.

## CPU transcription contract

`transcription_mode=local_cpu` runs Hugging Face Transformers on CPU only. The default
identity is:

```text
openai/whisper-small
revision 973afd24965f72e36ca33b3055d56a652f456b4d
```

Production rejects an unpinned local revision. The model and revision are persisted
on the job and every `transcription_runs` row.

`transcription_mode=openai` is opt-in and fail-closed. It requires all of:

- `CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_ENABLED=true`
- `OPENAI_API_KEY` in the worker secret environment
- a bounded `CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MAX_BYTES` (hard maximum 25 MB)
- a bounded `CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MAX_AUDIO_SECONDS`, enforced
  from an `ffprobe` readback before submission
- an explicit model in `CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MODEL` or the pinned
  release default

A timeout or connection loss after paid submission is persisted as `unknown`; the
job becomes terminal and is not retried blindly. An operator must reconcile it.

Audio is extracted as private mode-0600 mono FLAC under
`CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT`. The designated path is written to the
database before extraction. It is unlinked on terminal success and failure, and a
later worker reconciles orphan paths after a crashed job lease expires. This is
secure lifecycle deletion, not a claim of forensic SSD erasure.

## Required production tools and boundaries

The acquirer image must contain absolute, immutable `ffmpeg` and `ffprobe` binaries,
plus the repository's pinned Python dependencies. Provider egress belongs only to the
acquirer worker. The API must remain private behind the authenticated app gateway.
No Q, BX41 writer, Instagram, graph, pressure-guard, provider-account, or private-media
authority is part of this path.

Live enablement still requires public-provider canaries, the OpenAI secret/budget gate
if that backend is selected, model prewarming/capacity proof for local CPU mode, and
one bounded URL-to-transcript-to-query-to-clip E2E receipt on the production Linux host.
