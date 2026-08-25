from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from .canonical_media import publish_canonical_ingestion
from .channel_service_config import InternalRequestIdentity
from .channel_service_jobs import (
    IdempotencyConflict,
    claim_ingestion_jobs,
    complete_ingestion_job,
    ensure_channel_entitlement,
    ensure_source_channel,
    fail_ingestion_job,
    get_or_create_ingestion_request,
    renew_ingestion_job_lease,
    reserve_ingestion_effect,
)
from .channel_service_store import (
    IngestionEffect,
    IngestionJob,
    IngestionRequest,
    TranscriptionRun,
    clear_tenant_scope,
    session_scope,
    set_tenant_scope,
    utcnow,
)
from .public_acquisition import (
    AcquiredPublicItem,
    PublicAcquisitionError,
    PublicItemDescriptor,
    acquire_public_item,
    descriptor_from_target,
    discover_public_items,
)
from .public_platforms import (
    CanonicalPublicTarget,
    PublicTargetError,
    normalize_public_target,
)
from .transcription_runtime import (
    AmbiguousTranscriptionError,
    TranscriptResult,
    TranscriptionConfigurationError,
    TranscriptionContract,
    TranscriptionError,
    delete_temporary_audio,
    extract_temporary_audio,
    transcribe_audio,
    transcription_temp_path,
)


LOG = logging.getLogger(__name__)
PUBLIC_JOB_KINDS = ["public_source_discovery", "public_item_ingestion"]


@dataclass(frozen=True)
class PublicWorkerDependencies:
    discover: Callable[..., tuple[PublicItemDescriptor, ...]] = discover_public_items
    acquire: Callable[..., AcquiredPublicItem] = acquire_public_item
    extract_audio: Callable[..., tuple[str, int]] = extract_temporary_audio
    transcribe: Callable[..., TranscriptResult] = transcribe_audio
    delete_audio: Callable[[Path], None] = delete_temporary_audio


class _LeaseHeartbeat:
    def __init__(self, *, job_id: str, worker_id: str, interval_seconds: float = 60.0):
        self.job_id = job_id
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.failed: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"public-ingest-heartbeat-{job_id[-8:]}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self.stopped.wait(self.interval_seconds):
            try:
                with session_scope() as session:
                    renew_ingestion_job_lease(
                        session,
                        job_id=self.job_id,
                        worker_id=self.worker_id,
                        lease_seconds=600,
                    )
            except BaseException as exc:  # surfaced to the owning worker thread
                self.failed = exc
                self.stopped.set()

    def __enter__(self) -> "_LeaseHeartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stopped.set()
        self.thread.join(timeout=5)

    def assert_live(self) -> None:
        if self.failed is not None:
            raise RuntimeError(
                "public ingestion worker lost its durable lease"
            ) from self.failed


def process_next_public_ingestion_job(
    *,
    worker_id: str,
    dependencies: PublicWorkerDependencies | None = None,
) -> bool:
    """Claim and process at most one public discovery/item job."""
    dependencies = dependencies or PublicWorkerDependencies()
    reconcile_orphaned_transcription_audio(dependencies=dependencies)
    with session_scope() as session:
        jobs = claim_ingestion_jobs(
            session,
            worker_id=worker_id,
            limit=1,
            lease_seconds=600,
            job_kinds=PUBLIC_JOB_KINDS,
        )
        if not jobs:
            return False
        job_id = jobs[0].id
        job_kind = jobs[0].job_kind

    with _LeaseHeartbeat(job_id=job_id, worker_id=worker_id) as heartbeat:
        try:
            if job_kind == "public_source_discovery":
                _process_discovery(
                    job_id=job_id,
                    worker_id=worker_id,
                    heartbeat=heartbeat,
                    dependencies=dependencies,
                )
            elif job_kind == "public_item_ingestion":
                _process_item(
                    job_id=job_id,
                    worker_id=worker_id,
                    heartbeat=heartbeat,
                    dependencies=dependencies,
                )
            else:  # pragma: no cover - claim filter makes this unreachable
                raise RuntimeError("claimed unsupported public ingestion job")
        except AmbiguousTranscriptionError as exc:
            _fail_claimed_job(
                job_id=job_id,
                worker_id=worker_id,
                code="transcription_unknown_requires_reconciliation",
                detail=str(exc),
                retryable=False,
            )
        except (TranscriptionConfigurationError, IdempotencyConflict) as exc:
            _fail_claimed_job(
                job_id=job_id,
                worker_id=worker_id,
                code="public_ingestion_configuration_error",
                detail=str(exc),
                retryable=False,
            )
        except Exception as exc:
            _fail_claimed_job(
                job_id=job_id,
                worker_id=worker_id,
                code="public_ingestion_failed",
                detail=str(exc),
                retryable=True,
            )
            LOG.exception("public ingestion failed job=%s", job_id)
    return True


def reconcile_orphaned_transcription_audio(
    *, dependencies: PublicWorkerDependencies | None = None
) -> int:
    """Delete designated temp audio only after its job is terminal or lease-expired."""
    dependencies = dependencies or PublicWorkerDependencies()
    candidates: list[tuple[str, Path]] = []
    now = utcnow()
    with session_scope() as session:
        runs = list(
            session.execute(
                select(TranscriptionRun)
                .where(
                    TranscriptionRun.cleanup_status.in_(
                        ("pending", "not_created", "cleanup_failed")
                    )
                )
                .order_by(TranscriptionRun.created_at.asc(), TranscriptionRun.id.asc())
                .limit(100)
            ).scalars()
        )
        for run in runs:
            job = session.get(IngestionJob, run.job_id)
            terminal = run.status in {"succeeded", "failed", "unknown"}
            expired = bool(
                job is None
                or job.status != "running"
                or job.lease_expires_at is None
                or _as_utc(job.lease_expires_at) <= now
            )
            if terminal or expired:
                candidates.append((run.id, Path(run.temp_audio_path)))

    cleaned = 0
    for run_id, path in candidates:
        failure: BaseException | None = None
        try:
            dependencies.delete_audio(path)
        except BaseException as exc:
            failure = exc
        with session_scope() as session:
            run = session.get(TranscriptionRun, run_id)
            if run is None:
                continue
            if failure is None:
                run.cleanup_status = (
                    "deleted" if path.exists() is False else "cleanup_failed"
                )
                if run.cleanup_status == "deleted":
                    run.cleaned_at = utcnow()
                    cleaned += 1
            else:
                run.cleanup_status = "cleanup_failed"
                run.error_code = "temporary_audio_cleanup_failed"
                run.error_detail = str(failure)[:8000]
    return cleaned


def _process_discovery(
    *,
    job_id: str,
    worker_id: str,
    heartbeat: _LeaseHeartbeat,
    dependencies: PublicWorkerDependencies,
) -> None:
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        payload = dict(job.payload_json or {})
        target = _target_from_payload(payload.get("target"))
        if target.target_kind != "channel":
            raise PublicAcquisitionError(
                "discovery job does not contain a channel target"
            )
        max_items = int(payload.get("max_items") or 0)
        effect, _ = reserve_ingestion_effect(
            session,
            job_id=job.id,
            provider=f"{target.platform}_public",
            effect_kind="public_channel_discovery",
            idempotency_key=f"public-discovery-v1:{job.dedupe_key}",
            request_payload={"target": target.as_payload(), "max_items": max_items},
        )
        should_discover = effect.status != "succeeded"
        if effect.status == "succeeded":
            items = tuple(
                _descriptor(value)
                for value in (effect.response_json or {}).get("items", [])
            )
        elif effect.status in {"reserved", "retry", "running"}:
            effect.status = "running"
            effect_id = effect.id
            items = ()
        else:
            raise PublicAcquisitionError("public discovery effect is terminal")

    if should_discover:
        discovered = dependencies.discover(target, max_items=max_items)
        heartbeat.assert_live()
        items = tuple(discovered)
        with session_scope() as session:
            effect = session.get(IngestionEffect, effect_id)
            if effect is None:
                raise RuntimeError("public discovery effect disappeared")
            effect.status = "succeeded"
            effect.response_json = {"items": [item.as_payload() for item in items]}

    child_job_ids: set[str] = set()
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        parent_requests = _tenant_requests(session, job)
        for parent in parent_requests:
            for item in items:
                child_payload = {
                    **dict(job.payload_json or {}),
                    "target": {
                        "platform": item.platform,
                        "target_kind": "item",
                        "external_id": item.external_id,
                        "canonical_url": item.canonical_url,
                        "handle": item.channel_handle,
                        "channel_external_id": item.channel_external_id,
                        "platform_entity_id": (
                            item.channel_external_id if item.platform == "x" else None
                        ),
                    },
                    "item": item.as_payload(),
                    "parent_discovery_job_id": job.id,
                }
                digest = hashlib.sha256(
                    f"{item.platform}:{item.channel_external_id}:{item.external_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
                channel = ensure_source_channel(
                    session,
                    platform=item.platform,
                    external_id=item.channel_external_id,
                    handle=item.channel_handle,
                    canonical_url=None,
                    metadata={"public_discovery_job_id": job.id},
                )
                ensure_channel_entitlement(
                    session,
                    tenant_id=parent.tenant_id,
                    channel_id=channel.id,
                    granted_by_user_id=parent.requested_by_user_id,
                    access_level="query",
                )
                _, child, _ = get_or_create_ingestion_request(
                    session,
                    tenant_id=parent.tenant_id,
                    requested_by_user_id=parent.requested_by_user_id,
                    idempotency_key=f"discovery:{parent.id}:{digest}",
                    job_kind="public_item_ingestion",
                    source_kind=item.platform,
                    source_key=f"{item.channel_external_id}:{item.external_id}",
                    pipeline_version=job.pipeline_version,
                    request_payload=child_payload,
                    channel_id=channel.id,
                    max_attempts=job.max_attempts,
                )
                child_job_ids.add(child.id)
        clear_tenant_scope(session)
        complete_ingestion_job(
            session,
            job_id=job.id,
            worker_id=worker_id,
            result={
                "platform": target.platform,
                "target_kind": "channel",
                "discovered_items": len(items),
                "child_jobs": sorted(child_job_ids),
                "lifetime_complete": False if target.platform == "x" else None,
            },
        )


def _process_item(
    *,
    job_id: str,
    worker_id: str,
    heartbeat: _LeaseHeartbeat,
    dependencies: PublicWorkerDependencies,
) -> None:
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        payload = dict(job.payload_json or {})
        item = (
            _descriptor(payload["item"])
            if isinstance(payload.get("item"), dict)
            else descriptor_from_target(_target_from_payload(payload.get("target")))
        )
        clip_ready = bool(payload.get("clip_ready", True))
        language = str(payload.get("language") or "en")
        contract = _contract(payload.get("transcription"))
        acquisition_effect, _ = reserve_ingestion_effect(
            session,
            job_id=job.id,
            provider=f"{item.platform}_public",
            effect_kind="public_video_download",
            idempotency_key=f"public-media-v1:{job.dedupe_key}",
            request_payload={"item": item.as_payload()},
        )
        if acquisition_effect.status == "succeeded":
            acquired = _acquired_from_effect(acquisition_effect)
        elif acquisition_effect.status in {"reserved", "retry", "running"}:
            acquisition_effect.status = "running"
            acquisition_effect_id = acquisition_effect.id
            acquired = None
        else:
            raise PublicAcquisitionError("public media effect is terminal")

    if acquired is None:
        acquired = dependencies.acquire(item)
        heartbeat.assert_live()
        with session_scope() as session:
            effect = session.get(IngestionEffect, acquisition_effect_id)
            if effect is None:
                raise RuntimeError("public media effect disappeared")
            effect.status = "succeeded"
            effect.response_json = _acquired_payload(acquired)

    transcript = _obtain_transcript(
        job_id=job_id,
        worker_id=worker_id,
        acquired=acquired,
        contract=contract,
        language=language,
        heartbeat=heartbeat,
        dependencies=dependencies,
    )
    heartbeat.assert_live()

    published = []
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        for request in _tenant_requests(session, job):
            if request.requested_by_user_id is None:
                raise RuntimeError(
                    "public ingestion request lacks an authenticated user"
                )
            result = publish_canonical_ingestion(
                session,
                identity=InternalRequestIdentity(
                    user_id=request.requested_by_user_id,
                    tenant_id=request.tenant_id,
                ),
                platform=acquired.item.platform,
                provider_video_id=acquired.item.external_id,
                channel_external_id=acquired.item.channel_external_id,
                channel_handle=acquired.item.channel_handle,
                channel_name=acquired.item.channel_handle,
                canonical_url=acquired.item.canonical_url,
                title=acquired.item.title,
                description=acquired.item.description,
                published_at=acquired.item.published_at,
                duration_ms=acquired.item.duration_ms,
                language=language,
                transcript_provider=transcript.provider,
                transcript_segments=transcript.segments,
                # Public-platform transcription is derived from this exact source. Keep
                # it canonical even when the caller did not require immediate clip use;
                # this avoids an unreferenced CAS object and preserves provenance.
                hot_media=acquired.media,
                metadata={
                    **(acquired.item.metadata or {}),
                    "public_ingestion_job_id": job.id,
                    "temporary_audio_retained": False,
                    "transcription_contract": contract.as_payload(),
                },
            )
            published.append(
                {
                    "tenant_id": request.tenant_id,
                    "media_id": result.media_id,
                    "transcript_revision_id": result.transcript_revision_id,
                }
            )
        clear_tenant_scope(session)
        complete_ingestion_job(
            session,
            job_id=job.id,
            worker_id=worker_id,
            result={
                "platform": acquired.item.platform,
                "target_kind": "item",
                "external_id": acquired.item.external_id,
                "sha256": acquired.media.sha256,
                "size_bytes": acquired.media.size_bytes,
                "mime_type": acquired.media.mime_type,
                "clip_ready": True,
                "clip_ready_requested": clip_ready,
                "transcript_sha256": transcript.sha256,
                "transcript_provider": transcript.provider,
                "transcript_segments": len(transcript.segments),
                "tenant_publications": len(published),
            },
        )


def _obtain_transcript(
    *,
    job_id: str,
    worker_id: str,
    acquired: AcquiredPublicItem,
    contract: TranscriptionContract,
    language: str,
    heartbeat: _LeaseHeartbeat,
    dependencies: PublicWorkerDependencies,
) -> TranscriptResult:
    ambiguous_paid_replay = False
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        effect, effect_created = reserve_ingestion_effect(
            session,
            job_id=job.id,
            provider=f"transcription_{contract.mode}",
            effect_kind="timestamped_audio_transcription",
            idempotency_key=f"public-transcript-v1:{job.dedupe_key}",
            request_payload={
                "audio_source_sha256": acquired.media.sha256,
                "language": language,
                "contract": contract.as_payload(),
            },
        )
        if effect.status == "succeeded":
            return _transcript_from_effect(effect)
        if effect.status == "unknown":
            raise AmbiguousTranscriptionError(
                "paid transcription is already in an ambiguous durable state"
            )
        if (
            contract.mode == "openai"
            and not effect_created
            and effect.status == "running"
        ):
            # `running` is committed before the paid call starts.  After a lease
            # loss the provider may already have accepted it, so retry is unsafe.
            ambiguous_paid_replay = True
            effect.status = "unknown"
            effect.response_json = {
                "error": "paid transcription lease expired after submission became possible"
            }
            for prior_run in session.execute(
                select(TranscriptionRun).where(TranscriptionRun.job_id == job.id)
            ).scalars():
                if prior_run.mode == "openai" and prior_run.status in {
                    "prepared",
                    "running",
                }:
                    prior_run.status = "unknown"
                    prior_run.error_code = "provider_result_ambiguous_after_lease_loss"
                    prior_run.error_detail = "worker lease expired after paid provider submission became possible"
        else:
            if effect.status not in {"reserved", "retry", "running"}:
                raise TranscriptionError("transcription effect is terminal")
            effect.status = "running"
            effect_id = effect.id
            attempt = int(job.attempt_count)
            path = transcription_temp_path(job_id=job.id, attempt_number=attempt)
            run_id = (
                f"trn_{hashlib.sha256(f'{job.id}:{attempt}'.encode()).hexdigest()[:40]}"
            )
            run = session.get(TranscriptionRun, run_id)
            if run is None:
                run = TranscriptionRun(
                    id=run_id,
                    job_id=job.id,
                    attempt_number=attempt,
                    mode=contract.mode,
                    model_id=contract.model_id,
                    model_revision=contract.model_revision,
                    status="prepared",
                    temp_audio_path=str(path),
                    cleanup_status="not_created",
                )
                session.add(run)
            elif run.status == "unknown":
                raise AmbiguousTranscriptionError(
                    "paid transcription attempt is ambiguous and cannot be retried"
                )
            elif run.status == "succeeded" and effect.status == "succeeded":
                return _transcript_from_effect(effect)
            run.status = "running"
            run.started_at = utcnow()
            run.error_code = None
            run.error_detail = None

    if ambiguous_paid_replay:
        raise AmbiguousTranscriptionError(
            "paid transcription may have been submitted before a worker lease expired"
        )

    audio_created = False
    transcript: TranscriptResult | None = None
    failure: BaseException | None = None
    try:
        audio_sha256, _ = dependencies.extract_audio(
            video_path=acquired.media.path, audio_path=path
        )
        audio_created = True
        with session_scope() as session:
            run = session.get(TranscriptionRun, run_id)
            if run is None:
                raise RuntimeError("transcription run disappeared")
            run.audio_sha256 = audio_sha256
            run.cleanup_status = "pending"
        heartbeat.assert_live()
        transcript = dependencies.transcribe(
            audio_path=path,
            contract=contract,
            language=language,
        )
        heartbeat.assert_live()
    except BaseException as exc:
        failure = exc
    finally:
        cleanup_error: BaseException | None = None
        try:
            dependencies.delete_audio(path)
        except BaseException as exc:
            cleanup_error = exc
        with session_scope() as session:
            run = session.get(TranscriptionRun, run_id)
            effect = session.get(IngestionEffect, effect_id)
            if run is None or effect is None:
                raise RuntimeError("transcription durable state disappeared")
            now = utcnow()
            run.cleanup_status = (
                "cleanup_failed"
                if cleanup_error
                else ("deleted" if audio_created else "not_created")
            )
            run.cleaned_at = None if cleanup_error else now
            run.completed_at = now
            if cleanup_error:
                run.status = "failed"
                run.error_code = "temporary_audio_cleanup_failed"
                run.error_detail = str(cleanup_error)[:8000]
                effect.status = "failed"
                effect.response_json = {"error": "temporary audio cleanup failed"}
            elif isinstance(failure, AmbiguousTranscriptionError):
                run.status = "unknown"
                run.error_code = "provider_result_ambiguous"
                run.error_detail = str(failure)[:8000]
                effect.status = "unknown"
                effect.response_json = {"error": str(failure)[:2000]}
            elif failure is not None:
                run.status = "failed"
                run.error_code = "transcription_failed"
                run.error_detail = str(failure)[:8000]
                effect.status = "retry"
                effect.response_json = {"error": str(failure)[:2000]}
            elif transcript is not None:
                run.status = "succeeded"
                run.transcript_sha256 = transcript.sha256
                run.provider_request_id = transcript.provider_request_id
                effect.status = "succeeded"
                effect.provider_effect_id = transcript.provider_request_id
                effect.response_json = _transcript_payload(transcript)
        if cleanup_error is not None:
            raise TranscriptionError(
                "temporary audio cleanup failed"
            ) from cleanup_error
        if failure is not None:
            raise failure
    if transcript is None:  # pragma: no cover - guarded by failure handling
        raise TranscriptionError("transcription produced no result")
    return transcript


def _fail_claimed_job(
    *, job_id: str, worker_id: str, code: str, detail: str, retryable: bool
) -> None:
    with session_scope() as session:
        job = session.get(IngestionJob, job_id)
        if job is None or job.status != "running" or job.lease_owner != worker_id:
            return
        failed_job = fail_ingestion_job(
            session,
            job_id=job_id,
            worker_id=worker_id,
            error_code=code,
            error_detail=detail,
            retryable=retryable,
            retry_after_seconds=60,
        )
        next_effect_status = "retry" if failed_job.status == "retry" else "failed"
        for effect in session.execute(
            select(IngestionEffect).where(IngestionEffect.job_id == job_id)
        ).scalars():
            if effect.status in {"reserved", "running", "retry"}:
                effect.status = next_effect_status
                if not effect.response_json:
                    effect.response_json = {"error": detail[:2000]}


def _owned_job(session, *, job_id: str, worker_id: str) -> IngestionJob:
    job = session.get(IngestionJob, job_id)
    if (
        job is None
        or job.status != "running"
        or job.lease_owner != worker_id
        or job.lease_expires_at is None
        or _as_utc(job.lease_expires_at) <= utcnow()
    ):
        raise RuntimeError("public ingestion worker does not own a live job lease")
    return job


def _as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _tenant_requests(session, job: IngestionJob) -> list[IngestionRequest]:
    rows: list[IngestionRequest] = []
    for tenant_id in sorted(set(job.request_tenant_ids_json or [])):
        set_tenant_scope(session, tenant_id)
        row = (
            session.execute(
                select(IngestionRequest)
                .where(
                    IngestionRequest.tenant_id == tenant_id,
                    IngestionRequest.job_id == job.id,
                )
                .order_by(IngestionRequest.created_at.asc(), IngestionRequest.id.asc())
            )
            .scalars()
            .first()
        )
        if row is None:
            raise RuntimeError("public ingestion job tenant fanout is inconsistent")
        rows.append(row)
    if not rows:
        raise RuntimeError("public ingestion job has no tenant request")
    return rows


def _target_from_payload(value: Any) -> CanonicalPublicTarget:
    if not isinstance(value, dict):
        raise PublicAcquisitionError("public ingestion target payload is missing")
    platform = _exact_string(value, "platform", 32)
    target_kind = _exact_string(value, "target_kind", 32)
    external_id = _exact_string(value, "external_id", 255)
    canonical_url = _exact_string(value, "canonical_url", 8_000)
    platform_entity_id = _optional_exact_string(value, "platform_entity_id", 255)
    try:
        canonical = normalize_public_target(
            platform=platform,
            target_kind=target_kind,
            target=canonical_url,
            platform_entity_id=platform_entity_id,
        )
    except PublicTargetError as exc:
        raise PublicAcquisitionError(
            "stored public target identity is invalid"
        ) from exc
    if canonical.external_id != external_id:
        raise PublicAcquisitionError(
            "stored public target identity does not match its URL"
        )
    expected = canonical.as_payload()
    # Direct targets must be the exact normalizer output. Enriched child items are
    # validated through their retained item descriptor instead of this fallback.
    for key in ("platform", "target_kind", "external_id", "canonical_url"):
        if value.get(key) != expected[key]:
            raise PublicAcquisitionError("stored public target is not canonical")
    if canonical.target_kind == "channel" or canonical.platform in {"x", "pumpfun"}:
        for key in ("handle", "channel_external_id", "platform_entity_id"):
            if value.get(key) != expected[key]:
                raise PublicAcquisitionError(
                    "stored public target binding is not canonical"
                )
    return canonical


def _descriptor(value: Any) -> PublicItemDescriptor:
    if not isinstance(value, dict):
        raise PublicAcquisitionError("public item descriptor is invalid")
    platform = _exact_string(value, "platform", 32)
    external_id = _exact_string(value, "external_id", 255)
    channel_external_id = _exact_string(value, "channel_external_id", 255)
    channel_handle = _optional_exact_string(value, "channel_handle", 255)
    canonical_url = _exact_string(value, "canonical_url", 8_000)
    try:
        canonical = normalize_public_target(
            platform=platform,
            target_kind="item",
            target=canonical_url,
            platform_entity_id=channel_external_id
            if platform in {"x", "twitter"}
            else None,
        )
    except PublicTargetError as exc:
        raise PublicAcquisitionError("stored public item identity is invalid") from exc
    if canonical.platform != platform or canonical.external_id != external_id:
        raise PublicAcquisitionError(
            "stored public item identity does not match its URL"
        )
    if canonical.platform in {"pumpfun", "x"} and (
        canonical.channel_external_id != channel_external_id
        or canonical.handle != channel_handle
    ):
        raise PublicAcquisitionError("stored public item author binding is invalid")
    if canonical.platform == "twitch":
        if not re.fullmatch(r"[A-Za-z0-9_]{1,255}", channel_external_id):
            raise PublicAcquisitionError("stored Twitch channel identity is invalid")
        if channel_handle and not re.fullmatch(r"[A-Za-z0-9_]{1,25}", channel_handle):
            raise PublicAcquisitionError("stored Twitch channel handle is invalid")
    if canonical.platform == "youtube" and not re.fullmatch(
        r"[A-Za-z0-9_@.-]{1,255}", channel_external_id
    ):
        raise PublicAcquisitionError("stored YouTube channel identity is invalid")
    title = _optional_exact_string(value, "title", 100_000, allow_newlines=True)
    description = _optional_exact_string(
        value, "description", 1_000_000, allow_newlines=True
    )
    published_at = _optional_exact_string(value, "published_at", 64)
    duration_raw = value.get("duration_ms")
    if duration_raw is not None and (
        isinstance(duration_raw, bool)
        or not isinstance(duration_raw, int)
        or duration_raw < 0
        or duration_raw > 10_000_000_000
    ):
        raise PublicAcquisitionError("stored public item duration is invalid")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise PublicAcquisitionError("stored public item metadata is invalid")
    try:
        metadata_size = len(
            json.dumps(
                metadata,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )
    except (TypeError, ValueError) as exc:
        raise PublicAcquisitionError("stored public item metadata is invalid") from exc
    if metadata_size > 256_000:
        raise PublicAcquisitionError("stored public item metadata is too large")
    return PublicItemDescriptor(
        platform=canonical.platform,
        external_id=external_id,
        channel_external_id=channel_external_id,
        channel_handle=channel_handle,
        canonical_url=canonical.canonical_url,
        title=title,
        description=description,
        published_at=published_at,
        duration_ms=duration_raw,
        metadata=dict(metadata),
    )


def _exact_string(value: dict[str, Any], key: str, maximum: int) -> str:
    raw = value.get(key)
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw) > maximum
        or raw != raw.strip()
        or any(ord(char) < 32 for char in raw)
    ):
        raise PublicAcquisitionError(f"stored public {key} is invalid")
    return raw


def _optional_exact_string(
    value: dict[str, Any],
    key: str,
    maximum: int,
    *,
    allow_newlines: bool = False,
) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw or len(raw) > maximum or raw != raw.strip():
        raise PublicAcquisitionError(f"stored public {key} is invalid")
    if "\x00" in raw or (not allow_newlines and any(ord(char) < 32 for char in raw)):
        raise PublicAcquisitionError(f"stored public {key} is invalid")
    return raw


def _contract(value: Any) -> TranscriptionContract:
    if not isinstance(value, dict):
        raise TranscriptionConfigurationError("transcription contract is missing")
    mode = str(value.get("mode") or "")
    model_id = str(value.get("model_id") or "")
    revision = str(value.get("model_revision") or "") or None
    if mode not in {"openai", "local_cpu"} or not model_id:
        raise TranscriptionConfigurationError("transcription contract is invalid")
    if (
        mode == "local_cpu"
        and not revision
        and (__import__("os").getenv("CHANNEL_SERVICE_ENV") or "").lower()
        in {"prod", "production"}
    ):
        raise TranscriptionConfigurationError("local CPU model revision is not pinned")
    return TranscriptionContract(mode=mode, model_id=model_id, model_revision=revision)


def _acquired_payload(acquired: AcquiredPublicItem) -> dict[str, Any]:
    return {
        "item": acquired.item.as_payload(),
        "media": {
            "path": str(acquired.media.path),
            "sha256": acquired.media.sha256,
            "size_bytes": acquired.media.size_bytes,
            "mime_type": acquired.media.mime_type,
        },
    }


def _acquired_from_effect(effect: IngestionEffect) -> AcquiredPublicItem:
    payload = dict(effect.response_json or {})
    media = payload.get("media")
    if not isinstance(media, dict):
        raise PublicAcquisitionError("public media effect response is invalid")
    from .canonical_media import HotMediaSpec, verify_hot_media

    return AcquiredPublicItem(
        item=_descriptor(payload.get("item")),
        media=verify_hot_media(
            HotMediaSpec(
                path=Path(str(media.get("path") or "")),
                sha256=str(media.get("sha256") or ""),
                size_bytes=int(media.get("size_bytes") or 0),
                mime_type=str(media.get("mime_type") or ""),
            )
        ),
    )


def _transcript_payload(transcript: TranscriptResult) -> dict[str, Any]:
    return {
        "provider": transcript.provider,
        "provider_request_id": transcript.provider_request_id,
        "sha256": transcript.sha256,
        "segments": list(transcript.segments),
    }


def _transcript_from_effect(effect: IngestionEffect) -> TranscriptResult:
    payload = dict(effect.response_json or {})
    result = TranscriptResult(
        provider=str(payload.get("provider") or ""),
        provider_request_id=(
            str(payload["provider_request_id"])
            if payload.get("provider_request_id")
            else None
        ),
        segments=tuple(payload.get("segments") or ()),
    )
    if (
        not result.provider
        or not result.segments
        or result.sha256 != payload.get("sha256")
    ):
        raise TranscriptionError("durable transcript effect response is invalid")
    return result
