from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Self

from sqlalchemy import select, text

from .canonical_media import HotMediaSpec, publish_canonical_ingestion
from .canonical_vector_publication import publish_canonical_transcript_vectors
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
from .channel_service_logic import _build_pack_artifacts, _export_root
from .channel_service_store import (
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    CommerceScope,
    IngestionEffect,
    IngestionJob,
    IngestionRequest,
    MediaLocation,
    MediaObject,
    PackBatch,
    PackVideo,
    SourceChannel,
    SourceVideo,
    TranscriptionRun,
    TranscriptRevision,
    TranscriptSegment,
    VideoMediaRef,
    clear_tenant_scope,
    commerce_ownership_values,
    commerce_scope_predicates,
    gateway_commerce_scope,
    session_scope,
    set_commerce_scope,
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
    TranscriptionConfigurationError,
    TranscriptionContract,
    TranscriptionError,
    TranscriptResult,
    delete_temporary_audio,
    extract_temporary_audio,
    transcribe_audio,
    transcription_temp_path,
)

LOG = logging.getLogger(__name__)
PUBLIC_JOB_KINDS = [
    "public_source_discovery",
    "public_item_ingestion",
    "paid_pack_export_gc",
]
PAID_PUBLIC_INGESTION_SCHEMA = "icmfyi.paid-public-ingestion.v1"
_PAID_WORK_KEYS = {
    "schema",
    "intentId",
    "outboxId",
    "tenantId",
    "principalId",
    "orderId",
    "packId",
    "batchId",
    "quoteId",
    "quoteHash",
    "videoId",
    "position",
}
_PACK_EXPORT_LOCKS_GUARD = threading.Lock()
_PACK_EXPORT_LOCKS: dict[str, threading.Lock] = {}
_PAID_PACK_GENERATION_PATTERN = re.compile(
    r"paid-(?P<snapshot>[0-9a-f]{64})-(?P<nonce>[0-9a-f]{32})"
)
_PAID_PACK_GENERATION_GC_MIN_AGE_SECONDS = 60 * 60
_PAID_PACK_PUBLISHED_MARKER = ".icmfyi-paid-export-published"
_PAID_PACK_EXPORT_GC_SCHEMA = "icmfyi.paid-pack-export-gc.v1"


@dataclass(frozen=True)
class PublicWorkerDependencies:
    discover: Callable[..., tuple[PublicItemDescriptor, ...]] = discover_public_items
    acquire: Callable[..., AcquiredPublicItem] = acquire_public_item
    extract_audio: Callable[..., tuple[str, int]] = extract_temporary_audio
    transcribe: Callable[..., TranscriptResult] = transcribe_audio
    delete_audio: Callable[[Path], None] = delete_temporary_audio
    publish_vectors: Callable[..., dict[str, Any]] = (
        publish_canonical_transcript_vectors
    )


@dataclass(frozen=True)
class _PaidPackExportTarget:
    scope: CommerceScope
    order_id: str
    pack_id: str
    batch_id: str
    quote_id: str


@dataclass(frozen=True)
class _PaidPackExportState:
    pack: ChannelPack
    batches: tuple[PackBatch, ...]
    orders: tuple[ChannelOrder, ...]
    quotes_by_id: dict[str, ChannelQuote]
    checkouts_by_id: dict[str, CheckoutSessionRecord]
    rows: tuple[PackVideo, ...]

    @property
    def latest_batch(self) -> PackBatch:
        return self.batches[-1]

    @property
    def latest_quote(self) -> ChannelQuote:
        return self.quotes_by_id[self.latest_batch.quote_id]


@dataclass(frozen=True)
class _VerifiedPackExport:
    paths: dict[str, str]
    manifest: dict[str, Any]
    unpublished_root: Path | None


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
            except BaseException as exc:  # noqa: BLE001 - surfaced to owner thread
                self.failed = exc
                self.stopped.set()

    def __enter__(self) -> Self:
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
            elif job_kind == "paid_pack_export_gc":
                _process_paid_pack_export_gc(
                    job_id=job_id,
                    worker_id=worker_id,
                    heartbeat=heartbeat,
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
        except BaseException as exc:  # noqa: BLE001 - cleanup must survive cancellation
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
                    f"{item.platform}:{item.channel_external_id}:{item.external_id}".encode()
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
        canonical_ready = _canonical_ready_item(session, item=item)
        transcript = None
        if canonical_ready is not None:
            acquired, transcript = canonical_ready
        else:
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

    if transcript is None:
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
    paid_export_targets: dict[tuple[str, str, str], _PaidPackExportTarget] = {}
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
                    "request_id": request.id,
                    "tenant_id": request.tenant_id,
                    "principal_id": request.requested_by_user_id,
                    "media_id": result.media_id,
                    "transcript_revision_id": result.transcript_revision_id,
                }
            )
            export_target = _reconcile_paid_public_ingestion_success(
                session,
                request=request,
                media_id=result.media_id,
                transcript_provider=transcript.provider,
            )
            if export_target is not None:
                key = (
                    export_target.scope.authority_kind,
                    str(export_target.scope.tenant_id or ""),
                    export_target.pack_id,
                )
                paid_export_targets[key] = export_target
        clear_tenant_scope(session)

    canonical_publications = {
        (row["media_id"], row["transcript_revision_id"])
        for row in published
    }
    if len(canonical_publications) != 1:
        raise RuntimeError(
            "public ingestion did not resolve one canonical transcript publication"
        )
    media_id, transcript_revision_id = next(iter(canonical_publications))
    qdrant_publication = dependencies.publish_vectors(
        item=acquired.item,
        transcript=transcript,
        media_id=media_id,
        transcript_revision_id=transcript_revision_id,
        language=language,
    )
    heartbeat.assert_live()

    # Publication is durable before any potentially slow filesystem hashing or
    # ZIP work.  The per-pack finalizer serializes builders without holding
    # order, pack, batch, or video row locks across filesystem I/O.
    _finalize_paid_pack_exports(
        tuple(paid_export_targets.values()), heartbeat=heartbeat
    )

    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
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
                "request_publications": len(published),
                "principal_publications": len(
                    {
                        (row["tenant_id"], row["principal_id"])
                        for row in published
                    }
                ),
                "tenant_publications": len(
                    {row["tenant_id"] for row in published}
                ),
                "canonical_ready_reuse": canonical_ready is not None,
                "qdrant_publication": qdrant_publication,
            },
        )


def _finalize_paid_pack_exports(
    targets: tuple[_PaidPackExportTarget, ...], *, heartbeat: _LeaseHeartbeat
) -> None:
    export_errors: list[str] = []
    for export_target in targets:
        heartbeat.assert_live()
        try:
            _finalize_paid_pack_export(export_target)
        except Exception as exc:
            # Every deduplicated public item may fulfill multiple buyers. One
            # pack's filesystem failure must not prevent later packs from
            # validating or repairing their independent export custody. The
            # finalizer itself commits any incomplete state while it still
            # owns the pack advisory lock.
            export_errors.append(f"{export_target.pack_id}: {exc}")
    if export_errors:
        raise RuntimeError(
            "paid pack export finalization failed: " + "; ".join(export_errors)
        )


def _paid_pack_export_gc_payload(
    *,
    target: _PaidPackExportTarget,
    retired_generation: str,
    replacement_generation: str | None,
    not_before: datetime,
) -> dict[str, Any]:
    if (
        _PAID_PACK_GENERATION_PATTERN.fullmatch(retired_generation) is None
        or (
            replacement_generation is not None
            and _PAID_PACK_GENERATION_PATTERN.fullmatch(replacement_generation)
            is None
        )
        or retired_generation == replacement_generation
    ):
        raise RuntimeError("paid pack GC generation transition is invalid")
    ownership = commerce_ownership_values(target.scope)
    return {
        "schema": _PAID_PACK_EXPORT_GC_SCHEMA,
        "authorityKind": ownership["authority_kind"],
        "tenantId": ownership["tenant_id"],
        "principalUserId": ownership["principal_user_id"],
        "packId": target.pack_id,
        "orderId": target.order_id,
        "batchId": target.batch_id,
        "quoteId": target.quote_id,
        "retiredGeneration": retired_generation,
        "replacementGeneration": replacement_generation,
        "notBeforeUtc": _as_utc(not_before).isoformat(),
    }


def _paid_pack_export_gc_dedupe_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"paid-pack-export-gc:v1:{hashlib.sha256(encoded).hexdigest()}"


def _enqueue_paid_pack_export_gc(
    session,
    *,
    target: _PaidPackExportTarget,
    retired_generation: str,
    replacement_generation: str | None,
) -> IngestionJob:
    """Durably schedule the grace-expiry sweep in the export attach transaction."""

    not_before = utcnow() + timedelta(
        seconds=_PAID_PACK_GENERATION_GC_MIN_AGE_SECONDS + 5
    )
    payload = _paid_pack_export_gc_payload(
        target=target,
        retired_generation=retired_generation,
        replacement_generation=replacement_generation,
        not_before=not_before,
    )
    dedupe_key = _paid_pack_export_gc_dedupe_key(payload)
    existing = session.execute(
        select(IngestionJob).where(IngestionJob.dedupe_key == dedupe_key)
    ).scalar_one_or_none()
    expected_tenants = (
        [str(target.scope.tenant_id)]
        if target.scope.authority_kind == "gateway"
        else []
    )
    if existing is not None:
        if (
            existing.job_kind != "paid_pack_export_gc"
            or existing.source_kind != "paid_pack"
            or existing.source_key != f"{target.pack_id}:{retired_generation}"
            or existing.pipeline_version != "paid-pack-export-gc-v1"
            or existing.payload_json != payload
            or list(existing.request_tenant_ids_json or []) != expected_tenants
        ):
            raise RuntimeError("paid pack GC dedupe identity is inconsistent")
        return existing
    digest = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()
    job = IngestionJob(
        id=f"job_{digest[:40]}",
        dedupe_key=dedupe_key,
        job_kind="paid_pack_export_gc",
        source_kind="paid_pack",
        source_key=f"{target.pack_id}:{retired_generation}",
        pipeline_version="paid-pack-export-gc-v1",
        status="queued",
        # Match paid ingestion priority so FIFO order prevents a continuously
        # busy paid queue from starving an already-due custody cleanup.
        priority=10,
        max_attempts=100,
        next_run_at=not_before,
        payload_json=payload,
        request_tenant_ids_json=expected_tenants,
    )
    session.add(job)
    session.flush()
    return job


def _paid_pack_export_gc_target(job: IngestionJob) -> _PaidPackExportTarget:
    payload = job.payload_json
    expected_keys = {
        "schema",
        "authorityKind",
        "tenantId",
        "principalUserId",
        "packId",
        "orderId",
        "batchId",
        "quoteId",
        "retiredGeneration",
        "replacementGeneration",
        "notBeforeUtc",
    }
    if (
        job.job_kind != "paid_pack_export_gc"
        or job.source_kind != "paid_pack"
        or job.pipeline_version != "paid-pack-export-gc-v1"
        or not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema") != _PAID_PACK_EXPORT_GC_SCHEMA
        or not all(
            isinstance(payload.get(key), str) and payload.get(key)
            for key in (
                "authorityKind",
                "packId",
                "orderId",
                "batchId",
                "quoteId",
                "retiredGeneration",
                "notBeforeUtc",
            )
        )
        or _PAID_PACK_GENERATION_PATTERN.fullmatch(
            str(payload.get("retiredGeneration") or "")
        )
        is None
        or (
            payload.get("replacementGeneration") is not None
            and (
                not isinstance(payload.get("replacementGeneration"), str)
                or _PAID_PACK_GENERATION_PATTERN.fullmatch(
                    str(payload["replacementGeneration"])
                )
                is None
            )
        )
        or payload.get("retiredGeneration") == payload.get("replacementGeneration")
    ):
        raise RuntimeError("paid pack GC job payload is invalid")
    scope = CommerceScope(
        authority_kind=str(payload["authorityKind"]),
        tenant_id=payload.get("tenantId"),
        principal_user_id=payload.get("principalUserId"),
    )
    ownership = commerce_ownership_values(scope)
    if (
        payload.get("tenantId") != ownership["tenant_id"]
        or payload.get("principalUserId") != ownership["principal_user_id"]
    ):
        raise RuntimeError("paid pack GC commerce scope is invalid")
    target = _PaidPackExportTarget(
        scope=scope,
        order_id=str(payload["orderId"]),
        pack_id=str(payload["packId"]),
        batch_id=str(payload["batchId"]),
        quote_id=str(payload["quoteId"]),
    )
    try:
        not_before = _as_utc(datetime.fromisoformat(str(payload["notBeforeUtc"])))
    except (TypeError, ValueError):
        raise RuntimeError("paid pack GC grace deadline is invalid") from None
    expected_dedupe = _paid_pack_export_gc_dedupe_key(payload)
    expected_tenants = (
        [str(target.scope.tenant_id)]
        if target.scope.authority_kind == "gateway"
        else []
    )
    if (
        job.dedupe_key != expected_dedupe
        or job.source_key != f"{target.pack_id}:{payload['retiredGeneration']}"
        or list(job.request_tenant_ids_json or []) != expected_tenants
        or utcnow() < not_before
    ):
        raise RuntimeError("paid pack GC job identity is invalid")
    return target


def _process_paid_pack_export_gc(
    *, job_id: str, worker_id: str, heartbeat: _LeaseHeartbeat
) -> None:
    with session_scope() as session:
        job = _owned_job(session, job_id=job_id, worker_id=worker_id)
        target = _paid_pack_export_gc_target(job)
        retired_generation = str(job.payload_json["retiredGeneration"])
        replacement_generation = job.payload_json["replacementGeneration"]
        heartbeat.assert_live()
        with _serialized_pack_export(session, pack_id=target.pack_id):
            # Validate the exact queued order/batch/quote lineage under its
            # immutable commerce scope before consulting filesystem custody.
            _paid_export_rows(session, target=target, lock=False)
            deleted = _gc_unadvertised_paid_pack_generations(
                session,
                target=target,
                fail_on_ambiguity=True,
                only_generation=retired_generation,
            )
            heartbeat.assert_live()
            complete_ingestion_job(
                session,
                job_id=job_id,
                worker_id=worker_id,
                result={
                    "schema": _PAID_PACK_EXPORT_GC_SCHEMA,
                    "pack_id": target.pack_id,
                    "retired_generation": retired_generation,
                    "replacement_generation": replacement_generation,
                    "deleted_generations": [path.name for path in deleted],
                },
            )


def _canonical_ready_item(
    session, *, item: PublicItemDescriptor
) -> tuple[AcquiredPublicItem, TranscriptResult] | None:
    """Return exact retained canonical facts, or require the provider pipeline."""
    source = session.execute(
        select(SourceVideo).where(
            SourceVideo.platform == item.platform,
            SourceVideo.external_id == item.external_id,
            SourceVideo.status == "active",
            SourceVideo.clip_ready.is_(True),
        )
    ).scalar_one_or_none()
    if source is None:
        return None
    channel = session.get(SourceChannel, source.channel_id)
    if channel is None or channel.status != "active":
        return None
    revision = session.execute(
        select(TranscriptRevision).where(
            TranscriptRevision.video_id == source.id,
            TranscriptRevision.is_current.is_(True),
            TranscriptRevision.status == "active",
        )
    ).scalar_one_or_none()
    if revision is None:
        return None
    segments = list(
        session.execute(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.revision_id == revision.id,
                TranscriptSegment.status == "active",
            )
            .order_by(TranscriptSegment.ordinal.asc())
        ).scalars()
    )
    if not segments:
        return None
    reference = session.execute(
        select(VideoMediaRef).where(
            VideoMediaRef.video_id == source.id,
            VideoMediaRef.role == "source_video",
            VideoMediaRef.status == "active",
        )
    ).scalar_one_or_none()
    if reference is None:
        return None
    media_object = session.get(MediaObject, reference.media_sha256)
    location = session.execute(
        select(MediaLocation).where(
            MediaLocation.media_sha256 == reference.media_sha256,
            MediaLocation.backend == "hot_local",
            MediaLocation.status == "active",
        )
    ).scalar_one_or_none()
    if media_object is None or location is None:
        return None
    path = Path(location.location_key)
    if not path.is_absolute() or not path.is_file():
        return None
    retained_item = PublicItemDescriptor(
        platform=source.platform,
        external_id=source.external_id,
        channel_external_id=channel.external_id,
        channel_handle=channel.handle or item.channel_handle,
        canonical_url=source.canonical_url or item.canonical_url,
        title=source.title or item.title,
        description=source.description or item.description,
        published_at=(source.published_at.isoformat() if source.published_at else item.published_at),
        duration_ms=source.duration_ms,
        metadata={**dict(source.metadata_json or {}), "canonical_ready_reuse": True},
    )
    return (
        AcquiredPublicItem(
            item=retained_item,
            media=HotMediaSpec(
                path=path,
                sha256=media_object.sha256,
                size_bytes=int(media_object.size_bytes),
                mime_type=media_object.mime_type,
            ),
        ),
        TranscriptResult(
            provider=revision.provider,
            provider_request_id=None,
            segments=tuple(
                {
                    "ordinal": segment.ordinal,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "speaker_label": segment.speaker_label,
                    "text": segment.text,
                }
                for segment in segments
            ),
        ),
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
    except BaseException as exc:  # noqa: BLE001 - ambiguity is reconciled durably
        failure = exc
    finally:
        cleanup_error: BaseException | None = None
        try:
            dependencies.delete_audio(path)
        except BaseException as exc:  # noqa: BLE001 - cleanup must always run
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
        if (
            failed_job.status == "failed"
            and failed_job.job_kind == "public_item_ingestion"
        ):
            for request in _all_tenant_requests(session, failed_job):
                _reconcile_paid_public_ingestion_failure(
                    session,
                    request=request,
                    error_code=code,
                    error_detail=detail,
                )
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
    return _all_tenant_requests(session, job)


def _all_tenant_requests(
    session, job: IngestionJob
) -> list[IngestionRequest]:
    rows: list[IngestionRequest] = []
    for tenant_id in sorted(set(job.request_tenant_ids_json or [])):
        set_tenant_scope(session, tenant_id)
        rows.extend(
            session.execute(
                select(IngestionRequest)
                .where(
                    IngestionRequest.tenant_id == tenant_id,
                    IngestionRequest.job_id == job.id,
                )
                .order_by(IngestionRequest.created_at.asc(), IngestionRequest.id.asc())
            )
            .scalars()
            .all()
        )
    if not rows:
        raise RuntimeError("public ingestion job has no tenant request")
    return rows


def _paid_work_binding(request: IngestionRequest) -> dict[str, Any] | None:
    value = (request.request_json or {}).get("paidWork")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _PAID_WORK_KEYS:
        raise RuntimeError("paid public-ingestion binding shape is invalid")
    if (
        value.get("schema") != PAID_PUBLIC_INGESTION_SCHEMA
        or value.get("tenantId") != request.tenant_id
        or value.get("principalId") != request.requested_by_user_id
        or not all(
            isinstance(value.get(key), str) and value.get(key)
            for key in (
                "intentId",
                "outboxId",
                "orderId",
                "packId",
                "batchId",
                "quoteId",
                "quoteHash",
                "videoId",
            )
        )
        or isinstance(value.get("position"), bool)
        or not isinstance(value.get("position"), int)
        or int(value["position"]) < 1
    ):
        raise RuntimeError("paid public-ingestion binding identity is invalid")
    return value


def _locked_paid_rows(session, *, request: IngestionRequest, binding: dict):
    if request.requested_by_user_id is None:
        raise RuntimeError("paid public-ingestion request has no principal")
    scope = gateway_commerce_scope(
        tenant_id=request.tenant_id,
        principal_user_id=request.requested_by_user_id,
    )
    set_commerce_scope(session, scope)
    # Every paid-publication writer uses pack -> batch -> order -> video. The
    # pack row serializes same-pack reconcilers before they can take a narrower
    # order lock, matching the export finalizer's pack-wide lock order.
    pack = session.execute(
        select(ChannelPack)
        .where(
            ChannelPack.id == binding["packId"],
            *commerce_scope_predicates(ChannelPack, scope),
        )
        .with_for_update()
    ).scalar_one_or_none()
    batch = session.execute(
        select(PackBatch)
        .where(
            PackBatch.id == binding["batchId"],
            PackBatch.pack_id == binding["packId"],
            PackBatch.quote_id == binding["quoteId"],
            *commerce_scope_predicates(PackBatch, scope),
        )
        .with_for_update()
    ).scalar_one_or_none()
    order = session.execute(
        select(ChannelOrder)
        .where(
            ChannelOrder.id == binding["orderId"],
            ChannelOrder.pack_id == binding["packId"],
            ChannelOrder.batch_id == binding["batchId"],
            ChannelOrder.quote_id == binding["quoteId"],
            *commerce_scope_predicates(ChannelOrder, scope),
        )
        .with_for_update()
    ).scalar_one_or_none()
    quote = session.execute(
        select(ChannelQuote).where(
            ChannelQuote.id == binding["quoteId"],
            *commerce_scope_predicates(ChannelQuote, scope),
        )
    ).scalar_one_or_none()
    video = session.execute(
        select(PackVideo)
        .where(
            PackVideo.pack_id == binding["packId"],
            PackVideo.batch_id == binding["batchId"],
            PackVideo.quote_id == binding["quoteId"],
            PackVideo.video_id == binding["videoId"],
            PackVideo.position == binding["position"],
            *commerce_scope_predicates(PackVideo, scope),
        )
        .with_for_update()
    ).scalar_one_or_none()
    if None in (order, pack, batch, quote, video):
        raise RuntimeError("paid public-ingestion commerce lineage is missing")
    return scope, order, pack, batch, quote, video


def _canonical_transcript_rows(session, rows: list[PackVideo]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for pack_video in rows:
        source = session.execute(
            select(SourceVideo).where(
                SourceVideo.platform == "youtube",
                SourceVideo.external_id == pack_video.video_id,
                SourceVideo.status == "active",
            )
        ).scalar_one_or_none()
        if source is None:
            continue
        revision = session.execute(
            select(TranscriptRevision).where(
                TranscriptRevision.video_id == source.id,
                TranscriptRevision.is_current.is_(True),
                TranscriptRevision.status == "active",
            )
        ).scalar_one_or_none()
        if revision is None:
            continue
        segments = session.execute(
            select(TranscriptSegment)
            .where(
                TranscriptSegment.revision_id == revision.id,
                TranscriptSegment.status == "active",
            )
            .order_by(TranscriptSegment.ordinal.asc())
        ).scalars()
        output[pack_video.video_id] = [
            {
                "video_id": pack_video.video_id,
                "segment_id": segment.id,
                "start_s": float(segment.start_ms) / 1000.0,
                "end_s": float(segment.end_ms) / 1000.0,
                "speaker": segment.speaker_label,
                "text": segment.text,
                "source": pack_video.transcript_source,
            }
            for segment in segments
        ]
    return output


def _verified_pack_artifacts(
    session,
    *,
    pack: ChannelPack,
    batch: PackBatch,
    quote: ChannelQuote,
    rows: list[PackVideo],
    transcript_rows: dict[str, list[dict]],
    snapshot_sha256: str,
) -> _VerifiedPackExport:
    if any(not transcript_rows.get(row.video_id) for row in rows):
        raise RuntimeError("paid pack cannot export before every canonical transcript exists")
    # Each build gets a fresh immutable directory. The snapshot prefix binds
    # content while the nonce prevents a stale-artifact repair from truncating
    # files that a ready pack is still advertising to concurrent readers.
    generation_prefix = f"paid-{snapshot_sha256}-"
    generation = f"{generation_prefix}{uuid.uuid4().hex}"
    expected_pack_root = (_export_root() / pack.id).resolve()
    expected_generation_pattern = re.compile(
        rf"{re.escape(generation_prefix)}[0-9a-f]{{32}}"
    )
    prior = dict(pack.export_paths_json or {})
    expected_names = {
        "manifest": "manifest.json",
        "videos": "videos.ndjson",
        "links": "links.ndjson",
        "transcripts": "transcripts.ndjson",
        "archive": f"{pack.id}.bundle.zip",
    }
    prior_valid = bool(prior)
    verified_prior: dict[str, str] = {}
    prior_generation_root: Path | None = None
    if prior_valid:
        for key, expected_name in expected_names.items():
            raw_path = prior.get(f"{key}_path")
            raw_digest = prior.get(f"{key}_sha256")
            if (
                not isinstance(raw_path, str)
                or not isinstance(raw_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", raw_digest) is None
            ):
                prior_valid = False
                break
            try:
                path = Path(raw_path).resolve(strict=True)
                data = path.read_bytes()
            except OSError:
                prior_valid = False
                break
            if prior_generation_root is None:
                prior_generation_root = path.parent
            if (
                path.parent != prior_generation_root
                or path.parent.parent != expected_pack_root
                or expected_generation_pattern.fullmatch(path.parent.name) is None
                or path.name != expected_name
                or not data
                or hashlib.sha256(data).hexdigest() != raw_digest
            ):
                prior_valid = False
                break
            verified_prior[f"{key}_path"] = str(path)
            verified_prior[f"{key}_sha256"] = raw_digest
    if prior_valid:
        try:
            manifest = json.loads(
                Path(verified_prior["manifest_path"]).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            prior_valid = False
        else:
            prior_valid = bool(
                isinstance(manifest, dict)
                and manifest.get("pack_id") == pack.id
                and manifest.get("status") == "ready"
                and manifest.get("ready_video_count")
                == pack.total_purchased_video_count
                and manifest.get("total_purchased_video_count")
                == pack.total_purchased_video_count
                and manifest.get("batch_count") == pack.batch_count
                and manifest.get("latest_batch_index") == batch.batch_index
            )
    if prior_valid:
        manifest["verified_exports"] = {
            key: value
            for key, value in verified_prior.items()
            if key.endswith("_sha256")
        }
        return _VerifiedPackExport(
            paths=verified_prior,
            manifest=manifest,
            unpublished_root=None,
        )

    unpublished_root = expected_pack_root / generation
    try:
        paths = _build_pack_artifacts(
            session=session,
            pack=pack,
            batch=batch,
            language=str((quote.request_json or {}).get("language") or "en"),
            prefer_auto=bool((quote.request_json or {}).get("prefer_auto", True)),
            transcript_rows_by_video=transcript_rows,
            artifact_generation=generation,
            manifest_status="ready",
            batch_status="ready",
            authoritative_pack_rows=rows,
        )
        receipt: dict[str, str] = {}
        for key in ("manifest", "videos", "links", "transcripts", "archive"):
            path = Path(str(paths[f"{key}_path"])).resolve(strict=True)
            data = path.read_bytes()
            if path.parent != unpublished_root or not data:
                raise RuntimeError(f"paid pack {key} export is invalid")
            receipt[f"{key}_path"] = str(path)
            receipt[f"{key}_sha256"] = hashlib.sha256(data).hexdigest()
        manifest = json.loads(
            Path(receipt["manifest_path"]).read_text(encoding="utf-8")
        )
        manifest["verified_exports"] = {
            key: receipt[key]
            for key in receipt
            if key.endswith("_sha256")
        }
    except Exception:
        _cleanup_unpublished_pack_generation(
            root=unpublished_root,
            pack_id=pack.id,
            snapshot_sha256=snapshot_sha256,
        )
        raise
    return _VerifiedPackExport(
        paths=receipt,
        manifest=manifest,
        unpublished_root=unpublished_root,
    )


def _cleanup_unpublished_pack_generation(
    *, root: Path, pack_id: str, snapshot_sha256: str
) -> bool:
    expected_parent = (_export_root() / pack_id).resolve()
    candidate = Path(root)
    expected_name = re.compile(
        rf"paid-{re.escape(snapshot_sha256)}-[0-9a-f]{{32}}"
    )
    if candidate.parent != expected_parent or expected_name.fullmatch(candidate.name) is None:
        LOG.error("refused unsafe unpublished paid-pack cleanup path=%s", candidate)
        return False
    try:
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.exists():
            shutil.rmtree(candidate)
    except OSError:
        LOG.exception("could not clean unpublished paid-pack generation path=%s", candidate)
        return False
    return not candidate.exists() and not candidate.is_symlink()


def _authoritative_advertised_pack_generation(
    session,
    *,
    target: _PaidPackExportTarget,
    expected_pack_root: Path,
) -> tuple[bool, Path | None]:
    """Return one authoritative advertised root, preserving on any ambiguity."""

    try:
        # Keep an ordinary statement failure inside a savepoint so a failed GC
        # readback cannot poison the finalizer's outer transaction. A transport
        # failure may still make the session unusable, but it never authorizes a
        # filesystem deletion.
        with session.begin_nested():
            set_commerce_scope(session, target.scope)
            pack = session.execute(
                select(ChannelPack)
                .where(
                    ChannelPack.id == target.pack_id,
                    *commerce_scope_predicates(ChannelPack, target.scope),
                )
                .execution_options(populate_existing=True)
            ).scalar_one_or_none()
    except Exception:
        LOG.exception(
            "paid-pack generation GC could not read authoritative pack state pack=%s",
            target.pack_id,
        )
        return False, None
    if pack is None:
        LOG.error(
            "paid-pack generation GC found no authoritative pack row pack=%s",
            target.pack_id,
        )
        return False, None

    advertised = pack.export_paths_json
    if advertised is None:
        return True, None
    if not isinstance(advertised, dict) or not advertised:
        LOG.error(
            "paid-pack generation GC found ambiguous export metadata pack=%s",
            target.pack_id,
        )
        return False, None

    expected_names = {
        "manifest_path": "manifest.json",
        "videos_path": "videos.ndjson",
        "links_path": "links.ndjson",
        "transcripts_path": "transcripts.ndjson",
        "archive_path": f"{target.pack_id}.bundle.zip",
    }
    unknown_path_keys = {
        key
        for key in advertised
        if isinstance(key, str)
        and key.endswith("_path")
        and key not in expected_names
    }
    if unknown_path_keys:
        LOG.error(
            "paid-pack generation GC found unknown advertised paths pack=%s keys=%s",
            target.pack_id,
            sorted(unknown_path_keys),
        )
        return False, None

    advertised_root: Path | None = None
    for key, expected_name in expected_names.items():
        raw_path = advertised.get(key)
        if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
            LOG.error(
                "paid-pack generation GC found incomplete advertised paths pack=%s",
                target.pack_id,
            )
            return False, None
        try:
            path = Path(raw_path).resolve(strict=False)
        except (OSError, RuntimeError):
            LOG.exception(
                "paid-pack generation GC could not resolve advertised path pack=%s key=%s",
                target.pack_id,
                key,
            )
            return False, None
        if path.name != expected_name or path.parent.parent != expected_pack_root:
            LOG.error(
                "paid-pack generation GC found out-of-root advertised path pack=%s key=%s",
                target.pack_id,
                key,
            )
            return False, None
        if advertised_root is None:
            advertised_root = path.parent
        elif path.parent != advertised_root:
            LOG.error(
                "paid-pack generation GC found mixed advertised roots pack=%s",
                target.pack_id,
            )
            return False, None
    return True, advertised_root


def _mark_paid_pack_generation_published(*, root: Path, pack_id: str) -> bool:
    """Mark one DB-proven UUID generation without changing advertised artifacts."""

    expected_parent = (_export_root() / pack_id).resolve()
    candidate = Path(root)
    if (
        candidate.parent != expected_parent
        or _PAID_PACK_GENERATION_PATTERN.fullmatch(candidate.name) is None
        or candidate.is_symlink()
        or not candidate.is_dir()
    ):
        LOG.error("refused unsafe paid-pack publication marker path=%s", candidate)
        return False
    marker = candidate / _PAID_PACK_PUBLISHED_MARKER
    try:
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            LOG.error("refused ambiguous paid-pack publication marker path=%s", marker)
            return False
        if not marker.exists():
            marker.touch(exist_ok=False)
    except OSError:
        LOG.exception("could not mark paid-pack generation published path=%s", candidate)
        return False
    return True


def _gc_unadvertised_paid_pack_generations(
    session,
    *,
    target: _PaidPackExportTarget,
    fail_on_ambiguity: bool = False,
    only_generation: str | None = None,
) -> tuple[Path, ...]:
    """Delete only exact generations proven absent from authoritative DB state.

    The caller must hold ``_serialized_pack_export`` for this pack. Every pack
    export writer uses that same boundary, so the scoped DB read and candidate
    deletion form one custody interval without holding commerce ``FOR UPDATE``
    rows across filesystem I/O. Never-published crash roots have no publication
    marker and can be removed immediately. A published root is removed only by
    its exact durable grace-expiry job, never by an opportunistic later build.
    """

    if (
        only_generation is not None
        and _PAID_PACK_GENERATION_PATTERN.fullmatch(only_generation) is None
    ):
        raise RuntimeError("paid pack GC target generation is invalid")

    expected_pack_root = (_export_root() / target.pack_id).resolve()
    authoritative, advertised_root = _authoritative_advertised_pack_generation(
        session,
        target=target,
        expected_pack_root=expected_pack_root,
    )
    if not authoritative:
        if fail_on_ambiguity:
            raise RuntimeError("paid pack GC authoritative readback is ambiguous")
        return ()
    # Every UUID generation attached by this protocol receives this internal
    # marker only after COMMIT succeeds. An acknowledgement-lost COMMIT is
    # reconciled here from the authoritative current DB path before any scan.
    # If marking the current root is ambiguous, unmarked roots are preserved.
    unmarked_cleanup_safe = True
    if (
        advertised_root is not None
        and _PAID_PACK_GENERATION_PATTERN.fullmatch(advertised_root.name) is not None
    ):
        unmarked_cleanup_safe = _mark_paid_pack_generation_published(
            root=advertised_root,
            pack_id=target.pack_id,
        )
        if not unmarked_cleanup_safe:
            raise RuntimeError("paid pack GC could not protect advertised generation")
    try:
        candidates = tuple(expected_pack_root.iterdir())
    except FileNotFoundError:
        return ()
    except OSError:
        LOG.exception(
            "paid-pack generation GC could not enumerate pack root pack=%s",
            target.pack_id,
        )
        if fail_on_ambiguity:
            raise RuntimeError("paid pack GC could not enumerate pack root")
        return ()

    deleted: list[Path] = []
    for candidate in sorted(candidates, key=lambda item: item.name):
        match = _PAID_PACK_GENERATION_PATTERN.fullmatch(candidate.name)
        if match is None or (
            only_generation is not None and candidate.name != only_generation
        ):
            continue
        try:
            candidate.lstat()
            resolved_candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            LOG.exception(
                "paid-pack generation GC could not inspect candidate path=%s",
                candidate,
            )
            if fail_on_ambiguity:
                raise RuntimeError("paid pack GC candidate inspection is ambiguous")
            continue
        if not candidate.is_dir() and not candidate.is_symlink():
            continue
        if advertised_root is not None and resolved_candidate == advertised_root:
            continue
        if candidate.is_symlink():
            if only_generation is None:
                continue
            removed = _cleanup_unpublished_pack_generation(
                root=candidate,
                pack_id=target.pack_id,
                snapshot_sha256=match.group("snapshot"),
            )
            if removed:
                deleted.append(candidate)
            elif fail_on_ambiguity:
                raise RuntimeError("paid pack GC could not unlink candidate symlink")
            continue
        marker = candidate / _PAID_PACK_PUBLISHED_MARKER
        try:
            if marker.is_symlink() or (marker.exists() and not marker.is_file()):
                if fail_on_ambiguity:
                    raise RuntimeError("paid pack GC publication marker is ambiguous")
                continue
            published_marker = marker.is_file()
        except OSError:
            LOG.exception(
                "paid-pack generation GC could not inspect marker path=%s",
                marker,
            )
            if fail_on_ambiguity:
                raise RuntimeError("paid pack GC publication marker is ambiguous")
            continue
        if published_marker and only_generation is None:
            continue
        if not published_marker and not unmarked_cleanup_safe:
            continue
        removed = _cleanup_unpublished_pack_generation(
            root=candidate,
            pack_id=target.pack_id,
            snapshot_sha256=match.group("snapshot"),
        )
        if removed:
            deleted.append(candidate)
        elif fail_on_ambiguity:
            raise RuntimeError("paid pack GC could not remove candidate generation")
    if deleted:
        LOG.info(
            "paid-pack generation GC removed unadvertised roots pack=%s count=%s",
            target.pack_id,
            len(deleted),
        )
    return tuple(deleted)


def _paid_export_snapshot(
    session,
    *,
    pack: ChannelPack,
    batches: tuple[PackBatch, ...],
    orders: tuple[ChannelOrder, ...],
    quotes_by_id: dict[str, ChannelQuote],
    checkouts_by_id: dict[str, CheckoutSessionRecord],
    rows: list[PackVideo],
) -> tuple[str, dict[str, list[dict]]]:
    transcript_rows = _canonical_transcript_rows(session, rows)
    if any(not transcript_rows.get(row.video_id) for row in rows):
        raise RuntimeError("paid pack cannot export before every canonical transcript exists")
    payload = {
        "pack": {
            "id": pack.id,
            "mode": pack.mode,
            "namespace": pack.namespace,
            "channel_handle": pack.channel_handle,
            "resolved_channel_id": pack.resolved_channel_id,
            "resolved_channel_name": pack.resolved_channel_name,
            "batch_count": pack.batch_count,
            "total_purchased_video_count": pack.total_purchased_video_count,
        },
        "batches": [
            {
                "id": batch.id,
                "pack_id": batch.pack_id,
                "quote_id": batch.quote_id,
                "checkout_session_id": batch.checkout_session_id,
                "batch_index": batch.batch_index,
                "billable_video_count": batch.billable_video_count,
                "quote": {
                    "id": quotes_by_id[batch.quote_id].id,
                    "language": str(
                        (
                            quotes_by_id[batch.quote_id].request_json or {}
                        ).get("language")
                        or "en"
                    ),
                    "prefer_auto": bool(
                        (
                            quotes_by_id[batch.quote_id].request_json or {}
                        ).get("prefer_auto", True)
                    ),
                },
            }
            for batch in batches
        ],
        "orders": [
            {
                "id": order.id,
                "quote_id": order.quote_id,
                "checkout_session_id": order.checkout_session_id,
                "pack_id": order.pack_id,
                "batch_id": order.batch_id,
                "payment_status": order.payment_status,
                "payment_provider": order.payment_provider,
                "amount_cents": order.amount_cents,
                "currency": order.currency,
            }
            for order in orders
        ],
        "checkouts": [
            {
                "id": checkout.id,
                "status": checkout.status,
                "currency": checkout.currency,
                "total_amount_cents": checkout.total_amount_cents,
                "quote_ids": list(checkout.quote_ids_json or []),
                "line_items": list(checkout.line_items_json or []),
                "payment_provider": checkout.payment_provider,
                "payment_status": checkout.payment_status,
            }
            for checkout in (
                checkouts_by_id[key] for key in sorted(checkouts_by_id)
            )
        ],
        "videos": [
            {
                "id": row.id,
                "pack_id": row.pack_id,
                "batch_id": row.batch_id,
                "quote_id": row.quote_id,
                "position": row.position,
                "video_id": row.video_id,
                "title": row.title,
                "description": row.description,
                "channel_name": row.channel_name,
                "channel_handle": row.channel_handle,
                "published_at": row.published_at,
                "duration_s": row.duration_s,
                "video_url": row.video_url,
                "thumbnail_url": row.thumbnail_url,
                "transcript_source": row.transcript_source,
                "indexed_parent_id": row.indexed_parent_id,
                "status": row.status,
                "transcript_rows": transcript_rows[row.video_id],
            }
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), transcript_rows


def _aggregate_paid_pack(
    session,
    *,
    scope,
    order: ChannelOrder,
    pack: ChannelPack,
    batch: PackBatch,
    quote: ChannelQuote,
    export_result: tuple[dict[str, str], dict[str, Any]] | None = None,
) -> bool:
    batch_rows = list(
        session.execute(
            select(PackVideo)
            .where(
                PackVideo.batch_id == batch.id,
                *commerce_scope_predicates(PackVideo, scope),
            )
            .order_by(PackVideo.position.asc())
            .with_for_update()
        ).scalars()
    )
    pack_rows = list(
        session.execute(
            select(PackVideo)
            .where(
                PackVideo.pack_id == pack.id,
                *commerce_scope_predicates(PackVideo, scope),
            )
            .order_by(PackVideo.position.asc())
            .with_for_update()
        ).scalars()
    )
    batch.ready_video_count = sum(row.status == "ready" for row in batch_rows)
    pack.ready_video_count = sum(row.status == "ready" for row in pack_rows)
    failed = any(row.status == "failed" for row in batch_rows)
    batch_complete = (
        bool(batch_rows)
        and batch.ready_video_count == len(batch_rows) == batch.billable_video_count
    )
    pack_failed = any(row.status == "failed" for row in pack_rows)
    all_ready = (
        bool(pack_rows)
        and pack.ready_video_count == len(pack_rows) == pack.total_purchased_video_count
    )
    # Path strings and stored digests are not custody proof. A successful
    # reconciliation always schedules the unlocked finalizer unless this exact
    # call is attaching files it has just verified outside the row-lock phase.
    exports_ready = export_result is not None
    batch_was_ready = batch.status == "ready" and order.status == "ready"
    pack_was_ready = pack.status == "ready"
    batch.status = (
        "failed"
        if failed
        else "ready"
        if batch_complete
        and all_ready
        and (exports_ready or (batch_was_ready and pack_was_ready))
        else "partial"
        if batch.ready_video_count
        else "queued"
    )
    order.status = batch.status
    pack.status = (
        "failed"
        if pack_failed
        else "ready"
        if all_ready and (exports_ready or pack_was_ready)
        else "partial"
        if pack.ready_video_count
        else "queued"
    )
    return bool(all_ready and not pack_failed and not exports_ready)


def _paid_export_rows(
    session,
    *,
    target: _PaidPackExportTarget,
    lock: bool,
) -> _PaidPackExportState:
    set_commerce_scope(session, target.scope)

    def _maybe_lock(statement):
        return statement.with_for_update() if lock else statement

    pack = session.execute(
        _maybe_lock(
            select(ChannelPack).where(
                ChannelPack.id == target.pack_id,
                *commerce_scope_predicates(ChannelPack, target.scope),
            )
        )
    ).scalar_one_or_none()
    batches = tuple(
        session.execute(
            _maybe_lock(
                select(PackBatch)
                .where(
                    PackBatch.pack_id == target.pack_id,
                    *commerce_scope_predicates(PackBatch, target.scope),
                )
                .order_by(PackBatch.batch_index.asc(), PackBatch.id.asc())
            )
        ).scalars()
    )
    orders = tuple(
        session.execute(
            _maybe_lock(
                select(ChannelOrder)
                .where(
                    ChannelOrder.pack_id == target.pack_id,
                    *commerce_scope_predicates(ChannelOrder, target.scope),
                )
                .order_by(ChannelOrder.batch_id.asc(), ChannelOrder.id.asc())
            )
        ).scalars()
    )
    quote_ids = tuple(batch.quote_id for batch in batches)
    quotes = tuple(
        session.execute(
            _maybe_lock(
                select(ChannelQuote)
                .where(
                    ChannelQuote.id.in_(quote_ids),
                    *commerce_scope_predicates(ChannelQuote, target.scope),
                )
                .order_by(ChannelQuote.id.asc())
            )
        ).scalars()
    ) if quote_ids else ()
    checkout_ids = tuple(batch.checkout_session_id for batch in batches)
    checkouts = tuple(
        session.execute(
            _maybe_lock(
                select(CheckoutSessionRecord)
                .where(
                    CheckoutSessionRecord.id.in_(checkout_ids),
                    *commerce_scope_predicates(
                        CheckoutSessionRecord, target.scope
                    ),
                )
                .order_by(CheckoutSessionRecord.id.asc())
            )
        ).scalars()
    ) if checkout_ids else ()
    rows = tuple(
        session.execute(
            _maybe_lock(
                select(PackVideo)
                .join(PackBatch, PackBatch.id == PackVideo.batch_id)
                .where(
                    PackVideo.pack_id == target.pack_id,
                    *commerce_scope_predicates(PackVideo, target.scope),
                    *commerce_scope_predicates(PackBatch, target.scope),
                )
                .order_by(
                    PackBatch.batch_index.asc(),
                    PackVideo.position.asc(),
                    PackVideo.id.asc(),
                )
            )
        ).scalars()
    )
    if pack is None or not batches or not orders or not rows:
        raise RuntimeError("paid pack export lineage is missing")
    quotes_by_id = {quote.id: quote for quote in quotes}
    checkouts_by_id = {checkout.id: checkout for checkout in checkouts}
    batches_by_id = {batch.id: batch for batch in batches}
    orders_by_batch: dict[str, ChannelOrder] = {}
    for order in orders:
        if order.batch_id in orders_by_batch:
            raise RuntimeError("paid pack export contains duplicate batch orders")
        orders_by_batch[order.batch_id] = order
    if (
        len(batches) != int(pack.batch_count or 0)
        or [batch.batch_index for batch in batches]
        != list(range(1, len(batches) + 1))
        or set(orders_by_batch) != set(batches_by_id)
        or set(quotes_by_id) != set(quote_ids)
        or set(checkouts_by_id) != set(checkout_ids)
    ):
        raise RuntimeError("paid pack export lineage is incomplete")
    for batch in batches:
        order = orders_by_batch[batch.id]
        checkout = checkouts_by_id[batch.checkout_session_id]
        matching_lines = [
            line
            for line in (checkout.line_items_json or [])
            if isinstance(line, dict) and line.get("quote_id") == batch.quote_id
        ]
        checkout_quote_ids = list(checkout.quote_ids_json or [])
        line_amount = (
            matching_lines[0].get("amount_cents")
            if len(matching_lines) == 1
            else None
        )
        if (
            order.pack_id != pack.id
            or order.quote_id != batch.quote_id
            or order.checkout_session_id != batch.checkout_session_id
            or checkout_quote_ids.count(batch.quote_id) != 1
            or len(matching_lines) != 1
            or isinstance(line_amount, bool)
            or not isinstance(line_amount, int)
            or line_amount != int(batch.amount_cents)
            or int(order.amount_cents) != int(batch.amount_cents)
            or order.currency != checkout.currency
            or order.payment_provider != checkout.payment_provider
            or not _settled_payment_status(order.payment_status)
            or not _settled_payment_status(checkout.payment_status)
            or checkout.status not in {"open", "completed"}
        ):
            raise RuntimeError("paid pack export lineage is inconsistent")
    target_order = next((row for row in orders if row.id == target.order_id), None)
    target_batch = batches_by_id.get(target.batch_id)
    if (
        target_order is None
        or target_batch is None
        or target_order.batch_id != target_batch.id
        or target_order.quote_id != target.quote_id
        or target_batch.quote_id != target.quote_id
    ):
        raise RuntimeError("paid pack export target lineage is inconsistent")
    for row in rows:
        row_batch = batches_by_id.get(row.batch_id)
        if row_batch is None or row.quote_id != row_batch.quote_id:
            raise RuntimeError("paid pack video lineage is inconsistent")
    return _PaidPackExportState(
        pack=pack,
        batches=batches,
        orders=orders,
        quotes_by_id=quotes_by_id,
        checkouts_by_id=checkouts_by_id,
        rows=rows,
    )


def _settled_payment_status(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.startswith("settled_") or normalized in {
        "settled",
        "paid",
        "payment_confirmed",
        "succeeded",
        "development_bypass",
    }


def _attach_paid_pack_export(
    *,
    state: _PaidPackExportState,
    export_result: _VerifiedPackExport,
) -> None:
    pack = state.pack
    paths = export_result.paths
    manifest = export_result.manifest
    rows_by_batch: dict[str, list[PackVideo]] = {
        batch.id: [] for batch in state.batches
    }
    for row in state.rows:
        rows_by_batch[row.batch_id].append(row)
    if (
        len(state.rows) != pack.total_purchased_video_count
        or any(row.status != "ready" for row in state.rows)
    ):
        raise RuntimeError("paid pack is not fully ready for export attachment")
    orders_by_batch = {order.batch_id: order for order in state.orders}
    for batch in state.batches:
        batch_rows = rows_by_batch[batch.id]
        if len(batch_rows) != batch.billable_video_count:
            raise RuntimeError("paid pack batch size is inconsistent")
        batch.ready_video_count = len(batch_rows)
        batch.status = "ready"
        orders_by_batch[batch.id].status = "ready"
        batch.manifest_json = {
            "pack_id": pack.id,
            "batch_id": batch.id,
            "batch_index": batch.batch_index,
            "ready_video_count": batch.ready_video_count,
            "billable_video_count": batch.billable_video_count,
            "manifest_sha256": paths["manifest_sha256"],
        }
    pack.ready_video_count = len(state.rows)
    pack.status = "ready"
    pack.export_paths_json = paths
    pack.manifest_json = manifest


def _mark_paid_pack_export_state_incomplete(
    state: _PaidPackExportState, *, error: BaseException
) -> None:
    rows_by_batch: dict[str, list[PackVideo]] = {
        batch.id: [] for batch in state.batches
    }
    for row in state.rows:
        rows_by_batch[row.batch_id].append(row)
    orders_by_batch = {order.batch_id: order for order in state.orders}
    for batch in state.batches:
        batch_rows = rows_by_batch[batch.id]
        batch.ready_video_count = sum(row.status == "ready" for row in batch_rows)
        batch.status = (
            "failed"
            if any(row.status == "failed" for row in batch_rows)
            else "partial"
            if batch.ready_video_count
            else "queued"
        )
        batch.manifest_json = None
        order = orders_by_batch[batch.id]
        order.status = batch.status
        order.notes_json = {
            **dict(order.notes_json or {}),
            "export_failure": str(error)[:1000],
        }
    state.pack.ready_video_count = sum(row.status == "ready" for row in state.rows)
    state.pack.status = (
        "failed"
        if any(row.status == "failed" for row in state.rows)
        else "partial"
        if state.pack.ready_video_count
        else "queued"
    )
    state.pack.export_paths_json = None
    state.pack.manifest_json = None


@contextmanager
def _serialized_pack_export(session, *, pack_id: str):
    if session.get_bind().dialect.name == "postgresql":
        lock_value = int.from_bytes(
            hashlib.sha256(f"icmfyi:paid-pack-export:{pack_id}".encode()).digest()[:8],
            byteorder="big",
            signed=True,
        )
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_value)"),
            {"lock_value": lock_value},
        )
        yield
        return
    with _PACK_EXPORT_LOCKS_GUARD:
        process_lock = _PACK_EXPORT_LOCKS.setdefault(pack_id, threading.Lock())
    with process_lock:
        yield


def _finalize_paid_pack_export(target: _PaidPackExportTarget) -> None:
    failure: Exception | None = None
    snapshot_sha256: str | None = None
    export_result: _VerifiedPackExport | None = None
    prior_advertised_root: Path | None = None
    with session_scope() as session:
        with _serialized_pack_export(session, pack_id=target.pack_id):
            _gc_unadvertised_paid_pack_generations(session, target=target)
            prior_authoritative, prior_root = _authoritative_advertised_pack_generation(
                session,
                target=target,
                expected_pack_root=(_export_root() / target.pack_id).resolve(),
            )
            if (
                prior_authoritative
                and prior_root is not None
                and _PAID_PACK_GENERATION_PATTERN.fullmatch(prior_root.name)
                is not None
            ):
                prior_advertised_root = prior_root
            try:
                # A savepoint lets an ordinary query/build failure roll back
                # without releasing PostgreSQL's transaction-scoped advisory
                # lock. The incomplete state is then attached atomically under
                # the same ownership interval.
                with session.begin_nested():
                    state = _paid_export_rows(session, target=target, lock=False)
                    pack = state.pack
                    rows = list(state.rows)
                    if (
                        any(row.status != "ready" for row in rows)
                        or len(rows) != pack.total_purchased_video_count
                    ):
                        raise RuntimeError(
                            "paid pack changed before export generation"
                        )
                    snapshot_sha256, transcript_rows = _paid_export_snapshot(
                        session,
                        pack=pack,
                        batches=state.batches,
                        orders=state.orders,
                        quotes_by_id=state.quotes_by_id,
                        checkouts_by_id=state.checkouts_by_id,
                        rows=rows,
                    )
                    export_result = _verified_pack_artifacts(
                        session,
                        pack=pack,
                        batch=state.latest_batch,
                        quote=state.latest_quote,
                        rows=rows,
                        transcript_rows=transcript_rows,
                        snapshot_sha256=snapshot_sha256,
                    )

                    # No filesystem access occurs after this point. Re-read
                    # and lock the exact commerce lineage, then compare the
                    # complete DB snapshot before attaching immutable paths.
                    session.expire_all()
                    state = _paid_export_rows(session, target=target, lock=True)
                    pack = state.pack
                    rows = list(state.rows)
                    current_sha256, _ = _paid_export_snapshot(
                        session,
                        pack=pack,
                        batches=state.batches,
                        orders=state.orders,
                        quotes_by_id=state.quotes_by_id,
                        checkouts_by_id=state.checkouts_by_id,
                        rows=rows,
                    )
                    if current_sha256 != snapshot_sha256:
                        raise RuntimeError(
                            "paid pack changed during export generation"
                        )
                    _attach_paid_pack_export(
                        state=state, export_result=export_result
                    )
                    if pack.status != "ready" or any(
                        order.status != "ready" for order in state.orders
                    ):
                        raise RuntimeError(
                            "paid pack export could not reach ready state"
                        )
                    if (
                        export_result.unpublished_root is not None
                        and prior_advertised_root is not None
                        and prior_advertised_root
                        != export_result.unpublished_root
                    ):
                        _enqueue_paid_pack_export_gc(
                            session,
                            target=target,
                            retired_generation=prior_advertised_root.name,
                            replacement_generation=export_result.unpublished_root.name,
                        )
            except Exception as exc:
                failure = exc
                try:
                    state = _paid_export_rows(session, target=target, lock=True)
                    _mark_paid_pack_export_state_incomplete(state, error=exc)
                    if prior_advertised_root is not None:
                        _enqueue_paid_pack_export_gc(
                            session,
                            target=target,
                            retired_generation=prior_advertised_root.name,
                            replacement_generation=None,
                        )
                except Exception:
                    if (
                        export_result is not None
                        and export_result.unpublished_root is not None
                        and snapshot_sha256 is not None
                    ):
                        _cleanup_unpublished_pack_generation(
                            root=export_result.unpublished_root,
                            pack_id=target.pack_id,
                            snapshot_sha256=snapshot_sha256,
                        )
                    raise
            # Commit before releasing the process lock on SQLite. PostgreSQL's
            # transaction advisory lock is released by this exact commit.
            try:
                session.commit()
            except Exception:
                # A transport-level COMMIT error is outcome-ambiguous. Retain
                # the generation until an independent database readback proves
                # it was not published; deleting here could erase a committed
                # export whose acknowledgement was lost.
                LOG.exception(
                    "paid-pack export commit outcome is ambiguous pack=%s",
                    target.pack_id,
                )
                raise
            if (
                failure is None
                and export_result is not None
                and export_result.unpublished_root is not None
            ):
                _mark_paid_pack_generation_published(
                    root=export_result.unpublished_root,
                    pack_id=target.pack_id,
                )
            if (
                failure is not None
                and export_result is not None
                and export_result.unpublished_root is not None
                and snapshot_sha256 is not None
            ):
                _cleanup_unpublished_pack_generation(
                    root=export_result.unpublished_root,
                    pack_id=target.pack_id,
                    snapshot_sha256=snapshot_sha256,
                )
    if failure is not None:
        raise failure


def _reconcile_paid_public_ingestion_success(
    session,
    *,
    request: IngestionRequest,
    media_id: str,
    transcript_provider: str,
) -> _PaidPackExportTarget | None:
    binding = _paid_work_binding(request)
    if binding is None:
        return None
    scope, order, pack, batch, quote, video = _locked_paid_rows(
        session, request=request, binding=binding
    )
    video.status = "ready"
    video.indexed_parent_id = media_id
    video.transcript_source = transcript_provider
    needs_export = _aggregate_paid_pack(
        session, scope=scope, order=order, pack=pack, batch=batch, quote=quote
    )
    if not needs_export:
        return None
    return _PaidPackExportTarget(
        scope=scope,
        order_id=order.id,
        pack_id=pack.id,
        batch_id=batch.id,
        quote_id=quote.id,
    )


def _reconcile_paid_public_ingestion_failure(
    session,
    *,
    request: IngestionRequest,
    error_code: str,
    error_detail: str,
) -> None:
    binding = _paid_work_binding(request)
    if binding is None:
        return
    scope, order, pack, batch, quote, video = _locked_paid_rows(
        session, request=request, binding=binding
    )
    if video.status == "ready":
        # A later terminal replay must not demote a previously completed pack.
        # If export finalization had failed, its pre-existing state is already
        # partial and remains eligible for operational reconciliation.
        return
    video.status = "failed"
    batch.build_notes_json = {
        **dict(batch.build_notes_json or {}),
        video.video_id: f"{error_code}: {error_detail[:1000]}",
    }
    _aggregate_paid_pack(
        session, scope=scope, order=order, pack=pack, batch=batch, quote=quote
    )


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
