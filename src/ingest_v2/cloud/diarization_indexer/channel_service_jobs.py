from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Iterable

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .channel_service_store import (
    IngestionEffect,
    IngestionJob,
    IngestionRequest,
    SourceChannel,
    Tenant,
    TenantChannelEntitlement,
    UserAccount,
    utcnow,
)


class IdempotencyConflict(ValueError):
    """The same idempotency key was reused with different immutable inputs."""


class IngestionLeaseLost(RuntimeError):
    """A worker tried to finish work that it no longer owns."""


def _required(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    return normalized


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]
    return f"{prefix}_{digest}"


def _json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _insert_with_savepoint(session: Session, row: Any) -> bool:
    """Return False on a concurrent uniqueness winner without aborting the outer transaction."""
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
        return True
    except IntegrityError:
        return False


def ensure_source_channel(
    session: Session,
    *,
    platform: str,
    external_id: str,
    handle: str | None = None,
    display_name: str | None = None,
    canonical_url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SourceChannel:
    platform_normalized = _required(platform, "platform").lower()
    external_id_normalized = _required(external_id, "external_id")
    existing = session.execute(
        select(SourceChannel).where(
            SourceChannel.platform == platform_normalized,
            SourceChannel.external_id == external_id_normalized,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    row = SourceChannel(
        id=_stable_id("chn", f"{platform_normalized}:{external_id_normalized}"),
        platform=platform_normalized,
        external_id=external_id_normalized,
        handle=(str(handle).strip() or None) if handle is not None else None,
        display_name=(str(display_name).strip() or None) if display_name is not None else None,
        canonical_url=(str(canonical_url).strip() or None) if canonical_url is not None else None,
        metadata_json=dict(metadata or {}),
    )
    if _insert_with_savepoint(session, row):
        return row
    return session.execute(
        select(SourceChannel).where(
            SourceChannel.platform == platform_normalized,
            SourceChannel.external_id == external_id_normalized,
        )
    ).scalar_one()


def ensure_channel_entitlement(
    session: Session,
    *,
    tenant_id: str,
    channel_id: str,
    granted_by_user_id: str | None = None,
    access_level: str = "query",
) -> TenantChannelEntitlement:
    tenant_id = _required(tenant_id, "tenant_id")
    channel_id = _required(channel_id, "channel_id")
    access_level = _required(access_level, "access_level")
    if session.get(Tenant, tenant_id) is None:
        raise ValueError(f"tenant {tenant_id} does not exist")
    if session.get(SourceChannel, channel_id) is None:
        raise ValueError(f"source channel {channel_id} does not exist")
    if granted_by_user_id and session.get(UserAccount, granted_by_user_id) is None:
        raise ValueError(f"user {granted_by_user_id} does not exist")

    existing = session.execute(
        select(TenantChannelEntitlement).where(
            TenantChannelEntitlement.tenant_id == tenant_id,
            TenantChannelEntitlement.channel_id == channel_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.access_level != access_level:
            raise IdempotencyConflict(
                "channel entitlement already exists with a different access level"
            )
        return existing

    row = TenantChannelEntitlement(
        id=_stable_id("ent", f"{tenant_id}:{channel_id}"),
        tenant_id=tenant_id,
        channel_id=channel_id,
        granted_by_user_id=granted_by_user_id,
        access_level=access_level,
    )
    if _insert_with_savepoint(session, row):
        return row
    return session.execute(
        select(TenantChannelEntitlement).where(
            TenantChannelEntitlement.tenant_id == tenant_id,
            TenantChannelEntitlement.channel_id == channel_id,
        )
    ).scalar_one()


def get_or_create_ingestion_request(
    session: Session,
    *,
    tenant_id: str,
    idempotency_key: str,
    job_kind: str,
    source_kind: str,
    source_key: str,
    pipeline_version: str,
    request_payload: dict[str, Any],
    requested_by_user_id: str | None = None,
    channel_id: str | None = None,
    priority: int = 0,
    max_attempts: int = 5,
) -> tuple[IngestionRequest, IngestionJob, bool]:
    """
    Create one tenant request and reuse globally equivalent canonical work.

    The boolean is True only when this call created the tenant request. Reusing an
    idempotency key with different immutable inputs fails before any new work is queued.
    """
    tenant_id = _required(tenant_id, "tenant_id")
    idempotency_key = _required(idempotency_key, "idempotency_key")
    job_kind = _required(job_kind, "job_kind").lower()
    source_kind = _required(source_kind, "source_kind").lower()
    source_key = _required(source_key, "source_key")
    pipeline_version = _required(pipeline_version, "pipeline_version")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if session.get(Tenant, tenant_id) is None:
        raise ValueError(f"tenant {tenant_id} does not exist")
    if requested_by_user_id and session.get(UserAccount, requested_by_user_id) is None:
        raise ValueError(f"user {requested_by_user_id} does not exist")
    if channel_id and session.get(SourceChannel, channel_id) is None:
        raise ValueError(f"source channel {channel_id} does not exist")

    canonical = {
        "job_kind": job_kind,
        "source_kind": source_kind,
        "source_key": source_key,
        "pipeline_version": pipeline_version,
    }
    canonical_key = _json_fingerprint(canonical)
    fingerprint = _json_fingerprint({"canonical": canonical, "request": request_payload})

    existing_request = session.execute(
        select(IngestionRequest).where(
            IngestionRequest.tenant_id == tenant_id,
            IngestionRequest.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if existing_request is not None:
        if existing_request.request_fingerprint != fingerprint:
            raise IdempotencyConflict(
                "idempotency key already exists with different source or request payload"
            )
        return existing_request, session.get(IngestionJob, existing_request.job_id), False

    job = session.execute(
        select(IngestionJob).where(IngestionJob.dedupe_key == canonical_key)
    ).scalar_one_or_none()
    if job is None:
        candidate = IngestionJob(
            id=_stable_id("job", canonical_key),
            dedupe_key=canonical_key,
            channel_id=channel_id,
            job_kind=job_kind,
            source_kind=source_kind,
            source_key=source_key,
            pipeline_version=pipeline_version,
            priority=int(priority),
            max_attempts=int(max_attempts),
            payload_json=dict(request_payload),
        )
        if _insert_with_savepoint(session, candidate):
            job = candidate
        else:
            job = session.execute(
                select(IngestionJob).where(IngestionJob.dedupe_key == canonical_key)
            ).scalar_one()

    request = IngestionRequest(
        id=_stable_id("req", f"{tenant_id}:{idempotency_key}"),
        tenant_id=tenant_id,
        requested_by_user_id=requested_by_user_id,
        job_id=job.id,
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
        request_json=dict(request_payload),
        status="ready" if job.status == "succeeded" else "accepted",
    )
    if _insert_with_savepoint(session, request):
        return request, job, True

    concurrent = session.execute(
        select(IngestionRequest).where(
            IngestionRequest.tenant_id == tenant_id,
            IngestionRequest.idempotency_key == idempotency_key,
        )
    ).scalar_one()
    if concurrent.request_fingerprint != fingerprint:
        raise IdempotencyConflict("idempotency key was concurrently created with different inputs")
    return concurrent, session.get(IngestionJob, concurrent.job_id), False


def claim_ingestion_jobs(
    session: Session,
    *,
    worker_id: str,
    limit: int = 1,
    lease_seconds: int = 300,
    now: datetime | None = None,
    job_kinds: Iterable[str] | None = None,
) -> list[IngestionJob]:
    worker_id = _required(worker_id, "worker_id")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if lease_seconds < 10 or lease_seconds > 3600:
        raise ValueError("lease_seconds must be between 10 and 3600")
    now = now or utcnow()
    normalized_kinds = [
        str(value).strip().lower() for value in (job_kinds or []) if str(value).strip()
    ]

    _fail_exhausted_expired_jobs(session, now=now, job_kinds=normalized_kinds)

    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.attempt_count < IngestionJob.max_attempts,
            or_(
                and_(
                    IngestionJob.status.in_(("queued", "retry")),
                    or_(IngestionJob.next_run_at.is_(None), IngestionJob.next_run_at <= now),
                    or_(
                        IngestionJob.lease_expires_at.is_(None),
                        IngestionJob.lease_expires_at <= now,
                    ),
                ),
                and_(
                    IngestionJob.status == "running",
                    IngestionJob.lease_expires_at.is_not(None),
                    IngestionJob.lease_expires_at <= now,
                ),
            ),
        )
        .order_by(
            IngestionJob.priority.desc(), IngestionJob.created_at.asc(), IngestionJob.id.asc()
        )
        .limit(limit)
    )
    if normalized_kinds:
        statement = statement.where(IngestionJob.job_kind.in_(normalized_kinds))
    if session.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)

    jobs = list(session.execute(statement).scalars())
    lease_expires_at = now + timedelta(seconds=lease_seconds)
    for job in jobs:
        job.status = "running"
        job.attempt_count += 1
        job.lease_owner = worker_id
        job.lease_expires_at = lease_expires_at
        job.updated_at = now
    session.flush()
    return jobs


def complete_ingestion_job(
    session: Session,
    *,
    job_id: str,
    worker_id: str,
    result: dict[str, Any],
    now: datetime | None = None,
) -> IngestionJob:
    now = now or utcnow()
    job = _owned_running_job(session, job_id=job_id, worker_id=worker_id, now=now)
    job.status = "succeeded"
    job.result_json = dict(result)
    job.completed_at = now
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_error_code = None
    job.last_error_detail = None
    job.updated_at = now
    session.execute(
        update(IngestionRequest)
        .where(IngestionRequest.job_id == job.id)
        .values(status="ready", updated_at=now)
    )
    session.flush()
    return job


def fail_ingestion_job(
    session: Session,
    *,
    job_id: str,
    worker_id: str,
    error_code: str,
    error_detail: str,
    retryable: bool,
    retry_after_seconds: int = 60,
    now: datetime | None = None,
) -> IngestionJob:
    now = now or utcnow()
    job = _owned_running_job(session, job_id=job_id, worker_id=worker_id, now=now)
    can_retry = bool(retryable and job.attempt_count < job.max_attempts)
    job.status = "retry" if can_retry else "failed"
    job.next_run_at = now + timedelta(seconds=max(0, retry_after_seconds)) if can_retry else None
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_error_code = _required(error_code, "error_code")[:64]
    job.last_error_detail = str(error_detail or "")[:8000]
    job.completed_at = None if can_retry else now
    job.updated_at = now
    session.execute(
        update(IngestionRequest)
        .where(IngestionRequest.job_id == job.id)
        .values(status="accepted" if can_retry else "failed", updated_at=now)
    )
    session.flush()
    return job


def reserve_ingestion_effect(
    session: Session,
    *,
    job_id: str,
    provider: str,
    effect_kind: str,
    idempotency_key: str,
    request_payload: dict[str, Any],
) -> tuple[IngestionEffect, bool]:
    if session.get(IngestionJob, job_id) is None:
        raise ValueError(f"ingestion job {job_id} does not exist")
    provider = _required(provider, "provider").lower()
    effect_kind = _required(effect_kind, "effect_kind").lower()
    idempotency_key = _required(idempotency_key, "idempotency_key")
    fingerprint = _json_fingerprint(request_payload)

    existing = session.execute(
        select(IngestionEffect).where(
            IngestionEffect.provider == provider,
            IngestionEffect.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if existing is not None:
        if (
            existing.job_id != job_id
            or existing.effect_kind != effect_kind
            or existing.request_fingerprint != fingerprint
        ):
            raise IdempotencyConflict(
                "provider effect idempotency key already exists with different inputs"
            )
        return existing, False

    row = IngestionEffect(
        id=_stable_id("eff", f"{provider}:{idempotency_key}"),
        job_id=job_id,
        provider=provider,
        effect_kind=effect_kind,
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
        request_json=dict(request_payload),
    )
    if _insert_with_savepoint(session, row):
        return row, True
    concurrent = session.execute(
        select(IngestionEffect).where(
            IngestionEffect.provider == provider,
            IngestionEffect.idempotency_key == idempotency_key,
        )
    ).scalar_one()
    if (
        concurrent.job_id != job_id
        or concurrent.effect_kind != effect_kind
        or concurrent.request_fingerprint != fingerprint
    ):
        raise IdempotencyConflict(
            "provider effect was concurrently reserved with different inputs"
        )
    return concurrent, False


def _fail_exhausted_expired_jobs(
    session: Session,
    *,
    now: datetime,
    job_kinds: list[str],
) -> None:
    """Terminalize crashed final attempts so they cannot remain running forever."""
    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.status == "running",
            IngestionJob.attempt_count >= IngestionJob.max_attempts,
            IngestionJob.lease_expires_at.is_not(None),
            IngestionJob.lease_expires_at <= now,
        )
        .order_by(IngestionJob.lease_expires_at.asc(), IngestionJob.id.asc())
        .limit(100)
    )
    if job_kinds:
        statement = statement.where(IngestionJob.job_kind.in_(job_kinds))
    if session.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)

    exhausted = list(session.execute(statement).scalars())
    if not exhausted:
        return

    exhausted_ids: list[str] = []
    for job in exhausted:
        exhausted_ids.append(job.id)
        job.status = "failed"
        job.next_run_at = None
        job.lease_owner = None
        job.lease_expires_at = None
        job.last_error_code = "lease_expired_after_max_attempts"
        job.last_error_detail = "worker lease expired on the final allowed attempt"
        job.completed_at = now
        job.updated_at = now
    session.execute(
        update(IngestionRequest)
        .where(IngestionRequest.job_id.in_(exhausted_ids))
        .values(status="failed", updated_at=now)
    )
    session.flush()


def _owned_running_job(
    session: Session,
    *,
    job_id: str,
    worker_id: str,
    now: datetime,
) -> IngestionJob:
    job_id = _required(job_id, "job_id")
    worker_id = _required(worker_id, "worker_id")
    statement = select(IngestionJob).where(
        IngestionJob.id == job_id,
        IngestionJob.status == "running",
        IngestionJob.lease_owner == worker_id,
        IngestionJob.lease_expires_at.is_not(None),
        IngestionJob.lease_expires_at > now,
    )
    if session.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update()
    job = session.execute(statement).scalar_one_or_none()
    if job is None:
        raise IngestionLeaseLost(
            f"worker {worker_id} does not own a live running lease for ingestion job {job_id}"
        )
    return job
