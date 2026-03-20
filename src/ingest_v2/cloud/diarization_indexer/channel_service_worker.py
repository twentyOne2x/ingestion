from __future__ import annotations

import logging
import math
import os
import socket
import time
from datetime import timedelta, timezone
from typing import Optional

from sqlalchemy import and_, or_, select

from src.ingest_v2.pipelines.index_youtube_captions import _classify_transcript_fetch_error, fetch_transcript_cues

from .channel_service_logic import _transcript_rows_from_cues, _write_probe_artifact
from .channel_service_runtime import (
    distinct_health_group_count,
    pool_execution_env,
    pool_profile,
    pop_dispatch,
    redis_client,
    scheduler_enabled,
)
from .channel_service_store import (
    EgressPool,
    SchedulerAttempt,
    SchedulerJob,
    TranscriptProbe,
    init_db,
    session_scope,
    utcnow,
)


LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _player_clients() -> list[str]:
    raw = (os.getenv("CHANNEL_SERVICE_ACQUIRE_PLAYER_CLIENTS") or "").strip()
    if not raw:
        return ["android", "ios"]
    out = [part.strip() for part in raw.split(",") if part.strip()]
    return out or ["android", "ios"]


def _worker_id() -> str:
    host = (os.getenv("HOSTNAME") or "").strip()
    if not host:
        try:
            host = socket.gethostname().strip()
        except Exception:
            host = "host"
    return f"{host}-p{os.getpid()}"


def _normalize_utc(value):
    if value is None:
        return None
    if getattr(value, "tzinfo", None) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _pool_group_rows(session, *, pool: EgressPool) -> list[EgressPool]:
    group = str(pool.health_group or pool.id)
    return session.execute(
        select(EgressPool).where(
            or_(
                EgressPool.id == pool.id,
                EgressPool.health_group == group,
            )
        )
    ).scalars().all()


def _retry_after_s(
    *,
    attempt_count: int,
    retry_s: int,
    error_kind: str,
    rate_limit_cooldown_s: int,
    rate_limit_max_retry_s: int,
    error_max_retry_s: int,
) -> int:
    if error_kind == "rate_limited":
        exponent = max(0, attempt_count - 1)
        return min(rate_limit_max_retry_s, max(retry_s, rate_limit_cooldown_s * (2**exponent)))
    return min(error_max_retry_s, max(retry_s, retry_s * max(1, attempt_count)))


def _sleep_until(target_monotonic: float, *, poll_s: int) -> None:
    while True:
        remaining = target_monotonic - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(max(1.0, float(poll_s)), remaining, 30.0))


def claim_next_probe(*, worker_id: str) -> Optional[dict]:
    now = utcnow()
    with session_scope() as session:
        probe = (
            session.execute(
                select(TranscriptProbe)
                .where(
                    or_(
                        TranscriptProbe.status.in_(["queued", "retry"]),
                        and_(
                            TranscriptProbe.status == "running",
                            TranscriptProbe.lease_expires_at.is_not(None),
                            TranscriptProbe.lease_expires_at < now,
                        ),
                    ),
                    or_(
                        TranscriptProbe.next_attempt_at.is_(None),
                        TranscriptProbe.next_attempt_at <= now,
                    ),
                )
                .order_by(TranscriptProbe.created_at.asc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if probe is None:
            return None

        probe.status = "running"
        probe.attempt_count = int(probe.attempt_count or 0) + 1
        probe.last_attempted_at = now
        probe.next_attempt_at = None
        probe.lease_owner = worker_id
        probe.lease_expires_at = now + timedelta(minutes=10)
        session.flush()
        return {
            "key": probe.key,
            "video_id": probe.video_id,
            "video_url": probe.video_url,
            "language": probe.language,
            "prefer_auto": bool(probe.prefer_auto),
            "attempt_count": int(probe.attempt_count or 0),
            "scheduler_job_id": None,
            "pool_id": None,
            "pool_profile": None,
        }


def claim_scheduled_job(*, worker_id: str, timeout_s: int) -> Optional[dict]:
    try:
        client = redis_client()
        payload = pop_dispatch(client=client, timeout_s=timeout_s)
    except Exception as exc:
        LOG.warning("[channel-service-acquirer] scheduler pop failed err=%s", exc)
        return None
    if not payload:
        return None

    job_id = str(payload.get("job_id") or "").strip()
    probe_key = str(payload.get("probe_key") or "").strip()
    pool_id = str(payload.get("pool_id") or "").strip()
    if not job_id or not probe_key or not pool_id:
        return None

    now = utcnow()
    with session_scope() as session:
        job = session.get(SchedulerJob, job_id)
        probe = session.get(TranscriptProbe, probe_key)
        pool = session.get(EgressPool, pool_id)
        if job is None or probe is None:
            if job is not None:
                job.status = "dead_letter"
                job.lease_owner = None
                job.lease_expires_at = None
                job.dispatched_at = None
            return None
        if job.status not in {"dispatched", "queued"}:
            return None
        quarantine_until = _normalize_utc(pool.quarantine_until) if pool is not None else None
        if pool is None or pool.status == "disabled" or (quarantine_until and quarantine_until > now):
            job.status = "delayed"
            job.next_run_at = now + timedelta(seconds=max(30, timeout_s))
            job.last_error_kind = "blocked_by_egress"
            job.last_error_detail = "assigned pool unavailable at claim time"
            job.dispatched_at = None
            job.lease_owner = None
            job.lease_expires_at = None
            return None

        lease_until = now + timedelta(minutes=10)
        job.status = "running"
        job.lease_owner = worker_id
        job.lease_expires_at = lease_until
        job.attempt_count = int(job.attempt_count or 0) + 1
        probe.status = "running"
        probe.attempt_count = int(probe.attempt_count or 0) + 1
        probe.last_attempted_at = now
        probe.next_attempt_at = None
        probe.lease_owner = worker_id
        probe.lease_expires_at = lease_until
        session.flush()
        return {
            "key": probe.key,
            "video_id": probe.video_id,
            "video_url": probe.video_url,
            "language": probe.language,
            "prefer_auto": bool(probe.prefer_auto),
            "attempt_count": int(probe.attempt_count or 0),
            "scheduler_job_id": job.id,
            "pool_id": pool_id,
            "pool_profile": pool_profile(pool_id),
        }


def finalize_probe(
    *,
    key: str,
    status: str,
    transcript_source: Optional[str],
    artifact_path: Optional[str],
    error_detail: Optional[str],
    retry_after_s: Optional[int],
    scheduler_job_id: Optional[str],
    pool_id: Optional[str],
    worker_id: str,
    error_kind: Optional[str],
) -> None:
    now = utcnow()
    with session_scope() as session:
        probe = session.get(TranscriptProbe, key)
        if probe is not None:
            probe.status = status
            probe.transcript_source = transcript_source
            probe.artifact_path = artifact_path
            probe.error_detail = error_detail
            probe.lease_owner = None
            probe.lease_expires_at = None
            probe.next_attempt_at = now + timedelta(seconds=retry_after_s) if retry_after_s else None

        if scheduler_job_id:
            job = session.get(SchedulerJob, scheduler_job_id)
            if job is not None:
                if status == "ready":
                    job.status = "completed"
                    job.next_run_at = None
                elif status == "unavailable":
                    job.status = "dead_letter"
                    job.next_run_at = None
                else:
                    job.status = "delayed"
                    job.next_run_at = now + timedelta(seconds=retry_after_s) if retry_after_s else now
                job.last_error_kind = error_kind
                job.last_error_detail = error_detail
                job.lease_owner = None
                job.lease_expires_at = None
                job.dispatched_at = None

            session.add(
                SchedulerAttempt(
                    job_id=scheduler_job_id,
                    probe_key=key,
                    pool_id=pool_id,
                    worker_id=worker_id,
                    status=status,
                    error_kind=error_kind,
                    error_detail=error_detail,
                    finished_at=now,
                )
            )

        if pool_id:
            pool = session.get(EgressPool, pool_id)
            if pool is not None:
                quarantine_threshold = max(1, _env_int("CHANNEL_SERVICE_POOL_RATE_LIMIT_QUARANTINE_THRESHOLD", 3))
                quarantine_s = max(60, _env_int("CHANNEL_SERVICE_POOL_RATE_LIMIT_QUARANTINE_S", 1800))
                rate_limit_rows = _pool_group_rows(session, pool=pool) if error_kind == "rate_limited" else [pool]
                if status == "ready":
                    for row in _pool_group_rows(session, pool=pool):
                        if row.status == "disabled":
                            continue
                        row.status = "healthy"
                        row.quarantine_until = None
                        row.consecutive_rate_limit_count = 0
                        row.last_success_at = now
                        row.last_error_kind = None
                        row.last_error_detail = None
                else:
                    if error_kind == "rate_limited":
                        next_count = int(pool.consecutive_rate_limit_count or 0) + 1
                        next_status = "quarantined" if next_count >= quarantine_threshold else "degraded"
                        next_quarantine_until = now + timedelta(seconds=quarantine_s)
                        for row in rate_limit_rows:
                            if row.status == "disabled":
                                continue
                            row.last_error_kind = error_kind
                            row.last_error_detail = error_detail
                            row.last_failure_at = now
                            row.last_rate_limited_at = now
                            row.consecutive_rate_limit_count = max(int(row.consecutive_rate_limit_count or 0), next_count)
                            row.status = next_status
                            row.quarantine_until = next_quarantine_until if next_status == "quarantined" else row.quarantine_until
                    else:
                        pool.last_error_kind = error_kind
                        pool.last_error_detail = error_detail
                        pool.last_failure_at = now
                        pool.consecutive_rate_limit_count = 0
                        if pool.status != "disabled":
                            pool.status = "degraded"
        session.flush()


def run_forever() -> None:
    init_db()
    worker_id = _worker_id()
    poll_s = max(1, _env_int("CHANNEL_SERVICE_ACQUIRE_POLL_S", 3))
    retry_s = max(30, _env_int("CHANNEL_SERVICE_ACQUIRE_RETRY_S", 300))
    max_attempts = max(1, _env_int("CHANNEL_SERVICE_ACQUIRE_MAX_ATTEMPTS", 3))
    rate_limit_cooldown_s = max(retry_s, _env_int("CHANNEL_SERVICE_ACQUIRE_RATE_LIMIT_COOLDOWN_S", 900))
    rate_limit_max_retry_s = max(rate_limit_cooldown_s, _env_int("CHANNEL_SERVICE_ACQUIRE_RATE_LIMIT_MAX_RETRY_S", 14400))
    error_max_retry_s = max(retry_s, _env_int("CHANNEL_SERVICE_ACQUIRE_MAX_RETRY_S", 1800))
    reroute_retry_s = max(15, _env_int("CHANNEL_SERVICE_POOL_REROUTE_RETRY_S", 45))
    default_use_proxy_pool = _env_bool("CHANNEL_SERVICE_ACQUIRE_USE_PROXY_POOL", False)
    default_allow_transcript_api = _env_bool("CHANNEL_SERVICE_ACQUIRE_ALLOW_TRANSCRIPT_API", False)
    default_player_clients = _player_clients()
    use_scheduler = scheduler_enabled()
    multi_health_group_reroute = distinct_health_group_count() > 1
    LOG.info(
        "[channel-service-acquirer] starting worker_id=%s poll_s=%s max_attempts=%s scheduler=%s clients=%s",
        worker_id,
        poll_s,
        max_attempts,
        use_scheduler,
        ",".join(default_player_clients),
    )
    cooldown_until = 0.0

    while True:
        job = None
        fallback_mode = False
        if use_scheduler:
            job = claim_scheduled_job(worker_id=worker_id, timeout_s=poll_s)
            if job is None and _env_bool("CHANNEL_SERVICE_ACQUIRE_DB_FALLBACK", False):
                fallback_mode = True
        else:
            fallback_mode = True

        if fallback_mode:
            if cooldown_until > time.monotonic():
                _sleep_until(cooldown_until, poll_s=poll_s)
            job = claim_next_probe(worker_id=worker_id)
            if job is None:
                time.sleep(poll_s)
                continue
        elif job is None:
            continue

        pool = job.get("pool_id")
        pool_cfg = job.get("pool_profile") or {}
        scheduled_job = bool(job.get("scheduler_job_id"))
        can_fast_reroute = use_scheduler and scheduled_job and not fallback_mode and multi_health_group_reroute
        allow_transcript_api = bool(pool_cfg.get("allow_transcript_api", default_allow_transcript_api))
        use_proxy_pool = bool(pool_cfg.get("use_proxy_pool", default_use_proxy_pool))
        player_clients = list(pool_cfg.get("player_clients") or default_player_clients)

        LOG.info(
            "[channel-service-acquirer] claimed key=%s video=%s attempt=%s pool=%s",
            job["key"],
            job["video_id"],
            job["attempt_count"],
            pool or "fallback",
        )
        try:
            with pool_execution_env(pool_cfg):
                cues, transcript_source, _, resolved_vid, error_detail = fetch_transcript_cues(
                    video_url=job["video_url"],
                    video_id=job["video_id"],
                    language=job["language"],
                    prefer_auto=job["prefer_auto"],
                    allow_transcript_api=allow_transcript_api,
                    use_proxy_pool=use_proxy_pool,
                    player_clients=player_clients,
                )
            if cues:
                transcript_rows = _transcript_rows_from_cues(
                    row={"video_id": resolved_vid or job["video_id"]},
                    cues=cues,
                    source=transcript_source or "transcript",
                )
                artifact_path = _write_probe_artifact(
                    video_id=job["video_id"],
                    language=job["language"],
                    prefer_auto=job["prefer_auto"],
                    transcript_source=transcript_source or "transcript",
                    transcript_rows=transcript_rows,
                )
                finalize_probe(
                    key=job["key"],
                    status="ready",
                    transcript_source=transcript_source,
                    artifact_path=artifact_path,
                    error_detail=None,
                    retry_after_s=None,
                    scheduler_job_id=job.get("scheduler_job_id"),
                    pool_id=pool,
                    worker_id=worker_id,
                    error_kind=None,
                )
                LOG.info(
                    "[channel-service-acquirer] ready key=%s video=%s rows=%s source=%s pool=%s",
                    job["key"],
                    job["video_id"],
                    len(transcript_rows),
                    transcript_source,
                    pool or "fallback",
                )
                continue

            error_kind = _classify_transcript_fetch_error(error_detail)
            if error_kind in {"transcript_unavailable", "video_unavailable"}:
                finalize_probe(
                    key=job["key"],
                    status="unavailable",
                    transcript_source=None,
                    artifact_path=None,
                    error_detail=error_detail or error_kind,
                    retry_after_s=None,
                    scheduler_job_id=job.get("scheduler_job_id"),
                    pool_id=pool,
                    worker_id=worker_id,
                    error_kind=error_kind,
                )
                LOG.warning(
                    "[channel-service-acquirer] unavailable key=%s video=%s reason=%s pool=%s",
                    job["key"],
                    job["video_id"],
                    error_kind,
                    pool or "fallback",
                )
                continue

            retry_after = _retry_after_s(
                attempt_count=int(job["attempt_count"]),
                retry_s=retry_s,
                error_kind=error_kind,
                rate_limit_cooldown_s=rate_limit_cooldown_s,
                rate_limit_max_retry_s=rate_limit_max_retry_s,
                error_max_retry_s=error_max_retry_s,
            )
            if error_kind == "rate_limited" and can_fast_reroute:
                retry_after = reroute_retry_s
            if fallback_mode and error_kind == "rate_limited":
                cooldown_until = time.monotonic() + retry_after
                LOG.warning(
                    "[channel-service-acquirer] cooldown reason=%s sleep_s=%s",
                    error_kind,
                    int(math.ceil(retry_after)),
                )
            if error_kind != "rate_limited" and int(job["attempt_count"]) >= max_attempts:
                finalize_probe(
                    key=job["key"],
                    status="unavailable",
                    transcript_source=None,
                    artifact_path=None,
                    error_detail=error_detail or error_kind,
                    retry_after_s=None,
                    scheduler_job_id=job.get("scheduler_job_id"),
                    pool_id=pool,
                    worker_id=worker_id,
                    error_kind=error_kind,
                )
                LOG.warning(
                    "[channel-service-acquirer] unavailable key=%s video=%s attempts=%s reason=%s pool=%s",
                    job["key"],
                    job["video_id"],
                    job["attempt_count"],
                    error_kind,
                    pool or "fallback",
                )
                continue

            finalize_probe(
                key=job["key"],
                status="retry",
                transcript_source=None,
                artifact_path=None,
                error_detail=error_detail or error_kind,
                retry_after_s=retry_after,
                scheduler_job_id=job.get("scheduler_job_id"),
                pool_id=pool,
                worker_id=worker_id,
                error_kind=error_kind,
            )
            LOG.warning(
                "[channel-service-acquirer] retry key=%s video=%s next_in_s=%s reason=%s pool=%s",
                job["key"],
                job["video_id"],
                retry_after,
                error_kind,
                pool or "fallback",
            )
        except Exception as exc:
            error_detail = str(exc)[:2000]
            error_kind = _classify_transcript_fetch_error(error_detail)
            retry_after = None
            if error_kind in {"transcript_unavailable", "video_unavailable"}:
                status = "unavailable"
            else:
                if error_kind == "rate_limited":
                    retry_after = _retry_after_s(
                        attempt_count=int(job["attempt_count"]),
                        retry_s=retry_s,
                        error_kind=error_kind,
                        rate_limit_cooldown_s=rate_limit_cooldown_s,
                        rate_limit_max_retry_s=rate_limit_max_retry_s,
                        error_max_retry_s=error_max_retry_s,
                    )
                    if can_fast_reroute:
                        retry_after = reroute_retry_s
                    if fallback_mode:
                        cooldown_until = time.monotonic() + retry_after
                        LOG.warning(
                            "[channel-service-acquirer] cooldown reason=%s sleep_s=%s",
                            error_kind,
                            int(math.ceil(retry_after)),
                        )
                    status = "retry"
                else:
                    status = "unavailable" if int(job["attempt_count"]) >= max_attempts else "retry"
                    if status == "retry":
                        retry_after = _retry_after_s(
                            attempt_count=int(job["attempt_count"]),
                            retry_s=retry_s,
                            error_kind=error_kind,
                            rate_limit_cooldown_s=rate_limit_cooldown_s,
                            rate_limit_max_retry_s=rate_limit_max_retry_s,
                            error_max_retry_s=error_max_retry_s,
                        )
            finalize_probe(
                key=job["key"],
                status=status,
                transcript_source=None,
                artifact_path=None,
                error_detail=error_detail,
                retry_after_s=retry_after,
                scheduler_job_id=job.get("scheduler_job_id"),
                pool_id=pool,
                worker_id=worker_id,
                error_kind=error_kind,
            )
            LOG.exception(
                "[channel-service-acquirer] failed key=%s video=%s retry_after=%s reason=%s pool=%s",
                job["key"],
                job["video_id"],
                retry_after,
                error_kind,
                pool or "fallback",
            )


if __name__ == "__main__":
    run_forever()
