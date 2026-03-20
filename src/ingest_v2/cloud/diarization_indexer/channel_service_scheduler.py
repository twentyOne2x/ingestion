from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from collections import Counter
from datetime import timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import func, or_, select

from src.ingest_v2.pipelines.index_youtube_captions import _classify_transcript_fetch_error, fetch_transcript_cues

from .channel_service_runtime import (
    all_pool_profiles,
    dispatch_payload,
    dispatch_ttl_s,
    dispatch_token,
    distinct_health_group_count,
    enqueue_dispatch,
    global_dispatch_limit,
    lane_priority,
    per_channel_dispatch_limit,
    pool_execution_env,
    pool_profile,
    redis_client,
    scheduler_enabled,
)
from .channel_service_store import (
    EgressPool,
    QuoteVideo,
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


def new_job_id() -> str:
    return f"sjob_{uuid.uuid4().hex[:24]}"


def _scheduler_id() -> str:
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


def _pool_group_key(pool: EgressPool) -> str:
    return str(pool.health_group or pool.id)


def _pending_quote_summary(session) -> Dict[str, dict]:
    rows = session.execute(
        select(
            QuoteVideo.video_id,
            func.count(QuoteVideo.id),
            func.min(QuoteVideo.position),
        ).where(QuoteVideo.status == "pending_acquisition")
        .group_by(QuoteVideo.video_id)
    ).all()
    out: Dict[str, dict] = {}
    for video_id, count, min_position in rows:
        if not video_id:
            continue
        out[str(video_id)] = {
            "subscriber_count": int(count or 0),
            "min_position": int(min_position or 999999),
        }
    return out


def classify_probe_lane(*, subscriber_count: int, min_position: int) -> str:
    if subscriber_count <= 0:
        return "cache_refresh"
    if min_position <= 10:
        return "quote_starter_probe"
    return "quote_deferred_probe"


def ensure_egress_pools(session) -> list[EgressPool]:
    wanted_ids = set()
    for profile in all_pool_profiles():
        wanted_ids.add(profile["id"])
        row = session.get(EgressPool, profile["id"])
        if row is None:
            row = EgressPool(
                id=profile["id"],
                status="healthy",
                pool_kind=str(profile.get("pool_kind") or "direct"),
                display_name=str(profile.get("display_name") or profile["id"]),
                health_group=str(profile.get("health_group") or profile["id"]),
                concurrency_limit=int(profile.get("concurrency_limit") or 2),
            )
            session.add(row)
        else:
            row.pool_kind = str(profile.get("pool_kind") or row.pool_kind)
            row.display_name = str(profile.get("display_name") or row.display_name or row.id)
            row.health_group = str(profile.get("health_group") or row.health_group or row.id)
            row.concurrency_limit = int(profile.get("concurrency_limit") or row.concurrency_limit or 2)
            if row.status == "disabled":
                row.status = "healthy"
        session.flush()

    rows = session.execute(select(EgressPool)).scalars().all()
    for row in rows:
        if row.id not in wanted_ids and row.status != "disabled":
            row.status = "disabled"
    session.flush()
    return session.execute(select(EgressPool).order_by(EgressPool.id.asc())).scalars().all()


def recover_stale_jobs(session) -> int:
    now = utcnow()
    rows = session.execute(
        select(SchedulerJob).where(
            SchedulerJob.status.in_(["dispatched", "running"]),
            SchedulerJob.lease_expires_at.is_not(None),
            SchedulerJob.lease_expires_at < now,
        )
    ).scalars().all()
    recovered = 0
    for job in rows:
        probe = session.execute(
            select(TranscriptProbe).where(TranscriptProbe.key == job.probe_key)
        ).scalar_one_or_none()
        if probe is None or probe.status in {"ready", "unavailable"}:
            job.status = "completed" if probe and probe.status == "ready" else "dead_letter"
        else:
            job.status = "delayed" if probe.status == "retry" else "queued"
            job.next_run_at = probe.next_attempt_at
        job.dispatched_at = None
        job.lease_owner = None
        job.lease_expires_at = None
        recovered += 1
    session.flush()
    return recovered


def sync_scheduler_jobs(session) -> dict:
    now = utcnow()
    pending_summary = _pending_quote_summary(session)
    probes = session.execute(select(TranscriptProbe)).scalars().all()
    existing = {
        row.probe_key: row
        for row in session.execute(select(SchedulerJob)).scalars().all()
    }
    synced = 0
    for probe in probes:
        summary = pending_summary.get(probe.video_id, {"subscriber_count": 0, "min_position": 999999})
        lane = classify_probe_lane(
            subscriber_count=int(summary["subscriber_count"]),
            min_position=int(summary["min_position"]),
        )
        job = existing.get(probe.key)
        if probe.status in {"ready", "unavailable"}:
            if job is not None and job.status not in {"completed", "dead_letter"}:
                job.status = "completed" if probe.status == "ready" else "dead_letter"
                job.next_run_at = None
                job.dispatched_at = None
                job.lease_owner = None
                job.lease_expires_at = None
                synced += 1
            continue

        if job is None:
            job = SchedulerJob(
                id=new_job_id(),
                probe_key=probe.key,
                video_id=probe.video_id,
                channel_handle=probe.channel_handle,
            )
            session.add(job)
            existing[probe.key] = job

        job.video_id = probe.video_id
        job.channel_handle = probe.channel_handle
        job.lane = lane
        job.priority = lane_priority(lane)
        job.subscriber_count = int(summary["subscriber_count"])
        job.last_error_detail = probe.error_detail
        next_attempt_at = _normalize_utc(probe.next_attempt_at)
        lease_expires_at = _normalize_utc(probe.lease_expires_at)
        if probe.status == "retry":
            job.status = "delayed" if next_attempt_at and next_attempt_at > now else "queued"
            job.next_run_at = probe.next_attempt_at
        elif probe.status == "running" and lease_expires_at and lease_expires_at > now:
            job.status = "running"
            job.lease_owner = probe.lease_owner
            job.lease_expires_at = probe.lease_expires_at
        else:
            if job.status not in {"running", "dispatched"}:
                job.status = "queued"
            job.next_run_at = probe.next_attempt_at
        synced += 1

    session.flush()
    return {"synced_jobs": synced, "active_probes": len(probes)}


def _dispatchable_pools(session) -> list[EgressPool]:
    rows = session.execute(select(EgressPool).order_by(EgressPool.id.asc())).scalars().all()
    dispatchable = []
    for row in rows:
        if row.status == "disabled":
            continue
        if row.status == "quarantined":
            continue
        if pool_profile(row.id) is None:
            continue
        dispatchable.append(row)
    return dispatchable


def _active_usage(session) -> tuple[int, Counter, Counter]:
    now = utcnow()
    rows = session.execute(
        select(SchedulerJob).where(
            SchedulerJob.status.in_(["dispatched", "running"]),
            SchedulerJob.lease_expires_at.is_not(None),
            SchedulerJob.lease_expires_at > now,
        )
    ).scalars().all()
    by_channel: Counter = Counter()
    by_pool: Counter = Counter()
    for row in rows:
        if row.channel_handle:
            by_channel[row.channel_handle] += 1
        if row.assigned_pool_id:
            by_pool[row.assigned_pool_id] += 1
    return len(rows), by_channel, by_pool


def choose_pool(*, pools: list[EgressPool], active_by_pool: Counter, job: SchedulerJob) -> Optional[EgressPool]:
    candidates = []
    pools_by_id = {pool.id: pool for pool in pools}
    for pool in pools:
        if active_by_pool[pool.id] >= max(1, int(pool.concurrency_limit or 1)):
            continue
        if pool_profile(pool.id) is None:
            continue
        candidates.append(pool)
    if not candidates:
        return None

    last_pool_id = (job.assigned_pool_id or "").strip()
    last_pool = pools_by_id.get(last_pool_id)
    last_group = _pool_group_key(last_pool) if last_pool is not None else None
    if job.last_error_kind == "rate_limited":
        if last_group:
            group_candidates = [pool for pool in candidates if _pool_group_key(pool) != last_group]
            if group_candidates:
                candidates = group_candidates
        if last_pool_id:
            pool_candidates = [pool for pool in candidates if pool.id != last_pool_id]
            if pool_candidates:
                candidates = pool_candidates

    healthy_candidates = [pool for pool in candidates if pool.status == "healthy"]
    if healthy_candidates:
        candidates = healthy_candidates
    elif job.last_error_kind == "rate_limited":
        return None

    def sort_key(pool: EgressPool) -> tuple:
        status_penalty = 0 if pool.status == "healthy" else 1
        canary_state = str(pool.last_canary_status or "").strip().lower()
        canary_penalty = 0 if canary_state == "passed" else 1 if not canary_state else 2
        success_penalty = 0 if _normalize_utc(pool.last_success_at) else 1
        rate_limit_penalty = int(pool.consecutive_rate_limit_count or 0)
        return (
            status_penalty,
            canary_penalty,
            success_penalty,
            active_by_pool[pool.id],
            rate_limit_penalty,
            pool.id,
        )

    candidates.sort(key=sort_key)
    return candidates[0]


def _canary_interval_s() -> int:
    return max(60, _env_int("CHANNEL_SERVICE_POOL_CANARY_INTERVAL_S", 900))


def _canary_min_delay_s() -> int:
    return max(30, _env_int("CHANNEL_SERVICE_POOL_CANARY_MIN_DELAY_S", 180))


def _canary_retry_s() -> int:
    return max(60, _env_int("CHANNEL_SERVICE_POOL_CANARY_RETRY_S", 300))


def _canary_max_per_cycle() -> int:
    return max(1, _env_int("CHANNEL_SERVICE_POOL_CANARY_MAX_PER_CYCLE", 1))


def _canary_due(pool: EgressPool, *, now) -> bool:
    if pool.status == "disabled":
        return False
    last_finished = _normalize_utc(pool.last_canary_finished_at)
    if pool.status in {"quarantined", "degraded"}:
        recent_failure = _normalize_utc(pool.last_rate_limited_at) or _normalize_utc(pool.last_failure_at)
        if recent_failure and (now - recent_failure).total_seconds() < _canary_min_delay_s():
            return False
        if last_finished and (now - last_finished).total_seconds() < _canary_interval_s():
            return False
        return True
    if last_finished is None:
        return True
    return (now - last_finished).total_seconds() >= _canary_interval_s()


def _resolve_canary_target(*, session, pool: EgressPool, profile: dict) -> Optional[dict]:
    explicit_url = str(profile.get("canary_video_url") or "").strip()
    explicit_language = str(profile.get("canary_language") or "en").strip() or "en"
    if explicit_url:
        return {
            "video_url": explicit_url,
            "video_id": None,
            "language": explicit_language,
        }

    probe = session.execute(
        select(TranscriptProbe)
        .where(
            TranscriptProbe.status == "ready",
            TranscriptProbe.video_url.is_not(None),
        )
        .order_by(TranscriptProbe.updated_at.desc())
        .limit(1)
    ).scalars().first()
    if probe is None:
        quote_video = session.execute(
            select(QuoteVideo)
            .where(
                QuoteVideo.included.is_(True),
                QuoteVideo.video_url.is_not(None),
            )
            .order_by(QuoteVideo.id.desc())
            .limit(1)
        ).scalars().first()
        if quote_video is None:
            return None
        return {
            "video_url": quote_video.video_url,
            "video_id": quote_video.video_id,
            "language": explicit_language,
        }
    return {
        "video_url": probe.video_url,
        "video_id": probe.video_id,
        "language": str(probe.language or explicit_language).strip() or explicit_language,
    }


def _pool_rows_for_group(session, *, pool: EgressPool) -> list[EgressPool]:
    group = _pool_group_key(pool)
    return session.execute(
        select(EgressPool)
        .where(
            or_(
                EgressPool.id == pool.id,
                EgressPool.health_group == group,
            )
        )
        .order_by(EgressPool.id.asc())
    ).scalars().all()


def run_pool_canary(pool_id: str) -> dict:
    started_at = utcnow()
    with session_scope() as session:
        pool = session.get(EgressPool, pool_id)
        if pool is None:
            return {"pool_id": pool_id, "status": "missing"}
        profile = pool_profile(pool_id)
        if profile is None:
            pool.status = "disabled"
            pool.last_canary_started_at = started_at
            pool.last_canary_finished_at = started_at
            pool.last_canary_status = "disabled"
            return {"pool_id": pool_id, "status": "disabled"}
        target = _resolve_canary_target(session=session, pool=pool, profile=profile)
        pool.last_canary_started_at = started_at
        pool.last_canary_status = "running"
        pool.last_canary_error_kind = None
        pool.last_canary_error_detail = None
        session.flush()

    if target is None:
        finished_at = utcnow()
        with session_scope() as session:
            pool = session.get(EgressPool, pool_id)
            if pool is not None:
                pool.last_canary_finished_at = finished_at
                pool.last_canary_status = "skipped"
                pool.last_canary_error_kind = "no_target"
                pool.last_canary_error_detail = "no ready transcript probe available for canary"
        return {"pool_id": pool_id, "status": "skipped", "reason": "no_target"}

    error_kind = None
    error_detail = None
    resolved_video_id = str(target.get("video_id") or "").strip() or None
    try:
        with pool_execution_env(profile):
            cues, _, _, maybe_video_id, error_detail = fetch_transcript_cues(
                video_url=str(target["video_url"]),
                video_id=resolved_video_id,
                language=str(target["language"]),
                prefer_auto=True,
                allow_transcript_api=False,
                use_proxy_pool=bool(profile.get("use_proxy_pool")),
                player_clients=list(profile.get("player_clients") or []),
            )
        if cues:
            finished_at = utcnow()
            with session_scope() as session:
                pool = session.get(EgressPool, pool_id)
                if pool is not None:
                    rows = _pool_rows_for_group(session, pool=pool)
                    for row in rows:
                        if row.status == "disabled":
                            continue
                        row.status = "healthy"
                        row.quarantine_until = None
                        row.consecutive_rate_limit_count = 0
                        row.last_success_at = finished_at
                        row.last_canary_started_at = started_at
                        row.last_canary_finished_at = finished_at
                        row.last_canary_status = "passed"
                        row.last_canary_error_kind = None
                        row.last_canary_error_detail = None
                        row.last_canary_video_id = str(maybe_video_id or resolved_video_id or "")
            return {"pool_id": pool_id, "status": "passed", "video_id": maybe_video_id or resolved_video_id}
        error_kind = _classify_transcript_fetch_error(error_detail or "transcript_unavailable")
        resolved_video_id = maybe_video_id or resolved_video_id
    except Exception as exc:
        error_detail = str(exc)[:2000]
        error_kind = _classify_transcript_fetch_error(error_detail)

    finished_at = utcnow()
    with session_scope() as session:
        pool = session.get(EgressPool, pool_id)
        if pool is None:
            return {"pool_id": pool_id, "status": "missing_after_canary"}
        rows = _pool_rows_for_group(session, pool=pool)
        if error_kind in {"transcript_unavailable", "video_unavailable"}:
            for row in rows:
                row.last_canary_started_at = started_at
                row.last_canary_finished_at = finished_at
                row.last_canary_status = "bad_target"
                row.last_canary_error_kind = error_kind
                row.last_canary_error_detail = error_detail
                row.last_canary_video_id = str(resolved_video_id or "")
            return {"pool_id": pool_id, "status": "bad_target", "error_kind": error_kind}

        target_status = "quarantined" if error_kind == "rate_limited" else "degraded"
        retry_at = finished_at + timedelta(seconds=_canary_retry_s())
        for row in rows:
            if row.status == "disabled":
                continue
            row.status = target_status
            row.quarantine_until = retry_at
            row.last_failure_at = finished_at
            row.last_canary_started_at = started_at
            row.last_canary_finished_at = finished_at
            row.last_canary_status = "failed"
            row.last_canary_error_kind = error_kind
            row.last_canary_error_detail = error_detail
            row.last_canary_video_id = str(resolved_video_id or "")
            if error_kind == "rate_limited":
                row.last_rate_limited_at = finished_at
        return {"pool_id": pool_id, "status": "failed", "error_kind": error_kind}


def run_due_pool_canaries() -> dict:
    checked = 0
    passed = 0
    failed = 0
    skipped = 0
    results = []
    now = utcnow()
    with session_scope() as session:
        pools = ensure_egress_pools(session)
        due_ids = [
            pool.id
            for pool in sorted(
                pools,
                key=lambda row: (
                    0 if row.status == "quarantined" else 1 if row.status == "degraded" else 2,
                    row.id,
                ),
            )
            if _canary_due(pool, now=now)
        ]

    for pool_id in due_ids[: _canary_max_per_cycle()]:
        result = run_pool_canary(pool_id)
        checked += 1
        status = str(result.get("status") or "")
        if status == "passed":
            passed += 1
        elif status in {"failed", "bad_target"}:
            failed += 1
        else:
            skipped += 1
        results.append(result)

    return {
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }


def dispatch_ready_jobs() -> dict:
    if not scheduler_enabled():
        return {"scheduler_enabled": False, "dispatched": 0}

    canary_summary = run_due_pool_canaries()
    client = redis_client()
    now = utcnow()
    dispatched = 0
    delayed_for_capacity = 0
    delayed_for_egress = 0
    recovered = 0
    with session_scope() as session:
        ensure_egress_pools(session)
        sync_scheduler_jobs(session)
        recovered = recover_stale_jobs(session)
        active_total, active_by_channel, active_by_pool = _active_usage(session)
        global_limit = global_dispatch_limit()
        channel_limit = per_channel_dispatch_limit()
        dispatchable_pools = _dispatchable_pools(session)
        budget = max(0, global_limit - active_total)
        if budget <= 0:
            return {
                "scheduler_enabled": True,
                "dispatched": 0,
                "recovered": recovered,
                "budget": 0,
                "active_total": active_total,
                "canaries": canary_summary,
            }

        rows = session.execute(
            select(SchedulerJob)
            .where(
                SchedulerJob.status.in_(["queued", "delayed"]),
                or_(SchedulerJob.next_run_at.is_(None), SchedulerJob.next_run_at <= now),
                or_(
                    SchedulerJob.dispatched_at.is_(None),
                    SchedulerJob.lease_expires_at.is_(None),
                    SchedulerJob.lease_expires_at < now,
                ),
            )
            .order_by(
                SchedulerJob.priority.asc(),
                SchedulerJob.subscriber_count.desc(),
                SchedulerJob.created_at.asc(),
            )
            .limit(max(32, global_limit * 4))
        ).scalars().all()

        for job in rows:
            if dispatched >= budget:
                break
            if job.channel_handle and active_by_channel[job.channel_handle] >= channel_limit:
                delayed_for_capacity += 1
                continue
            pool = choose_pool(pools=dispatchable_pools, active_by_pool=active_by_pool, job=job)
            if pool is None:
                delayed_for_egress += 1
                job.status = "delayed"
                job.next_run_at = now + timedelta(seconds=_env_int("CHANNEL_SERVICE_SCHEDULER_EGRESS_RETRY_S", 300))
                job.last_error_kind = "blocked_by_egress"
                job.last_error_detail = "no dispatchable egress pool available"
                continue

            token = dispatch_token(job.id, pool.id)
            payload = dispatch_payload(
                job_id=job.id,
                probe_key=job.probe_key,
                pool_id=pool.id,
                lane=job.lane,
                token=token,
            )
            if not enqueue_dispatch(client=client, lane=job.lane, payload=payload, token=token):
                continue
            job.status = "dispatched"
            job.assigned_pool_id = pool.id
            job.dispatched_at = now
            job.lease_expires_at = now + timedelta(seconds=dispatch_ttl_s())
            active_by_pool[pool.id] += 1
            if job.channel_handle:
                active_by_channel[job.channel_handle] += 1
            dispatched += 1

    return {
        "scheduler_enabled": True,
        "dispatched": dispatched,
        "recovered": recovered,
        "delayed_for_capacity": delayed_for_capacity,
        "delayed_for_egress": delayed_for_egress,
        "canaries": canary_summary,
    }


def serialize_scheduler_summary(*, session) -> dict:
    counts = Counter(
        row.status
        for row in session.execute(select(SchedulerJob)).scalars().all()
    )
    return {
        "ok": True,
        "jobs": dict(counts),
        "global_dispatch_limit": global_dispatch_limit(),
        "per_channel_dispatch_limit": per_channel_dispatch_limit(),
        "dispatchable_pool_count": len(_dispatchable_pools(session)),
        "distinct_health_group_count": distinct_health_group_count(),
        "lanes": [
            {"lane": str(lane), "count": int(count or 0)}
            for lane, count in session.execute(
                select(SchedulerJob.lane, func.count(SchedulerJob.id))
                .group_by(SchedulerJob.lane)
                .order_by(SchedulerJob.lane.asc())
            ).all()
        ],
    }


def serialize_egress_pools(*, session) -> dict:
    rows = session.execute(select(EgressPool).order_by(EgressPool.id.asc())).scalars().all()
    return {
        "ok": True,
        "configured_pool_count": len(all_pool_profiles()),
        "distinct_health_group_count": distinct_health_group_count(),
        "pools": [
            {
                "pool_id": row.id,
                "status": row.status,
                "pool_kind": row.pool_kind,
                "display_name": row.display_name,
                "health_group": row.health_group,
                "concurrency_limit": row.concurrency_limit,
                "last_error_kind": row.last_error_kind,
                "last_error_detail": row.last_error_detail,
                "last_success_at": row.last_success_at.isoformat() if row.last_success_at else None,
                "last_failure_at": row.last_failure_at.isoformat() if row.last_failure_at else None,
                "last_rate_limited_at": row.last_rate_limited_at.isoformat() if row.last_rate_limited_at else None,
                "consecutive_rate_limit_count": row.consecutive_rate_limit_count,
                "quarantine_until": _normalize_utc(row.quarantine_until).isoformat()
                if _normalize_utc(row.quarantine_until)
                else None,
                "quarantined": row.status == "quarantined",
                "dispatchable": row.status not in {"disabled", "quarantined"},
                "last_canary_started_at": _normalize_utc(row.last_canary_started_at).isoformat()
                if _normalize_utc(row.last_canary_started_at)
                else None,
                "last_canary_finished_at": _normalize_utc(row.last_canary_finished_at).isoformat()
                if _normalize_utc(row.last_canary_finished_at)
                else None,
                "last_canary_status": row.last_canary_status,
                "last_canary_error_kind": row.last_canary_error_kind,
                "last_canary_error_detail": row.last_canary_error_detail,
                "last_canary_video_id": row.last_canary_video_id,
            }
            for row in rows
        ],
    }


def run_forever() -> None:
    init_db()
    scheduler_id = _scheduler_id()
    poll_s = max(1, _env_int("CHANNEL_SERVICE_SCHEDULER_POLL_S", 3))
    LOG.info("[channel-service-scheduler] starting scheduler_id=%s poll_s=%s", scheduler_id, poll_s)
    while True:
        try:
            result = dispatch_ready_jobs()
            if result.get("dispatched") or result.get("recovered") or result.get("canaries", {}).get("checked"):
                LOG.info("[channel-service-scheduler] cycle=%s", result)
        except Exception:
            LOG.exception("[channel-service-scheduler] dispatch cycle failed")
        time.sleep(poll_s)


if __name__ == "__main__":
    run_forever()
