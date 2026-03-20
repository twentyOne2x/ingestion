from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import desc, func, or_, select

from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels

from .channel_service_acp import ACP_OFFERINGS
from .channel_service_logic import (
    create_checkout_session_with_payment,
    create_order_from_quote,
    new_id,
    persist_quote,
    plan_quote,
)
from .channel_service_runtime import global_dispatch_limit
from .channel_service_store import (
    AcpJobBridge,
    ChannelOrder,
    ChannelQuote,
    CheckoutSessionRecord,
    EgressPool,
    OfferingReadinessOverride,
    OfferingReadinessSnapshot,
    SchedulerJob,
    SoakRun,
    SoakSample,
    SyntheticRun,
    SyntheticStep,
    session_scope,
    utcnow,
)

PUBLICATION_STATES = {"supported", "degraded", "paused", "internal_only"}
ACCEPTANCE_SCOPES = {"catalog_only", "catalog_and_arbitrary"}


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _string_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_utc(value):
    if value is None:
        return None
    if getattr(value, "tzinfo", None) is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _round_float(value: float, digits: int = 1) -> float:
    return float(round(float(value or 0.0), digits))


def _normalize_handle(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not raw.startswith("@"):
        raw = f"@{raw}"
    return raw.lower()


def _quote_handle(quote: ChannelQuote) -> str:
    request_json = dict(quote.request_json or {})
    return (
        _normalize_handle(request_json.get("channel_handle"))
        or _normalize_handle(quote.channel_handle)
        or _normalize_handle(quote.resolved_channel_name)
    )


def _supported_catalog_handles(namespace: str) -> set[str]:
    return {
        _normalize_handle(handle)
        for handle in load_namespace_channels(namespace)
        if _normalize_handle(handle)
    }


def channel_is_catalog_supported(*, namespace: str, channel_handle: Optional[str]) -> bool:
    normalized = _normalize_handle(channel_handle)
    if not normalized:
        return False
    return normalized in _supported_catalog_handles(namespace)


def quote_is_catalog_supported(quote: ChannelQuote) -> bool:
    return channel_is_catalog_supported(namespace=str(quote.namespace or "videos"), channel_handle=_quote_handle(quote))


def quote_is_immediately_fulfillable(quote: ChannelQuote) -> bool:
    current_batch_index = max(1, int(quote.current_batch_index or 1))
    expected_count = int(quote.current_batch_video_count or 0)
    if expected_count <= 0:
        return False
    current_rows = [row for row in quote.videos if int(row.batch_index or 0) == current_batch_index and bool(row.included)]
    if len(current_rows) < expected_count:
        return False
    for row in current_rows:
        if str(row.status or "") != "included":
            return False
        if not _string_or_none(row.transcript_source):
            return False
    return True


def _proof_root() -> Path:
    raw = os.getenv("CHANNEL_SERVICE_READINESS_PROOF_ROOT") or ".local-data/proof/channel-knowledge-service-readiness"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _run_proof_dir(run_id: str) -> Path:
    path = _proof_root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _catalog_canary_request() -> dict:
    namespace = (os.getenv("CHANNEL_SERVICE_CATALOG_CANARY_NAMESPACE") or "videos").strip() or "videos"
    configured_handle = _string_or_none(os.getenv("CHANNEL_SERVICE_CATALOG_CANARY_HANDLE"))
    default_request = {
        "namespace": namespace,
        "mode": "recent_pack",
        "max_videos": max(1, min(10, _env_int("CHANNEL_SERVICE_CATALOG_CANARY_MAX_VIDEOS", 4))),
        "language": (os.getenv("CHANNEL_SERVICE_CATALOG_CANARY_LANGUAGE") or "en").strip() or "en",
        "prefer_auto": _env_bool("CHANNEL_SERVICE_CATALOG_CANARY_PREFER_AUTO", True),
        "published_after": _string_or_none(os.getenv("CHANNEL_SERVICE_CATALOG_CANARY_PUBLISHED_AFTER")),
        "published_before": _string_or_none(os.getenv("CHANNEL_SERVICE_CATALOG_CANARY_PUBLISHED_BEFORE")),
    }
    if configured_handle:
        return {
            "channel_handle": configured_handle,
            **default_request,
        }

    inferred = _discover_catalog_canary_request(
        namespace=namespace,
        default_max_videos=int(default_request["max_videos"]),
        default_language=str(default_request["language"]),
        default_prefer_auto=bool(default_request["prefer_auto"]),
    )
    if inferred is not None:
        return inferred

    handles = [row for row in load_namespace_channels(namespace) if str(row or "").strip()]
    handle = handles[0] if handles else None
    return {
        "channel_handle": handle,
        **default_request,
    }


def _quote_request_fixture(
    *,
    quote: ChannelQuote,
    namespace: str,
    default_max_videos: int,
    default_language: str,
    default_prefer_auto: bool,
) -> Optional[dict]:
    request_json = dict(quote.request_json or {})
    channel_handle = _string_or_none(request_json.get("channel_handle")) or _string_or_none(quote.channel_handle)
    if not channel_handle:
        return None
    if not channel_is_catalog_supported(namespace=namespace, channel_handle=channel_handle):
        resolved_handle = _string_or_none(quote.channel_handle)
        if not channel_is_catalog_supported(namespace=namespace, channel_handle=resolved_handle):
            return None
    return {
        "channel_handle": channel_handle,
        "namespace": _string_or_none(request_json.get("namespace")) or namespace,
        "mode": _string_or_none(request_json.get("mode")) or _string_or_none(quote.mode) or "recent_pack",
        "max_videos": max(
            1,
            min(
                int(request_json.get("max_videos") or quote.current_batch_video_count or default_max_videos),
                default_max_videos,
            ),
        ),
        "language": _string_or_none(request_json.get("language")) or default_language,
        "prefer_auto": bool(request_json.get("prefer_auto", default_prefer_auto)),
        "published_after": _string_or_none(request_json.get("published_after")),
        "published_before": _string_or_none(request_json.get("published_before")),
    }


def _discover_catalog_canary_request(
    *,
    namespace: str,
    default_max_videos: int,
    default_language: str,
    default_prefer_auto: bool,
) -> Optional[dict]:
    with session_scope() as session:
        rows = session.execute(
            select(SyntheticRun)
            .where(
                SyntheticRun.run_kind == "catalog_starter",
                SyntheticRun.status == "passed",
            )
            .order_by(desc(SyntheticRun.started_at), desc(SyntheticRun.id))
            .limit(5)
        ).scalars().all()
        for row in rows:
            request_json = dict((row.result_json or {}).get("request") or {})
            channel_handle = _string_or_none(request_json.get("channel_handle")) or _string_or_none(row.channel_handle)
            if not channel_is_catalog_supported(namespace=namespace, channel_handle=channel_handle):
                continue
            return {
                "channel_handle": channel_handle,
                "namespace": _string_or_none(request_json.get("namespace")) or namespace,
                "mode": _string_or_none(request_json.get("mode")) or "recent_pack",
                "max_videos": max(
                    1,
                    min(int(request_json.get("max_videos") or default_max_videos), default_max_videos),
                ),
                "language": _string_or_none(request_json.get("language")) or default_language,
                "prefer_auto": bool(request_json.get("prefer_auto", default_prefer_auto)),
                "published_after": _string_or_none(request_json.get("published_after")),
                "published_before": _string_or_none(request_json.get("published_before")),
            }

        orders = session.execute(
            select(ChannelOrder)
            .where(ChannelOrder.status == "ready")
            .order_by(desc(ChannelOrder.created_at), desc(ChannelOrder.id))
            .limit(20)
        ).scalars().all()
        for order in orders:
            quote = session.get(ChannelQuote, order.quote_id)
            if quote is None:
                continue
            fixture = _quote_request_fixture(
                quote=quote,
                namespace=namespace,
                default_max_videos=default_max_videos,
                default_language=default_language,
                default_prefer_auto=default_prefer_auto,
            )
            if fixture is not None:
                return fixture

        quotes = session.execute(
            select(ChannelQuote)
            .where(
                ChannelQuote.status == "open",
                ChannelQuote.current_batch_video_count > 0,
            )
            .order_by(desc(ChannelQuote.created_at), desc(ChannelQuote.id))
            .limit(20)
        ).scalars().all()
        for quote in quotes:
            fixture = _quote_request_fixture(
                quote=quote,
                namespace=namespace,
                default_max_videos=default_max_videos,
                default_language=default_language,
                default_prefer_auto=default_prefer_auto,
            )
            if fixture is not None:
                return fixture
    return None


def _arbitrary_canary_request() -> Optional[dict]:
    handle = _string_or_none(os.getenv("CHANNEL_SERVICE_ARBITRARY_CANARY_HANDLE"))
    if not handle:
        return None
    return {
        "channel_handle": handle,
        "namespace": (os.getenv("CHANNEL_SERVICE_ARBITRARY_CANARY_NAMESPACE") or "videos").strip() or "videos",
        "mode": "recent_pack",
        "max_videos": max(1, min(25, _env_int("CHANNEL_SERVICE_ARBITRARY_CANARY_MAX_VIDEOS", 5))),
        "language": (os.getenv("CHANNEL_SERVICE_ARBITRARY_CANARY_LANGUAGE") or "en").strip() or "en",
        "prefer_auto": _env_bool("CHANNEL_SERVICE_ARBITRARY_CANARY_PREFER_AUTO", True),
        "published_after": _string_or_none(os.getenv("CHANNEL_SERVICE_ARBITRARY_CANARY_PUBLISHED_AFTER")),
        "published_before": _string_or_none(os.getenv("CHANNEL_SERVICE_ARBITRARY_CANARY_PUBLISHED_BEFORE")),
    }


def _start_synthetic_run(*, run_kind: str, request_json: dict) -> str:
    run = SyntheticRun(
        id=new_id("synthetic"),
        run_kind=run_kind,
        status="running",
        channel_handle=_string_or_none(request_json.get("channel_handle")),
        namespace=_string_or_none(request_json.get("namespace")),
        mode=_string_or_none(request_json.get("mode")),
        max_videos=int(request_json.get("max_videos") or 0) or None,
        published_after=_string_or_none(request_json.get("published_after")),
        published_before=_string_or_none(request_json.get("published_before")),
        result_json={"request": dict(request_json or {})},
    )
    run_id = str(run.id)
    with session_scope() as session:
        session.add(run)
    return run_id


def _record_synthetic_step(
    *,
    synthetic_run_id: str,
    step_name: str,
    status: str,
    payload_json: Optional[dict] = None,
    detail: Optional[str] = None,
) -> None:
    with session_scope() as session:
        session.add(
            SyntheticStep(
                synthetic_run_id=synthetic_run_id,
                step_name=step_name,
                status=status,
                payload_json=dict(payload_json or {}),
                detail=_string_or_none(detail),
            )
        )


def _safe_record_synthetic_step(
    *,
    synthetic_run_id: str,
    step_name: str,
    status: str,
    payload_json: Optional[dict] = None,
    detail: Optional[str] = None,
) -> None:
    try:
        _record_synthetic_step(
            synthetic_run_id=synthetic_run_id,
            step_name=step_name,
            status=status,
            payload_json=payload_json,
            detail=detail,
        )
    except Exception:
        return


def _finish_synthetic_run(*, run_id: str, status: str, result_json: dict) -> dict:
    finished_at = utcnow()
    with session_scope() as session:
        run = session.get(SyntheticRun, run_id)
        if run is None:
            raise ValueError(f"synthetic run {run_id} was not found")
        run.status = status
        run.finished_at = finished_at
        run.result_json = dict(result_json or {})
        payload = serialize_synthetic_run(run)
    proof_dir = _run_proof_dir(run_id)
    _write_json(proof_dir / "synthetic-run.json", payload)
    return payload


def _start_soak_run(*, requested_jobs: int, max_workers: int) -> str:
    run = SoakRun(
        id=new_id("soak"),
        status="running",
        requested_jobs=int(requested_jobs),
        success_count=0,
        failure_count=0,
        result_json={"max_workers": int(max_workers)},
    )
    run_id = str(run.id)
    with session_scope() as session:
        session.add(run)
    return run_id


def _record_soak_sample(*, soak_run_id: str, sample_index: int, status: str, result_json: dict) -> None:
    with session_scope() as session:
        session.add(
            SoakSample(
                soak_run_id=soak_run_id,
                sample_index=int(sample_index),
                status=status,
                result_json=dict(result_json or {}),
            )
        )


def _safe_record_soak_sample(*, soak_run_id: str, sample_index: int, status: str, result_json: dict) -> None:
    try:
        _record_soak_sample(
            soak_run_id=soak_run_id,
            sample_index=sample_index,
            status=status,
            result_json=result_json,
        )
    except Exception:
        return


def _finish_soak_run(*, soak_run_id: str, status: str, success_count: int, failure_count: int, result_json: dict) -> dict:
    finished_at = utcnow()
    with session_scope() as session:
        run = session.get(SoakRun, soak_run_id)
        if run is None:
            raise ValueError(f"soak run {soak_run_id} was not found")
        run.status = status
        run.success_count = int(success_count)
        run.failure_count = int(failure_count)
        run.finished_at = finished_at
        run.result_json = dict(result_json or {})
        payload = serialize_soak_run(session=session, run=run)
    proof_dir = _run_proof_dir(soak_run_id)
    _write_json(proof_dir / "soak-run.json", payload)
    return payload


def _percentile(values: Iterable[int], q: float) -> Optional[int]:
    ordered = sorted(int(value) for value in values if int(value) >= 0)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return int(round(ordered[low] + (ordered[high] - ordered[low]) * weight))


def _latest_active_override(session) -> Optional[OfferingReadinessOverride]:
    now = utcnow()
    rows = session.execute(
        select(OfferingReadinessOverride)
        .where(
            OfferingReadinessOverride.active.is_(True),
            OfferingReadinessOverride.starts_at <= now,
            or_(
                OfferingReadinessOverride.expires_at.is_(None),
                OfferingReadinessOverride.expires_at > now,
            ),
        )
        .order_by(OfferingReadinessOverride.created_at.desc(), OfferingReadinessOverride.id.desc())
        .limit(1)
    ).scalars().all()
    return rows[0] if rows else None


def _healthy_pool_group_count(session) -> int:
    rows = session.execute(
        select(EgressPool)
        .where(EgressPool.status == "healthy")
        .order_by(EgressPool.id.asc())
    ).scalars().all()
    groups = {
        _string_or_none(row.health_group) or row.id
        for row in rows
        if row.status == "healthy"
    }
    return len(groups)


def _dispatch_usage(session) -> dict:
    now = utcnow()
    active_total = session.execute(
        select(func.count(SchedulerJob.id)).where(
            SchedulerJob.status.in_(["dispatched", "running"]),
            SchedulerJob.lease_expires_at.is_not(None),
            SchedulerJob.lease_expires_at > now,
        )
    ).scalar_one()
    queued_total = session.execute(
        select(func.count(SchedulerJob.id)).where(SchedulerJob.status == "queued")
    ).scalar_one()
    delayed_total = session.execute(
        select(func.count(SchedulerJob.id)).where(SchedulerJob.status == "delayed")
    ).scalar_one()
    running_total = session.execute(
        select(func.count(SchedulerJob.id)).where(SchedulerJob.status == "running")
    ).scalar_one()
    dispatch_limit = max(1, int(global_dispatch_limit()))
    headroom_percent = max(0.0, ((dispatch_limit - int(active_total or 0)) / dispatch_limit) * 100.0)
    return {
        "dispatch_limit": dispatch_limit,
        "active_total": int(active_total or 0),
        "queued_total": int(queued_total or 0),
        "delayed_total": int(delayed_total or 0),
        "running_total": int(running_total or 0),
        "queue_headroom_percent": _round_float(headroom_percent, 1),
    }


def _recent_quote_stats(session) -> dict:
    now = utcnow()
    window = now - timedelta(minutes=max(5, _env_int("CHANNEL_SERVICE_READINESS_QUOTE_WINDOW_MINUTES", 30)))
    rows = session.execute(
        select(ChannelQuote.planning_latency_ms)
        .where(
            ChannelQuote.created_at >= window,
            ChannelQuote.planning_latency_ms > 0,
        )
        .order_by(ChannelQuote.created_at.desc())
    ).all()
    values = [int(value or 0) for (value,) in rows if int(value or 0) > 0]
    return {
        "quote_count": len(values),
        "quote_p95_ms": _percentile(values, 0.95),
        "quote_p50_ms": _percentile(values, 0.50),
    }


def _latest_synthetic(session, run_kind: str) -> Optional[SyntheticRun]:
    return session.execute(
        select(SyntheticRun)
        .where(SyntheticRun.run_kind == run_kind)
        .order_by(desc(SyntheticRun.started_at), desc(SyntheticRun.id))
        .limit(1)
    ).scalars().first()


def _recent_synthetic_runs(session, run_kind: str, limit: int) -> list[SyntheticRun]:
    return session.execute(
        select(SyntheticRun)
        .where(SyntheticRun.run_kind == run_kind)
        .order_by(desc(SyntheticRun.started_at), desc(SyntheticRun.id))
        .limit(max(1, limit))
    ).scalars().all()


def _latest_soak(session) -> Optional[SoakRun]:
    return session.execute(
        select(SoakRun)
        .order_by(desc(SoakRun.started_at), desc(SoakRun.id))
        .limit(1)
    ).scalars().first()


def _run_age_seconds(run) -> Optional[float]:
    if run is None or getattr(run, "finished_at", None) is None:
        return None
    finished_at = _normalize_utc(run.finished_at)
    return max(0.0, (utcnow() - finished_at).total_seconds())


def _synthetic_success_ratio(rows: list[SyntheticRun]) -> Optional[float]:
    if not rows:
        return None
    successes = sum(1 for row in rows if row.status == "passed")
    return successes / len(rows)


def _is_recent_success(run, *, max_age_s: int) -> bool:
    age = _run_age_seconds(run)
    return bool(run is not None and run.status == "passed" and age is not None and age <= max_age_s)


def _soak_success(run: Optional[SoakRun]) -> bool:
    return bool(run is not None and run.status == "passed")


def _public_offering_ids(*, publication_state: str) -> list[str]:
    if publication_state != "supported":
        return []
    offering_ids = []
    if _env_bool("CHANNEL_SERVICE_ENABLE_ACP_STARTER_OFFER", True):
        offering_ids.append("transcript_pack_starter_10")
    if _env_bool("CHANNEL_SERVICE_ENABLE_ACP_EXPANSION_OFFER", False):
        offering_ids.append("transcript_pack_expansion_25")
    return offering_ids


def _offering_statuses(*, publication_state: str, acceptance_scope: str, reason_codes: list[str]) -> list[dict]:
    visible = set(_public_offering_ids(publication_state=publication_state))
    statuses = []
    for offering in ACP_OFFERINGS:
        published = offering.offering_id in visible
        statuses.append(
            {
                "offering_id": offering.offering_id,
                "published": published,
                "purchasable": published,
                "acceptance_scope": acceptance_scope if published else "catalog_only",
                "reason_codes": [] if published else list(reason_codes),
            }
        )
    return statuses


def _direct_sales_allowed(publication_state: str) -> bool:
    return publication_state in {"supported", "degraded"}


def compute_readiness(session, *, persist: bool = True) -> dict:
    healthy_pool_group_count = _healthy_pool_group_count(session)
    dispatch_usage = _dispatch_usage(session)
    quote_stats = _recent_quote_stats(session)
    latest_catalog = _latest_synthetic(session, "catalog_starter")
    latest_arbitrary = _latest_synthetic(session, "arbitrary_probe")
    latest_soak = _latest_soak(session)
    recent_catalog = _recent_synthetic_runs(
        session,
        "catalog_starter",
        max(2, _env_int("CHANNEL_SERVICE_READINESS_STARTER_RECENT_LIMIT", 20)),
    )
    recent_catalog_for_ratio = list(recent_catalog)
    if latest_soak is not None and latest_soak.status == "passed" and latest_soak.started_at is not None:
        soak_started_at = _normalize_utc(latest_soak.started_at)
        windowed = [
            row
            for row in recent_catalog
            if row.started_at is not None and _normalize_utc(row.started_at) >= soak_started_at
        ]
        if windowed:
            recent_catalog_for_ratio = windowed
    recent_catalog_ratio = _synthetic_success_ratio(recent_catalog_for_ratio)
    recent_catalog_failures = 0
    for row in recent_catalog[:2]:
        if row.status != "passed":
            recent_catalog_failures += 1
        else:
            break

    hard_reasons: list[str] = []
    soft_reasons: list[str] = []
    catalog_success_window_s = max(60, _env_int("CHANNEL_SERVICE_READINESS_CATALOG_SUCCESS_WINDOW_S", 900))
    catalog_pause_window_s = max(catalog_success_window_s, _env_int("CHANNEL_SERVICE_READINESS_CATALOG_PAUSE_WINDOW_S", 1800))
    arbitrary_success_window_s = max(60, _env_int("CHANNEL_SERVICE_READINESS_ARBITRARY_SUCCESS_WINDOW_S", 1800))
    soak_success_window_s = max(300, _env_int("CHANNEL_SERVICE_READINESS_SOAK_SUCCESS_WINDOW_S", 86400))
    queue_soft_headroom_percent = max(1, _env_int("CHANNEL_SERVICE_READINESS_QUEUE_SOFT_HEADROOM_PERCENT", 20))
    quote_soft_ms = max(100, _env_int("CHANNEL_SERVICE_READINESS_QUOTE_P95_SOFT_MS", 15000))
    quote_hard_ms = max(quote_soft_ms, _env_int("CHANNEL_SERVICE_READINESS_QUOTE_P95_HARD_MS", 30000))
    starter_success_ratio_pause = max(1, _env_int("CHANNEL_SERVICE_READINESS_STARTER_SUCCESS_RATIO_PAUSE_PERCENT", 80)) / 100.0

    catalog_has_recent_success = _is_recent_success(latest_catalog, max_age_s=catalog_success_window_s)
    catalog_is_stale = False
    latest_catalog_age_s = _run_age_seconds(latest_catalog)
    if latest_catalog is None:
        hard_reasons.append("catalog_canary_missing")
    elif latest_catalog.status != "passed" and recent_catalog_failures >= 2:
        soft_reasons.append("catalog_canary_consecutive_failures")
    if latest_catalog_age_s is None or latest_catalog_age_s > catalog_pause_window_s or latest_catalog.status != "passed":
        catalog_is_stale = not catalog_has_recent_success
    if catalog_is_stale:
        hard_reasons.append("catalog_canary_stale")

    if healthy_pool_group_count <= 0:
        hard_reasons.append("no_healthy_pool_groups")
    elif healthy_pool_group_count == 1:
        soft_reasons.append("single_healthy_pool_group")

    if dispatch_usage["queue_headroom_percent"] <= 0.0:
        hard_reasons.append("queue_headroom_exhausted")
    elif dispatch_usage["queue_headroom_percent"] < float(queue_soft_headroom_percent):
        soft_reasons.append("queue_headroom_low")

    quote_p95_ms = quote_stats["quote_p95_ms"]
    if quote_p95_ms is not None and quote_p95_ms > quote_hard_ms:
        hard_reasons.append("quote_latency_above_hard_threshold")
    elif quote_p95_ms is not None and quote_p95_ms > quote_soft_ms:
        soft_reasons.append("quote_latency_above_soft_threshold")

    soak_recent_success = bool(
        latest_soak is not None
        and latest_soak.status == "passed"
        and _run_age_seconds(latest_soak) is not None
        and _run_age_seconds(latest_soak) <= soak_success_window_s
    )
    if latest_soak is None:
        hard_reasons.append("soak_proof_missing")
    elif not soak_recent_success:
        hard_reasons.append("soak_proof_stale")

    if (
        recent_catalog_ratio is not None
        and len(recent_catalog_for_ratio) >= 5
        and recent_catalog_ratio < starter_success_ratio_pause
    ):
        hard_reasons.append("starter_success_rate_below_threshold")

    baseline_proven = any(row.status == "passed" for row in recent_catalog) and _soak_success(latest_soak)

    # Catalog-backed transcript packs can still be saleable when acquisition pools are unhealthy,
    # as long as recent catalog canary + soak proof show the indexed serving path is working.
    if baseline_proven and "no_healthy_pool_groups" in hard_reasons:
        hard_reasons = [reason for reason in hard_reasons if reason != "no_healthy_pool_groups"]
        soft_reasons.append("no_healthy_pool_groups")

    publication_state = "supported"
    acceptance_scope = "catalog_only"
    if not baseline_proven:
        publication_state = "internal_only"
    elif hard_reasons:
        publication_state = "paused"
    elif soft_reasons:
        publication_state = "degraded"

    arbitrary_configured = _arbitrary_canary_request() is not None
    arbitrary_recent_success = _is_recent_success(latest_arbitrary, max_age_s=arbitrary_success_window_s)
    if arbitrary_configured:
        if publication_state == "supported" and arbitrary_recent_success and healthy_pool_group_count >= 2:
            acceptance_scope = "catalog_and_arbitrary"
        else:
            if latest_arbitrary is None:
                soft_reasons.append("arbitrary_canary_missing")
            elif latest_arbitrary.status != "passed":
                soft_reasons.append("arbitrary_canary_failed")
            else:
                soft_reasons.append("arbitrary_scope_unproven")
    else:
        soft_reasons.append("arbitrary_canary_not_configured")

    hard_reasons = sorted(set(hard_reasons))
    soft_reasons = sorted(set(reason for reason in soft_reasons if reason not in hard_reasons))

    capacity_score = 100
    capacity_score -= 35 * len(hard_reasons)
    capacity_score -= 10 * len(soft_reasons)
    if dispatch_usage["queue_headroom_percent"] < 100.0:
        capacity_score -= int(round((100.0 - dispatch_usage["queue_headroom_percent"]) / 10.0))
    if quote_p95_ms:
        capacity_score -= min(15, int(quote_p95_ms / 5000))
    capacity_score = max(0, min(100, capacity_score))

    override = _latest_active_override(session)
    override_payload = serialize_readiness_override(override) if override is not None else None
    if override is not None:
        if override.publication_state:
            publication_state = override.publication_state
        if override.acceptance_scope:
            acceptance_scope = override.acceptance_scope
        if "manual_override_active" not in soft_reasons:
            soft_reasons.append("manual_override_active")

    reason_codes = sorted(set(hard_reasons + soft_reasons))
    direct_sales_allowed = _direct_sales_allowed(publication_state)
    offering_statuses = _offering_statuses(
        publication_state=publication_state,
        acceptance_scope=acceptance_scope,
        reason_codes=reason_codes,
    )
    payload = {
        "ok": True,
        "publication_state": publication_state,
        "acceptance_scope": acceptance_scope,
        "capacity_score": int(capacity_score),
        "purchasable": direct_sales_allowed,
        "direct_sales_allowed": direct_sales_allowed,
        "acp_publication_allowed": publication_state == "supported",
        "reason_codes": reason_codes,
        "hard_stop_reasons": hard_reasons,
        "soft_warning_reasons": soft_reasons,
        "healthy_pool_group_count": healthy_pool_group_count,
        "queue_headroom_percent": dispatch_usage["queue_headroom_percent"],
        "latest_catalog_canary_status": latest_catalog.status if latest_catalog is not None else None,
        "latest_arbitrary_canary_status": latest_arbitrary.status if latest_arbitrary is not None else None,
        "latest_soak_status": latest_soak.status if latest_soak is not None else None,
        "metrics": {
            **dispatch_usage,
            **quote_stats,
            "latest_catalog_canary_age_s": _run_age_seconds(latest_catalog),
            "latest_arbitrary_canary_age_s": _run_age_seconds(latest_arbitrary),
            "latest_soak_age_s": _run_age_seconds(latest_soak),
            "recent_catalog_success_ratio": recent_catalog_ratio,
            "recent_catalog_success_sample_count": len(recent_catalog_for_ratio),
        },
        "sources": {
            "catalog_canary_run_id": latest_catalog.id if latest_catalog is not None else None,
            "arbitrary_canary_run_id": latest_arbitrary.id if latest_arbitrary is not None else None,
            "soak_run_id": latest_soak.id if latest_soak is not None else None,
            "override_id": override.id if override is not None else None,
        },
        "override": override_payload,
        "offerings": offering_statuses,
    }
    if persist:
        snapshot = OfferingReadinessSnapshot(
            publication_state=publication_state,
            acceptance_scope=acceptance_scope,
            capacity_score=int(capacity_score),
            purchasable=direct_sales_allowed,
            hard_stop_reasons_json=list(hard_reasons),
            soft_warning_reasons_json=list(soft_reasons),
            healthy_pool_group_count=healthy_pool_group_count,
            queue_headroom_percent=dispatch_usage["queue_headroom_percent"],
            latest_catalog_canary_status=latest_catalog.status if latest_catalog is not None else None,
            latest_arbitrary_canary_status=latest_arbitrary.status if latest_arbitrary is not None else None,
            latest_soak_status=latest_soak.status if latest_soak is not None else None,
            metrics_json=payload["metrics"],
            source_json=payload["sources"],
        )
        session.add(snapshot)
        session.flush()
        payload["snapshot_id"] = int(snapshot.id)
        payload["effective_at"] = snapshot.created_at.isoformat()
    else:
        payload["snapshot_id"] = None
        payload["effective_at"] = utcnow().isoformat()
    return payload


def serialize_readiness_snapshot(snapshot: OfferingReadinessSnapshot, override: Optional[OfferingReadinessOverride] = None) -> dict:
    reason_codes = list(snapshot.hard_stop_reasons_json or []) + list(snapshot.soft_warning_reasons_json or [])
    publication_state = str(snapshot.publication_state or "internal_only")
    acceptance_scope = str(snapshot.acceptance_scope or "catalog_only")
    return {
        "snapshot_id": int(snapshot.id),
        "publication_state": publication_state,
        "acceptance_scope": acceptance_scope,
        "capacity_score": int(snapshot.capacity_score or 0),
        "purchasable": bool(snapshot.purchasable),
        "direct_sales_allowed": bool(snapshot.purchasable),
        "acp_publication_allowed": publication_state == "supported",
        "reason_codes": sorted(set(reason_codes)),
        "hard_stop_reasons": list(snapshot.hard_stop_reasons_json or []),
        "soft_warning_reasons": list(snapshot.soft_warning_reasons_json or []),
        "healthy_pool_group_count": int(snapshot.healthy_pool_group_count or 0),
        "queue_headroom_percent": _round_float(snapshot.queue_headroom_percent or 0.0),
        "latest_catalog_canary_status": snapshot.latest_catalog_canary_status,
        "latest_arbitrary_canary_status": snapshot.latest_arbitrary_canary_status,
        "latest_soak_status": snapshot.latest_soak_status,
        "metrics": dict(snapshot.metrics_json or {}),
        "sources": dict(snapshot.source_json or {}),
        "override": serialize_readiness_override(override) if override is not None else None,
        "effective_at": snapshot.created_at.isoformat(),
        "offerings": _offering_statuses(
            publication_state=publication_state,
            acceptance_scope=acceptance_scope,
            reason_codes=sorted(set(reason_codes)),
        ),
    }


def serialize_readiness_history(session, *, limit: int = 20) -> dict:
    rows = session.execute(
        select(OfferingReadinessSnapshot)
        .order_by(desc(OfferingReadinessSnapshot.created_at), desc(OfferingReadinessSnapshot.id))
        .limit(max(1, min(200, int(limit))))
    ).scalars().all()
    override = _latest_active_override(session)
    return {
        "ok": True,
        "history": [serialize_readiness_snapshot(row, override=None) for row in rows],
        "active_override": serialize_readiness_override(override) if override is not None else None,
    }


def create_readiness_override(
    *,
    session,
    publication_state: Optional[str],
    acceptance_scope: Optional[str],
    reason: str,
    created_by: Optional[str],
    expires_in_minutes: Optional[int],
    clear_existing: bool,
) -> OfferingReadinessOverride:
    publication_state = _string_or_none(publication_state)
    acceptance_scope = _string_or_none(acceptance_scope)
    if publication_state and publication_state not in PUBLICATION_STATES:
        raise ValueError(f"unsupported publication_state {publication_state}")
    if acceptance_scope and acceptance_scope not in ACCEPTANCE_SCOPES:
        raise ValueError(f"unsupported acceptance_scope {acceptance_scope}")
    if not publication_state and not acceptance_scope:
        raise ValueError("publication_state or acceptance_scope is required")
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("reason is required")

    if clear_existing:
        rows = session.execute(
            select(OfferingReadinessOverride).where(OfferingReadinessOverride.active.is_(True))
        ).scalars().all()
        for row in rows:
            row.active = False

    expires_at = None
    if expires_in_minutes is not None:
        expires_at = utcnow() + timedelta(minutes=max(1, int(expires_in_minutes)))
    override = OfferingReadinessOverride(
        publication_state=publication_state,
        acceptance_scope=acceptance_scope,
        reason=reason,
        created_by=_string_or_none(created_by),
        active=True,
        expires_at=expires_at,
    )
    session.add(override)
    session.flush()
    return override


def serialize_readiness_override(override: Optional[OfferingReadinessOverride]) -> Optional[dict]:
    if override is None:
        return None
    return {
        "override_id": int(override.id),
        "publication_state": override.publication_state,
        "acceptance_scope": override.acceptance_scope,
        "reason": override.reason,
        "created_by": override.created_by,
        "active": bool(override.active),
        "starts_at": override.starts_at.isoformat() if override.starts_at else None,
        "expires_at": override.expires_at.isoformat() if override.expires_at else None,
        "created_at": override.created_at.isoformat() if override.created_at else None,
    }


def enforce_checkout_allowed(*, session, quotes: list[ChannelQuote]) -> dict:
    readiness = compute_readiness(session, persist=True)
    if not bool(readiness.get("direct_sales_allowed")):
        raise ValueError(
            f"Transcript Pack sales are currently {readiness['publication_state']}: "
            + ", ".join(readiness["reason_codes"] or ["readiness_gate_blocked"])
        )
    if readiness["acceptance_scope"] != "catalog_and_arbitrary":
        arbitrary_quotes = [quote for quote in quotes if not quote_is_catalog_supported(quote)]
        if arbitrary_quotes:
            details = ", ".join(f"{quote.id}:{_quote_handle(quote) or quote.channel_handle}" for quote in arbitrary_quotes)
            raise ValueError(
                "Arbitrary-channel transcript-pack sales are not enabled yet; "
                f"unsupported checkout quotes: {details}"
            )
        blocked_quotes = [quote for quote in quotes if not quote_is_immediately_fulfillable(quote)]
        if blocked_quotes:
            details = ", ".join(f"{quote.id}:{_quote_handle(quote) or quote.channel_handle}" for quote in blocked_quotes)
            raise ValueError(
                "Catalog-only transcript-pack sales currently require fully deliverable indexed quotes; "
                f"blocked checkout quotes: {details}"
            )
    return readiness


def enforce_acp_job_allowed(*, session, offering_id: str, channel_handle: Optional[str], namespace: str) -> dict:
    readiness = compute_readiness(session, persist=True)
    public_ids = {item["offering_id"] for item in readiness["offerings"] if item["published"]}
    if offering_id not in public_ids:
        raise ValueError(
            f"ACP offering {offering_id} is not currently published: "
            + ", ".join(readiness["reason_codes"] or ["readiness_gate_blocked"])
        )
    if readiness["acceptance_scope"] != "catalog_and_arbitrary" and not channel_is_catalog_supported(
        namespace=namespace,
        channel_handle=channel_handle,
    ):
        raise ValueError("ACP transcript-pack offering currently accepts supported catalog channels only")
    return readiness


def run_catalog_starter_canary(*, request_overrides: Optional[dict] = None) -> dict:
    request_json = {**_catalog_canary_request(), **dict(request_overrides or {})}
    step_events: list[dict] = []
    if not _string_or_none(request_json.get("channel_handle")):
        run_id = _start_synthetic_run(run_kind="catalog_starter", request_json=request_json)
        result = {
            "ok": False,
            "run_kind": "catalog_starter",
            "status": "failed",
            "error": "catalog canary handle is not configured and no supported channel was found",
        }
        return _finish_synthetic_run(run_id=run_id, status="failed", result_json=result)

    run_id = _start_synthetic_run(run_kind="catalog_starter", request_json=request_json)
    proof_dir = _run_proof_dir(run_id)
    _write_json(proof_dir / "request.json", request_json)
    start = time.perf_counter()
    try:
        with session_scope() as session:
            plan_started = time.perf_counter()
            plan = plan_quote(
                session=session,
                channel_handle=str(request_json["channel_handle"]),
                namespace=str(request_json["namespace"]),
                mode=str(request_json["mode"]),
                max_videos=int(request_json["max_videos"]),
                language=str(request_json["language"]),
                prefer_auto=bool(request_json.get("prefer_auto", True)),
                pack_id=None,
                published_after=_string_or_none(request_json.get("published_after")),
                published_before=_string_or_none(request_json.get("published_before")),
            )
            planning_latency_ms = int((time.perf_counter() - plan_started) * 1000)
            step_events.append(
                {
                    "step_name": "plan_quote",
                    "status": "passed",
                    "payload_json": {
                        "planning_latency_ms": planning_latency_ms,
                        "included_video_count": len(plan.included_rows),
                        "pending_video_count": len(plan.pending_rows),
                        "excluded_video_count": len(plan.excluded_rows),
                    },
                    "detail": None,
                }
            )
            quote = persist_quote(
                session=session,
                request_payload={**request_json, "source": "synthetic", "run_id": run_id},
                plan=plan,
                planning_latency_ms=planning_latency_ms,
            )
            step_events.append(
                {
                    "step_name": "persist_quote",
                    "status": "passed",
                    "payload_json": {"quote_id": quote.id, "status": quote.status},
                    "detail": None,
                }
            )
            if quote.status != "open" or int(quote.current_batch_video_count or 0) <= 0:
                raise ValueError(
                    f"catalog canary quote {quote.id} is not billable yet "
                    f"(status={quote.status} current_batch_video_count={quote.current_batch_video_count})"
                )
            checkout = create_checkout_session_with_payment(
                session=session,
                quote_ids=[quote.id],
                idempotency_key=f"synthetic:{run_id}",
                payment_provider="synthetic",
                payment_status="settled_synthetic",
            )
            step_events.append(
                {
                    "step_name": "create_checkout",
                    "status": "passed",
                    "payload_json": {"checkout_session_id": checkout.id, "status": checkout.status},
                    "detail": None,
                }
            )
            pack, batch, order = create_order_from_quote(
                session=session,
                quote=quote,
                checkout=checkout,
                pack_id=None,
                buyer_subject_type=None,
                buyer_subject_id=None,
                external_payment={
                    "provider": "synthetic",
                    "payment_status": "settled_synthetic",
                    "amount_cents": int(quote.current_batch_amount_cents or 0),
                    "receipt_status": "settled",
                    "receipt_json": {
                        "provider": "synthetic",
                        "run_id": run_id,
                        "request": request_json,
                    },
                },
            )
            success = (
                order.status == "ready"
                and int(batch.ready_video_count or 0) == int(batch.billable_video_count or 0)
                and bool(pack.export_paths_json)
            )
            final_status = "passed" if success else "failed"
            step_events.append(
                {
                    "step_name": "create_order",
                    "status": final_status,
                    "payload_json": {
                        "order_id": order.id,
                        "pack_id": pack.id,
                        "batch_id": batch.id,
                        "order_status": order.status,
                        "batch_status": batch.status,
                        "ready_video_count": int(batch.ready_video_count or 0),
                        "billable_video_count": int(batch.billable_video_count or 0),
                    },
                    "detail": None if success else "starter batch did not become fully ready",
                }
            )
            result = {
                "ok": success,
                "run_kind": "catalog_starter",
                "status": final_status,
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "request": request_json,
                "quote_id": quote.id,
                "checkout_session_id": checkout.id,
                "order_id": order.id,
                "pack_id": pack.id,
                "batch_id": batch.id,
                "order_status": order.status,
                "batch_status": batch.status,
                "ready_video_count": int(batch.ready_video_count or 0),
                "billable_video_count": int(batch.billable_video_count or 0),
                "archive_path": (pack.export_paths_json or {}).get("archive_path"),
                "channel_handle": quote.channel_handle,
                "planning_latency_ms": planning_latency_ms,
            }
        for event in step_events:
            _safe_record_synthetic_step(synthetic_run_id=run_id, **event)
        return _finish_synthetic_run(run_id=run_id, status=final_status, result_json=result)
    except Exception as exc:
        detail = str(exc)[:2000]
        step_events.append(
            {
                "step_name": "failure",
                "status": "failed",
                "payload_json": {},
                "detail": detail,
            }
        )
        for event in step_events:
            _safe_record_synthetic_step(synthetic_run_id=run_id, **event)
        return _finish_synthetic_run(
            run_id=run_id,
            status="failed",
            result_json={
                "ok": False,
                "run_kind": "catalog_starter",
                "status": "failed",
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "request": request_json,
                "error": detail,
            },
        )


def run_arbitrary_probe_canary(*, request_overrides: Optional[dict] = None) -> dict:
    configured = _arbitrary_canary_request()
    request_json = {**(configured or {}), **dict(request_overrides or {})}
    run_id = _start_synthetic_run(run_kind="arbitrary_probe", request_json=request_json)
    proof_dir = _run_proof_dir(run_id)
    _write_json(proof_dir / "request.json", request_json)
    step_events: list[dict] = []
    if not _string_or_none(request_json.get("channel_handle")):
        return _finish_synthetic_run(
            run_id=run_id,
            status="skipped",
            result_json={
                "ok": False,
                "run_kind": "arbitrary_probe",
                "status": "skipped",
                "request": request_json,
                "error": "CHANNEL_SERVICE_ARBITRARY_CANARY_HANDLE is not configured",
            },
        )

    start = time.perf_counter()
    try:
        with session_scope() as session:
            plan_started = time.perf_counter()
            plan = plan_quote(
                session=session,
                channel_handle=str(request_json["channel_handle"]),
                namespace=str(request_json["namespace"]),
                mode=str(request_json["mode"]),
                max_videos=int(request_json["max_videos"]),
                language=str(request_json["language"]),
                prefer_auto=bool(request_json.get("prefer_auto", True)),
                pack_id=None,
                published_after=_string_or_none(request_json.get("published_after")),
                published_before=_string_or_none(request_json.get("published_before")),
            )
            planning_latency_ms = int((time.perf_counter() - plan_started) * 1000)
            quote = persist_quote(
                session=session,
                request_payload={**request_json, "source": "synthetic", "run_id": run_id},
                plan=plan,
                planning_latency_ms=planning_latency_ms,
            )
            success = quote.status == "open" and int(quote.current_batch_video_count or 0) > 0
            step_events.append(
                {
                    "step_name": "probe_quote",
                    "status": "passed" if success else "failed",
                    "payload_json": {
                        "quote_id": quote.id,
                        "status": quote.status,
                        "included_video_count": int(quote.included_video_count or 0),
                        "pending_video_count": sum(1 for row in quote.videos if row.status == "pending_acquisition"),
                        "planning_latency_ms": planning_latency_ms,
                    },
                    "detail": None if success else "arbitrary probe did not produce a billable starter batch",
                }
            )
            final_status = "passed" if success else "failed"
            result = {
                "ok": success,
                "run_kind": "arbitrary_probe",
                "status": final_status,
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "request": request_json,
                "quote_id": quote.id,
                "quote_status": quote.status,
                "included_video_count": int(quote.included_video_count or 0),
                "current_batch_video_count": int(quote.current_batch_video_count or 0),
                "planning_latency_ms": planning_latency_ms,
            }
        for event in step_events:
            _safe_record_synthetic_step(synthetic_run_id=run_id, **event)
        return _finish_synthetic_run(run_id=run_id, status=final_status, result_json=result)
    except Exception as exc:
        detail = str(exc)[:2000]
        step_events.append(
            {
                "step_name": "failure",
                "status": "failed",
                "payload_json": {},
                "detail": detail,
            }
        )
        for event in step_events:
            _safe_record_synthetic_step(synthetic_run_id=run_id, **event)
        return _finish_synthetic_run(
            run_id=run_id,
            status="failed",
            result_json={
                "ok": False,
                "run_kind": "arbitrary_probe",
                "status": "failed",
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "request": request_json,
                "error": detail,
            },
        )


def run_catalog_soak(*, jobs: int, max_workers: Optional[int] = None) -> dict:
    requested_jobs = max(1, int(jobs))
    worker_count = max(1, min(requested_jobs, int(max_workers or _env_int("CHANNEL_SERVICE_SOAK_MAX_WORKERS", 4))))
    run_id = _start_soak_run(requested_jobs=requested_jobs, max_workers=worker_count)
    proof_dir = _run_proof_dir(run_id)
    start = time.perf_counter()
    results: list[dict] = []
    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_catalog_starter_canary): sample_index
            for sample_index in range(1, requested_jobs + 1)
        }
        for future in as_completed(futures):
            sample_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive; runner already handles errors
                result = {
                    "ok": False,
                    "status": "failed",
                    "error": str(exc)[:2000],
                }
            results.append({"sample_index": sample_index, **dict(result or {})})
            if result.get("status") == "passed":
                success_count += 1
                sample_status = "passed"
            else:
                failure_count += 1
                sample_status = "failed"
            _safe_record_soak_sample(
                soak_run_id=run_id,
                sample_index=sample_index,
                status=sample_status,
                result_json=result,
            )

    results.sort(key=lambda item: int(item.get("sample_index") or 0))
    success_ratio = success_count / requested_jobs if requested_jobs else 0.0
    required_ratio = max(1, _env_int("CHANNEL_SERVICE_SOAK_REQUIRED_SUCCESS_RATIO_PERCENT", 100)) / 100.0
    final_status = "passed" if success_ratio >= required_ratio and failure_count == 0 else "failed"
    payload = {
        "ok": final_status == "passed",
        "soak_run_id": run_id,
        "status": final_status,
        "requested_jobs": requested_jobs,
        "max_workers": worker_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_ratio": success_ratio,
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "samples": results,
    }
    _write_json(proof_dir / "samples.json", payload)
    return _finish_soak_run(
        soak_run_id=run_id,
        status=final_status,
        success_count=success_count,
        failure_count=failure_count,
        result_json=payload,
    )


def serialize_synthetic_run(run: SyntheticRun) -> dict:
    return {
        "run_id": run.id,
        "run_kind": run.run_kind,
        "status": run.status,
        "channel_handle": run.channel_handle,
        "namespace": run.namespace,
        "mode": run.mode,
        "max_videos": run.max_videos,
        "published_after": run.published_after,
        "published_before": run.published_before,
        "result": dict(run.result_json or {}),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def serialize_soak_run(*, session, run: SoakRun) -> dict:
    samples = session.execute(
        select(SoakSample)
        .where(SoakSample.soak_run_id == run.id)
        .order_by(SoakSample.sample_index.asc(), SoakSample.id.asc())
    ).scalars().all()
    return {
        "soak_run_id": run.id,
        "status": run.status,
        "requested_jobs": int(run.requested_jobs or 0),
        "success_count": int(run.success_count or 0),
        "failure_count": int(run.failure_count or 0),
        "result": dict(run.result_json or {}),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "samples": [
            {
                "sample_index": int(sample.sample_index or 0),
                "status": sample.status,
                "result": dict(sample.result_json or {}),
                "created_at": sample.created_at.isoformat() if sample.created_at else None,
            }
            for sample in samples
        ],
    }


def latest_readiness_snapshot(session) -> Optional[OfferingReadinessSnapshot]:
    return session.execute(
        select(OfferingReadinessSnapshot)
        .order_by(desc(OfferingReadinessSnapshot.created_at), desc(OfferingReadinessSnapshot.id))
        .limit(1)
    ).scalars().first()


def get_existing_checkout_by_idempotency_key(session, *, idempotency_key: str) -> Optional[CheckoutSessionRecord]:
    return session.execute(
        select(CheckoutSessionRecord)
        .where(CheckoutSessionRecord.idempotency_key == idempotency_key)
        .limit(1)
    ).scalars().first()


def get_existing_acp_bridge(session, *, acp_job_id: str) -> Optional[AcpJobBridge]:
    return session.get(AcpJobBridge, acp_job_id)
