from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChannelQuote(Base):
    __tablename__ = "channel_quotes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="open", nullable=False)
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    channel_handle: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_channel_id: Mapped[str | None] = mapped_column(String(255))
    resolved_channel_name: Mapped[str | None] = mapped_column(String(255))
    requested_max_videos: Mapped[int] = mapped_column(Integer, nullable=False)
    included_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    excluded_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_batch_index: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    current_batch_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_batch_amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_included_amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    per_video_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_ready_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    eta_confidence: Mapped[str] = mapped_column(String(32), default="low", nullable=False)
    recommended_starter_batch_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    planning_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    batch_plan_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    price_breakdown_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    videos: Mapped[list["QuoteVideo"]] = relationship(
        back_populates="quote", cascade="all, delete-orphan", order_by="QuoteVideo.position"
    )


class QuoteVideo(Base):
    __tablename__ = "quote_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("channel_quotes.id"), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    included: Mapped[bool] = mapped_column(Boolean, nullable=False)
    video_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    channel_name: Mapped[str | None] = mapped_column(String(255))
    channel_handle: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[str | None] = mapped_column(String(32))
    duration_s: Mapped[float | None] = mapped_column(Float)
    video_url: Mapped[str | None] = mapped_column(Text)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    transcript_source: Mapped[str | None] = mapped_column(String(64))
    indexed_parent_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255))
    detail: Mapped[str | None] = mapped_column(Text)

    quote: Mapped["ChannelQuote"] = relationship(back_populates="videos")


class CheckoutSessionRecord(Base):
    __tablename__ = "checkout_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="open", nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    total_amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quote_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    line_items_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    payment_provider: Mapped[str] = mapped_column(String(64), default="x402", nullable=False)
    payment_status: Mapped[str] = mapped_column(String(64), default="not_implemented", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ChannelPack(Base):
    __tablename__ = "channel_packs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    channel_handle: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_channel_id: Mapped[str | None] = mapped_column(String(255))
    resolved_channel_name: Mapped[str | None] = mapped_column(String(255))
    total_purchased_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ready_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    batch_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    manifest_json: Mapped[dict | None] = mapped_column(JSON)
    export_paths_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PackBatch(Base):
    __tablename__ = "pack_batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(ForeignKey("channel_packs.id"), nullable=False, index=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("channel_quotes.id"), nullable=False, index=True)
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    billable_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ready_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_ready_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    build_notes_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    manifest_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PackVideo(Base):
    __tablename__ = "pack_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pack_id: Mapped[str] = mapped_column(ForeignKey("channel_packs.id"), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("pack_batches.id"), nullable=False, index=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("channel_quotes.id"), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    video_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    channel_name: Mapped[str | None] = mapped_column(String(255))
    channel_handle: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[str | None] = mapped_column(String(32))
    duration_s: Mapped[float | None] = mapped_column(Float)
    video_url: Mapped[str | None] = mapped_column(Text)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    transcript_source: Mapped[str | None] = mapped_column(String(64))
    indexed_parent_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ChannelOrder(Base):
    __tablename__ = "channel_orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("channel_quotes.id"), nullable=False, index=True)
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    pack_id: Mapped[str] = mapped_column(ForeignKey("channel_packs.id"), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("pack_batches.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    payment_status: Mapped[str] = mapped_column(String(64), default="pending", nullable=False)
    payment_provider: Mapped[str] = mapped_column(String(64), default="x402", nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    notes_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PaymentReceipt(Base):
    __tablename__ = "payment_receipts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    order_id: Mapped[str | None] = mapped_column(ForeignKey("channel_orders.id"), index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    receipt_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AcpJobBridge(Base):
    __tablename__ = "acp_job_bridges"

    acp_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    offering_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), default="received", nullable=False)
    quote_id: Mapped[str | None] = mapped_column(ForeignKey("channel_quotes.id"), index=True)
    checkout_session_id: Mapped[str | None] = mapped_column(ForeignKey("checkout_sessions.id"), index=True)
    order_id: Mapped[str | None] = mapped_column(ForeignKey("channel_orders.id"), index=True)
    pack_id: Mapped[str | None] = mapped_column(ForeignKey("channel_packs.id"), index=True)
    fixed_price_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    payment_provider: Mapped[str] = mapped_column(String(64), default="acp", nullable=False)
    payment_status: Mapped[str] = mapped_column(String(64), default="settled_acp", nullable=False)
    buyer_subject_type: Mapped[str | None] = mapped_column(String(64))
    buyer_subject_id: Mapped[str | None] = mapped_column(String(255))
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    delivery_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class OfferingReadinessSnapshot(Base):
    __tablename__ = "offering_readiness_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publication_state: Mapped[str] = mapped_column(String(32), nullable=False, default="internal_only")
    acceptance_scope: Mapped[str] = mapped_column(String(32), nullable=False, default="catalog_only")
    capacity_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    purchasable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hard_stop_reasons_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    soft_warning_reasons_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    healthy_pool_group_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    queue_headroom_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latest_catalog_canary_status: Mapped[str | None] = mapped_column(String(32))
    latest_arbitrary_canary_status: Mapped[str | None] = mapped_column(String(32))
    latest_soak_status: Mapped[str | None] = mapped_column(String(32))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    source_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class OfferingReadinessOverride(Base):
    __tablename__ = "offering_readiness_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publication_state: Mapped[str | None] = mapped_column(String(32))
    acceptance_scope: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SyntheticRun(Base):
    __tablename__ = "synthetic_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running", index=True)
    channel_handle: Mapped[str | None] = mapped_column(String(255), index=True)
    namespace: Mapped[str | None] = mapped_column(String(128))
    mode: Mapped[str | None] = mapped_column(String(64))
    max_videos: Mapped[int | None] = mapped_column(Integer)
    published_after: Mapped[str | None] = mapped_column(String(32))
    published_before: Mapped[str | None] = mapped_column(String(32))
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SyntheticStep(Base):
    __tablename__ = "synthetic_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    synthetic_run_id: Mapped[str] = mapped_column(ForeignKey("synthetic_runs.id"), nullable=False, index=True)
    step_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SoakRun(Base):
    __tablename__ = "soak_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running", index=True)
    requested_jobs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SoakSample(Base):
    __tablename__ = "soak_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    soak_run_id: Mapped[str] = mapped_column(ForeignKey("soak_runs.id"), nullable=False, index=True)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Entitlement(Base):
    __tablename__ = "entitlements"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(ForeignKey("channel_packs.id"), nullable=False, index=True)
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class TranscriptProbe(Base):
    __tablename__ = "transcript_probes"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    video_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    video_url: Mapped[str] = mapped_column(Text, nullable=False)
    channel_handle: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    prefer_auto: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    transcript_source: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_attempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class EgressPool(Base):
    __tablename__ = "egress_pools"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="healthy")
    pool_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="direct")
    display_name: Mapped[str | None] = mapped_column(String(255))
    health_group: Mapped[str | None] = mapped_column(String(128), index=True)
    concurrency_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    last_error_kind: Mapped[str | None] = mapped_column(String(64))
    last_error_detail: Mapped[str | None] = mapped_column(Text)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_rate_limited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consecutive_rate_limit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quarantine_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_canary_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_canary_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_canary_status: Mapped[str | None] = mapped_column(String(32))
    last_canary_error_kind: Mapped[str | None] = mapped_column(String(64))
    last_canary_error_detail: Mapped[str | None] = mapped_column(Text)
    last_canary_video_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class SchedulerJob(Base):
    __tablename__ = "scheduler_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    probe_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    video_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    channel_handle: Mapped[str | None] = mapped_column(String(255), index=True)
    lane: Mapped[str] = mapped_column(String(64), nullable=False, default="quote_starter_probe")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    subscriber_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assigned_pool_id: Mapped[str | None] = mapped_column(String(64), index=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error_kind: Mapped[str | None] = mapped_column(String(64))
    last_error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class SchedulerAttempt(Base):
    __tablename__ = "scheduler_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    probe_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pool_id: Mapped[str | None] = mapped_column(String(64), index=True)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_kind: Mapped[str | None] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


_ENGINE = None
_SESSION_FACTORY = None


def _database_url() -> str:
    return (
        os.getenv("CHANNEL_SERVICE_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "sqlite:///./.local-data/channel-service.db"
    )


def _ensure_sqlite_parent(url: str) -> dict:
    if not url.startswith("sqlite:///"):
        return {}
    path = url.replace("sqlite:///", "", 1)
    if path != ":memory:":
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return {"check_same_thread": False}


def get_engine():
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is None:
        url = _database_url()
        connect_args = _ensure_sqlite_parent(url)
        _ENGINE = create_engine(url, future=True, connect_args=connect_args)
        _SESSION_FACTORY = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)
    return _ENGINE


@contextmanager
def _sqlite_schema_lock(url: str) -> Iterator[None]:
    if not url.startswith("sqlite:///"):
        yield
        return

    path = url.replace("sqlite:///", "", 1)
    if path == ":memory:":
        yield
        return

    db_path = Path(path).expanduser().resolve()
    lock_path = db_path.with_suffix(f"{db_path.suffix}.schema.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def init_db() -> None:
    engine = get_engine()
    url = str(engine.url)
    try:
        with _sqlite_schema_lock(url):
            Base.metadata.create_all(bind=engine)
            _apply_lightweight_migrations(engine)
    except OperationalError as exc:
        # SQLite can still race across multi-worker startup even with checkfirst.
        # If another worker created the table first, the schema is already usable.
        if "already exists" not in str(exc).lower():
            raise


@contextmanager
def session_scope() -> Iterator:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        get_engine()
    session = _SESSION_FACTORY()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _apply_lightweight_migrations(engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "channel_quotes" in tables:
        _ensure_columns(
            engine,
            table_name="channel_quotes",
            wanted={
                "planning_latency_ms": "INTEGER DEFAULT 0",
            },
        )
    if "egress_pools" in tables:
        _ensure_columns(
            engine,
            table_name="egress_pools",
            wanted={
                "health_group": "VARCHAR(128)",
                "last_canary_started_at": "DATETIME",
                "last_canary_finished_at": "DATETIME",
                "last_canary_status": "VARCHAR(32)",
                "last_canary_error_kind": "VARCHAR(64)",
                "last_canary_error_detail": "TEXT",
                "last_canary_video_id": "VARCHAR(64)",
            },
        )


def _ensure_columns(engine, *, table_name: str, wanted: dict[str, str]) -> None:
    existing = {str(column["name"]) for column in inspect(engine).get_columns(table_name)}
    statements = []
    for column_name, ddl in wanted.items():
        if column_name in existing:
            continue
        statements.append(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")
    if not statements:
        return
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
