from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Index,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from .channel_service_config import (
    channel_service_database_url,
    validate_production_runtime,
)


ALEMBIC_HEAD_REVISION = "20260825_0004"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "user_accounts"
    __table_args__ = (
        UniqueConstraint(
            "auth_provider", "auth_subject", name="uq_user_accounts_provider_subject"
        ),
    )

    id: Mapped[str] = mapped_column(String(68), primary_key=True)
    auth_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    auth_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(68), primary_key=True)
    slug: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class TenantMembership(Base):
    __tablename__ = "tenant_memberships"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "user_id", name="uq_tenant_memberships_tenant_user"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("tenants.id"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("user_accounts.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="member")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class SourceChannel(Base):
    __tablename__ = "source_channels"
    __table_args__ = (
        UniqueConstraint(
            "platform", "external_id", name="uq_source_channels_platform_external"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    platform: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    handle: Mapped[str | None] = mapped_column(String(255), index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    canonical_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class SourceVideo(Base):
    """Canonical provider video identity shared by every entitled tenant."""

    __tablename__ = "source_videos"
    __table_args__ = (
        UniqueConstraint(
            "platform", "external_id", name="uq_source_videos_provider_item"
        ),
        CheckConstraint(
            "archive_state IN ('pending_discovery', 'retained_remote_verified', "
            "'retained_hot_verified', 'partial_only')",
            name="ck_source_videos_archive_state",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel_id: Mapped[str] = mapped_column(
        ForeignKey("source_channels.id"), nullable=False, index=True
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    archive_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending_discovery", index=True
    )
    clip_candidate: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    clip_ready: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class TranscriptRevision(Base):
    """Immutable transcript revision with one explicitly current revision per video."""

    __tablename__ = "transcript_revisions"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_revision_id",
            name="uq_transcript_revisions_provider_item",
        ),
        Index(
            "uq_transcript_revisions_current_video",
            "video_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current = 1"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("source_videos.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider_revision_id: Mapped[str] = mapped_column(String(255), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint(
            "revision_id", "ordinal", name="uq_transcript_segments_ordinal"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("transcript_revisions.id"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    speaker_label: Mapped[str | None] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class MediaObject(Base):
    """Content-addressed media fact; the digest is the canonical object identity."""

    __tablename__ = "media_objects"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class MediaLocation(Base):
    """Verified location of a content-addressed object on hot or archive storage."""

    __tablename__ = "media_locations"
    __table_args__ = (
        UniqueConstraint(
            "backend", "location_key", name="uq_media_locations_backend_key"
        ),
        CheckConstraint(
            "backend IN ('hot_local', 'storagebox')",
            name="ck_media_locations_backend",
        ),
        CheckConstraint(
            "status IN ('active', 'pending', 'missing', 'corrupt')",
            name="ck_media_locations_status",
        ),
        Index(
            "uq_media_locations_active_hot_local",
            "media_sha256",
            unique=True,
            postgresql_where=text("status = 'active' AND backend = 'hot_local'"),
            sqlite_where=text("status = 'active' AND backend = 'hot_local'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    media_sha256: Mapped[str] = mapped_column(
        ForeignKey("media_objects.sha256"), nullable=False, index=True
    )
    backend: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    location_key: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class VideoMediaRef(Base):
    __tablename__ = "video_media_refs"
    __table_args__ = (
        UniqueConstraint(
            "video_id", "media_sha256", "role", name="uq_video_media_refs_identity"
        ),
        CheckConstraint(
            "role IN ('source_video', 'proxy', 'audio', 'thumbnail')",
            name="ck_video_media_refs_role",
        ),
        CheckConstraint(
            "status IN ('active', 'inactive')",
            name="ck_video_media_refs_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("source_videos.id"), nullable=False, index=True
    )
    media_sha256: Mapped[str] = mapped_column(
        ForeignKey("media_objects.sha256"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class TenantChannelEntitlement(Base):
    __tablename__ = "tenant_channel_entitlements"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "channel_id", name="uq_tenant_channel_entitlements_scope"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("tenants.id"), nullable=False, index=True
    )
    channel_id: Mapped[str] = mapped_column(
        ForeignKey("source_channels.id"), nullable=False, index=True
    )
    granted_by_user_id: Mapped[str | None] = mapped_column(
        String(68), ForeignKey("user_accounts.id"), index=True
    )
    access_level: Mapped[str] = mapped_column(
        String(32), nullable=False, default="query"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ArchiveCatalogImport(Base):
    """Authoritative apply ledger for one immutable archive catalog packet."""

    __tablename__ = "archive_catalog_imports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    jsonl_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    sidecar_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    receipt_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    receipt_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_keys_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    video_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    media_sha256s_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="applied", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ArchiveTenantClaim(Base):
    """Admin-only immutable receipt for granting imported channels to a tenant."""

    __tablename__ = "archive_tenant_claims"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_archive_tenant_claims_key"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("tenants.id"), nullable=False, index=True
    )
    admin_user_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("user_accounts.id"), nullable=False, index=True
    )
    catalog_import_id: Mapped[str] = mapped_column(
        ForeignKey("archive_catalog_imports.id"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    receipt_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    receipt_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="applied", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ArchiveHydrationRegistration(Base):
    """Authoritative registration of a verified archive object in the hot-media CAS."""

    __tablename__ = "archive_hydration_registrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_receipt_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    media_sha256: Mapped[str] = mapped_column(
        String(64), ForeignKey("media_objects.sha256"), nullable=False, index=True
    )
    location_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("media_locations.id"), nullable=False, index=True
    )
    receipt_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    receipt_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="applied", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class IngestionJob(Base):
    """Globally deduplicated work; tenant ownership lives in IngestionRequest."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dedupe_key: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    channel_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_channels.id"), index=True
    )
    job_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_key: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued", index=True
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    request_tenant_ids_json: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    last_error_code: Mapped[str | None] = mapped_column(String(64))
    last_error_detail: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class IngestionRequest(Base):
    __tablename__ = "ingestion_requests"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_ingestion_requests_tenant_idempotency",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("tenants.id"), nullable=False, index=True
    )
    requested_by_user_id: Mapped[str | None] = mapped_column(
        String(68), ForeignKey("user_accounts.id"), index=True
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="accepted", index=True
    )
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class TenantExport(Base):
    """Durable, idempotent metadata for one immutable tenant export artifact."""

    __tablename__ = "tenant_exports"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_tenant_exports_tenant_idempotency"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("tenants.id"), nullable=False, index=True
    )
    requested_by_user_id: Mapped[str] = mapped_column(
        String(68), ForeignKey("user_accounts.id"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running", index=True
    )
    snapshot_sha256: Mapped[str | None] = mapped_column(String(64))
    database_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    database_path: Mapped[str | None] = mapped_column(Text)
    manifest_path: Mapped[str | None] = mapped_column(Text)
    counts_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class IngestionEffect(Base):
    """Durable reservation for an external provider effect such as transcription submission."""

    __tablename__ = "ingestion_effects"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "idempotency_key",
            name="uq_ingestion_effects_provider_idempotency",
        ),
        UniqueConstraint(
            "provider",
            "provider_effect_id",
            name="uq_ingestion_effects_provider_effect",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    effect_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="reserved", index=True
    )
    provider_effect_id: Mapped[str | None] = mapped_column(String(255))
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class TranscriptionRun(Base):
    """Crash-reconcilable transcript attempt and temporary-audio deletion ledger."""

    __tablename__ = "transcription_runs"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "attempt_number", name="uq_transcription_runs_attempt"
        ),
        CheckConstraint(
            "mode IN ('openai', 'local_cpu')", name="ck_transcription_runs_mode"
        ),
        CheckConstraint(
            "status IN ('prepared', 'running', 'succeeded', 'failed', 'unknown')",
            name="ck_transcription_runs_status",
        ),
        CheckConstraint(
            "cleanup_status IN ('pending', 'deleted', 'not_created', 'cleanup_failed')",
            name="ck_transcription_runs_cleanup_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.id"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="prepared", index=True
    )
    temp_audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    cleanup_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_created", index=True
    )
    audio_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    transcript_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    provider_request_id: Mapped[str | None] = mapped_column(String(255), index=True)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cleaned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


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
    included_video_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    excluded_video_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    current_batch_index: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    current_batch_video_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    current_batch_amount_cents: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_included_amount_cents: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    per_video_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_ready_minutes: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    eta_confidence: Mapped[str] = mapped_column(
        String(32), default="low", nullable=False
    )
    recommended_starter_batch_size: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    planning_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    batch_plan_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    price_breakdown_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    videos: Mapped[list["QuoteVideo"]] = relationship(
        back_populates="quote",
        cascade="all, delete-orphan",
        order_by="QuoteVideo.position",
    )


class QuoteVideo(Base):
    __tablename__ = "quote_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quote_id: Mapped[str] = mapped_column(
        ForeignKey("channel_quotes.id"), nullable=False, index=True
    )
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
    idempotency_key: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    total_amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    quote_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    line_items_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    payment_provider: Mapped[str] = mapped_column(
        String(64), default="x402", nullable=False
    )
    payment_status: Mapped[str] = mapped_column(
        String(64), default="not_implemented", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
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
    total_purchased_video_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    ready_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    batch_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    manifest_json: Mapped[dict | None] = mapped_column(JSON)
    export_paths_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PackBatch(Base):
    __tablename__ = "pack_batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("channel_packs.id"), nullable=False, index=True
    )
    quote_id: Mapped[str] = mapped_column(
        ForeignKey("channel_quotes.id"), nullable=False, index=True
    )
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    billable_video_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    ready_video_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_ready_minutes: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    build_notes_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    manifest_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PackVideo(Base):
    __tablename__ = "pack_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("channel_packs.id"), nullable=False, index=True
    )
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("pack_batches.id"), nullable=False, index=True
    )
    quote_id: Mapped[str] = mapped_column(
        ForeignKey("channel_quotes.id"), nullable=False, index=True
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ChannelOrder(Base):
    __tablename__ = "channel_orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    quote_id: Mapped[str] = mapped_column(
        ForeignKey("channel_quotes.id"), nullable=False, index=True
    )
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("channel_packs.id"), nullable=False, index=True
    )
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("pack_batches.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    payment_status: Mapped[str] = mapped_column(
        String(64), default="pending", nullable=False
    )
    payment_provider: Mapped[str] = mapped_column(
        String(64), default="x402", nullable=False
    )
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    notes_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PaymentReceipt(Base):
    __tablename__ = "payment_receipts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    checkout_session_id: Mapped[str] = mapped_column(
        ForeignKey("checkout_sessions.id"), nullable=False, index=True
    )
    order_id: Mapped[str | None] = mapped_column(
        ForeignKey("channel_orders.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    receipt_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class AcpJobBridge(Base):
    __tablename__ = "acp_job_bridges"

    acp_job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    offering_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), default="received", nullable=False)
    quote_id: Mapped[str | None] = mapped_column(
        ForeignKey("channel_quotes.id"), index=True
    )
    checkout_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("checkout_sessions.id"), index=True
    )
    order_id: Mapped[str | None] = mapped_column(
        ForeignKey("channel_orders.id"), index=True
    )
    pack_id: Mapped[str | None] = mapped_column(
        ForeignKey("channel_packs.id"), index=True
    )
    fixed_price_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="USD", nullable=False)
    payment_provider: Mapped[str] = mapped_column(
        String(64), default="acp", nullable=False
    )
    payment_status: Mapped[str] = mapped_column(
        String(64), default="settled_acp", nullable=False
    )
    buyer_subject_type: Mapped[str | None] = mapped_column(String(64))
    buyer_subject_id: Mapped[str | None] = mapped_column(String(255))
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    delivery_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class OfferingReadinessSnapshot(Base):
    __tablename__ = "offering_readiness_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publication_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="internal_only"
    )
    acceptance_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default="catalog_only"
    )
    capacity_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    purchasable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hard_stop_reasons_json: Mapped[list] = mapped_column(
        JSON, default=list, nullable=False
    )
    soft_warning_reasons_json: Mapped[list] = mapped_column(
        JSON, default=list, nullable=False
    )
    healthy_pool_group_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    queue_headroom_percent: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    latest_catalog_canary_status: Mapped[str | None] = mapped_column(String(32))
    latest_arbitrary_canary_status: Mapped[str | None] = mapped_column(String(32))
    latest_soak_status: Mapped[str | None] = mapped_column(String(32))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    source_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class OfferingReadinessOverride(Base):
    __tablename__ = "offering_readiness_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    publication_state: Mapped[str | None] = mapped_column(String(32))
    acceptance_scope: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    starts_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class SyntheticRun(Base):
    __tablename__ = "synthetic_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running", index=True
    )
    channel_handle: Mapped[str | None] = mapped_column(String(255), index=True)
    namespace: Mapped[str | None] = mapped_column(String(128))
    mode: Mapped[str | None] = mapped_column(String(64))
    max_videos: Mapped[int | None] = mapped_column(Integer)
    published_after: Mapped[str | None] = mapped_column(String(32))
    published_before: Mapped[str | None] = mapped_column(String(32))
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SyntheticStep(Base):
    __tablename__ = "synthetic_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    synthetic_run_id: Mapped[str] = mapped_column(
        ForeignKey("synthetic_runs.id"), nullable=False, index=True
    )
    step_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class SoakRun(Base):
    __tablename__ = "soak_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running", index=True
    )
    requested_jobs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SoakSample(Base):
    __tablename__ = "soak_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    soak_run_id: Mapped[str] = mapped_column(
        ForeignKey("soak_runs.id"), nullable=False, index=True
    )
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Entitlement(Base):
    __tablename__ = "entitlements"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(
        ForeignKey("channel_packs.id"), nullable=False, index=True
    )
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
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
    last_rate_limited_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    consecutive_rate_limit_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    quarantine_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_canary_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_canary_finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_canary_status: Mapped[str | None] = mapped_column(String(32))
    last_canary_error_kind: Mapped[str | None] = mapped_column(String(64))
    last_canary_error_detail: Mapped[str | None] = mapped_column(Text)
    last_canary_video_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class SchedulerJob(Base):
    __tablename__ = "scheduler_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    probe_key: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    video_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    channel_handle: Mapped[str | None] = mapped_column(String(255), index=True)
    lane: Mapped[str] = mapped_column(
        String(64), nullable=False, default="quote_starter_probe"
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
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
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


_ENGINE = None
_SESSION_FACTORY = None


def _database_url() -> str:
    return channel_service_database_url()


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
        engine_kwargs = {
            "future": True,
            "connect_args": connect_args,
            "pool_pre_ping": True,
        }
        if url.startswith("postgresql"):
            engine_kwargs.update(
                pool_size=int(os.getenv("CHANNEL_SERVICE_DB_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("CHANNEL_SERVICE_DB_MAX_OVERFLOW", "10")),
                pool_recycle=int(
                    os.getenv("CHANNEL_SERVICE_DB_POOL_RECYCLE_SECONDS", "1800")
                ),
            )
        _ENGINE = create_engine(url, **engine_kwargs)
        _SESSION_FACTORY = sessionmaker(
            bind=_ENGINE, autoflush=False, autocommit=False, future=True
        )
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
    validate_production_runtime()
    engine = get_engine()
    url = str(engine.url)
    if engine.url.get_backend_name() == "postgresql":
        _assert_alembic_head(engine)
        return
    try:
        with _sqlite_schema_lock(url):
            Base.metadata.create_all(bind=engine)
            _apply_lightweight_migrations(engine)
    except OperationalError as exc:
        # SQLite can still race across multi-worker startup even with checkfirst.
        # If another worker created the table first, the schema is already usable.
        if "already exists" not in str(exc).lower():
            raise


def _assert_alembic_head(engine) -> None:
    """Production PostgreSQL is migration-managed; application startup never mutates it."""
    inspector = inspect(engine)
    if "alembic_version" not in set(inspector.get_table_names()):
        raise RuntimeError(
            "PostgreSQL schema is unversioned; run `alembic upgrade head` before starting the service"
        )
    with engine.connect() as conn:
        revisions = {
            str(row[0])
            for row in conn.execute(text("SELECT version_num FROM alembic_version"))
        }
    if revisions != {ALEMBIC_HEAD_REVISION}:
        current = ",".join(sorted(revisions)) or "none"
        raise RuntimeError(
            f"PostgreSQL schema is at {current}; expected Alembic head {ALEMBIC_HEAD_REVISION}"
        )


def dispose_engine() -> None:
    """Dispose cached connections, primarily for tests and controlled process shutdown."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SESSION_FACTORY = None


def set_tenant_scope(session, tenant_id: str) -> None:
    """Set the transaction-local PostgreSQL RLS scope for tenant-owned rows."""
    from .channel_service_config import validate_tenant_id

    tenant_id = validate_tenant_id(tenant_id)
    if session.get_bind().dialect.name == "postgresql":
        session.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": tenant_id},
        )


def clear_tenant_scope(session) -> None:
    """Return a PostgreSQL transaction to its fail-closed tenant state."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT set_config('app.tenant_id', '', true)"))


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
    existing = {
        str(column["name"]) for column in inspect(engine).get_columns(table_name)
    }
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
