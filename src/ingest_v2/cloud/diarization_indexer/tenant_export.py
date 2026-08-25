from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .channel_service_config import InternalRequestIdentity
from .channel_service_store import (
    MediaObject,
    SourceChannel,
    SourceVideo,
    Tenant,
    TenantChannelEntitlement,
    TenantExport,
    TenantMembership,
    TranscriptRevision,
    TranscriptSegment,
    UserAccount,
    VideoMediaRef,
    set_tenant_scope,
    utcnow,
)


TENANT_EXPORT_SCHEMA_VERSION = "tenant-sqlite-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class TenantExportError(RuntimeError):
    """The requested tenant export could not be created or verified."""


@dataclass(frozen=True)
class TenantExportSnapshot:
    tenant_id: str
    channels: tuple[dict[str, Any], ...]
    videos: tuple[dict[str, Any], ...]
    transcript_revisions: tuple[dict[str, Any], ...]
    transcript_segments: tuple[dict[str, Any], ...]
    media_refs: tuple[dict[str, Any], ...]
    snapshot_sha256: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            "channels": len(self.channels),
            "media_refs": len(self.media_refs),
            "transcript_revisions": len(self.transcript_revisions),
            "transcript_segments": len(self.transcript_segments),
            "videos": len(self.videos),
        }


def ensure_gateway_principals(
    session: Session,
    identity: InternalRequestIdentity,
) -> None:
    """Materialize the gateway-authenticated principal without widening entitlements."""
    _insert_do_nothing(
        session,
        UserAccount,
        {
            "id": identity.user_id,
            "auth_provider": "internal_gateway",
            "auth_subject": identity.user_id,
            "status": "active",
        },
        ("id",),
    )
    _insert_do_nothing(
        session,
        Tenant,
        {
            "id": identity.tenant_id,
            "slug": f"gateway-{identity.tenant_id[4:]}",
            "display_name": f"Tenant {identity.tenant_id[4:16]}",
            "status": "active",
        },
        ("id",),
    )
    session.flush()

    user = session.get(UserAccount, identity.user_id)
    tenant = session.get(Tenant, identity.tenant_id)
    if user is None or user.status != "active":
        raise TenantExportError("gateway user is not active")
    if tenant is None or tenant.status != "active":
        raise TenantExportError("gateway tenant is not active")

    _insert_do_nothing(
        session,
        TenantMembership,
        {
            "tenant_id": identity.tenant_id,
            "user_id": identity.user_id,
            "role": "member",
            "status": "active",
        },
        ("tenant_id", "user_id"),
    )
    session.flush()
    membership = session.execute(
        select(TenantMembership).where(
            TenantMembership.tenant_id == identity.tenant_id,
            TenantMembership.user_id == identity.user_id,
        )
    ).scalar_one()
    if membership.status != "active":
        raise TenantExportError("gateway tenant membership is not active")


def build_tenant_export(
    session: Session,
    *,
    identity: InternalRequestIdentity,
    idempotency_key: str,
    export_root: Path | None = None,
) -> TenantExport:
    """Build or reuse one durable, deterministic, tenant-scoped SQLite export."""
    key = str(idempotency_key or "").strip()
    if not key or len(key) > 255 or any(ord(character) < 32 for character in key):
        raise TenantExportError(
            "idempotency_key must contain 1 to 255 printable characters"
        )

    ensure_gateway_principals(session, identity)
    set_tenant_scope(session, identity.tenant_id)

    request_fingerprint = _sha256_json(
        {
            "schema_version": TENANT_EXPORT_SCHEMA_VERSION,
            "tenant_id": identity.tenant_id,
        }
    )
    row = session.execute(
        select(TenantExport).where(
            TenantExport.tenant_id == identity.tenant_id,
            TenantExport.idempotency_key == key,
        )
    ).scalar_one_or_none()
    if row is not None and row.request_fingerprint != request_fingerprint:
        raise TenantExportError(
            "idempotency key already exists with different immutable inputs"
        )

    if row is None:
        export_id = _stable_id("tex", f"{identity.tenant_id}:{key}")
        _insert_do_nothing(
            session,
            TenantExport,
            {
                "id": export_id,
                "tenant_id": identity.tenant_id,
                "requested_by_user_id": identity.user_id,
                "idempotency_key": key,
                "request_fingerprint": request_fingerprint,
                "schema_version": TENANT_EXPORT_SCHEMA_VERSION,
                "status": "running",
                "counts_json": {},
                "manifest_json": {},
            },
            ("tenant_id", "idempotency_key"),
        )
        session.flush()
        row = session.execute(
            select(TenantExport).where(
                TenantExport.tenant_id == identity.tenant_id,
                TenantExport.idempotency_key == key,
            )
        ).scalar_one()
        if row.request_fingerprint != request_fingerprint:
            raise TenantExportError(
                "idempotency key was concurrently created with different immutable inputs"
            )

    if row.status == "completed" and tenant_export_artifacts_are_valid(row):
        return row

    row.status = "running"
    row.error_detail = None
    row.updated_at = utcnow()
    session.flush()
    try:
        snapshot = load_tenant_export_snapshot(session, tenant_id=identity.tenant_id)
        artifacts = _write_export_artifacts(
            snapshot,
            export_root=export_root or tenant_export_root(),
        )
        row.snapshot_sha256 = snapshot.snapshot_sha256
        row.database_sha256 = artifacts["database_sha256"]
        row.manifest_sha256 = artifacts["manifest_sha256"]
        row.database_path = str(artifacts["database_path"])
        row.manifest_path = str(artifacts["manifest_path"])
        row.counts_json = snapshot.counts
        row.manifest_json = artifacts["manifest"]
        row.status = "completed"
        row.completed_at = utcnow()
        row.updated_at = row.completed_at
    except Exception as exc:
        row.status = "failed"
        row.error_detail = f"{type(exc).__name__}: {exc}"[:8000]
        row.completed_at = utcnow()
        row.updated_at = row.completed_at
    session.flush()
    return row


def load_tenant_export_snapshot(
    session: Session, *, tenant_id: str
) -> TenantExportSnapshot:
    """Select only active canonical rows reachable through this tenant's active entitlements."""
    channel_models = list(
        session.execute(
            select(SourceChannel)
            .join(
                TenantChannelEntitlement,
                TenantChannelEntitlement.channel_id == SourceChannel.id,
            )
            .where(
                TenantChannelEntitlement.tenant_id == tenant_id,
                TenantChannelEntitlement.status == "active",
                SourceChannel.status == "active",
            )
            .order_by(SourceChannel.id.asc())
        ).scalars()
    )
    channel_ids = [channel.id for channel in channel_models]
    video_models = (
        list(
            session.execute(
                select(SourceVideo)
                .where(
                    SourceVideo.channel_id.in_(channel_ids),
                    SourceVideo.status == "active",
                )
                .order_by(SourceVideo.channel_id.asc(), SourceVideo.id.asc())
            ).scalars()
        )
        if channel_ids
        else []
    )
    video_ids = [video.id for video in video_models]
    revision_models = (
        list(
            session.execute(
                select(TranscriptRevision)
                .where(
                    TranscriptRevision.video_id.in_(video_ids),
                    TranscriptRevision.status == "active",
                    TranscriptRevision.is_current.is_(True),
                )
                .order_by(
                    TranscriptRevision.video_id.asc(), TranscriptRevision.id.asc()
                )
            ).scalars()
        )
        if video_ids
        else []
    )
    revision_ids = [revision.id for revision in revision_models]
    segment_models = (
        list(
            session.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.revision_id.in_(revision_ids),
                    TranscriptSegment.status == "active",
                )
                .order_by(
                    TranscriptSegment.revision_id.asc(),
                    TranscriptSegment.ordinal.asc(),
                    TranscriptSegment.id.asc(),
                )
            ).scalars()
        )
        if revision_ids
        else []
    )
    media_models = (
        list(
            session.execute(
                select(VideoMediaRef, MediaObject)
                .join(MediaObject, MediaObject.sha256 == VideoMediaRef.media_sha256)
                .where(
                    VideoMediaRef.video_id.in_(video_ids),
                    VideoMediaRef.status == "active",
                    MediaObject.status == "active",
                )
                .order_by(
                    VideoMediaRef.video_id.asc(),
                    VideoMediaRef.role.asc(),
                    VideoMediaRef.media_sha256.asc(),
                )
            ).all()
        )
        if video_ids
        else []
    )

    revisions_by_id = {revision.id: revision for revision in revision_models}
    channels = tuple(
        {
            "id": channel.id,
            "platform": channel.platform,
            "external_id": channel.external_id,
            "handle": channel.handle,
            "display_name": channel.display_name,
            "canonical_url": channel.canonical_url,
        }
        for channel in channel_models
    )
    videos = tuple(
        {
            "id": video.id,
            "channel_id": video.channel_id,
            "platform": video.platform,
            "external_id": video.external_id,
            "canonical_url": video.canonical_url,
            "title": video.title,
            "description": video.description,
            "published_at": _canonical_timestamp(video.published_at),
            "duration_ms": video.duration_ms,
            "archive_state": video.archive_state,
            "clip_candidate": video.clip_candidate,
            "clip_ready": video.clip_ready,
        }
        for video in video_models
    )
    revisions = tuple(
        {
            "id": revision.id,
            "video_id": revision.video_id,
            "provider": revision.provider,
            "provider_revision_id": revision.provider_revision_id,
            "language": revision.language,
            "content_sha256": _validated_sha256(revision.content_sha256),
            "captured_at": _canonical_timestamp(revision.captured_at),
        }
        for revision in revision_models
    )
    segments: list[dict[str, Any]] = []
    for segment in segment_models:
        if segment.end_ms < segment.start_ms or segment.ordinal < 0:
            raise TenantExportError(f"invalid transcript segment timing: {segment.id}")
        revision = revisions_by_id[segment.revision_id]
        segments.append(
            {
                "id": segment.id,
                "revision_id": segment.revision_id,
                "video_id": revision.video_id,
                "ordinal": segment.ordinal,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "speaker_label": segment.speaker_label,
                "text": segment.text,
            }
        )
    media_refs: list[dict[str, Any]] = []
    for reference, media in media_models:
        digest = _validated_sha256(media.sha256)
        if media.size_bytes < 0:
            raise TenantExportError(f"invalid media size for {digest}")
        media_refs.append(
            {
                "video_id": reference.video_id,
                "role": reference.role,
                "sha256": digest,
                "size_bytes": media.size_bytes,
                "mime_type": media.mime_type,
                "content_uri": f"sha256:{digest}",
            }
        )

    snapshot_payload = {
        "schema_version": TENANT_EXPORT_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "channels": channels,
        "videos": videos,
        "transcript_revisions": revisions,
        "transcript_segments": segments,
        "media_refs": media_refs,
    }
    return TenantExportSnapshot(
        tenant_id=tenant_id,
        channels=channels,
        videos=videos,
        transcript_revisions=revisions,
        transcript_segments=tuple(segments),
        media_refs=tuple(media_refs),
        snapshot_sha256=_sha256_json(snapshot_payload),
    )


def tenant_export_root() -> Path:
    value = (
        os.getenv("CHANNEL_SERVICE_TENANT_EXPORT_ROOT") or "/data/exports/tenants"
    ).strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TenantExportError("CHANNEL_SERVICE_TENANT_EXPORT_ROOT must be absolute")
    return path


def tenant_export_artifacts_are_valid(row: TenantExport) -> bool:
    if not all(
        (
            row.database_path,
            row.manifest_path,
            row.database_sha256,
            row.manifest_sha256,
        )
    ):
        return False
    database_path = Path(str(row.database_path))
    manifest_path = Path(str(row.manifest_path))
    return (
        database_path.is_file()
        and manifest_path.is_file()
        and _sha256_file(database_path) == row.database_sha256
        and _sha256_file(manifest_path) == row.manifest_sha256
    )


def tenant_export_artifact_path(row: TenantExport, name: str) -> Path:
    if name == "database":
        raw_path = row.database_path
        expected_sha256 = row.database_sha256
    elif name == "manifest":
        raw_path = row.manifest_path
        expected_sha256 = row.manifest_sha256
    else:
        raise TenantExportError("unsupported tenant export artifact")
    if not raw_path or not expected_sha256:
        raise TenantExportError("tenant export artifact is not ready")
    path = Path(raw_path).resolve()
    expected_parent = (tenant_export_root() / row.tenant_id).resolve()
    if path.parent != expected_parent:
        raise TenantExportError("tenant export artifact escaped its tenant root")
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise TenantExportError("tenant export artifact failed SHA-256 verification")
    return path


def serialize_tenant_export(row: TenantExport) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "schema_version": row.schema_version,
        "status": row.status,
        "snapshot_sha256": row.snapshot_sha256,
        "database_sha256": row.database_sha256,
        "manifest_sha256": row.manifest_sha256,
        "counts": dict(row.counts_json or {}),
        "error_detail": row.error_detail,
    }


def _write_export_artifacts(
    snapshot: TenantExportSnapshot,
    *,
    export_root: Path,
) -> dict[str, Any]:
    tenant_root = export_root / snapshot.tenant_id
    tenant_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tenant-export-", dir=tenant_root
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_sqlite_database(temporary_path, snapshot)
        database_sha256 = _sha256_file(temporary_path)
        database_name = f"{TENANT_EXPORT_SCHEMA_VERSION}-{database_sha256}.sqlite3"
        database_path = tenant_root / database_name
        _install_immutable_file(temporary_path, database_path, database_sha256)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": TENANT_EXPORT_SCHEMA_VERSION,
        "tenant_id": snapshot.tenant_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "database": {
            "filename": database_name,
            "sha256": database_sha256,
            "size_bytes": database_path.stat().st_size,
        },
        "counts": snapshot.counts,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = (
        tenant_root / f"{TENANT_EXPORT_SCHEMA_VERSION}-{manifest_sha256}.manifest.json"
    )
    descriptor, manifest_temporary_name = tempfile.mkstemp(
        prefix=".tenant-manifest-", dir=tenant_root
    )
    manifest_temporary_path = Path(manifest_temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _install_immutable_file(
            manifest_temporary_path,
            manifest_path,
            manifest_sha256,
        )
    finally:
        manifest_temporary_path.unlink(missing_ok=True)

    return {
        "database_path": database_path,
        "database_sha256": database_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
    }


def _write_sqlite_database(path: Path, snapshot: TenantExportSnapshot) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA page_size = 4096;
            PRAGMA auto_vacuum = NONE;
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            PRAGMA temp_store = MEMORY;
            PRAGMA foreign_keys = ON;
            PRAGMA application_id = 0x49434d46;
            PRAGMA user_version = 1;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE channels (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                external_id TEXT NOT NULL,
                handle TEXT,
                display_name TEXT,
                canonical_url TEXT
            ) WITHOUT ROWID;
            CREATE TABLE videos (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL REFERENCES channels(id),
                platform TEXT NOT NULL,
                external_id TEXT NOT NULL,
                canonical_url TEXT,
                title TEXT,
                description TEXT,
                published_at TEXT,
                duration_ms INTEGER,
                archive_state TEXT NOT NULL,
                clip_candidate INTEGER NOT NULL,
                clip_ready INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE transcript_revisions (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id),
                provider TEXT NOT NULL,
                provider_revision_id TEXT NOT NULL,
                language TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                captured_at TEXT
            ) WITHOUT ROWID;
            CREATE TABLE transcript_segments (
                row_id INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                revision_id TEXT NOT NULL REFERENCES transcript_revisions(id),
                video_id TEXT NOT NULL REFERENCES videos(id),
                ordinal INTEGER NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                speaker_label TEXT,
                text TEXT NOT NULL
            );
            CREATE UNIQUE INDEX uq_export_segments_revision_ordinal
                ON transcript_segments(revision_id, ordinal);
            CREATE TABLE media_refs (
                video_id TEXT NOT NULL REFERENCES videos(id),
                role TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                content_uri TEXT NOT NULL,
                PRIMARY KEY (video_id, role, sha256)
            ) WITHOUT ROWID;
            CREATE VIRTUAL TABLE transcript_segments_fts USING fts5(
                text,
                video_id UNINDEXED,
                revision_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", TENANT_EXPORT_SCHEMA_VERSION),
                ("snapshot_sha256", snapshot.snapshot_sha256),
                ("tenant_id", snapshot.tenant_id),
            ),
        )
        connection.executemany(
            "INSERT INTO channels VALUES (:id, :platform, :external_id, :handle, :display_name, :canonical_url)",
            snapshot.channels,
        )
        connection.executemany(
            """
            INSERT INTO videos VALUES (
                :id, :channel_id, :platform, :external_id, :canonical_url,
                :title, :description, :published_at, :duration_ms, :archive_state,
                :clip_candidate, :clip_ready
            )
            """,
            snapshot.videos,
        )
        connection.executemany(
            """
            INSERT INTO transcript_revisions VALUES (
                :id, :video_id, :provider, :provider_revision_id,
                :language, :content_sha256, :captured_at
            )
            """,
            snapshot.transcript_revisions,
        )
        segment_rows = [
            {"row_id": row_id, **segment}
            for row_id, segment in enumerate(snapshot.transcript_segments, start=1)
        ]
        connection.executemany(
            """
            INSERT INTO transcript_segments VALUES (
                :row_id, :id, :revision_id, :video_id, :ordinal,
                :start_ms, :end_ms, :speaker_label, :text
            )
            """,
            segment_rows,
        )
        connection.executemany(
            """
            INSERT INTO transcript_segments_fts(rowid, text, video_id, revision_id)
            VALUES (:row_id, :text, :video_id, :revision_id)
            """,
            segment_rows,
        )
        connection.executemany(
            """
            INSERT INTO media_refs VALUES (
                :video_id, :role, :sha256, :size_bytes, :mime_type, :content_uri
            )
            """,
            snapshot.media_refs,
        )
        connection.commit()
        result = connection.execute(
            "SELECT value FROM metadata WHERE key = 'snapshot_sha256'"
        ).fetchone()
        if result != (snapshot.snapshot_sha256,):
            raise TenantExportError(
                "SQLite knowledge check failed for snapshot metadata"
            )
        if connection.execute(
            "SELECT count(*) FROM transcript_segments_fts"
        ).fetchone() != (len(snapshot.transcript_segments),):
            raise TenantExportError("SQLite knowledge check failed for FTS row count")
        connection.execute("VACUUM")
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            raise TenantExportError(
                "SQLite runtime does not provide required FTS5 support"
            ) from exc
        raise
    finally:
        connection.close()


def _install_immutable_file(
    source: Path, destination: Path, expected_sha256: str
) -> None:
    os.chmod(source, 0o444)
    try:
        os.link(source, destination)
    except FileExistsError:
        if not destination.is_file() or _sha256_file(destination) != expected_sha256:
            raise TenantExportError(f"immutable artifact collision at {destination}")
    os.chmod(destination, 0o444)


def _insert_do_nothing(
    session: Session,
    model: type,
    values: dict[str, Any],
    conflict_columns: tuple[str, ...],
) -> None:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        statement = (
            insert(model)
            .values(**values)
            .on_conflict_do_nothing(index_elements=list(conflict_columns))
        )
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        statement = (
            insert(model)
            .values(**values)
            .on_conflict_do_nothing(index_elements=list(conflict_columns))
        )
    else:  # pragma: no cover - production and tests use PostgreSQL/SQLite
        raise TenantExportError(f"unsupported database dialect: {dialect}")
    session.execute(statement)


def _canonical_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validated_sha256(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise TenantExportError("canonical content digest must be lowercase SHA-256")
    return normalized


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:40]}"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
