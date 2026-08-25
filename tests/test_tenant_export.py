from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.channel_service_config import (
    InternalRequestIdentity,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    Base,
    MediaLocation,
    MediaObject,
    SourceChannel,
    SourceVideo,
    TenantChannelEntitlement,
    TenantExport,
    TranscriptRevision,
    TranscriptSegment,
    VideoMediaRef,
)
from src.ingest_v2.cloud.diarization_indexer.tenant_export import (
    build_tenant_export,
    ensure_gateway_principals,
)


IDENTITY_A = InternalRequestIdentity(
    user_id=f"usr_{'a' * 64}", tenant_id=f"ten_{'1' * 64}"
)
IDENTITY_B = InternalRequestIdentity(
    user_id=f"usr_{'b' * 64}", tenant_id=f"ten_{'2' * 64}"
)
MEDIA_A_SHA = hashlib.sha256(b"tenant-a-media").hexdigest()
MEDIA_B_SHA = hashlib.sha256(b"tenant-b-media").hexdigest()
TRANSCRIPT_A_SHA = hashlib.sha256(b"tenant-a-transcript").hexdigest()
TRANSCRIPT_B_SHA = hashlib.sha256(b"tenant-b-transcript").hexdigest()


def _seed_canonical_data(session: Session) -> None:
    ensure_gateway_principals(session, IDENTITY_A)
    ensure_gateway_principals(session, IDENTITY_B)
    session.add_all(
        [
            SourceChannel(
                id="chn-a",
                platform="youtube",
                external_id="channel-a",
                handle="@tenant-a",
                display_name="Tenant A Channel",
                canonical_url="https://youtube.example/channel-a",
            ),
            SourceChannel(
                id="chn-b",
                platform="twitch",
                external_id="channel-b",
                handle="tenant-b",
                display_name="Tenant B Channel",
                canonical_url="https://twitch.example/channel-b",
            ),
            TenantChannelEntitlement(
                id="ent-a",
                tenant_id=IDENTITY_A.tenant_id,
                channel_id="chn-a",
                granted_by_user_id=IDENTITY_A.user_id,
            ),
            TenantChannelEntitlement(
                id="ent-b",
                tenant_id=IDENTITY_B.tenant_id,
                channel_id="chn-b",
                granted_by_user_id=IDENTITY_B.user_id,
            ),
            SourceVideo(
                id="vid-a",
                channel_id="chn-a",
                platform="youtube",
                external_id="video-a",
                canonical_url="https://youtube.example/watch/video-a",
                title="Alpha",
                description="Tenant A only",
                published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                duration_ms=12_000,
            ),
            SourceVideo(
                id="vid-b",
                channel_id="chn-b",
                platform="twitch",
                external_id="video-b",
                canonical_url="https://twitch.example/videos/video-b",
                title="Beta",
                description="Tenant B only",
                published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                duration_ms=15_000,
            ),
            TranscriptRevision(
                id="rev-a",
                video_id="vid-a",
                provider="assemblyai",
                provider_revision_id="provider-a",
                language="en",
                content_sha256=TRANSCRIPT_A_SHA,
                captured_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            ),
            TranscriptRevision(
                id="rev-b",
                video_id="vid-b",
                provider="assemblyai",
                provider_revision_id="provider-b",
                language="en",
                content_sha256=TRANSCRIPT_B_SHA,
                captured_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
            ),
            TranscriptSegment(
                id="seg-a",
                revision_id="rev-a",
                ordinal=0,
                start_ms=0,
                end_ms=5_000,
                speaker_label="speaker-a",
                text="Solana knowledge belongs only to tenant alpha.",
            ),
            TranscriptSegment(
                id="seg-b",
                revision_id="rev-b",
                ordinal=0,
                start_ms=0,
                end_ms=6_000,
                speaker_label="speaker-b",
                text="Ethereum knowledge belongs only to tenant beta.",
            ),
            MediaObject(
                sha256=MEDIA_A_SHA,
                size_bytes=1234,
                mime_type="video/mp4",
            ),
            MediaObject(
                sha256=MEDIA_B_SHA,
                size_bytes=5678,
                mime_type="video/mp4",
            ),
            VideoMediaRef(
                id="ref-a",
                video_id="vid-a",
                media_sha256=MEDIA_A_SHA,
                role="source_video",
            ),
            VideoMediaRef(
                id="ref-b",
                video_id="vid-b",
                media_sha256=MEDIA_B_SHA,
                role="proxy",
            ),
            MediaLocation(
                id="loc-a",
                media_sha256=MEDIA_A_SHA,
                backend="hot_local",
                location_key=f"objects/sha256/{MEDIA_A_SHA[:2]}/{MEDIA_A_SHA}",
                status="active",
                bytes=1234,
                verified_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            ),
            MediaLocation(
                id="loc-b",
                media_sha256=MEDIA_B_SHA,
                backend="storagebox",
                location_key=f"objects/sha256/{MEDIA_B_SHA[:2]}/{MEDIA_B_SHA}",
                status="active",
                bytes=5678,
                verified_at=datetime(2026, 1, 6, tzinfo=timezone.utc),
            ),
        ]
    )
    session.commit()


def _database_rows(path: str, query: str) -> list[tuple]:
    connection = sqlite3.connect(
        f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True
    )
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def test_tenant_export_isolated_deterministic_and_searchable(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed_canonical_data(session)
        first = build_tenant_export(
            session,
            identity=IDENTITY_A,
            idempotency_key="export-a-1",
            export_root=tmp_path / "first",
        )
        second = build_tenant_export(
            session,
            identity=IDENTITY_A,
            idempotency_key="export-a-2",
            export_root=tmp_path / "second",
        )
        session.commit()

        assert first.status == "completed"
        assert second.status == "completed"
        assert first.database_sha256 == second.database_sha256
        assert first.manifest_sha256 == second.manifest_sha256
        assert (
            Path(first.database_path).read_bytes()
            == Path(second.database_path).read_bytes()
        )
        assert first.counts_json == {
            "channels": 1,
            "media_refs": 1,
            "transcript_revisions": 1,
            "transcript_segments": 1,
            "videos": 1,
        }
        assert Path(first.database_path).stat().st_mode & 0o222 == 0
        assert Path(first.manifest_path).stat().st_mode & 0o222 == 0

        assert _database_rows(first.database_path, "SELECT id FROM channels") == [
            ("chn-a",)
        ]
        assert _database_rows(first.database_path, "SELECT id FROM videos") == [
            ("vid-a",)
        ]
        assert _database_rows(
            first.database_path,
            "SELECT text FROM transcript_segments_fts WHERE transcript_segments_fts MATCH 'solana'",
        ) == [("Solana knowledge belongs only to tenant alpha.",)]
        assert _database_rows(
            first.database_path,
            "SELECT content_uri FROM media_refs",
        ) == [(f"sha256:{MEDIA_A_SHA}",)]


def test_idempotency_and_second_tenant_never_cross_scope(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed_canonical_data(session)
        first = build_tenant_export(
            session,
            identity=IDENTITY_A,
            idempotency_key="same-key",
            export_root=tmp_path,
        )
        same = build_tenant_export(
            session,
            identity=IDENTITY_A,
            idempotency_key="same-key",
            export_root=tmp_path,
        )
        other = build_tenant_export(
            session,
            identity=IDENTITY_B,
            idempotency_key="same-key",
            export_root=tmp_path,
        )
        session.commit()

        assert same.id == first.id
        assert session.execute(select(TenantExport)).scalars().all() == [first, other]
        assert other.database_sha256 != first.database_sha256
        assert _database_rows(other.database_path, "SELECT id FROM channels") == [
            ("chn-b",)
        ]
        assert _database_rows(other.database_path, "SELECT id FROM videos") == [
            ("vid-b",)
        ]
        assert _database_rows(
            other.database_path,
            "SELECT text FROM transcript_segments_fts WHERE transcript_segments_fts MATCH 'ethereum'",
        ) == [("Ethereum knowledge belongs only to tenant beta.",)]
