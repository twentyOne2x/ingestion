from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.canonical_media import HotMediaSpec
from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    reserve_ingestion_effect,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    IngestionEffect,
    IngestionJob,
    IngestionRequest,
    SourceVideo,
    TenantChannelEntitlement,
    TranscriptionRun,
    TranscriptRevision,
    dispose_engine,
    get_engine,
    init_db,
    utcnow,
)
from src.ingest_v2.cloud.diarization_indexer.public_acquisition import (
    AcquiredPublicItem,
    PublicAcquisitionError,
    PublicItemDescriptor,
    discover_public_items,
)
from src.ingest_v2.cloud.diarization_indexer.public_ingestion_worker import (
    PublicWorkerDependencies,
    process_next_public_ingestion_job,
    reconcile_orphaned_transcription_audio,
)
from src.ingest_v2.cloud.diarization_indexer.public_platforms import (
    PublicTargetError,
    normalize_public_target,
)
from src.ingest_v2.cloud.diarization_indexer.transcription_runtime import (
    AmbiguousTranscriptionError,
    TranscriptResult,
)

USER_A = f"usr_{'a' * 64}"
TENANT_A = f"ten_{'1' * 64}"
USER_B = f"usr_{'b' * 64}"
TENANT_B = f"ten_{'2' * 64}"


def _headers(user: str = USER_A, tenant: str = TENANT_A) -> dict[str, str]:
    return {"x-icmfyi-user-id": user, "x-icmfyi-tenant-id": tenant}


def _successful_qdrant_publication(**kwargs) -> dict:
    return {
        "collection": "icmfyi-v2__canonical",
        "media_id": kwargs["media_id"],
        "transcript_revision_id": kwargs["transcript_revision_id"],
        "point_count": 1,
        "readback_sha256": "f" * 64,
    }


@pytest.fixture()
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "public-ingestion.sqlite3"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv("CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{path}")
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(tmp_path / "hot"))
    monkeypatch.setenv(
        "CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT", str(tmp_path / "transcription-tmp")
    )
    dispose_engine()
    init_db()
    yield path
    dispose_engine()


def test_provider_target_normalization_is_exact_and_canonical() -> None:
    twitch = normalize_public_target(
        platform="twitch",
        target_kind="item",
        target="https://www.twitch.tv/videos/2845546003",
    )
    assert twitch.external_id == "2845546003"
    assert twitch.canonical_url == "https://www.twitch.tv/videos/2845546003"

    x = normalize_public_target(
        platform="twitter",
        target_kind="channel",
        target="https://twitter.com/OttaBag",
        platform_entity_id="1640167205072412672",
    )
    assert x.platform == "x"
    assert x.handle == "ottabag"
    assert x.external_id == "1640167205072412672"
    assert x.canonical_url == "https://x.com/ottabag"

    room = "4Nd1mYtPfpLfr8VZC6Uj7jZGwKkNjBP3wnrhX1WJpump"
    pump = normalize_public_target(
        platform="pumpfun", target_kind="item", target=f"pumpfun:{room}:clip_123"
    )
    assert pump.channel_external_id == room
    assert pump.external_id == "clip_123"

    with pytest.raises(PublicTargetError):
        normalize_public_target(
            platform="twitch",
            target_kind="item",
            target="https://evil.example/videos/2845546003",
        )
    with pytest.raises(PublicTargetError, match="numeric platform_entity_id"):
        normalize_public_target(
            platform="x", target_kind="channel", target="https://x.com/ottabag"
        )
    with pytest.raises(PublicTargetError):
        normalize_public_target(
            platform="pumpfun",
            target_kind="item",
            target="https://livestream-api.pump.fun@evil.example/clips/a/b",
        )


def test_generic_endpoint_requires_exact_idempotency_and_cannot_widen_tenant(
    database: Path,
) -> None:
    from src.ingest_v2.cloud.diarization_indexer.service import app

    client = TestClient(app)
    payload = {
        "platform": "twitch",
        "target_kind": "item",
        "target": "https://www.twitch.tv/videos/2845546003",
        "clip_ready": True,
        "transcription_mode": "local_cpu",
        "tenant_id": TENANT_B,
    }
    assert (
        client.post("/v1/ingest", headers=_headers(), json=payload).status_code == 422
    )
    payload.pop("tenant_id")
    missing = client.post("/v1/ingest", headers=_headers(), json=payload)
    assert missing.status_code == 422
    spaced = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "not exact"},
        json=payload,
    )
    assert spaced.status_code == 400

    first = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "twitch-vod-2845546003"},
        json=payload,
    )
    replay = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "twitch-vod-2845546003"},
        json=payload,
    )
    assert first.status_code == replay.status_code == 202
    assert first.json()["created"] is True
    assert replay.json()["created"] is False
    assert replay.json()["job_id"] == first.json()["job_id"]

    conflict_payload = {**payload, "clip_ready": False}
    conflict = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "twitch-vod-2845546003"},
        json=conflict_payload,
    )
    assert conflict.status_code == 409

    other = client.post(
        "/v1/ingest",
        headers={**_headers(USER_B, TENANT_B), "Idempotency-Key": "other-tenant"},
        json=payload,
    )
    same_tenant_new_key = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "same-tenant-new-key"},
        json=payload,
    )
    assert other.status_code == 202
    assert other.json()["job_id"] == first.json()["job_id"]
    assert same_tenant_new_key.status_code == 202
    assert same_tenant_new_key.json()["job_id"] == first.json()["job_id"]
    status = client.get(
        f"/v1/ingestion-jobs/{first.json()['job_id']}", headers=_headers()
    )
    assert status.status_code == 200
    assert status.json()["job_id"] == first.json()["job_id"]
    with Session(get_engine()) as session:
        assert session.scalar(select(func.count()).select_from(IngestionJob)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionRequest)) == 3


def test_global_deduplication_binds_output_affecting_policy(database: Path) -> None:
    from src.ingest_v2.cloud.diarization_indexer.service import app

    client = TestClient(app)
    base = {
        "platform": "twitch",
        "target_kind": "channel",
        "target": "https://www.twitch.tv/megga",
        "max_items": 2,
        "transcription_mode": "local_cpu",
        "language": "en",
    }

    first = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "policy-a"},
        json=base,
    )
    same = client.post(
        "/v1/ingest",
        headers={**_headers(USER_B, TENANT_B), "Idempotency-Key": "policy-b"},
        json=base,
    )
    different_limit = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "policy-c"},
        json={**base, "max_items": 3},
    )
    different_language = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "policy-d"},
        json={**base, "language": "fr"},
    )

    assert first.status_code == same.status_code == 202
    assert different_limit.status_code == different_language.status_code == 202
    assert first.json()["job_id"] == same.json()["job_id"]
    assert different_limit.json()["job_id"] != first.json()["job_id"]
    assert different_language.json()["job_id"] != first.json()["job_id"]


def test_channel_discovery_creates_deduplicated_child_jobs_for_each_tenant(
    database: Path,
) -> None:
    from src.ingest_v2.cloud.diarization_indexer.service import app

    client = TestClient(app)
    payload = {
        "platform": "twitch",
        "target_kind": "channel",
        "target": "https://www.twitch.tv/megga",
        "max_items": 2,
        "transcription_mode": "local_cpu",
    }
    for user, tenant, key in (
        (USER_A, TENANT_A, "megga-a"),
        (USER_A, TENANT_A, "megga-a-duplicate"),
        (USER_B, TENANT_B, "megga-b"),
    ):
        response = client.post(
            "/v1/ingest",
            headers={**_headers(user, tenant), "Idempotency-Key": key},
            json=payload,
        )
        assert response.status_code == 202

    def discover(target, *, max_items):
        assert target.handle == "megga"
        assert max_items == 2
        return tuple(
            PublicItemDescriptor(
                platform="twitch",
                external_id=value,
                channel_external_id="megga",
                channel_handle="megga",
                canonical_url=f"https://www.twitch.tv/videos/{value}",
                title=f"VOD {value}",
            )
            for value in ("2831021540", "2830228699")
        )

    dependencies = PublicWorkerDependencies(discover=discover)
    assert process_next_public_ingestion_job(
        worker_id="public-discovery-test", dependencies=dependencies
    )
    with Session(get_engine()) as session:
        jobs = (
            session.execute(select(IngestionJob).order_by(IngestionJob.job_kind))
            .scalars()
            .all()
        )
        assert len(jobs) == 3
        assert [row.job_kind for row in jobs].count("public_item_ingestion") == 2
        assert (
            sum(
                len(row.request_tenant_ids_json)
                for row in jobs
                if row.job_kind == "public_item_ingestion"
            )
            == 4
        )
        # Every accepted request receives child lineage, including a second
        # idempotent request by the same principal in the same tenant.
        assert session.scalar(select(func.count()).select_from(IngestionRequest)) == 9
        assert (
            session.scalar(select(func.count()).select_from(TenantChannelEntitlement))
            == 2
        )


def test_item_worker_publishes_once_per_tenant_and_deletes_temporary_audio(
    database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical
    from src.ingest_v2.cloud.diarization_indexer.service import app

    media_path = tmp_path / "hot" / "sha256" / "aa" / f"{'a' * 64}.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"offline-fixture-video")
    media = HotMediaSpec(
        path=media_path,
        sha256=hashlib.sha256(media_path.read_bytes()).hexdigest(),
        size_bytes=media_path.stat().st_size,
        mime_type="video/mp4",
    )
    monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)
    client = TestClient(app)
    payload = {
        "platform": "twitch",
        "target_kind": "item",
        "target": "https://www.twitch.tv/videos/2845546003",
        "clip_ready": True,
        "transcription_mode": "local_cpu",
    }
    for user, tenant, key in (
        (USER_A, TENANT_A, "vod-a"),
        (USER_A, TENANT_A, "vod-a-duplicate"),
        (USER_B, TENANT_B, "vod-b"),
    ):
        assert (
            client.post(
                "/v1/ingest",
                headers={**_headers(user, tenant), "Idempotency-Key": key},
                json=payload,
            ).status_code
            == 202
        )

    item = PublicItemDescriptor(
        platform="twitch",
        external_id="2845546003",
        channel_external_id="236171146",
        channel_handle="cented",
        canonical_url="https://www.twitch.tv/videos/2845546003",
        title="Offline fixture",
        duration_ms=2_000,
    )
    observed_audio: list[Path] = []

    def acquire(_):
        return AcquiredPublicItem(item=item, media=media)

    def extract_audio(*, video_path: Path, audio_path: Path):
        assert video_path == media_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"temporary-audio")
        observed_audio.append(audio_path)
        return hashlib.sha256(b"temporary-audio").hexdigest(), 15

    def transcribe(*, audio_path: Path, contract, language: str):
        assert audio_path.exists()
        assert contract.mode == "local_cpu"
        assert language == "en"
        return TranscriptResult(
            provider="local_cpu:fixture@deadbeef",
            provider_request_id=None,
            segments=(
                {
                    "ordinal": 0,
                    "start_ms": 0,
                    "end_ms": 2000,
                    "speaker_label": None,
                    "text": "A timestamped offline transcript.",
                },
            ),
        )

    dependencies = PublicWorkerDependencies(
        acquire=acquire,
        extract_audio=extract_audio,
        transcribe=transcribe,
        delete_audio=lambda path: path.unlink(missing_ok=True),
        publish_vectors=_successful_qdrant_publication,
    )
    assert process_next_public_ingestion_job(
        worker_id="public-item-test", dependencies=dependencies
    )
    assert observed_audio and not observed_audio[0].exists()
    with Session(get_engine()) as session:
        assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
        assert session.scalar(select(func.count()).select_from(TranscriptRevision)) == 1
        assert (
            session.scalar(select(func.count()).select_from(TenantChannelEntitlement))
            == 2
        )
        run = session.execute(select(TranscriptionRun)).scalar_one()
        assert run.status == "succeeded"
        assert run.cleanup_status == "deleted"
        job = session.execute(select(IngestionJob)).scalar_one()
        assert job.status == "succeeded"
        assert job.result_json["tenant_publications"] == 2
        assert job.result_json["transcript_segments"] == 1


def test_item_worker_requires_qdrant_readback_without_repeating_paid_effects(
    database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical
    from src.ingest_v2.cloud.diarization_indexer.service import app

    media_path = tmp_path / "hot" / "sha256" / "bb" / f"{'b' * 64}.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"qdrant-publication-retry-video")
    media = HotMediaSpec(
        path=media_path,
        sha256=hashlib.sha256(media_path.read_bytes()).hexdigest(),
        size_bytes=media_path.stat().st_size,
        mime_type="video/mp4",
    )
    monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)

    response = TestClient(app).post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "qdrant-readback-retry"},
        json={
            "platform": "twitch",
            "target_kind": "item",
            "target": "https://www.twitch.tv/videos/2845546003",
            "clip_ready": True,
            "transcription_mode": "openai",
        },
    )
    assert response.status_code == 202

    item = PublicItemDescriptor(
        platform="twitch",
        external_id="2845546003",
        channel_external_id="236171146",
        channel_handle="cented",
        canonical_url="https://www.twitch.tv/videos/2845546003",
        title="Qdrant publication retry fixture",
        duration_ms=2_000,
    )
    provider_calls: list[str] = []
    vector_attempts: list[dict] = []

    def acquire(_):
        provider_calls.append("acquire")
        return AcquiredPublicItem(item=item, media=media)

    def extract_audio(*, video_path: Path, audio_path: Path):
        provider_calls.append("extract")
        assert video_path == media_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"paid-temporary-audio")
        return hashlib.sha256(b"paid-temporary-audio").hexdigest(), 20

    def transcribe(*, audio_path: Path, contract, language: str):
        provider_calls.append("transcribe")
        assert audio_path.exists()
        assert contract.mode == "openai"
        assert language == "en"
        return TranscriptResult(
            provider="openai:gpt-4o-mini-transcribe",
            provider_request_id="provider-request-once",
            segments=(
                {
                    "ordinal": 0,
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "speaker_label": None,
                    "text": "The canonical transcript must be readable from Qdrant.",
                },
            ),
        )

    def publish_vectors(**kwargs):
        vector_attempts.append(kwargs)
        if len(vector_attempts) == 1:
            raise RuntimeError("canonical Qdrant readback is incomplete")
        return {
            "collection": "icmfyi-v2__streams",
            "media_id": kwargs["media_id"],
            "transcript_revision_id": kwargs["transcript_revision_id"],
            "point_count": 1,
            "readback_sha256": "c" * 64,
        }

    dependencies = PublicWorkerDependencies(
        acquire=acquire,
        extract_audio=extract_audio,
        transcribe=transcribe,
        delete_audio=lambda path: path.unlink(missing_ok=True),
        publish_vectors=publish_vectors,
    )

    assert process_next_public_ingestion_job(
        worker_id="qdrant-readback-first", dependencies=dependencies
    )
    assert provider_calls == ["acquire", "extract", "transcribe"]
    assert len(vector_attempts) == 1
    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        assert job is not None and job.status == "retry"
        assert job.last_error_detail == "canonical Qdrant readback is incomplete"
        assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
        assert session.scalar(select(func.count()).select_from(TranscriptRevision)) == 1
        effects = list(session.scalars(select(IngestionEffect)))
        assert len(effects) == 2
        assert {effect.status for effect in effects} == {"succeeded"}
        assert {
            effect.provider_effect_id
            for effect in effects
            if effect.provider == "transcription_openai"
        } == {"provider-request-once"}
        job.next_run_at = utcnow()
        session.commit()

    assert process_next_public_ingestion_job(
        worker_id="qdrant-readback-retry", dependencies=dependencies
    )
    assert provider_calls == ["acquire", "extract", "transcribe"]
    assert len(vector_attempts) == 2
    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        assert job is not None and job.status == "succeeded"
        assert job.attempt_count == 2
        assert job.result_json["canonical_ready_reuse"] is True
        assert job.result_json["qdrant_publication"] == {
            "collection": "icmfyi-v2__streams",
            "media_id": vector_attempts[-1]["media_id"],
            "transcript_revision_id": vector_attempts[-1]["transcript_revision_id"],
            "point_count": 1,
            "readback_sha256": "c" * 64,
        }
        assert session.scalar(select(func.count()).select_from(IngestionEffect)) == 2


def test_paid_transcription_ambiguity_is_terminal_and_audio_is_deleted(
    database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical
    from src.ingest_v2.cloud.diarization_indexer.service import app

    media_path = tmp_path / "hot" / "ambiguous.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"video")
    media = HotMediaSpec(
        path=media_path,
        sha256=hashlib.sha256(b"video").hexdigest(),
        size_bytes=5,
        mime_type="video/mp4",
    )
    monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)
    client = TestClient(app)
    response = client.post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "ambiguous-paid"},
        json={
            "platform": "twitch",
            "target_kind": "item",
            "target": "https://www.twitch.tv/videos/2845546003",
            "transcription_mode": "openai",
        },
    )
    assert response.status_code == 202
    item = PublicItemDescriptor(
        platform="twitch",
        external_id="2845546003",
        channel_external_id="cented",
        channel_handle="cented",
        canonical_url="https://www.twitch.tv/videos/2845546003",
    )
    audio_paths: list[Path] = []

    def extract_audio(*, video_path: Path, audio_path: Path):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")
        audio_paths.append(audio_path)
        return hashlib.sha256(b"audio").hexdigest(), 5

    dependencies = PublicWorkerDependencies(
        acquire=lambda _: AcquiredPublicItem(item=item, media=media),
        extract_audio=extract_audio,
        transcribe=lambda **_: (_ for _ in ()).throw(
            AmbiguousTranscriptionError("timeout after provider submission")
        ),
        delete_audio=lambda path: path.unlink(missing_ok=True),
    )
    assert process_next_public_ingestion_job(
        worker_id="ambiguous-worker", dependencies=dependencies
    )
    assert audio_paths and not audio_paths[0].exists()
    with Session(get_engine()) as session:
        job = session.execute(select(IngestionJob)).scalar_one()
        run = session.execute(select(TranscriptionRun)).scalar_one()
        assert job.status == "failed"
        assert job.last_error_code == "transcription_unknown_requires_reconciliation"
        assert run.status == "unknown"
        assert run.cleanup_status == "deleted"


def test_expired_paid_running_effect_is_unknown_and_never_resubmitted(
    database: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical
    from src.ingest_v2.cloud.diarization_indexer.service import app

    media_path = tmp_path / "hot" / "already-downloaded.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"video")
    media = HotMediaSpec(
        path=media_path,
        sha256=hashlib.sha256(b"video").hexdigest(),
        size_bytes=5,
        mime_type="video/mp4",
    )
    monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)
    response = TestClient(app).post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "crashed-paid"},
        json={
            "platform": "twitch",
            "target_kind": "item",
            "target": "https://www.twitch.tv/videos/2845546003",
            "transcription_mode": "openai",
        },
    )
    assert response.status_code == 202
    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        job.status = "running"
        job.attempt_count = 1
        job.lease_owner = "crashed-paid-worker"
        job.lease_expires_at = utcnow() - timedelta(seconds=1)
        effect, _ = reserve_ingestion_effect(
            session,
            job_id=job.id,
            provider="transcription_openai",
            effect_kind="timestamped_audio_transcription",
            idempotency_key=f"public-transcript-v1:{job.dedupe_key}",
            request_payload={
                "audio_source_sha256": media.sha256,
                "language": "en",
                "contract": job.payload_json["transcription"],
            },
        )
        effect.status = "running"
        session.add(
            TranscriptionRun(
                id=f"trn_{'e' * 40}",
                job_id=job.id,
                attempt_number=1,
                mode="openai",
                model_id=job.payload_json["transcription"]["model_id"],
                model_revision=None,
                status="running",
                temp_audio_path=str(
                    tmp_path / "transcription-tmp" / job.id / "attempt-1.flac"
                ),
                cleanup_status="deleted",
                cleaned_at=utcnow(),
            )
        )
        session.commit()

    item = PublicItemDescriptor(
        platform="twitch",
        external_id="2845546003",
        channel_external_id="236171146",
        channel_handle="cented",
        canonical_url="https://www.twitch.tv/videos/2845546003",
    )
    provider_calls: list[str] = []
    dependencies = PublicWorkerDependencies(
        acquire=lambda _: AcquiredPublicItem(item=item, media=media),
        extract_audio=lambda **_: (_ for _ in ()).throw(
            AssertionError("audio extraction must not restart after paid ambiguity")
        ),
        transcribe=lambda **_: provider_calls.append("called"),
    )
    assert process_next_public_ingestion_job(
        worker_id="paid-recovery-worker", dependencies=dependencies
    )
    assert provider_calls == []
    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        effect = session.execute(
            select(IngestionEffect).where(
                IngestionEffect.provider == "transcription_openai"
            )
        ).scalar_one()
        run = session.get(TranscriptionRun, f"trn_{'e' * 40}")
        assert job.status == "failed"
        assert job.last_error_code == "transcription_unknown_requires_reconciliation"
        assert effect.status == "unknown"
        assert run.status == "unknown"


def test_orphan_reconciler_waits_for_lease_expiry_then_deletes_exact_path(
    database: Path,
    tmp_path: Path,
) -> None:
    from src.ingest_v2.cloud.diarization_indexer.service import app

    response = TestClient(app).post(
        "/v1/ingest",
        headers={**_headers(), "Idempotency-Key": "orphan-audio"},
        json={
            "platform": "twitch",
            "target_kind": "item",
            "target": "https://www.twitch.tv/videos/2845546003",
            "transcription_mode": "local_cpu",
        },
    )
    assert response.status_code == 202
    audio = (
        tmp_path / "transcription-tmp" / response.json()["job_id"] / "attempt-1.flac"
    )
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"orphaned-private-audio")
    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        job.status = "running"
        job.attempt_count = 1
        job.lease_owner = "crashed-worker"
        job.lease_expires_at = utcnow() + timedelta(minutes=5)
        session.add(
            TranscriptionRun(
                id=f"trn_{'f' * 40}",
                job_id=job.id,
                attempt_number=1,
                mode="local_cpu",
                model_id="fixture",
                model_revision="a" * 40,
                status="running",
                temp_audio_path=str(audio),
                cleanup_status="pending",
            )
        )
        session.commit()

    deleted: list[Path] = []

    def delete(path: Path) -> None:
        deleted.append(path)
        path.unlink(missing_ok=True)

    dependencies = PublicWorkerDependencies(delete_audio=delete)
    assert reconcile_orphaned_transcription_audio(dependencies=dependencies) == 0
    assert audio.exists()
    assert deleted == []

    with Session(get_engine()) as session:
        job = session.get(IngestionJob, response.json()["job_id"])
        job.lease_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    assert reconcile_orphaned_transcription_audio(dependencies=dependencies) == 1
    assert deleted == [audio]
    assert not audio.exists()
    with Session(get_engine()) as session:
        run = session.get(TranscriptionRun, f"trn_{'f' * 40}")
        assert run.cleanup_status == "deleted"
        assert run.cleaned_at is not None


class _FakeResponse:
    def __init__(self, payload: dict | None = None, content: bytes = b"") -> None:
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_pumpfun_discovery_is_bounded_and_x_is_numeric_handle_bound() -> None:
    room = "4Nd1mYtPfpLfr8VZC6Uj7jZGwKkNjBP3wnrhX1WJpump"
    pump_target = normalize_public_target(
        platform="pumpfun", target_kind="channel", target=room
    )

    class PumpHTTP:
        def get(self, url, **kwargs):
            assert url.endswith(room)
            return _FakeResponse(
                {
                    "clips": [
                        {
                            "clipId": "clip-1",
                            "playlistUrl": "https://clips.pump.fun/live/clip-1.m3u8",
                            "duration": 4,
                        }
                    ],
                    "hasMore": False,
                }
            )

    pump_items = discover_public_items(pump_target, max_items=1, http=PumpHTTP())
    assert len(pump_items) == 1
    assert pump_items[0].channel_external_id == room
    assert pump_items[0].metadata["discovery"] == "pumpfun_public_clips_api"

    x_target = normalize_public_target(
        platform="x",
        target_kind="channel",
        target="https://x.com/Tester",
        platform_entity_id="123456789012345678",
    )
    document = {
        "props": {
            "pageProps": {
                "timeline": {
                    "entries": [
                        {
                            "content": {
                                "tweet": {
                                    "id_str": "1888888888888888888",
                                    "user": {
                                        "id_str": "123456789012345678",
                                        "screen_name": "tester",
                                    },
                                    "full_text": "public video",
                                    "extended_entities": {
                                        "media": [
                                            {"id_str": "777777777777", "type": "video"}
                                        ]
                                    },
                                }
                            }
                        }
                    ]
                }
            }
        }
    }
    body = (
        b'<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(document).encode()
        + b"</script>"
    )

    class XHTTP:
        def get(self, url, **kwargs):
            return _FakeResponse(content=body)

    x_items = discover_public_items(x_target, max_items=5, http=XHTTP())
    assert len(x_items) == 1
    assert x_items[0].external_id == "1888888888888888888"
    assert x_items[0].metadata["lifetime_complete"] is False

    bad = body.replace(b'"screen_name": "tester"', b'"screen_name": "impostor"')

    class BadXHTTP:
        def get(self, url, **kwargs):
            return _FakeResponse(content=bad)

    with pytest.raises(PublicAcquisitionError, match="identity mismatch"):
        discover_public_items(x_target, max_items=5, http=BadXHTTP())
