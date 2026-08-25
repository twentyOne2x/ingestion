from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.channel_service_config import (
    ChannelServiceConfigurationError,
    channel_service_database_url,
    forwarded_internal_identity,
    internal_request_is_authorized,
    validate_production_runtime,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    IdempotencyConflict,
    IngestionLeaseLost,
    claim_ingestion_jobs,
    complete_ingestion_job,
    get_or_create_ingestion_request,
    reserve_ingestion_effect,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    Base,
    Tenant,
    UserAccount,
    utcnow,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as database:
        database.add_all(
            [
                Tenant(id="tenant-a", slug="tenant-a", display_name="Tenant A"),
                Tenant(id="tenant-b", slug="tenant-b", display_name="Tenant B"),
                UserAccount(id="user-a", auth_provider="test", auth_subject="a"),
                UserAccount(id="user-b", auth_provider="test", auth_subject="b"),
            ]
        )
        database.commit()
        yield database


def test_production_rejects_implicit_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.delenv("CHANNEL_SERVICE_DATABASE_URL", raising=False)
    with pytest.raises(ChannelServiceConfigurationError, match="production requires"):
        channel_service_database_url()


def test_production_rejects_missing_internal_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL",
        "postgresql+psycopg://icmfyi:unused@postgres/icmfyi",
    )
    monkeypatch.delenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", raising=False)
    with pytest.raises(ChannelServiceConfigurationError, match="at least 32 characters"):
        validate_production_runtime()


def test_production_internal_request_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "s" * 32
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", secret)

    assert internal_request_is_authorized("/v1/channel-packs/quotes", {}) is False
    assert (
        internal_request_is_authorized(
            "/v1/channel-packs/quotes",
            {"x-icmfyi-internal-secret": "wrong"},
        )
        is False
    )
    assert (
        internal_request_is_authorized(
            "/v1/channel-packs/quotes",
            {"X-ICMFYI-INTERNAL-SECRET": secret},
        )
        is True
    )
    assert internal_request_is_authorized("/healthz", {}) is True
    assert internal_request_is_authorized("/pubsub/push", {}) is True


def test_tenant_identity_is_derived_only_from_gateway_headers() -> None:
    identity = forwarded_internal_identity(
        {
            "x-icmfyi-user-id": "user-a",
            "x-icmfyi-tenant-id": "tenant-a",
        }
    )
    assert identity.user_id == "user-a"
    assert identity.tenant_id == "tenant-a"

    with pytest.raises(ChannelServiceConfigurationError, match="x-icmfyi-tenant-id"):
        forwarded_internal_identity({"x-icmfyi-user-id": "user-a"})


def test_two_tenants_reuse_one_canonical_ingestion_job(session: Session) -> None:
    request_a, job_a, created_a = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-a",
        requested_by_user_id="user-a",
        idempotency_key="request-a",
        job_kind="video",
        source_kind="youtube",
        source_key="video-123",
        pipeline_version="qwen06-v1",
        request_payload={"retention": "clip_ready"},
    )
    request_b, job_b, created_b = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-b",
        requested_by_user_id="user-b",
        idempotency_key="request-b",
        job_kind="video",
        source_kind="youtube",
        source_key="video-123",
        pipeline_version="qwen06-v1",
        request_payload={"retention": "clip_ready"},
    )
    session.commit()

    assert created_a and created_b
    assert request_a.id != request_b.id
    assert job_a.id == job_b.id


def test_tenant_idempotency_key_cannot_change_effect(session: Session) -> None:
    arguments = dict(
        session=session,
        tenant_id="tenant-a",
        requested_by_user_id="user-a",
        idempotency_key="stable-key",
        job_kind="video",
        source_kind="youtube",
        source_key="video-123",
        pipeline_version="qwen06-v1",
    )
    get_or_create_ingestion_request(**arguments, request_payload={"retention": "query_only"})
    with pytest.raises(IdempotencyConflict):
        get_or_create_ingestion_request(**arguments, request_payload={"retention": "clip_ready"})


def test_expired_lease_is_recovered_once(session: Session) -> None:
    _, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-a",
        idempotency_key="lease-test",
        job_kind="video",
        source_kind="twitch",
        source_key="vod-1",
        pipeline_version="v1",
        request_payload={},
    )
    now = utcnow()
    first = claim_ingestion_jobs(session, worker_id="worker-a", now=now, lease_seconds=10)
    assert [item.id for item in first] == [job.id]
    assert claim_ingestion_jobs(session, worker_id="worker-b", now=now, lease_seconds=10) == []

    recovered = claim_ingestion_jobs(
        session,
        worker_id="worker-b",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )
    assert [item.id for item in recovered] == [job.id]
    complete_ingestion_job(session, job_id=job.id, worker_id="worker-b", result={"ok": True})
    assert (
        claim_ingestion_jobs(
            session,
            worker_id="worker-c",
            now=now + timedelta(seconds=22),
            lease_seconds=10,
        )
        == []
    )


def test_expired_worker_cannot_complete_before_reclaim(session: Session) -> None:
    _, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-a",
        idempotency_key="stale-owner-test",
        job_kind="video",
        source_kind="youtube",
        source_key="video-stale",
        pipeline_version="v1",
        request_payload={},
    )
    started_at = utcnow()
    claim_ingestion_jobs(
        session,
        worker_id="worker-a",
        now=started_at,
        lease_seconds=10,
    )

    with pytest.raises(IngestionLeaseLost, match="live running lease"):
        complete_ingestion_job(
            session,
            job_id=job.id,
            worker_id="worker-a",
            result={"must_not_commit": True},
            now=started_at + timedelta(seconds=11),
        )


def test_expired_final_attempt_becomes_terminal_failure(session: Session) -> None:
    request, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-a",
        idempotency_key="exhausted-lease-test",
        job_kind="video",
        source_kind="youtube",
        source_key="video-exhausted",
        pipeline_version="v1",
        request_payload={},
        max_attempts=1,
    )
    started_at = utcnow()
    claimed = claim_ingestion_jobs(
        session,
        worker_id="worker-a",
        now=started_at,
        lease_seconds=10,
    )
    assert [item.id for item in claimed] == [job.id]

    assert (
        claim_ingestion_jobs(
            session,
            worker_id="worker-b",
            now=started_at + timedelta(seconds=11),
            lease_seconds=10,
        )
        == []
    )
    session.refresh(job)
    session.refresh(request)
    assert job.status == "failed"
    assert job.last_error_code == "lease_expired_after_max_attempts"
    assert job.lease_owner is None
    assert job.lease_expires_at is None
    assert request.status == "failed"


def test_provider_effect_is_reserved_before_submit(session: Session) -> None:
    _, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id="tenant-a",
        idempotency_key="effect-test",
        job_kind="transcript",
        source_kind="youtube",
        source_key="video-456",
        pipeline_version="v1",
        request_payload={},
    )
    effect, created = reserve_ingestion_effect(
        session,
        job_id=job.id,
        provider="assemblyai",
        effect_kind="transcription",
        idempotency_key="assemblyai:video-456:v1",
        request_payload={"audio_sha256": "a" * 64},
    )
    same, created_again = reserve_ingestion_effect(
        session,
        job_id=job.id,
        provider="assemblyai",
        effect_kind="transcription",
        idempotency_key="assemblyai:video-456:v1",
        request_payload={"audio_sha256": "a" * 64},
    )
    assert created is True
    assert created_again is False
    assert same.id == effect.id


def test_production_image_is_immutable_and_non_root() -> None:
    dockerfile = (REPOSITORY_ROOT / "services" / "diarization_indexer" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "deno.land/install.sh" not in dockerfile
    assert "apt-get" not in dockerfile
    assert dockerfile.count("@sha256:") == 3
    assert "denoland/deno:2.5.6@sha256:" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "alembic==1.16.5" in requirements
    assert "psycopg[binary]==3.2.10" in requirements


def test_initial_alembic_revision_is_a_static_schema_snapshot() -> None:
    revisions = sorted((REPOSITORY_ROOT / "alembic" / "versions").glob("*.py"))
    assert len(revisions) == 1
    migration = revisions[0].read_text(encoding="utf-8")

    assert 'revision: str = "20260825_0001"' in migration
    assert "op.create_table(" in migration
    assert "Base.metadata" not in migration
    assert "create_all(" not in migration
