from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.channel_service_config import (
    ChannelServiceConfigurationError,
    channel_service_database_url,
    embedding_contract,
    enforce_canonical_namespace,
    forwarded_internal_identity,
    internal_request_is_authorized,
    validate_production_runtime,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    IdempotencyConflict,
    IngestionLeaseLost,
    claim_ingestion_job,
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
TENANT_A = f"ten_{'a' * 64}"
TENANT_B = f"ten_{'b' * 64}"
USER_A = f"usr_{'1' * 64}"
USER_B = f"usr_{'2' * 64}"


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as database:
        database.add_all(
            [
                Tenant(id=TENANT_A, slug="tenant-a", display_name="Tenant A"),
                Tenant(id=TENANT_B, slug="tenant-b", display_name="Tenant B"),
                UserAccount(id=USER_A, auth_provider="test", auth_subject="a"),
                UserAccount(id=USER_B, auth_provider="test", auth_subject="b"),
            ]
        )
        database.commit()
        yield database


def test_production_rejects_implicit_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.delenv("CHANNEL_SERVICE_DATABASE_URL", raising=False)
    with pytest.raises(ChannelServiceConfigurationError, match="production requires"):
        channel_service_database_url()


def test_production_rejects_missing_internal_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL",
        "postgresql+psycopg://icmfyi:unused@postgres/icmfyi",
    )
    monkeypatch.delenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", raising=False)
    with pytest.raises(
        ChannelServiceConfigurationError, match="at least 32 characters"
    ):
        validate_production_runtime()


def test_production_requires_exact_embedding_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL",
        "postgresql+psycopg://icmfyi:unused@postgres/icmfyi",
    )
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    monkeypatch.setenv("EMBED_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.delenv("EMBED_MODEL_REVISION", raising=False)

    with pytest.raises(ChannelServiceConfigurationError, match="EMBED_MODEL_REVISION"):
        validate_production_runtime()
    monkeypatch.setenv("EMBED_MODEL_REVISION", "main")
    with pytest.raises(ChannelServiceConfigurationError, match="EMBED_MODEL_REVISION"):
        validate_production_runtime()

    revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    monkeypatch.setenv("EMBED_MODEL_REVISION", revision)
    validate_production_runtime()
    assert embedding_contract() == {
        "provider": "sentence-transformers",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": revision,
        "dimension": 1024,
    }


def test_sentence_transformer_uses_revision_and_normalized_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.pipelines.upsert_pinecone as module

    calls: dict = {}

    class Encoded:
        def tolist(self):
            return [[1.0, 0.0]]

    class FakeSentenceTransformer:
        def __init__(self, model: str, *, revision: str | None) -> None:
            calls["model"] = model
            calls["revision"] = revision

        def encode(self, texts, *, batch_size: int, normalize_embeddings: bool):
            calls["texts"] = list(texts)
            calls["batch_size"] = batch_size
            calls["normalize_embeddings"] = normalize_embeddings
            return Encoded()

    revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    monkeypatch.setattr(
        module,
        "settings_v2",
        SimpleNamespace(
            EMBED_PROVIDER="sentence-transformers",
            EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B",
            EMBED_MODEL_REVISION=revision,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    assert module._embedder()(["fact"]) == [[1.0, 0.0]]
    assert calls == {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": revision,
        "texts": ["fact"],
        "batch_size": 64,
        "normalize_embeddings": True,
    }


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
    user_id = f"usr_{'a' * 64}"
    tenant_id = f"ten_{'b' * 64}"
    identity = forwarded_internal_identity(
        {
            "x-icmfyi-user-id": user_id,
            "x-icmfyi-tenant-id": tenant_id,
        }
    )
    assert identity.user_id == user_id
    assert identity.tenant_id == tenant_id

    with pytest.raises(ChannelServiceConfigurationError, match="x-icmfyi-tenant-id"):
        forwarded_internal_identity({"x-icmfyi-user-id": user_id})
    with pytest.raises(ChannelServiceConfigurationError, match="x-icmfyi-user-id"):
        forwarded_internal_identity(
            {
                "x-icmfyi-user-id": tenant_id,
                "x-icmfyi-tenant-id": user_id,
            }
        )


def test_production_namespace_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    assert enforce_canonical_namespace("videos") == "videos"
    with pytest.raises(ChannelServiceConfigurationError, match="must equal"):
        enforce_canonical_namespace("bnb")


def test_two_tenants_reuse_one_canonical_ingestion_job(session: Session) -> None:
    request_a, job_a, created_a = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        requested_by_user_id=USER_A,
        idempotency_key="request-a",
        job_kind="video",
        source_kind="youtube",
        source_key="video-123",
        pipeline_version="qwen06-v1",
        request_payload={"retention": "clip_ready"},
    )
    request_b, job_b, created_b = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_B,
        requested_by_user_id=USER_B,
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
        tenant_id=TENANT_A,
        requested_by_user_id=USER_A,
        idempotency_key="stable-key",
        job_kind="video",
        source_kind="youtube",
        source_key="video-123",
        pipeline_version="qwen06-v1",
    )
    get_or_create_ingestion_request(
        **arguments, request_payload={"retention": "query_only"}
    )
    with pytest.raises(IdempotencyConflict):
        get_or_create_ingestion_request(
            **arguments, request_payload={"retention": "clip_ready"}
        )


def test_expired_lease_is_recovered_once(session: Session) -> None:
    _, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        idempotency_key="lease-test",
        job_kind="video",
        source_kind="twitch",
        source_key="vod-1",
        pipeline_version="v1",
        request_payload={},
    )
    now = utcnow()
    first = claim_ingestion_jobs(
        session, worker_id="worker-a", now=now, lease_seconds=10
    )
    assert [item.id for item in first] == [job.id]
    assert (
        claim_ingestion_jobs(session, worker_id="worker-b", now=now, lease_seconds=10)
        == []
    )

    recovered = claim_ingestion_jobs(
        session,
        worker_id="worker-b",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )
    assert [item.id for item in recovered] == [job.id]
    complete_ingestion_job(
        session, job_id=job.id, worker_id="worker-b", result={"ok": True}
    )
    assert (
        claim_ingestion_jobs(
            session,
            worker_id="worker-c",
            now=now + timedelta(seconds=22),
            lease_seconds=10,
        )
        == []
    )


def test_killed_hot_media_acquirer_is_reclaimed_without_claiming_other_kinds(
    session: Session,
) -> None:
    _, hot_job, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        idempotency_key="hot-media-killed",
        job_kind="youtube_hot_media",
        source_kind="youtube",
        source_key="dQw4w9WgXcQ",
        pipeline_version="youtube-hot-media-v1",
        request_payload={
            "video_id": "dQw4w9WgXcQ",
            "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        },
    )
    _, other_job, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        idempotency_key="other-kind",
        job_kind="transcript",
        source_kind="youtube",
        source_key="other-video",
        pipeline_version="v1",
        request_payload={},
    )
    started_at = utcnow()
    first = claim_ingestion_jobs(
        session,
        worker_id="acquirer-killed",
        now=started_at,
        lease_seconds=10,
        job_kinds=["youtube_hot_media"],
    )
    assert [row.id for row in first] == [hot_job.id]
    reclaimed = claim_ingestion_jobs(
        session,
        worker_id="acquirer-restarted",
        now=started_at + timedelta(seconds=11),
        lease_seconds=10,
        job_kinds=["youtube_hot_media"],
    )
    assert [row.id for row in reclaimed] == [hot_job.id]
    session.refresh(other_job)
    assert other_job.status == "queued"


def test_exact_claim_never_consumes_an_unrelated_job(session: Session) -> None:
    _, first, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        idempotency_key="exact-claim-first",
        job_kind="video",
        source_kind="youtube",
        source_key="first",
        pipeline_version="v1",
        request_payload={},
    )
    _, second, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
        idempotency_key="exact-claim-second",
        job_kind="video",
        source_kind="youtube",
        source_key="second",
        pipeline_version="v1",
        request_payload={},
    )

    claimed = claim_ingestion_job(
        session, job_id=second.id, worker_id="direct-worker", lease_seconds=10
    )
    assert claimed is not None
    assert claimed.id == second.id
    session.refresh(first)
    assert first.status == "queued"


def test_expired_worker_cannot_complete_before_reclaim(session: Session) -> None:
    _, job, _ = get_or_create_ingestion_request(
        session,
        tenant_id=TENANT_A,
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
        tenant_id=TENANT_A,
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
        tenant_id=TENANT_A,
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
    dockerfile = (
        REPOSITORY_ROOT / "services" / "diarization_indexer" / "Dockerfile"
    ).read_text(encoding="utf-8")
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "deno.land/install.sh" not in dockerfile
    assert "apt-get" not in dockerfile
    assert dockerfile.count("@sha256:") == 4
    assert "denoland/deno:2.5.6@sha256:" in dockerfile
    assert "mwader/static-ffmpeg:7.1.1@sha256:" in dockerfile
    assert "COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg" in dockerfile
    assert "COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "COPY --chown=0:0 . /app" in dockerfile
    assert "RUN chmod -R a-w /app" in dockerfile
    assert "alembic==1.16.5" in requirements
    assert "psycopg[binary]==3.2.10" in requirements


def test_initial_alembic_revision_is_a_static_schema_snapshot() -> None:
    revisions = sorted((REPOSITORY_ROOT / "alembic" / "versions").glob("*.py"))
    assert len(revisions) == 2
    initial_migration = revisions[0].read_text(encoding="utf-8")
    export_migration = revisions[1].read_text(encoding="utf-8")

    assert 'revision: str = "20260825_0001"' in initial_migration
    assert 'revision: str = "20260825_0002"' in export_migration
    assert "op.create_table(" in initial_migration
    assert "op.create_table(" in export_migration
    assert "ENABLE ROW LEVEL SECURITY" in export_migration
    assert "FORCE ROW LEVEL SECURITY" in export_migration
    assert "current_setting('app.tenant_id', true)" in export_migration
    assert "Base.metadata" not in initial_migration + export_migration
    assert "create_all(" not in initial_migration + export_migration
