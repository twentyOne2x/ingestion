import base64
import json
import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session


def build_pubsub_payload(namespace: str, body: dict) -> dict:
    encoded = base64.b64encode(json.dumps(body).encode()).decode()
    return {
        "message": {
            "data": encoded,
            "attributes": {"namespace": namespace},
            "messageId": "abc",
        },
        "subscription": "projects/test/subscriptions/diarization-ready",
    }


@pytest.fixture(autouse=True)
def namespace_config(monkeypatch):
    config = {
        "namespaces": {
            "videos": {"channels": ["@SolanaFndn"]},
            "bnb": {"channels": ["@BinanceYoutube"]},
        }
    }
    monkeypatch.setenv("YT_NAMESPACE_CONFIG_JSON", json.dumps(config))
    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
    return config


@pytest.fixture(autouse=True)
def stub_pubsub_verification(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.verify_pubsub_push", mock
    )
    return mock


def get_app():
    from src.ingest_v2.cloud.diarization_indexer.service import app

    return app


def test_endpoint_skips_unknown_namespace(monkeypatch):
    app = get_app()
    client = TestClient(app)
    mock_service = Mock()
    mock_service.handle_event = Mock()
    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.create_ingest_service",
        lambda *args, **kwargs: mock_service,
    )

    payload = build_pubsub_payload(
        namespace="unknown",
        body={
            "mp3_uri": "gs://bucket/foo.mp3",
            "diarized_uri": "gs://bucket/bar.json",
            "metadata_uri": "gs://bucket/meta.json",
            "video_id": "video123",
            "entities_uri": None,
        },
    )

    resp = client.post("/pubsub/push", json=payload)
    assert resp.status_code == 204
    mock_service.handle_event.assert_not_called()


def test_endpoint_invokes_ingest_for_configured_namespace(monkeypatch):
    app = get_app()
    client = TestClient(app)
    mock_service = Mock()
    mock_service.handle_event = Mock()
    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.create_ingest_service",
        lambda *args, **kwargs: mock_service,
    )

    payload = build_pubsub_payload(
        namespace="videos",
        body={
            "mp3_uri": "gs://bucket/youtube_audio/2024-01-01_abc123/title.mp3",
            "diarized_uri": "gs://bucket/youtube_diarized/abc123/abc123_diarized.json",
            "metadata_uri": "gs://bucket/youtube_diarized/abc123/abc123_metadata.json",
            "video_id": "abc123",
            "entities_uri": "gs://bucket/youtube_diarized/abc123/abc123_entities.json",
        },
    )

    resp = client.post("/pubsub/push", json=payload)
    assert resp.status_code == 204
    mock_service.handle_event.assert_called_once()


def test_production_rejects_namespace_widening_on_every_write_route(monkeypatch):
    secret = "s" * 32
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", secret)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    app = get_app()
    client = TestClient(app)
    internal_headers = {"x-icmfyi-internal-secret": secret}

    youtube = client.post(
        "/index/youtube",
        headers=internal_headers,
        json={
            "video_urls": ["https://youtube.example/watch?v=abc"],
            "namespace": "bnb",
        },
    )
    diarized = client.post(
        "/index/diarized",
        headers=internal_headers,
        json={"video_id": "abc", "diarized_uri": "/unread", "namespace": "bnb"},
    )
    quote = client.post(
        "/v1/channel-packs/quotes",
        headers=internal_headers,
        json={"channel_handle": "@example", "namespace": "bnb"},
    )
    pubsub = client.post(
        "/pubsub/push",
        json=build_pubsub_payload(
            namespace="bnb",
            body={
                "mp3_uri": "gs://bucket/foo.mp3",
                "diarized_uri": "gs://bucket/bar.json",
                "metadata_uri": "gs://bucket/meta.json",
                "video_id": "abc",
                "entities_uri": None,
            },
        ),
    )

    assert youtube.status_code == 400
    assert diarized.status_code == 400
    assert quote.status_code == 400
    assert pubsub.status_code == 400
    assert "CHANNEL_SERVICE_CANONICAL_NAMESPACE" in youtube.json()["detail"]


def test_production_namespace_route_still_requires_gateway_secret(monkeypatch):
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    client = TestClient(get_app())

    response = client.post(
        "/index/youtube",
        json={
            "video_urls": ["https://youtube.example/watch?v=abc"],
            "namespace": "bnb",
        },
    )

    assert response.status_code == 401


def test_tenant_export_route_rejects_untrusted_or_swapped_principals(monkeypatch):
    secret = "s" * 32
    user_id = f"usr_{'a' * 64}"
    tenant_id = f"ten_{'b' * 64}"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", secret)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    client = TestClient(get_app())

    missing_secret = client.post(
        "/v1/tenant-exports",
        headers={
            "x-icmfyi-user-id": user_id,
            "x-icmfyi-tenant-id": tenant_id,
        },
        json={"idempotency_key": "export-1"},
    )
    swapped = client.post(
        "/v1/tenant-exports",
        headers={
            "x-icmfyi-internal-secret": secret,
            "x-icmfyi-user-id": tenant_id,
            "x-icmfyi-tenant-id": user_id,
        },
        json={"idempotency_key": "export-1"},
    )

    assert missing_secret.status_code == 401
    assert swapped.status_code == 401


def test_direct_youtube_index_materializes_two_tenant_entitlements(
    tmp_path,
    monkeypatch,
):
    from src.ingest_v2.cloud.diarization_indexer.canonical_media import (
        canonical_source_video_id,
    )
    from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
        SourceVideo,
        TenantChannelEntitlement,
        dispose_engine,
        get_engine,
        init_db,
    )

    database_path = tmp_path / "direct-index.sqlite3"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database_path}"
    )
    dispose_engine()
    init_db()
    video_id = "dQw4w9WgXcQ"

    def fake_index(**kwargs):
        kwargs["canonical_publish"](
            {
                "platform": "youtube",
                "provider_video_id": video_id,
                "channel_external_id": "UC-direct",
                "channel_handle": "@direct",
                "channel_name": "Direct",
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "Direct video",
                "description": "Direct tenant publication.",
                "published_at": "2026-08-25T00:00:00Z",
                "duration_ms": 1_000,
                "language": "en",
                "transcript_provider": "test",
                "transcript_segments": [
                    {"start": 0.0, "end": 1.0, "text": "Tenant-scoped fact."}
                ],
                "metadata": {},
            }
        )
        return {
            "video_id": video_id,
            "media_id": canonical_source_video_id("youtube", video_id),
            "segments": 1,
        }

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.index_youtube_video_captions",
        fake_index,
    )
    client = TestClient(get_app())
    for marker in ("a", "b"):
        response = client.post(
            "/index/youtube",
            headers={
                "x-icmfyi-user-id": f"usr_{marker * 64}",
                "x-icmfyi-tenant-id": f"ten_{marker * 64}",
            },
            json={
                "video_urls": [f"https://www.youtube.com/watch?v={video_id}"],
                "clip_ready": False,
                "tenant_id": f"ten_{'f' * 64}",
            },
        )
        assert response.status_code == 200
        assert response.json()["failed"] == []

    with Session(get_engine()) as session:
        assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
        assert (
            session.scalar(select(func.count()).select_from(TenantChannelEntitlement))
            == 2
        )
    dispose_engine()


def test_direct_youtube_all_failed_is_non_2xx_and_not_ok(monkeypatch):
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.index_youtube_video_captions",
        Mock(side_effect=RuntimeError("provider unavailable")),
    )
    response = TestClient(get_app()).post(
        "/index/youtube",
        json={"video_urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]},
    )

    assert response.status_code == 502
    assert response.json()["ok"] is False
    assert response.json()["indexed"] == []
    assert response.json()["failed"][0]["error"] == "provider unavailable"


def test_hot_media_post_is_202_and_acquirer_is_the_only_serial_writer(
    tmp_path,
    monkeypatch,
):
    from src.ingest_v2.cloud.diarization_indexer.canonical_media import HotMediaSpec
    from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
        IngestionEffect,
        IngestionJob,
        dispose_engine,
        get_engine,
        init_db,
    )
    from src.ingest_v2.cloud.diarization_indexer.channel_service_worker import (
        process_next_hot_media_job,
    )

    database_path = tmp_path / "concurrent-direct-index.sqlite3"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database_path}"
    )
    dispose_engine()
    init_db()
    headers = {
        "x-icmfyi-user-id": f"usr_{'a' * 64}",
        "x-icmfyi-tenant-id": f"ten_{'b' * 64}",
    }
    media_path = tmp_path / "retained.mp4"
    media_path.write_bytes(b"test-video")
    spec = HotMediaSpec(
        path=media_path,
        sha256="c" * 64,
        size_bytes=media_path.stat().st_size,
        mime_type="video/mp4",
    )
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    errors: list[Exception] = []

    def fake_acquire(url: str, video_id: str) -> HotMediaSpec:
        calls.append(video_id)
        entered.set()
        assert release.wait(timeout=10)
        return spec

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.channel_service_worker.acquire_youtube_hot_media",
        fake_acquire,
    )
    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.verify_hot_media",
        lambda value: value,
    )

    def fake_index(**kwargs):
        retained = kwargs["media_acquire"](kwargs["video_url"], "dQw4w9WgXcQ")
        return {
            "video_id": "dQw4w9WgXcQ",
            "media_id": "vid_test",
            "segments": 1,
            "clip_ready": retained == spec,
        }

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.service.index_youtube_video_captions",
        fake_index,
    )
    client = TestClient(get_app())
    payload = {
        "video_urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        "clip_ready": True,
    }
    first = client.post("/index/youtube", headers=headers, json=payload)
    second = client.post("/index/youtube", headers=headers, json=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["pending"] == second.json()["pending"]
    job_id = first.json()["pending"][0]["job_id"]
    assert calls == []
    with Session(get_engine()) as session:
        assert session.scalar(select(func.count()).select_from(IngestionJob)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionEffect)) == 0

    assert (
        client.get(f"/v1/ingestion-jobs/{job_id}", headers=headers).json()["status"]
        == "queued"
    )
    other_headers = {
        "x-icmfyi-user-id": f"usr_{'c' * 64}",
        "x-icmfyi-tenant-id": f"ten_{'d' * 64}",
    }
    assert (
        client.get(f"/v1/ingestion-jobs/{job_id}", headers=other_headers).status_code
        == 404
    )

    def first_worker() -> None:
        try:
            assert process_next_hot_media_job(worker_id="acquirer-a") is True
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=first_worker)
    thread.start()
    assert entered.wait(timeout=10)
    assert process_next_hot_media_job(worker_id="acquirer-b") is False
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []

    ready = client.get(f"/v1/ingestion-jobs/{job_id}", headers=headers)
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    reused = client.post("/index/youtube", headers=headers, json=payload)
    assert reused.status_code == 200
    assert reused.json()["indexed"][0]["clip_ready"] is True
    assert calls == ["dQw4w9WgXcQ"]
    with Session(get_engine()) as session:
        job = session.execute(select(IngestionJob)).scalar_one()
        effect = session.execute(select(IngestionEffect)).scalar_one()
        assert job.status == "succeeded"
        assert job.lease_owner is None
        assert effect.status == "succeeded"
        assert effect.provider_effect_id == "dQw4w9WgXcQ"
    dispose_engine()
