import base64
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


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
    monkeypatch.setattr("src.ingest_v2.cloud.diarization_indexer.service.verify_pubsub_push", mock)
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
        "src.ingest_v2.cloud.diarization_indexer.service.create_ingest_service", lambda *args, **kwargs: mock_service
    )

    payload = build_pubsub_payload(
        namespace="unknown",
        body={
            "mp3_uri": "gs://bucket/foo.mp3",
            "diarized_uri": "gs://bucket/bar.json",
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
        "src.ingest_v2.cloud.diarization_indexer.service.create_ingest_service", lambda *args, **kwargs: mock_service
    )

    payload = build_pubsub_payload(
        namespace="videos",
        body={
            "mp3_uri": "gs://bucket/youtube_audio/2024-01-01_abc123/title.mp3",
            "diarized_uri": "gs://bucket/youtube_diarized/abc123/abc123_diarized.json",
            "entities_uri": "gs://bucket/youtube_diarized/abc123/abc123_entities.json",
        },
    )

    resp = client.post("/pubsub/push", json=payload)
    assert resp.status_code == 204
    mock_service.handle_event.assert_called_once()
