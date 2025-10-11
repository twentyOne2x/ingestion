import pytest
from fastapi import HTTPException


def test_verification_disabled_skips_all_checks(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.setenv("PUBSUB_VERIFY_SIGNATURE", "0")

    # Should not raise even with missing headers/audience.
    verify_pubsub_push({}, audience=None)


def test_missing_bearer_token_raises(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.delenv("PUBSUB_VERIFY_SIGNATURE", raising=False)
    monkeypatch.setenv("PUBSUB_TOKEN_AUDIENCE", "https://example.com/pubsub")

    with pytest.raises(HTTPException) as exc:
        verify_pubsub_push({}, audience=None)
    assert exc.value.status_code == 401


def test_principal_mismatch_results_in_forbidden(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.delenv("PUBSUB_VERIFY_SIGNATURE", raising=False)
    monkeypatch.setenv("PUBSUB_TOKEN_AUDIENCE", "https://example.com/pubsub")
    monkeypatch.setenv("PUBSUB_SERVICE_ACCOUNT", "expected@project.iam.gserviceaccount.com")

    def fake_verify(token, request, audience):
        assert audience == "https://example.com/pubsub"
        return {"email": "other@project.iam.gserviceaccount.com", "iss": "https://accounts.google.com"}

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.pubsub.id_token.verify_oauth2_token",
        fake_verify,
    )

    with pytest.raises(HTTPException) as exc:
        verify_pubsub_push({"Authorization": "Bearer token"}, audience=None)
    assert exc.value.status_code == 403


def test_successful_verification_returns_claims(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.delenv("PUBSUB_VERIFY_SIGNATURE", raising=False)
    monkeypatch.setenv("PUBSUB_TOKEN_AUDIENCE", "https://example.com/pubsub")
    monkeypatch.setenv("PUBSUB_SERVICE_ACCOUNT", "expected@project.iam.gserviceaccount.com")
    monkeypatch.setenv("PUBSUB_TOKEN_ISSUER", "https://accounts.google.com")

    def fake_verify(token, request, audience):
        assert audience == "https://example.com/pubsub"
        return {"email": "expected@project.iam.gserviceaccount.com", "iss": "https://accounts.google.com"}

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.pubsub.id_token.verify_oauth2_token",
        fake_verify,
    )

    claims = verify_pubsub_push({"Authorization": "Bearer token"}, audience=None)
    assert claims["email"] == "expected@project.iam.gserviceaccount.com"


def test_multiple_audiences(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.delenv("PUBSUB_VERIFY_SIGNATURE", raising=False)
    monkeypatch.setenv(
        "PUBSUB_TOKEN_AUDIENCE",
        "https://example.com/base, https://example.com/pubsub",
    )
    monkeypatch.setenv("PUBSUB_SERVICE_ACCOUNT", "expected@project.iam.gserviceaccount.com")
    monkeypatch.setenv("PUBSUB_TOKEN_ISSUER", "https://accounts.google.com")

    attempts = []

    def fake_verify(token, request, audience):
        attempts.append(audience)
        if audience == "https://example.com/pubsub":
            return {"email": "expected@project.iam.gserviceaccount.com", "iss": "https://accounts.google.com"}
        raise ValueError("Token has wrong audience")

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.pubsub.id_token.verify_oauth2_token",
        fake_verify,
    )

    claims = verify_pubsub_push({"Authorization": "Bearer token"}, audience="https://fallback/audience")
    assert claims["email"] == "expected@project.iam.gserviceaccount.com"
    assert attempts == ["https://example.com/base", "https://example.com/pubsub"]


def test_request_url_http_scheme_is_normalized(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.pubsub import verify_pubsub_push

    monkeypatch.delenv("PUBSUB_VERIFY_SIGNATURE", raising=False)
    monkeypatch.setenv("PUBSUB_TOKEN_AUDIENCE", "https://example.com/pubsub")
    monkeypatch.setenv("PUBSUB_SERVICE_ACCOUNT", "expected@project.iam.gserviceaccount.com")
    monkeypatch.setenv("PUBSUB_TOKEN_ISSUER", "https://accounts.google.com")

    def fake_verify(token, request, audience):
        assert audience == "https://example.com/pubsub"
        return {"email": "expected@project.iam.gserviceaccount.com", "iss": "https://accounts.google.com"}

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.pubsub.id_token.verify_oauth2_token",
        fake_verify,
    )

    claims = verify_pubsub_push({"Authorization": "Bearer token"}, audience="http://example.com/pubsub")
    assert claims["email"] == "expected@project.iam.gserviceaccount.com"
