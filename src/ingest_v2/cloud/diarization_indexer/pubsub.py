from __future__ import annotations

import logging
import os
from typing import List, Mapping, Optional

from fastapi import HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from starlette.datastructures import Headers

LOG = logging.getLogger(__name__)

_FALSEY = {"0", "false", "no", "off", "disabled"}


def _env_var_truthy(key: str, *, default: bool = True) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() not in _FALSEY


def verify_pubsub_push(
    headers: Mapping[str, str],
    *,
    audience: Optional[str],
) -> dict:
    """
    Validate the Authorization header of a Pub/Sub push request.

    When ``PUBSUB_VERIFY_SIGNATURE`` is falsey (0, false, no, off, disabled) we skip
    validation entirely. This is primarily used for local testing.

    Otherwise we require a bearer token and verify it using the Google public keys.
    The expected audience defaults to the request URL supplied by the caller but can
    be overridden with ``PUBSUB_TOKEN_AUDIENCE``.
    """

    if not _env_var_truthy("PUBSUB_VERIFY_SIGNATURE", default=True):
        return {}

    audience_override = os.getenv("PUBSUB_TOKEN_AUDIENCE")
    allowed_audiences: List[str] = []

    def _append_candidate(value: Optional[str]) -> None:
        if not value:
            return
        candidate = value.strip()
        if not candidate:
            return

        candidates = [candidate]
        if candidate.startswith("http://"):
            candidates.append("https://" + candidate[len("http://") :])

        for item in candidates:
            if item and item not in allowed_audiences:
                allowed_audiences.append(item)

    if audience_override:
        for candidate in audience_override.split(","):
            _append_candidate(candidate)

    _append_candidate(audience)

    if not allowed_audiences:
        raise RuntimeError(
            "PUBSUB_TOKEN_AUDIENCE must be set (or audience provided) when "
            "PUBSUB_VERIFY_SIGNATURE is enabled."
        )

    header_obj = Headers(headers)
    auth_header = header_obj.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    verify_request = google_requests.Request()
    last_error: Optional[Exception] = None
    claims = None
    for candidate in allowed_audiences:
        try:
            claims = id_token.verify_oauth2_token(token, verify_request, audience=candidate)
            break
        except ValueError as exc:
            last_error = exc
            continue
        except Exception as exc:  # pragma: no cover - network errors simply map to 401
            LOG.warning("Pub/Sub token verification failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Pub/Sub token") from exc

    if claims is None:
        msg = f"Token audience mismatch; expected one of {allowed_audiences}"
        if last_error:
            LOG.warning("%s (%s)", msg, last_error)
        else:
            LOG.warning("%s", msg)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Pub/Sub token")

    expected_issuer = os.getenv("PUBSUB_TOKEN_ISSUER")
    if expected_issuer and claims.get("iss") != expected_issuer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unexpected token issuer")

    expected_principal = os.getenv("PUBSUB_SERVICE_ACCOUNT")
    if expected_principal:
        token_principal = claims.get("email") or claims.get("sub")
        if token_principal != expected_principal:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unexpected token principal")

    return claims
