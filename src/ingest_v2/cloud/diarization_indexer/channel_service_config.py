from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Mapping

from sqlalchemy.engine import make_url


PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})
INTERNAL_SECRET_ENV = "CHANNEL_SERVICE_INTERNAL_SHARED_SECRET"
INTERNAL_SECRET_HEADER = "x-icmfyi-internal-secret"
INTERNAL_USER_HEADER = "x-icmfyi-user-id"
INTERNAL_TENANT_HEADER = "x-icmfyi-tenant-id"


class ChannelServiceConfigurationError(RuntimeError):
    """Raised when the channel service would start with an unsafe configuration."""


@dataclass(frozen=True)
class InternalRequestIdentity:
    """Identity asserted by the authenticated app gateway over the private network."""

    user_id: str
    tenant_id: str


def runtime_environment() -> str:
    return (
        (os.getenv("CHANNEL_SERVICE_ENV") or os.getenv("ICMFYI_ENV") or "development")
        .strip()
        .lower()
    )


def is_production_environment() -> bool:
    return runtime_environment() in PRODUCTION_ENVIRONMENTS


def normalize_database_url(raw_url: str) -> str:
    """Use psycopg 3 for PostgreSQL while retaining explicit driver URLs."""
    value = str(raw_url or "").strip()
    if value.startswith("postgres://"):
        return f"postgresql+psycopg://{value[len('postgres://') :]}"
    if value.startswith("postgresql://"):
        return f"postgresql+psycopg://{value[len('postgresql://') :]}"
    return value


def channel_service_database_url() -> str:
    explicit = (os.getenv("CHANNEL_SERVICE_DATABASE_URL") or "").strip()
    if is_production_environment() and not explicit:
        raise ChannelServiceConfigurationError(
            "production requires CHANNEL_SERVICE_DATABASE_URL; DATABASE_URL and SQLite fallbacks are disabled"
        )
    return normalize_database_url(
        explicit or os.getenv("DATABASE_URL") or "sqlite:///./.local-data/channel-service.db"
    )


def internal_shared_secret() -> str:
    return (os.getenv(INTERNAL_SECRET_ENV) or "").strip()


def validate_production_runtime() -> None:
    """Validate only production-critical settings without disclosing their values."""
    if not is_production_environment():
        return

    url = make_url(channel_service_database_url())
    if url.get_backend_name() != "postgresql":
        raise ChannelServiceConfigurationError(
            "production requires a PostgreSQL CHANNEL_SERVICE_DATABASE_URL"
        )

    secret = internal_shared_secret()
    if len(secret) < 32:
        raise ChannelServiceConfigurationError(
            f"production requires {INTERNAL_SECRET_ENV} with at least 32 characters"
        )


def is_internal_auth_exempt_path(path: str) -> bool:
    """Paths with their own authentication or intentionally public health/catalog semantics."""
    normalized = f"/{str(path or '').lstrip('/')}"
    if normalized == "/healthz":
        return True
    if normalized == "/pubsub/push":
        return True
    if normalized == "/v1/channel-packs/acp/offerings":
        return True
    if normalized.startswith("/v1/channel-packs/acp/jobs"):
        return True
    return False


def internal_request_is_authorized(path: str, headers: Mapping[str, str]) -> bool:
    """Validate the gateway-to-ingestion secret for non-public production routes."""
    if not is_production_environment() or is_internal_auth_exempt_path(path):
        return True

    expected = internal_shared_secret()
    presented = _header_value(headers, INTERNAL_SECRET_HEADER)
    return bool(expected and presented and secrets.compare_digest(presented, expected))


def forwarded_internal_identity(headers: Mapping[str, str]) -> InternalRequestIdentity:
    """
    Read tenant identity only from gateway-authenticated headers.

    Tenant-scoped handlers must call this after the internal-secret middleware and
    must not accept a tenant or user identity from request bodies or query strings.
    """
    user_id = _bounded_identity_header(headers, INTERNAL_USER_HEADER)
    tenant_id = _bounded_identity_header(headers, INTERNAL_TENANT_HEADER)
    return InternalRequestIdentity(user_id=user_id, tenant_id=tenant_id)


def _bounded_identity_header(headers: Mapping[str, str], name: str) -> str:
    value = _header_value(headers, name)
    if not value or len(value) > 255 or any(ord(character) < 32 for character in value):
        raise ChannelServiceConfigurationError(f"missing or invalid {name}")
    return value


def _header_value(headers: Mapping[str, str], name: str) -> str:
    direct = headers.get(name)
    if direct is not None:
        return str(direct).strip()
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value).strip()
    return ""
