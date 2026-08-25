from __future__ import annotations

import pytest

import src.ingest_v2.cloud.diarization_indexer.channel_service_readiness as readiness
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    SYSTEM_COMMERCE_SCOPE,
    gateway_commerce_scope,
)


def test_readiness_restores_gateway_scope_after_success(monkeypatch) -> None:
    gateway_scope = gateway_commerce_scope(
        tenant_id=f"ten_{'a' * 64}",
        principal_user_id=f"usr_{'a' * 64}",
    )
    observed = []
    monkeypatch.setattr(
        readiness,
        "set_commerce_scope",
        lambda _session, scope: observed.append(scope),
    )
    monkeypatch.setattr(
        readiness,
        "_compute_readiness",
        lambda _session, *, persist: {"persist": persist},
    )

    result = readiness.compute_readiness(
        object(),
        persist=True,
        restore_commerce_scope=gateway_scope,
    )

    assert result == {"persist": True}
    assert observed == [SYSTEM_COMMERCE_SCOPE, gateway_scope]


def test_readiness_restores_gateway_scope_after_failure(monkeypatch) -> None:
    gateway_scope = gateway_commerce_scope(
        tenant_id=f"ten_{'b' * 64}",
        principal_user_id=f"usr_{'b' * 64}",
    )
    observed = []
    monkeypatch.setattr(
        readiness,
        "set_commerce_scope",
        lambda _session, scope: observed.append(scope),
    )

    def fail(_session, *, persist):
        assert persist is False
        raise RuntimeError("readiness failed")

    monkeypatch.setattr(readiness, "_compute_readiness", fail)

    with pytest.raises(RuntimeError, match="readiness failed"):
        readiness.compute_readiness(
            object(),
            persist=False,
            restore_commerce_scope=gateway_scope,
        )

    assert observed == [SYSTEM_COMMERCE_SCOPE, gateway_scope]
