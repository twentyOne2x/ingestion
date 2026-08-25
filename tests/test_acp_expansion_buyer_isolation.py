from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer import channel_service_acp as acp
from src.ingest_v2.cloud.diarization_indexer import channel_service_logic as logic
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    ACP_COMMERCE_SCOPE,
    SYSTEM_COMMERCE_SCOPE,
    AcpJobBridge,
    Base,
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    Entitlement,
    commerce_ownership_values,
)

OWNER = ("acp_client", "0x" + "ab" * 20)
ATTACKER = ("acp_client", "0x" + "cd" * 20)
PACK_ID = "pack_acp_owner"


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as database:
        yield database


def _seed_pack(
    session: Session,
    *,
    pack_id: str = PACK_ID,
    buyer: tuple[str, str] = OWNER,
    entitlement_status: str = "active",
    pack_status: str = "ready",
    commerce_scope=ACP_COMMERCE_SCOPE,
    create_entitlement: bool = True,
    create_starter_origin: bool = True,
) -> ChannelPack:
    ownership = commerce_ownership_values(commerce_scope)
    pack = ChannelPack(
        **ownership,
        id=pack_id,
        status=pack_status,
        mode="recent_pack",
        namespace="videos",
        channel_handle="@owner",
        resolved_channel_id="channel-owner",
        resolved_channel_name="Owner",
        total_purchased_video_count=10,
        ready_video_count=10,
        batch_count=1,
        manifest_json={},
        export_paths_json={},
    )
    session.add(pack)
    session.flush()
    if create_entitlement:
        session.add(
            Entitlement(
                **ownership,
                id=f"entitlement_{pack_id}_{buyer[1]}",
                pack_id=pack.id,
                subject_type=buyer[0],
                subject_id=buyer[1],
                status=entitlement_status,
            )
        )
        session.flush()
    if create_starter_origin:
        order_id = f"order_starter_{pack_id}"
        session.add(
            ChannelOrder(
                **ownership,
                id=order_id,
                quote_id=f"quote_starter_{pack_id}",
                checkout_session_id=f"checkout_starter_{pack_id}",
                pack_id=pack.id,
                batch_id=f"batch_starter_{pack_id}",
                status="ready",
                payment_status="settled_acp",
                payment_provider="acp",
                amount_cents=1,
                currency="USD",
                notes_json={},
            )
        )
        session.add(
            AcpJobBridge(
                **ownership,
                acp_job_id=f"job_starter_{pack_id}",
                offering_id="transcript_pack_starter_10",
                status="ready_for_delivery",
                order_id=order_id,
                pack_id=pack.id,
                fixed_price_cents=1,
                currency="USD",
                payment_provider="acp",
                payment_status="settled_acp",
                buyer_subject_type=buyer[0],
                buyer_subject_id=buyer[1],
                request_json={
                    "acp_job_id": f"job_starter_{pack_id}",
                    "offering_id": "transcript_pack_starter_10",
                    "channel_handle": "@owner",
                    "pack_id": None,
                    "mode": "recent_pack",
                    "namespace": "videos",
                    "buyer_subject_type": buyer[0],
                    "buyer_subject_id": buyer[1],
                },
                delivery_json={"pack_id": pack.id},
            )
        )
        session.flush()
    return pack


def _payload(
    *,
    job_id: str,
    pack_id: str = PACK_ID,
    buyer: tuple[str, str] = OWNER,
    payment_status: str = "settled_acp",
) -> dict:
    return {
        "acp_job_id": job_id,
        "offering_id": "transcript_pack_expansion_25",
        "input": {
            "channel_handle": "@owner",
            "pack_id": pack_id,
            "max_videos": 1,
            "namespace": "videos",
            "language": "en",
            "prefer_auto": True,
        },
        "buyer": {"subject_type": buyer[0], "subject_id": buyer[1]},
        "payment": {"provider": "icmfyi-acp", "status": payment_status},
    }


def _ready_plan(**kwargs) -> logic.QuotePlan:
    row = {
        "position": 1,
        "batch_index": 2,
        "status": "included",
        "video_id": "video-expansion-1",
        "title": "Expansion video",
        "description": "",
        "channel_name": "Owner",
        "channel_handle": "@owner",
        "published_at": "2026-08-25",
        "duration_s": 120.0,
        "video_url": "https://www.youtube.com/watch?v=video-expansion-1",
        "thumbnail_url": None,
        "transcript_source": "indexed",
        "indexed_parent_id": "parent-expansion-1",
    }
    return logic.QuotePlan(
        channel_handle=kwargs["channel_handle"],
        channel_name="Owner",
        channel_id="channel-owner",
        namespace=kwargs["namespace"],
        mode=kwargs["mode"],
        included_rows=[row],
        pending_rows=[],
        excluded_rows=[],
        batch_plan=[{"batch_index": 2, "video_count": 1}],
        per_video_cents=500,
        current_batch_index=2,
        current_batch_amount_cents=500,
        current_batch_video_count=1,
        total_included_amount_cents=500,
        estimated_ready_minutes=1,
        eta_confidence="high",
        recommended_starter_batch_size=1,
        existing_pack_id=kwargs["pack_id"],
        existing_batch_count=1,
    )


def test_shared_acp_realm_does_not_reveal_or_expand_another_buyers_pack(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_pack(session)
    _seed_pack(
        session,
        pack_id="pack_inactive_entitlement",
        entitlement_status="revoked",
    )
    _seed_pack(session, pack_id="pack_bad_status", pack_status="failed")
    _seed_pack(
        session,
        pack_id="pack_system_realm",
        commerce_scope=SYSTEM_COMMERCE_SCOPE,
    )
    session.add(
        Entitlement(
            **commerce_ownership_values(ACP_COMMERCE_SCOPE),
            id="entitlement_forged_attacker",
            pack_id=PACK_ID,
            subject_type=ATTACKER[0],
            subject_id=ATTACKER[1],
            status="active",
        )
    )
    session.flush()
    entitlement_count = session.scalar(select(func.count()).select_from(Entitlement))
    bridge_count = session.scalar(select(func.count()).select_from(AcpJobBridge))
    order_count = session.scalar(select(func.count()).select_from(ChannelOrder))
    planning_calls = 0

    def _unexpected_plan(**kwargs):
        nonlocal planning_calls
        planning_calls += 1
        raise AssertionError("unauthorized expansion reached quote planning")

    monkeypatch.setattr(logic, "plan_quote", _unexpected_plan)

    attempts = [
        _payload(job_id="job-attacker", buyer=ATTACKER),
        _payload(
            job_id="job-wrong-subject-type",
            buyer=("different_subject_type", OWNER[1]),
        ),
        _payload(job_id="job-unknown", pack_id="pack_missing", buyer=ATTACKER),
        _payload(job_id="job-inactive", pack_id="pack_inactive_entitlement"),
        _payload(job_id="job-bad-status", pack_id="pack_bad_status"),
        _payload(job_id="job-wrong-realm", pack_id="pack_system_realm"),
        {
            **_payload(job_id="job-no-buyer"),
            "buyer": {"subject_type": OWNER[0], "subject_id": ""},
        },
        {
            **_payload(job_id="job-wrong-channel"),
            "input": {
                **_payload(job_id="unused")["input"],
                "channel_handle": "@other",
            },
        },
        {
            **_payload(job_id="job-wrong-namespace"),
            "input": {
                **_payload(job_id="unused")["input"],
                "namespace": "other-namespace",
            },
        },
    ]
    errors = []
    for payload in attempts:
        with pytest.raises(ValueError) as exc_info:
            acp.create_or_sync_acp_job(
                session=session, payload=payload, base_url="https://service.test"
            )
        errors.append(str(exc_info.value))

    assert errors == ["Expansion pack is not available"] * len(attempts)
    assert planning_calls == 0
    assert (
        session.scalar(select(func.count()).select_from(AcpJobBridge)) == bridge_count
    )
    assert session.scalar(select(func.count()).select_from(ChannelQuote)) == 0
    assert session.scalar(select(func.count()).select_from(ChannelOrder)) == order_count
    assert (
        session.scalar(select(func.count()).select_from(Entitlement))
        == entitlement_count
    )


def test_expansion_rechecks_active_entitlement_before_ordering(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_pack(session)
    monkeypatch.setattr(logic, "plan_quote", _ready_plan)

    bridge = acp.create_or_sync_acp_job(
        session=session,
        payload=_payload(job_id="job-revoked", payment_status="pending"),
        base_url="https://service.test",
    )
    assert bridge.order_id is None
    order_count = session.scalar(select(func.count()).select_from(ChannelOrder))

    entitlement = session.scalar(select(Entitlement))
    assert entitlement is not None
    entitlement.status = "revoked"
    bridge.payment_status = "settled_acp"
    session.flush()

    refresh_calls = 0

    def _unexpected_refresh(**kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        raise AssertionError("revoked expansion refreshed its quote")

    monkeypatch.setattr(logic, "refresh_quote_state", _unexpected_refresh)
    with pytest.raises(ValueError, match="^Expansion pack is not available$"):
        acp.refresh_acp_job(
            session=session, bridge=bridge, base_url="https://service.test"
        )

    assert refresh_calls == 0
    assert session.scalar(select(func.count()).select_from(CheckoutSessionRecord)) == 0
    assert session.scalar(select(func.count()).select_from(ChannelOrder)) == order_count


def test_authorized_legacy_quote_is_safely_bound_to_its_bridge_buyer(
    session: Session,
) -> None:
    _seed_pack(session)
    payload = _payload(job_id="job-legacy", payment_status="pending")
    offering = acp.get_acp_offering(payload["offering_id"])
    normalized = acp.normalize_acp_job_payload(payload=payload, offering=offering)
    legacy_quote_request = {
        "channel_handle": normalized["channel_handle"],
        "max_videos": normalized["max_videos"],
        "namespace": normalized["namespace"],
        "mode": normalized["mode"],
        "language": normalized["language"],
        "prefer_auto": normalized["prefer_auto"],
        "pack_id": normalized["pack_id"],
        "published_after": normalized["published_after"],
        "published_before": normalized["published_before"],
        "source": "acp",
        "acp_job_id": normalized["acp_job_id"],
        "offering_id": normalized["offering_id"],
    }
    quote = logic.persist_quote(
        session=session,
        commerce_scope=ACP_COMMERCE_SCOPE,
        request_payload=legacy_quote_request,
        plan=_ready_plan(
            channel_handle="@owner",
            namespace="videos",
            mode="recent_pack",
            pack_id=PACK_ID,
        ),
    )
    bridge = AcpJobBridge(
        **commerce_ownership_values(ACP_COMMERCE_SCOPE),
        acp_job_id=normalized["acp_job_id"],
        offering_id=normalized["offering_id"],
        status="received",
        quote_id=quote.id,
        fixed_price_cents=offering.fixed_price_cents,
        currency="USD",
        payment_provider=normalized["payment_provider"],
        payment_status=normalized["payment_status"],
        buyer_subject_type=normalized["buyer_subject_type"],
        buyer_subject_id=normalized["buyer_subject_id"],
        request_json=normalized,
        delivery_json={},
    )
    session.add(bridge)
    session.flush()

    acp.refresh_acp_job(session=session, bridge=bridge, base_url="https://service.test")

    assert quote.request_json["buyer_subject_type"] == OWNER[0]
    assert quote.request_json["buyer_subject_id"] == OWNER[1]
    assert bridge.order_id is None


def test_entitlement_created_after_hostile_quote_cannot_authorize_it(
    session: Session,
) -> None:
    pack = _seed_pack(session, create_entitlement=False)
    payload = _payload(job_id="job-stored-hostile")
    offering = acp.get_acp_offering(payload["offering_id"])
    normalized = acp.normalize_acp_job_payload(payload=payload, offering=offering)
    quote_request = {
        key: normalized[key]
        for key in (
            "channel_handle",
            "max_videos",
            "namespace",
            "mode",
            "language",
            "prefer_auto",
            "pack_id",
            "published_after",
            "published_before",
            "acp_job_id",
            "offering_id",
            "buyer_subject_type",
            "buyer_subject_id",
        )
    }
    quote_request["source"] = "acp"
    quote = logic.persist_quote(
        session=session,
        commerce_scope=ACP_COMMERCE_SCOPE,
        request_payload=quote_request,
        plan=_ready_plan(
            channel_handle="@owner",
            namespace="videos",
            mode="recent_pack",
            pack_id=PACK_ID,
        ),
    )
    ownership = commerce_ownership_values(ACP_COMMERCE_SCOPE)
    session.add(
        Entitlement(
            **ownership,
            id="entitlement_postdated_hostile_quote",
            pack_id=pack.id,
            subject_type=OWNER[0],
            subject_id=OWNER[1],
            status="active",
        )
    )
    bridge = AcpJobBridge(
        **ownership,
        acp_job_id=normalized["acp_job_id"],
        offering_id=normalized["offering_id"],
        status="received",
        quote_id=quote.id,
        fixed_price_cents=offering.fixed_price_cents,
        currency="USD",
        payment_provider=normalized["payment_provider"],
        payment_status="settled_acp",
        buyer_subject_type=normalized["buyer_subject_type"],
        buyer_subject_id=normalized["buyer_subject_id"],
        request_json=normalized,
        delivery_json={},
    )
    session.add(bridge)
    session.flush()

    with pytest.raises(ValueError, match="^Expansion pack is not available$"):
        acp.refresh_acp_job(
            session=session, bridge=bridge, base_url="https://service.test"
        )

    assert session.scalar(select(func.count()).select_from(CheckoutSessionRecord)) == 0
    assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 1
    assert pack.total_purchased_video_count == 10


def test_same_buyer_expansion_and_exact_replay_preserve_one_lineage(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_pack(session)
    planning_calls = 0

    def _plan(**kwargs):
        nonlocal planning_calls
        planning_calls += 1
        return _ready_plan(**kwargs)

    monkeypatch.setattr(logic, "plan_quote", _plan)
    monkeypatch.setattr(logic, "child_segments_by_parent", lambda *args, **kwargs: {})
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    payload = _payload(job_id="job-owner")
    payload["buyer"]["subject_id"] = "0x" + "Ab" * 20

    bridge = acp.create_or_sync_acp_job(
        session=session, payload=payload, base_url="https://service.test"
    )
    quote = session.get(ChannelQuote, bridge.quote_id)
    order = session.get(ChannelOrder, bridge.order_id)
    assert quote is not None
    assert order is not None
    assert bridge.pack_id == PACK_ID
    assert order.pack_id == PACK_ID
    assert quote.request_json["pack_id"] == PACK_ID
    assert quote.request_json["buyer_subject_type"] == OWNER[0]
    assert quote.request_json["buyer_subject_id"] == OWNER[1]
    assert bridge.buyer_subject_id == OWNER[1]
    assert session.scalar(select(func.count()).select_from(ChannelQuote)) == 1
    assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 2
    assert session.scalar(select(func.count()).select_from(Entitlement)) == 2

    first_ids = (
        bridge.quote_id,
        bridge.checkout_session_id,
        bridge.order_id,
        bridge.pack_id,
    )
    replay = acp.create_or_sync_acp_job(
        session=session, payload=payload, base_url="https://service.test"
    )

    assert planning_calls == 1
    assert (
        replay.quote_id,
        replay.checkout_session_id,
        replay.order_id,
        replay.pack_id,
    ) == first_ids
    assert session.scalar(select(func.count()).select_from(AcpJobBridge)) == 2
    assert session.scalar(select(func.count()).select_from(ChannelQuote)) == 1
    assert session.scalar(select(func.count()).select_from(CheckoutSessionRecord)) == 1
    assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 2
    assert session.scalar(select(func.count()).select_from(Entitlement)) == 2
