from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.canonical_media import HotMediaSpec
from src.ingest_v2.cloud.diarization_indexer.channel_service_commerce import (
    PAYMENT_IDEMPOTENCY_KEY,
    YOUTUBE_INGEST_TOOL,
    CommerceConfigurationError,
    CommerceResolutionError,
    bind_gateway_commerce_quote,
    resolve_authoritative_commerce_quote,
    validate_x402_commerce_runtime,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    ensure_channel_entitlement,
    get_or_create_ingestion_request,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_logic import (
    enforce_direct_order_allowed,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    Base,
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    Entitlement,
    IngestionJob,
    IngestionRequest,
    PackBatch,
    PackVideo,
    PaymentReceipt,
    QuoteVideo,
    SourceChannel,
    SourceVideo,
    Tenant,
    TenantMembership,
    UserAccount,
    commerce_ownership_values,
    dispose_engine,
    gateway_commerce_scope,
    get_engine,
    init_db,
    utcnow,
)
from src.ingest_v2.cloud.diarization_indexer.paid_work_worker import (
    PAID_WORK_SCHEMA,
    PAID_WORK_TOPIC,
    PaidWorkError,
    SettledPaidWorkClaim,
    fulfill_claimed_paid_work,
    payment_worker_database_url,
)
from src.ingest_v2.cloud.diarization_indexer.public_acquisition import (
    AcquiredPublicItem,
    PublicItemDescriptor,
)
from src.ingest_v2.cloud.diarization_indexer.public_ingestion_worker import (
    PublicWorkerDependencies,
    process_next_public_ingestion_job,
)
from src.ingest_v2.cloud.diarization_indexer.transcription_runtime import (
    TranscriptResult,
)

TENANT_ID = f"ten_{'a' * 64}"
PRINCIPAL_ID = f"usr_{'a' * 64}"
OTHER_TENANT_ID = f"ten_{'b' * 64}"
OTHER_PRINCIPAL_ID = f"usr_{'b' * 64}"
SAME_TENANT_PRINCIPAL_ID = f"usr_{'c' * 64}"
ASSET = "eip155:8453/erc20:0x0000000000000000000000000000000000000001"


def _successful_qdrant_publication(**kwargs) -> dict:
    return {
        "collection": "icmfyi-v2__canonical",
        "media_id": kwargs["media_id"],
        "transcript_revision_id": kwargs["transcript_revision_id"],
        "point_count": 1,
        "readback_sha256": "f" * 64,
    }


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'paid-work.sqlite3'}")
    Base.metadata.create_all(engine)
    return engine


def _seed_quote(session: Session, monkeypatch: pytest.MonkeyPatch) -> ChannelQuote:
    monkeypatch.setenv("CHANNEL_SERVICE_X402_COMMERCE_ENABLED", "true")
    monkeypatch.setenv("CHANNEL_SERVICE_X402_ASSET", ASSET)
    monkeypatch.setenv("CHANNEL_SERVICE_X402_ATOMIC_UNITS_PER_CENT", "10000")
    for tenant_id, principal_id, slug in (
        (TENANT_ID, PRINCIPAL_ID, "alpha"),
        (OTHER_TENANT_ID, OTHER_PRINCIPAL_ID, "beta"),
    ):
        session.add(
            UserAccount(
                id=principal_id,
                auth_provider="test",
                auth_subject=principal_id,
            )
        )
        session.add(Tenant(id=tenant_id, slug=slug, display_name=slug.title()))
        session.flush()
        session.add(
            TenantMembership(
                tenant_id=tenant_id,
                user_id=principal_id,
                role="owner",
            )
        )
    session.flush()

    scope = gateway_commerce_scope(tenant_id=TENANT_ID, principal_user_id=PRINCIPAL_ID)
    ownership = commerce_ownership_values(scope)
    quote = ChannelQuote(
        **ownership,
        id="quote_paid_1",
        status="open",
        mode="recent_pack",
        namespace="videos",
        channel_handle="@example",
        resolved_channel_id="UCexample",
        resolved_channel_name="Example",
        requested_max_videos=1,
        included_video_count=1,
        excluded_video_count=0,
        current_batch_index=1,
        current_batch_video_count=1,
        current_batch_amount_cents=100,
        total_included_amount_cents=100,
        per_video_cents=100,
        estimated_ready_minutes=5,
        eta_confidence="high",
        recommended_starter_batch_size=1,
        planning_latency_ms=1,
        request_json={
            "channel_handle": "@example",
            "max_videos": 1,
            "mode": "recent_pack",
            "namespace": "videos",
            "language": "en",
            "prefer_auto": True,
            "pack_id": None,
        },
        batch_plan_json=[
            {
                "batch_index": 1,
                "billable_video_count": 1,
                "amount_cents": 100,
            }
        ],
        price_breakdown_json={"currency": "USD", "amount_cents": 100},
        commerce_json={},
        expires_at=utcnow() + timedelta(minutes=30),
    )
    quote.videos.append(
        QuoteVideo(
            **ownership,
            position=1,
            batch_index=1,
            included=True,
            video_id="dQw4w9WgXcQ",
            title="Example",
            video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            status="included",
        )
    )
    session.add(quote)
    session.flush()
    bind_gateway_commerce_quote(quote, scope)
    session.flush()
    return quote


def _claim_for(
    quote: ChannelQuote,
    *,
    tenant_id: str,
    principal_id: str,
    ordinal: int,
) -> SettledPaidWorkClaim:
    idempotency_key = f"paid-work-{ordinal}"
    stored = {
        **dict(quote.commerce_json),
        PAYMENT_IDEMPOTENCY_KEY: idempotency_key,
    }
    quote.commerce_json = stored
    payload = {
        "schema": PAID_WORK_SCHEMA,
        "tenantId": tenant_id,
        "principalId": principal_id,
        "toolName": YOUTUBE_INGEST_TOOL,
        "idempotencyKey": idempotency_key,
        "requestHash": stored["requestHash"],
        "commerce": {
            "provider": "icmfyi-acp",
            "quoteId": quote.id,
            "offeringId": stored["offeringId"],
            "quoteHash": stored["quoteHash"],
        },
        "work": copy.deepcopy(stored["workPayload"]),
    }
    return SettledPaidWorkClaim(
        outbox_id=f"10000000-0000-4000-8000-{ordinal:012d}",
        intent_id=f"20000000-0000-4000-8000-{ordinal:012d}",
        tenant_id=tenant_id,
        principal_id=principal_id,
        topic=PAID_WORK_TOPIC,
        idempotency_key=idempotency_key,
        request_hash=stored["requestHash"],
        tool_name=YOUTUBE_INGEST_TOOL,
        commerce_quote_id=quote.id,
        commerce_quote_hash=stored["quoteHash"],
        asset=stored["asset"],
        amount_atomic=Decimal(stored["amountAtomic"]),
        payload=payload,
        settlement_network="eip155:8453",
        settlement_transaction="0x" + f"{ordinal:x}"[-1] * 64,
        settlement_recorded_at=utcnow(),
    )


def _claim(quote: ChannelQuote) -> SettledPaidWorkClaim:
    return _claim_for(
        quote,
        tenant_id=TENANT_ID,
        principal_id=PRINCIPAL_ID,
        ordinal=1,
    )


def test_paid_work_creates_one_owned_deferred_order_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    export_root = tmp_path / "exports"
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    with Session(engine) as session:
        quote = _seed_quote(session, monkeypatch)
        claim = _claim(quote)
        acknowledgements: list[tuple[str, str]] = []
        result = fulfill_claimed_paid_work(
            session,
            claim,
            acknowledge=lambda _session, claimed, order_id: acknowledgements.append(
                (claimed.outbox_id, order_id)
            ),
        )
        session.commit()

        assert result.created is True
        assert acknowledgements == [(claim.outbox_id, result.order_id)]
        assert export_root.exists() is False
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 1
        assert session.scalar(select(func.count()).select_from(ChannelPack)) == 1
        assert session.scalar(select(func.count()).select_from(PackBatch)) == 1
        assert session.scalar(select(func.count()).select_from(PackVideo)) == 1
        assert session.scalar(select(func.count()).select_from(PaymentReceipt)) == 1
        assert session.scalar(select(func.count()).select_from(Entitlement)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionRequest)) == 1
        assert session.scalar(select(func.count()).select_from(IngestionJob)) == 1
        assert len(result.request_ids) == 1

        order = session.get(ChannelOrder, result.order_id)
        checkout = session.get(CheckoutSessionRecord, result.checkout_session_id)
        assert order is not None and checkout is not None
        assert (order.authority_kind, order.tenant_id, order.principal_user_id) == (
            "gateway",
            TENANT_ID,
            PRINCIPAL_ID,
        )
        assert order.payment_provider == "x402"
        assert order.payment_status == "settled_x402"
        assert order.status == "queued"
        assert checkout.status == "completed"
        assert checkout.total_amount_cents == 1
        assert session.scalars(select(PackVideo)).one().status == "queued"

        replay_acks: list[str] = []
        replay = fulfill_claimed_paid_work(
            session,
            claim,
            acknowledge=lambda _session, _claimed, order_id: replay_acks.append(
                order_id
            ),
        )
        session.commit()
        assert replay.created is False
        assert replay.order_id == result.order_id
        assert replay_acks == [result.order_id]
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 1
        assert session.scalar(select(func.count()).select_from(PaymentReceipt)) == 1

        duplicate_payload = copy.deepcopy(claim.payload)
        duplicate_payload["idempotencyKey"] = "paid-work-2"
        second_settlement = replace(
            claim,
            outbox_id="10000000-0000-4000-8000-000000000002",
            intent_id="20000000-0000-4000-8000-000000000002",
            idempotency_key="paid-work-2",
            settlement_transaction="0x" + "4" * 64,
            payload=duplicate_payload,
        )
        with pytest.raises(PaidWorkError, match="does not match the authoritative quote"):
            fulfill_claimed_paid_work(
                session, second_settlement, acknowledge=lambda *_args: None
            )
        session.rollback()
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 1
        assert session.scalar(select(func.count()).select_from(PaymentReceipt)) == 1


def test_ack_failure_rolls_back_order_and_retry_creates_exactly_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        claim = _claim(_seed_quote(session, monkeypatch))
        session.commit()

    with Session(engine) as session:
        with pytest.raises(RuntimeError, match="ack failed"):
            fulfill_claimed_paid_work(
                session,
                claim,
                acknowledge=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("ack failed")
                ),
            )
        session.rollback()

    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 0
        result = fulfill_claimed_paid_work(
            session,
            claim,
            acknowledge=lambda *_args: None,
        )
        session.commit()
        assert result.created is True
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 1
        assert session.scalar(select(func.count()).select_from(PaymentReceipt)) == 1


@pytest.mark.parametrize(
    ("mutator", "error"),
    (
        (
            lambda claim: replace(claim, tool_name="icmfyi.ingest.twitch"),
            "topic or tool",
        ),
        (
            lambda claim: replace(claim, idempotency_key="bad,key"),
            "idempotency key",
        ),
        (
            lambda claim: replace(claim, request_hash="0" * 64),
            "payload does not match",
        ),
        (
            lambda claim: replace(
                claim, payload={**claim.payload, "tenantId": OTHER_TENANT_ID}
            ),
            "payload does not match",
        ),
        (
            lambda claim: replace(
                claim,
                payload={**claim.payload, "untrustedWork": {"url": "https://evil"}},
            ),
            "payload shape",
        ),
    ),
)
def test_hostile_paid_work_never_creates_or_acknowledges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    error: str,
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        claim = mutator(_claim(_seed_quote(session, monkeypatch)))
        session.commit()
    acknowledged: list[str] = []
    with Session(engine) as session:
        with pytest.raises(PaidWorkError, match=error):
            fulfill_claimed_paid_work(
                session,
                claim,
                acknowledge=lambda _session, _claim, order_id: acknowledged.append(
                    order_id
                ),
            )
        session.rollback()
        assert acknowledged == []
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 0


def test_cross_principal_claim_cannot_resolve_owned_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        original = _claim(_seed_quote(session, monkeypatch))
        payload = copy.deepcopy(original.payload)
        payload["tenantId"] = OTHER_TENANT_ID
        payload["principalId"] = OTHER_PRINCIPAL_ID
        claim = replace(
            original,
            tenant_id=OTHER_TENANT_ID,
            principal_id=OTHER_PRINCIPAL_ID,
            payload=payload,
        )
        session.commit()
    with Session(engine) as session, pytest.raises(PaidWorkError, match="not owned"):
        fulfill_claimed_paid_work(session, claim, acknowledge=lambda *_args: None)


def test_authoritative_resolver_binds_one_exact_payment_intention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        quote = _seed_quote(session, monkeypatch)
        stored = dict(quote.commerce_json)
        scope = gateway_commerce_scope(
            tenant_id=TENANT_ID, principal_user_id=PRINCIPAL_ID
        )
        resolved = resolve_authoritative_commerce_quote(
            session=session,
            scope=scope,
            quote_id=quote.id,
            tool_name=YOUTUBE_INGEST_TOOL,
            idempotency_key="paid-work-1",
            request_hash=stored["requestHash"],
        )
        assert resolved["quoteHash"] == stored["quoteHash"]
        assert resolved["workPayload"] == stored["workPayload"]
        assert quote.commerce_json[PAYMENT_IDEMPOTENCY_KEY] == "paid-work-1"
        assert session.scalar(select(func.count()).select_from(ChannelOrder)) == 0

        replay = resolve_authoritative_commerce_quote(
            session=session,
            scope=scope,
            quote_id=quote.id,
            tool_name=YOUTUBE_INGEST_TOOL,
            idempotency_key="paid-work-1",
            request_hash=stored["requestHash"],
        )
        assert replay == resolved
        with pytest.raises(CommerceResolutionError, match="different idempotency key"):
            resolve_authoritative_commerce_quote(
                session=session,
                scope=scope,
                quote_id=quote.id,
                tool_name=YOUTUBE_INGEST_TOOL,
                idempotency_key="paid-work-2",
                request_hash=stored["requestHash"],
            )
        with pytest.raises(CommerceResolutionError, match="idempotency key is invalid"):
            resolve_authoritative_commerce_quote(
                session=session,
                scope=scope,
                quote_id=quote.id,
                tool_name=YOUTUBE_INGEST_TOOL,
                idempotency_key="bad,key",
                request_hash=stored["requestHash"],
            )

        other_scope = gateway_commerce_scope(
            tenant_id=OTHER_TENANT_ID, principal_user_id=OTHER_PRINCIPAL_ID
        )
        with pytest.raises(CommerceResolutionError, match="not found"):
            resolve_authoritative_commerce_quote(
                session=session,
                scope=other_scope,
                quote_id=quote.id,
                tool_name=YOUTUBE_INGEST_TOOL,
                idempotency_key="paid-work-1",
                request_hash=stored["requestHash"],
            )


def test_authoritative_resolver_never_retrofits_an_unbound_settled_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    with Session(engine) as session:
        quote = _seed_quote(session, monkeypatch)
        claim = _claim(quote)
        fulfill_claimed_paid_work(
            session,
            claim,
            acknowledge=lambda *_args: None,
        )
        session.flush()
        legacy_projection = dict(quote.commerce_json)
        legacy_projection.pop(PAYMENT_IDEMPOTENCY_KEY)
        quote.commerce_json = legacy_projection
        session.flush()

        scope = gateway_commerce_scope(
            tenant_id=TENANT_ID, principal_user_id=PRINCIPAL_ID
        )
        with pytest.raises(CommerceResolutionError, match="lacks an idempotency binding"):
            resolve_authoritative_commerce_quote(
                session=session,
                scope=scope,
                quote_id=quote.id,
                tool_name=claim.tool_name,
                idempotency_key="fresh-payment-attempt",
                request_hash=claim.request_hash,
            )


def test_security_definer_contract_is_static_and_fail_closed() -> None:
    sql_path = (
        Path(__file__).parents[1] / "sql" / "001_payment_worker_security_definer.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")
    assert sql.count("SECURITY DEFINER") == 3
    assert sql.count("SET search_path = pg_catalog, public") == 3
    assert "BEGIN;" in sql and sql.rstrip().endswith("COMMIT;")
    assert sql.index("BEGIN;") < sql.index("CREATE OR REPLACE FUNCTION")
    assert sql.rindex("REVOKE ALL ON FUNCTION") < sql.rindex("COMMIT;")
    assert "FOR UPDATE OF outbox SKIP LOCKED" in sql
    assert "pg_try_advisory_xact_lock" in sql
    assert "hashtextextended(claimed.claimed_commerce_quote_id, 0)" in sql
    assert "quote.commerce_json ->> 'paymentIdempotencyKey'" in sql
    assert "paymentIdempotencyKey" in sql
    assert "outbox.next_delivery_attempt_at" in sql
    assert "outbox.dead_lettered_at IS NULL" in sql
    assert "intent.status = 'settled'" in sql
    assert "intent.tool_name = 'icmfyi.ingest.youtube'" in sql
    assert "icmfyi.paid-work-request.v1" in sql
    assert "create_settled_channel_pack_order" in sql
    assert "channel_order.payment_status = 'settled_x402'" in sql
    assert "icmfyi_fail_settled_paid_work" in sql
    assert "public.ingestion_requests" in sql
    assert "REVOKE ALL ON FUNCTION" in sql
    assert "FROM PUBLIC" in sql
    assert "GRANT SELECT" not in sql
    assert "EXECUTE format" not in sql


def test_settled_work_reaches_canonical_corpus_and_verified_pack_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical

    database = tmp_path / "paid-e2e.sqlite3"
    hot_root = tmp_path / "hot"
    export_root = tmp_path / "exports"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(hot_root))
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    monkeypatch.setenv(
        "CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT", str(tmp_path / "transcription")
    )
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            claim = _claim(_seed_quote(session, monkeypatch))
            accepted = fulfill_claimed_paid_work(
                session, claim, acknowledge=lambda *_args: None
            )
            session.commit()
            assert accepted.request_ids

        media_path = hot_root / "fixture.mp4"
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(b"paid public item fixture")
        media = HotMediaSpec(
            path=media_path,
            sha256=hashlib.sha256(media_path.read_bytes()).hexdigest(),
            size_bytes=media_path.stat().st_size,
            mime_type="video/mp4",
        )
        monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)
        item = PublicItemDescriptor(
            platform="youtube",
            external_id="dQw4w9WgXcQ",
            channel_external_id="UCexample",
            channel_handle="@example",
            canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Example",
            duration_ms=2_000,
        )

        def extract_audio(*, video_path: Path, audio_path: Path):
            assert video_path == media_path
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"audio")
            return hashlib.sha256(b"audio").hexdigest(), 5

        dependencies = PublicWorkerDependencies(
            acquire=lambda _item: AcquiredPublicItem(item=item, media=media),
            extract_audio=extract_audio,
            transcribe=lambda **_kwargs: TranscriptResult(
                provider="local_cpu:test@fixture",
                provider_request_id=None,
                segments=(
                    {
                        "ordinal": 0,
                        "start_ms": 0,
                        "end_ms": 2_000,
                        "speaker_label": None,
                        "text": "A paid channel transcript.",
                    },
                ),
            ),
            delete_audio=lambda path: path.unlink(missing_ok=True),
            publish_vectors=_successful_qdrant_publication,
        )
        assert process_next_public_ingestion_job(
            worker_id="paid-e2e-worker", dependencies=dependencies
        )
        with Session(engine) as session:
            order = session.get(ChannelOrder, accepted.order_id)
            pack = session.get(ChannelPack, accepted.pack_id)
            video = session.execute(select(PackVideo)).scalar_one()
            assert order is not None and order.status == "ready"
            assert pack is not None and pack.status == "ready"
            assert video.status == "ready" and video.indexed_parent_id
            assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
            receipt = dict(pack.export_paths_json or {})
            for key in ("manifest", "videos", "links", "transcripts", "archive"):
                path = Path(receipt[f"{key}_path"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt[
                    f"{key}_sha256"
                ]
            original_job = session.execute(
                select(IngestionJob).where(
                    IngestionJob.job_kind == "public_item_ingestion"
                )
            ).scalar_one()
            # A completed canonical job may have exhausted its original
            # provider retry budget. Tenant fanout gets a new DB-only cycle.
            original_job.attempt_count = original_job.max_attempts
            channel = session.execute(select(SourceChannel)).scalar_one()
            payload = copy.deepcopy(original_job.payload_json)
            payload.pop("paidWork", None)
            ensure_channel_entitlement(
                session,
                tenant_id=OTHER_TENANT_ID,
                channel_id=channel.id,
                granted_by_user_id=OTHER_PRINCIPAL_ID,
            )
            _, replay_job, created = get_or_create_ingestion_request(
                session,
                tenant_id=OTHER_TENANT_ID,
                requested_by_user_id=OTHER_PRINCIPAL_ID,
                idempotency_key="canonical-ready-second-principal",
                job_kind=original_job.job_kind,
                source_kind=original_job.source_kind,
                source_key=original_job.source_key,
                pipeline_version=original_job.pipeline_version,
                request_payload=payload,
                channel_id=channel.id,
            )
            session.commit()
            assert created is True and replay_job.id == original_job.id
            assert replay_job.status == "queued"
            assert replay_job.attempt_count == 0

        provider_calls: list[str] = []
        reuse_dependencies = PublicWorkerDependencies(
            acquire=lambda _item: provider_calls.append("acquire"),
            extract_audio=lambda **_kwargs: provider_calls.append("extract"),
            transcribe=lambda **_kwargs: provider_calls.append("transcribe"),
            delete_audio=lambda _path: provider_calls.append("delete"),
            publish_vectors=_successful_qdrant_publication,
        )
        assert process_next_public_ingestion_job(
            worker_id="paid-canonical-reuse", dependencies=reuse_dependencies
        )
        assert provider_calls == []
        with Session(engine) as session:
            replay_job = session.execute(
                select(IngestionJob).where(
                    IngestionJob.job_kind == "public_item_ingestion"
                )
            ).scalar_one()
            assert replay_job.result_json["canonical_ready_reuse"] is True

        # Terminal replay does no new provider work and preserves exact receipts.
        assert not process_next_public_ingestion_job(
            worker_id="paid-e2e-worker-replay", dependencies=dependencies
        )
    finally:
        dispose_engine()


def test_same_tenant_paid_principals_both_receive_the_deduped_canonical_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest_v2.cloud.diarization_indexer.canonical_media as canonical

    database = tmp_path / "same-tenant-paid.sqlite3"
    hot_root = tmp_path / "hot"
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(hot_root))
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv(
        "CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT", str(tmp_path / "transcription")
    )
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            first_quote = _seed_quote(session, monkeypatch)
            session.add(
                UserAccount(
                    id=SAME_TENANT_PRINCIPAL_ID,
                    auth_provider="test",
                    auth_subject=SAME_TENANT_PRINCIPAL_ID,
                )
            )
            session.flush()
            session.add(
                TenantMembership(
                    tenant_id=TENANT_ID,
                    user_id=SAME_TENANT_PRINCIPAL_ID,
                    role="member",
                )
            )
            second_scope = gateway_commerce_scope(
                tenant_id=TENANT_ID,
                principal_user_id=SAME_TENANT_PRINCIPAL_ID,
            )
            ownership = commerce_ownership_values(second_scope)
            second_quote = ChannelQuote(
                **ownership,
                id="quote_paid_same_tenant_2",
                status="open",
                mode=first_quote.mode,
                namespace=first_quote.namespace,
                channel_handle=first_quote.channel_handle,
                resolved_channel_id=first_quote.resolved_channel_id,
                resolved_channel_name=first_quote.resolved_channel_name,
                requested_max_videos=1,
                included_video_count=1,
                excluded_video_count=0,
                current_batch_index=1,
                current_batch_video_count=1,
                current_batch_amount_cents=100,
                total_included_amount_cents=100,
                per_video_cents=100,
                estimated_ready_minutes=5,
                eta_confidence="high",
                recommended_starter_batch_size=1,
                planning_latency_ms=1,
                request_json=copy.deepcopy(first_quote.request_json),
                batch_plan_json=copy.deepcopy(first_quote.batch_plan_json),
                price_breakdown_json=copy.deepcopy(
                    first_quote.price_breakdown_json
                ),
                commerce_json={},
                expires_at=utcnow() + timedelta(minutes=30),
            )
            second_quote.videos.append(
                QuoteVideo(
                    **ownership,
                    position=1,
                    batch_index=1,
                    included=True,
                    video_id="dQw4w9WgXcQ",
                    title="Example",
                    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    status="included",
                )
            )
            session.add(second_quote)
            session.flush()
            bind_gateway_commerce_quote(second_quote, second_scope)
            first = fulfill_claimed_paid_work(
                session, _claim(first_quote), acknowledge=lambda *_args: None
            )
            second = fulfill_claimed_paid_work(
                session,
                _claim_for(
                    second_quote,
                    tenant_id=TENANT_ID,
                    principal_id=SAME_TENANT_PRINCIPAL_ID,
                    ordinal=2,
                ),
                acknowledge=lambda *_args: None,
            )
            session.commit()
            assert first.pack_id != second.pack_id
            assert session.scalar(select(func.count()).select_from(IngestionJob)) == 1
            assert (
                session.scalar(select(func.count()).select_from(IngestionRequest)) == 2
            )

        media_path = hot_root / "same-tenant-fixture.mp4"
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(b"same tenant paid fixture")
        media = HotMediaSpec(
            path=media_path,
            sha256=hashlib.sha256(media_path.read_bytes()).hexdigest(),
            size_bytes=media_path.stat().st_size,
            mime_type="video/mp4",
        )
        monkeypatch.setattr(canonical, "_verify_hot_media", lambda value: value)
        item = PublicItemDescriptor(
            platform="youtube",
            external_id="dQw4w9WgXcQ",
            channel_external_id="UCexample",
            channel_handle="@example",
            canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Example",
            duration_ms=2_000,
        )
        provider_calls: list[str] = []

        def extract_audio(*, video_path: Path, audio_path: Path):
            provider_calls.append("extract")
            assert video_path == media_path
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"audio")
            return hashlib.sha256(b"audio").hexdigest(), 5

        dependencies = PublicWorkerDependencies(
            acquire=lambda _item: (
                provider_calls.append("acquire")
                or AcquiredPublicItem(item=item, media=media)
            ),
            extract_audio=extract_audio,
            transcribe=lambda **_kwargs: (
                provider_calls.append("transcribe")
                or TranscriptResult(
                    provider="local_cpu:test@fixture",
                    provider_request_id=None,
                    segments=(
                        {
                            "ordinal": 0,
                            "start_ms": 0,
                            "end_ms": 2_000,
                            "speaker_label": None,
                            "text": "One canonical item for two paid principals.",
                        },
                    ),
                )
            ),
            delete_audio=lambda path: path.unlink(missing_ok=True),
            publish_vectors=_successful_qdrant_publication,
        )
        assert process_next_public_ingestion_job(
            worker_id="same-tenant-paid-worker", dependencies=dependencies
        )
        assert provider_calls == ["acquire", "extract", "transcribe"]

        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
            assert session.scalar(select(func.count()).select_from(ChannelPack)) == 2
            assert session.scalar(select(func.count()).select_from(PackVideo)) == 2
            assert {row.status for row in session.scalars(select(ChannelOrder))} == {
                "ready"
            }
            packs = list(session.scalars(select(ChannelPack)))
            assert {pack.status for pack in packs} == {"ready"}
            assert all(pack.export_paths_json for pack in packs)
            job = session.scalars(
                select(IngestionJob).where(
                    IngestionJob.job_kind == "public_item_ingestion"
                )
            ).one()
            assert job.result_json["request_publications"] == 2
            assert job.result_json["principal_publications"] == 2
            assert job.result_json["tenant_publications"] == 1
    finally:
        dispose_engine()


def test_payment_worker_url_requires_distinct_exact_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_ROLE", raising=False
    )
    monkeypatch.delenv("CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_URL", raising=False)
    with pytest.raises(PaidWorkError, match="is required"):
        payment_worker_database_url()
    monkeypatch.setenv(
        "CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_URL",
        "postgresql+psycopg://icmfyi_runtime:secret@postgres/icmfyi",
    )
    with pytest.raises(PaidWorkError, match="exact icmfyi_payment_worker role"):
        payment_worker_database_url()
    exact = "postgresql+psycopg://icmfyi_payment_worker:secret@postgres/icmfyi"
    monkeypatch.setenv("CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_URL", exact)
    assert payment_worker_database_url() == exact

    monkeypatch.setenv(
        "CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_ROLE",
        "icmfyi_payment_worker_custom",
    )
    with pytest.raises(PaidWorkError, match="must remain exactly"):
        payment_worker_database_url()


def test_x402_runtime_cannot_enable_a_parallel_free_order_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHANNEL_SERVICE_X402_COMMERCE_ENABLED", "true")
    monkeypatch.setenv("CHANNEL_SERVICE_X402_ASSET", ASSET)
    monkeypatch.setenv("CHANNEL_SERVICE_X402_ATOMIC_UNITS_PER_CENT", "10000")
    monkeypatch.delenv("CHANNEL_SERVICE_REQUIRE_PAYMENT", raising=False)
    with pytest.raises(CommerceConfigurationError, match="REQUIRE_PAYMENT=true"):
        validate_x402_commerce_runtime()
    monkeypatch.setenv("CHANNEL_SERVICE_REQUIRE_PAYMENT", "true")
    validate_x402_commerce_runtime()
    with pytest.raises(ValueError, match="settled x402 work outbox"):
        enforce_direct_order_allowed()
