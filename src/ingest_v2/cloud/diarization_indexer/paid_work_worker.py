from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .channel_service_acp import get_acp_offering
from .channel_service_commerce import (
    COMMERCE_SCHEMA,
    PAYMENT_IDEMPOTENCY_KEY,
    WORK_OPERATION,
    WORK_SCHEMA,
    YOUTUBE_INGEST_TOOL,
    validate_stored_commerce_projection,
)
from .channel_service_jobs import (
    ensure_channel_entitlement,
    ensure_source_channel,
    get_or_create_ingestion_request,
)
from .channel_service_logic import (
    create_checkout_session_with_payment,
    create_order_from_quote,
)
from .channel_service_store import (
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    PackBatch,
    PackVideo,
    PaymentReceipt,
    commerce_scope_predicates,
    gateway_commerce_scope,
    set_commerce_scope,
)
from .public_platforms import PublicTargetError, normalize_public_target
from .transcription_runtime import (
    TranscriptionConfigurationError,
    resolve_transcription_contract,
)

LOG = logging.getLogger(__name__)
PAYMENT_WORKER_DATABASE_URL_ENV = "CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_URL"
PAYMENT_WORKER_ROLE_ENV = "CHANNEL_SERVICE_PAYMENT_WORKER_DATABASE_ROLE"
PAYMENT_WORKER_ROLE = "icmfyi_payment_worker"
PAID_WORK_TOPIC = "icmfyi.work.requested.v1"
PAID_WORK_SCHEMA = "icmfyi.paid-work-request.v1"
SETTLEMENT_RECEIPT_SCHEMA = "icmfyi.x402-settlement-receipt.v1"
PAID_PUBLIC_INGESTION_SCHEMA = "icmfyi.paid-public-ingestion.v1"
PUBLIC_INGESTION_SCHEMA = "icmfyi.public-ingestion-request.v1"
MAX_DELIVERY_ATTEMPTS = 5
_TABLE_PRIVILEGES = (
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "REFERENCES",
    "TRIGGER",
)
_EXPECTED_TABLE_PRIVILEGES = {
    "channel_quotes": {"SELECT"},
    "quote_videos": {"SELECT"},
    "checkout_sessions": {"SELECT", "INSERT", "UPDATE"},
    "channel_packs": {"SELECT", "INSERT", "UPDATE"},
    "pack_batches": {"SELECT", "INSERT", "UPDATE"},
    "pack_videos": {"INSERT"},
    "channel_orders": {"SELECT", "INSERT", "UPDATE"},
    "payment_receipts": {"SELECT", "INSERT"},
    "entitlements": {"INSERT"},
    "tenants": {"SELECT"},
    "user_accounts": {"SELECT"},
    "source_channels": {"SELECT", "INSERT"},
    "tenant_channel_entitlements": {"SELECT", "INSERT"},
    "ingestion_jobs": {"SELECT", "INSERT", "UPDATE"},
    "ingestion_requests": {"SELECT", "INSERT"},
}
_EXPECTED_FUNCTIONS = {
    "icmfyi_claim_settled_paid_work()",
    "icmfyi_ack_settled_paid_work(uuid,uuid,text)",
    "icmfyi_fail_settled_paid_work(uuid,uuid,text,text,boolean,integer,integer)",
}
_IDEMPOTENCY_PATTERN = re.compile(r"[\x21-\x2b\x2d-\x7e]{1,255}\Z")
_PRINTABLE_PATTERN = re.compile(r"[\x21-\x7e]+\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)


class PaidWorkError(RuntimeError):
    """A settled paid-work row cannot be consumed without ambiguity."""


def _payment_worker_role() -> str:
    """Return the one database principal bound into the PostgreSQL RLS policy."""
    configured = (os.getenv(PAYMENT_WORKER_ROLE_ENV) or PAYMENT_WORKER_ROLE).strip()
    if configured != PAYMENT_WORKER_ROLE:
        raise PaidWorkError(
            f"{PAYMENT_WORKER_ROLE_ENV} must remain exactly {PAYMENT_WORKER_ROLE}"
        )
    return PAYMENT_WORKER_ROLE


@dataclass(frozen=True)
class SettledPaidWorkClaim:
    outbox_id: str
    intent_id: str
    tenant_id: str
    principal_id: str
    topic: str
    idempotency_key: str
    request_hash: str
    tool_name: str
    commerce_quote_id: str
    commerce_quote_hash: str
    asset: str
    amount_atomic: Decimal
    payload: dict
    settlement_network: str
    settlement_transaction: str
    settlement_recorded_at: datetime

    @classmethod
    def from_row(cls, row: Mapping) -> SettledPaidWorkClaim:
        try:
            amount_atomic = Decimal(str(row["amount_atomic"]))
            recorded_at = row["settlement_recorded_at"]
            payload = row["payload"]
            claim = cls(
                outbox_id=str(row["outbox_id"]),
                intent_id=str(row["intent_id"]),
                tenant_id=str(row["tenant_id"]),
                principal_id=str(row["principal_id"]),
                topic=str(row["topic"]),
                idempotency_key=str(row["idempotency_key"]),
                request_hash=str(row["request_hash"]),
                tool_name=str(row["tool_name"]),
                commerce_quote_id=str(row["commerce_quote_id"]),
                commerce_quote_hash=str(row["commerce_quote_hash"]),
                asset=str(row["asset"]),
                amount_atomic=amount_atomic,
                payload=dict(payload),
                settlement_network=str(row["settlement_network"]),
                settlement_transaction=str(row["settlement_transaction"]),
                settlement_recorded_at=recorded_at,
            )
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise PaidWorkError("paid-work claim has invalid database types") from exc
        if (
            not isinstance(recorded_at, datetime)
            or recorded_at.tzinfo is None
            or not isinstance(payload, dict)
        ):
            raise PaidWorkError(
                "paid-work claim has invalid settlement or payload types"
            )
        return claim


@dataclass(frozen=True)
class PaidWorkResult:
    outbox_id: str
    intent_id: str
    order_id: str
    checkout_session_id: str
    pack_id: str
    request_ids: tuple[str, ...]
    created: bool


@dataclass(frozen=True)
class PaidWorkFailure:
    outbox_id: str
    intent_id: str
    error_code: str
    retryable: bool


def _same(left: object, right: object) -> bool:
    return hmac.compare_digest(str(left or ""), str(right or ""))


def _exact_keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise PaidWorkError(f"{label} shape is invalid")
    return value


def _validate_claim_shape(claim: SettledPaidWorkClaim) -> tuple[dict, dict]:
    if not _UUID_PATTERN.fullmatch(claim.outbox_id) or not _UUID_PATTERN.fullmatch(
        claim.intent_id
    ):
        raise PaidWorkError("paid-work database identity is invalid")
    if claim.topic != PAID_WORK_TOPIC or claim.tool_name != YOUTUBE_INGEST_TOOL:
        raise PaidWorkError("paid-work topic or tool is invalid")
    if not _IDEMPOTENCY_PATTERN.fullmatch(claim.idempotency_key):
        raise PaidWorkError("paid-work idempotency key is invalid")
    if not _SHA256_PATTERN.fullmatch(
        claim.request_hash
    ) or not _SHA256_PATTERN.fullmatch(claim.commerce_quote_hash):
        raise PaidWorkError("paid-work hash identity is invalid")
    if (
        claim.amount_atomic != claim.amount_atomic.to_integral_value()
        or claim.amount_atomic <= 0
        or not _PRINTABLE_PATTERN.fullmatch(claim.asset)
        or len(claim.asset) > 255
        or not _PRINTABLE_PATTERN.fullmatch(claim.settlement_network)
        or len(claim.settlement_network) > 255
        or not _PRINTABLE_PATTERN.fullmatch(claim.settlement_transaction)
        or len(claim.settlement_transaction) > 512
    ):
        raise PaidWorkError("paid-work settlement facts are invalid")

    payload = _exact_keys(
        claim.payload,
        {
            "schema",
            "tenantId",
            "principalId",
            "toolName",
            "idempotencyKey",
            "requestHash",
            "commerce",
            "work",
        },
        "paid-work payload",
    )
    commerce = _exact_keys(
        payload.get("commerce"),
        {"provider", "quoteId", "offeringId", "quoteHash"},
        "paid-work commerce payload",
    )
    work = _exact_keys(
        payload.get("work"),
        {"schema", "operation", "quoteId", "packId"},
        "paid-work work payload",
    )
    expected_scalars = {
        "schema": PAID_WORK_SCHEMA,
        "tenantId": claim.tenant_id,
        "principalId": claim.principal_id,
        "toolName": claim.tool_name,
        "idempotencyKey": claim.idempotency_key,
        "requestHash": claim.request_hash,
    }
    if any(
        not _same(payload.get(key), value) for key, value in expected_scalars.items()
    ):
        raise PaidWorkError("paid-work payload does not match settled database facts")
    if (
        commerce.get("provider") != "icmfyi-acp"
        or not _same(commerce.get("quoteId"), claim.commerce_quote_id)
        or not _same(commerce.get("quoteHash"), claim.commerce_quote_hash)
        or not str(commerce.get("offeringId") or "")
        or work.get("schema") != WORK_SCHEMA
        or work.get("operation") != WORK_OPERATION
        or not _same(work.get("quoteId"), claim.commerce_quote_id)
        or (work.get("packId") is not None and not isinstance(work.get("packId"), str))
    ):
        raise PaidWorkError("paid-work commerce or work identity is invalid")
    return commerce, work


def _validate_quote_against_claim(
    quote: ChannelQuote,
    claim: SettledPaidWorkClaim,
    *,
    commerce: dict,
    work: dict,
) -> tuple[int, str | None]:
    stored = dict(quote.commerce_json or {})
    try:
        validate_stored_commerce_projection(stored)
    except Exception as exc:
        raise PaidWorkError("stored commerce projection is invalid") from exc
    expected = {
        "schema": COMMERCE_SCHEMA,
        "tenantId": claim.tenant_id,
        "principalId": claim.principal_id,
        "quoteId": claim.commerce_quote_id,
        "quoteHash": claim.commerce_quote_hash,
        "requestHash": claim.request_hash,
        "toolName": claim.tool_name,
        "asset": claim.asset,
        "amountAtomic": str(claim.amount_atomic.to_integral_value()),
        "offeringId": commerce["offeringId"],
        PAYMENT_IDEMPOTENCY_KEY: claim.idempotency_key,
    }
    if any(not _same(stored.get(key), value) for key, value in expected.items()):
        raise PaidWorkError("settled intent does not match the authoritative quote")
    if stored.get("workPayload") != work:
        raise PaidWorkError("settled work does not match the authoritative quote")

    requested_pack_id = str((quote.request_json or {}).get("pack_id") or "") or None
    work_pack_id = str(work.get("packId") or "") or None
    if requested_pack_id != work_pack_id:
        raise PaidWorkError("settled work pack does not match the quote request")
    try:
        offering = get_acp_offering(str(stored["offeringId"]))
    except ValueError as exc:
        raise PaidWorkError("settled work offering is invalid") from exc
    return int(offering.fixed_price_cents), requested_pack_id


def _exact_existing_order(
    session: Session,
    *,
    claim: SettledPaidWorkClaim,
    scope,
    expected_amount_cents: int,
) -> tuple[ChannelOrder, PaymentReceipt] | None:
    orders = (
        session.execute(
            select(ChannelOrder)
            .where(
                ChannelOrder.quote_id == claim.commerce_quote_id,
                *commerce_scope_predicates(ChannelOrder, scope),
            )
            .with_for_update()
        )
        .scalars()
        .all()
    )
    if not orders:
        return None
    if len(orders) != 1:
        raise PaidWorkError("settled quote has ambiguous order history")
    order = orders[0]
    checkout = session.execute(
        select(CheckoutSessionRecord).where(
            CheckoutSessionRecord.id == order.checkout_session_id,
            *commerce_scope_predicates(CheckoutSessionRecord, scope),
        )
    ).scalar_one_or_none()
    pack = session.execute(
        select(ChannelPack).where(
            ChannelPack.id == order.pack_id,
            *commerce_scope_predicates(ChannelPack, scope),
        )
    ).scalar_one_or_none()
    batch = session.execute(
        select(PackBatch).where(
            PackBatch.id == order.batch_id,
            *commerce_scope_predicates(PackBatch, scope),
        )
    ).scalar_one_or_none()
    receipt = session.execute(
        select(PaymentReceipt).where(
            PaymentReceipt.order_id == order.id,
            *commerce_scope_predicates(PaymentReceipt, scope),
        )
    ).scalar_one_or_none()
    receipt_json = dict(receipt.receipt_json or {}) if receipt is not None else {}
    if (
        checkout is None
        or checkout.status != "completed"
        or checkout.payment_provider != "x402"
        or checkout.payment_status != "settled_x402"
        or list(checkout.quote_ids_json or []) != [claim.commerce_quote_id]
        or int(checkout.total_amount_cents or 0) != expected_amount_cents
        or pack is None
        or batch is None
        or batch.pack_id != pack.id
        or batch.quote_id != claim.commerce_quote_id
        or batch.checkout_session_id != checkout.id
        or int(batch.amount_cents or 0) != expected_amount_cents
        or order.payment_provider != "x402"
        or order.payment_status != "settled_x402"
        or order.amount_cents != expected_amount_cents
        or order.currency != "USD"
        or receipt is None
        or receipt.checkout_session_id != checkout.id
        or receipt.provider != "x402"
        or receipt.status != "settled"
        or receipt.amount_cents != expected_amount_cents
        or receipt.currency != "USD"
        or receipt_json.get("schema") != SETTLEMENT_RECEIPT_SCHEMA
        or not _same(receipt_json.get("paymentIntentId"), claim.intent_id)
        or not _same(receipt_json.get("outboxId"), claim.outbox_id)
        or not _same(receipt_json.get("idempotencyKey"), claim.idempotency_key)
        or not _same(receipt_json.get("requestHash"), claim.request_hash)
        or not _same(receipt_json.get("quoteHash"), claim.commerce_quote_hash)
        or not _same(receipt_json.get("network"), claim.settlement_network)
        or not _same(receipt_json.get("transaction"), claim.settlement_transaction)
    ):
        raise PaidWorkError("settled quote already has a different order or receipt")
    return order, receipt


def _public_ingestion_policy(*, language: str, transcript: dict) -> tuple[str, dict]:
    policy = {
        "clip_ready": True,
        "language": language,
        "max_items": None,
        "transcription": transcript,
    }
    digest = hashlib.sha256(
        json.dumps(
            policy, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    return digest, policy


def _enqueue_paid_public_ingestion(
    session: Session,
    *,
    claim: SettledPaidWorkClaim,
    quote: ChannelQuote,
    order: ChannelOrder,
    pack_id: str,
    batch_id: str,
) -> tuple[str, ...]:
    """Create exact durable public-ingestion requests without external effects."""
    language = str((quote.request_json or {}).get("language") or "en").strip()
    if not language or len(language) > 32:
        raise PaidWorkError("settled quote language is invalid")
    try:
        contract = resolve_transcription_contract("auto")
    except TranscriptionConfigurationError as exc:
        raise PaidWorkError("paid public-ingestion transcription contract is invalid") from exc
    policy_hash, _ = _public_ingestion_policy(
        language=language, transcript=contract.as_payload()
    )
    channel_external_id = str(
        quote.resolved_channel_id or quote.channel_handle.lstrip("@")
    ).strip()
    if not channel_external_id:
        raise PaidWorkError("settled quote channel identity is invalid")
    channel = ensure_source_channel(
        session,
        platform="youtube",
        external_id=channel_external_id,
        handle=quote.channel_handle,
        display_name=quote.resolved_channel_name,
        canonical_url=(
            f"https://www.youtube.com/channel/{channel_external_id}"
            if quote.resolved_channel_id
            else f"https://www.youtube.com/{quote.channel_handle}"
        ),
        metadata={"paid_public_ingestion": {"quote_id": quote.id}},
    )
    ensure_channel_entitlement(
        session,
        tenant_id=claim.tenant_id,
        channel_id=channel.id,
        granted_by_user_id=claim.principal_id,
        access_level="query",
    )

    quote_rows = sorted(
        (
            row
            for row in quote.videos
            if row.included and int(row.batch_index) == int(quote.current_batch_index)
        ),
        key=lambda row: (int(row.position), int(row.id or 0)),
    )
    if len(quote_rows) != int(quote.current_batch_video_count or 0) or not quote_rows:
        raise PaidWorkError("settled quote billed-video inventory is inconsistent")

    request_ids: list[str] = []
    for row in quote_rows:
        video_id = str(row.video_id or "").strip()
        canonical_url = str(row.video_url or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise PaidWorkError("settled quote contains an invalid YouTube video identity")
        if not canonical_url:
            canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            target = normalize_public_target(
                platform="youtube", target_kind="item", target=canonical_url
            )
        except PublicTargetError as exc:
            raise PaidWorkError("settled quote contains a noncanonical video URL") from exc
        if target.external_id != video_id:
            raise PaidWorkError("settled quote video URL and identity disagree")
        duration_ms = None
        if row.duration_s is not None:
            duration = float(row.duration_s)
            if not math.isfinite(duration) or duration < 0:
                raise PaidWorkError("settled quote video duration is invalid")
            duration_ms = round(duration * 1000)
        paid_work = {
            "schema": PAID_PUBLIC_INGESTION_SCHEMA,
            "intentId": claim.intent_id,
            "outboxId": claim.outbox_id,
            "tenantId": claim.tenant_id,
            "principalId": claim.principal_id,
            "orderId": order.id,
            "packId": pack_id,
            "batchId": batch_id,
            "quoteId": quote.id,
            "quoteHash": claim.commerce_quote_hash,
            "videoId": video_id,
            "position": int(row.position),
        }
        item = {
            "platform": "youtube",
            "external_id": video_id,
            "channel_external_id": channel_external_id,
            "channel_handle": quote.channel_handle,
            "canonical_url": target.canonical_url,
            "title": row.title,
            "description": row.description,
            "published_at": row.published_at,
            "duration_ms": duration_ms,
            "metadata": {"paidWork": paid_work},
        }
        payload = {
            "schema": PUBLIC_INGESTION_SCHEMA,
            "target": target.as_payload(),
            "max_items": 1,
            "clip_ready": True,
            "language": language,
            "transcription": contract.as_payload(),
            "item": item,
            "paidWork": paid_work,
        }
        request, _, _ = get_or_create_ingestion_request(
            session,
            tenant_id=claim.tenant_id,
            requested_by_user_id=claim.principal_id,
            idempotency_key=(
                f"paid:{claim.intent_id}:{int(row.position)}:{video_id}"
            ),
            job_kind="public_item_ingestion",
            source_kind="youtube",
            source_key=f"{channel_external_id}:{video_id}",
            # Provider work stays globally canonical. A newly purchased request
            # attached to a succeeded job requeues only DB/filesystem fanout; the
            # public worker proves canonical readiness before skipping providers.
            pipeline_version=f"public-ingest-v1:{policy_hash}",
            request_payload=payload,
            channel_id=channel.id,
            priority=10,
            max_attempts=5,
        )
        request_ids.append(request.id)
    return tuple(request_ids)


def fulfill_claimed_paid_work(
    session: Session,
    claim: SettledPaidWorkClaim,
    *,
    acknowledge: Callable[[Session, SettledPaidWorkClaim, str], None],
) -> PaidWorkResult:
    """Create or reconcile one paid order and acknowledge in the same transaction.

    The function performs database work only.  It deliberately disables quote
    refresh, provider calls, canonical publication, and filesystem exports while
    the payment outbox row is locked.
    """
    commerce, work = _validate_claim_shape(claim)
    try:
        scope = gateway_commerce_scope(
            tenant_id=claim.tenant_id,
            principal_user_id=claim.principal_id,
        )
        set_commerce_scope(session, scope)
    except ValueError as exc:
        raise PaidWorkError("settled principal or tenant identity is invalid") from exc

    quote = session.execute(
        select(ChannelQuote).where(
            ChannelQuote.id == claim.commerce_quote_id,
            *commerce_scope_predicates(ChannelQuote, scope),
        )
    ).scalar_one_or_none()
    if quote is None:
        raise PaidWorkError("settled commerce quote is not owned by this principal")
    amount_cents, pack_id = _validate_quote_against_claim(
        quote, claim, commerce=commerce, work=work
    )

    existing = _exact_existing_order(
        session,
        claim=claim,
        scope=scope,
        expected_amount_cents=amount_cents,
    )
    if existing is not None:
        order, _ = existing
        pack_id = order.pack_id
        request_ids = _enqueue_paid_public_ingestion(
            session,
            claim=claim,
            quote=quote,
            order=order,
            pack_id=pack_id,
            batch_id=order.batch_id,
        )
        acknowledge(session, claim, order.id)
        return PaidWorkResult(
            outbox_id=claim.outbox_id,
            intent_id=claim.intent_id,
            order_id=order.id,
            checkout_session_id=order.checkout_session_id,
            pack_id=order.pack_id,
            request_ids=request_ids,
            created=False,
        )

    checkout = create_checkout_session_with_payment(
        session=session,
        commerce_scope=scope,
        quote_ids=[quote.id],
        idempotency_key=f"x402:{claim.intent_id}",
        payment_provider="x402",
        payment_status="settled_x402",
        line_item_amount_overrides={quote.id: amount_cents},
        refresh_quotes=False,
    )
    if (
        checkout.payment_provider != "x402"
        or checkout.payment_status != "settled_x402"
        or list(checkout.quote_ids_json or []) != [quote.id]
        or int(checkout.total_amount_cents or 0) != amount_cents
    ):
        raise PaidWorkError("settled checkout idempotency state is ambiguous")

    recorded_at = claim.settlement_recorded_at
    receipt_json = {
        "schema": SETTLEMENT_RECEIPT_SCHEMA,
        "paymentIntentId": claim.intent_id,
        "outboxId": claim.outbox_id,
        "idempotencyKey": claim.idempotency_key,
        "quoteId": quote.id,
        "quoteHash": claim.commerce_quote_hash,
        "requestHash": claim.request_hash,
        "network": claim.settlement_network,
        "transaction": claim.settlement_transaction,
        "settledAt": recorded_at.astimezone(timezone.utc).isoformat(),
    }
    # PostgreSQL INSERT .. RETURNING requires a column SELECT grant. Pre-fetch
    # the owned sequence value instead, preserving INSERT-only PackVideo access.
    PackVideo.__table__.implicit_returning = False
    pack, _, order = create_order_from_quote(
        session=session,
        commerce_scope=scope,
        quote=quote,
        checkout=checkout,
        pack_id=pack_id,
        buyer_subject_type="tenant",
        buyer_subject_id=claim.tenant_id,
        external_payment={
            "provider": "x402",
            "payment_status": "settled_x402",
            "receipt_status": "settled",
            "amount_cents": amount_cents,
            "receipt_json": receipt_json,
        },
        canonical_publish=None,
        acquire_hot_media=False,
        media_acquire=None,
        refresh_quote=False,
        defer_fulfillment=True,
    )
    checkout.status = "completed"
    request_ids = _enqueue_paid_public_ingestion(
        session,
        claim=claim,
        quote=quote,
        order=order,
        pack_id=pack.id,
        batch_id=order.batch_id,
    )
    order.notes_json = {
        "paymentIntentId": claim.intent_id,
        "outboxId": claim.outbox_id,
        "idempotencyKey": claim.idempotency_key,
        "quoteHash": claim.commerce_quote_hash,
        "requestHash": claim.request_hash,
        "publicIngestionRequestIds": list(request_ids),
    }
    session.flush()
    _exact_existing_order(
        session,
        claim=claim,
        scope=scope,
        expected_amount_cents=amount_cents,
    )
    acknowledge(session, claim, order.id)
    return PaidWorkResult(
        outbox_id=claim.outbox_id,
        intent_id=claim.intent_id,
        order_id=order.id,
        checkout_session_id=checkout.id,
        pack_id=pack.id,
        request_ids=request_ids,
        created=True,
    )


def claim_settled_paid_work(session: Session) -> SettledPaidWorkClaim | None:
    row = (
        session.execute(text("SELECT * FROM public.icmfyi_claim_settled_paid_work()"))
        .mappings()
        .one_or_none()
    )
    return SettledPaidWorkClaim.from_row(row) if row is not None else None


def acknowledge_settled_paid_work(
    session: Session, claim: SettledPaidWorkClaim, order_id: str
) -> None:
    acknowledged = session.execute(
        text(
            "SELECT public.icmfyi_ack_settled_paid_work("
            "CAST(:outbox_id AS uuid), CAST(:intent_id AS uuid), :order_id)"
        ),
        {
            "outbox_id": claim.outbox_id,
            "intent_id": claim.intent_id,
            "order_id": order_id,
        },
    ).scalar_one()
    if acknowledged is not True:
        raise PaidWorkError("paid-work acknowledgement did not confirm exact readback")


def fail_settled_paid_work(
    session: Session,
    claim: SettledPaidWorkClaim,
    *,
    error_code: str,
    error_detail: str,
    retryable: bool,
    retry_delay_seconds: int,
) -> None:
    recorded = session.execute(
        text(
            "SELECT public.icmfyi_fail_settled_paid_work("
            "CAST(:outbox_id AS uuid), CAST(:intent_id AS uuid), :error_code, "
            ":error_detail, :retryable, :maximum_attempts, :retry_delay_seconds)"
        ),
        {
            "outbox_id": claim.outbox_id,
            "intent_id": claim.intent_id,
            "error_code": error_code,
            "error_detail": error_detail[:8000],
            "retryable": retryable,
            "maximum_attempts": MAX_DELIVERY_ATTEMPTS,
            "retry_delay_seconds": retry_delay_seconds,
        },
    ).scalar_one()
    if recorded is not True:
        raise PaidWorkError("paid-work failure did not confirm exact readback")


def _failure_contract(exc: BaseException) -> tuple[str, bool, int]:
    if isinstance(exc, PaidWorkError):
        return "paid_work_invalid", False, 0
    if isinstance(exc, (OperationalError, DBAPIError)):
        return "paid_work_database_transient", True, 30
    if isinstance(exc, (ValueError, TypeError)):
        return "paid_work_invalid", False, 0
    return "paid_work_internal_error", True, 60


def consume_one_settled_paid_work(
    session: Session,
) -> PaidWorkResult | PaidWorkFailure | None:
    claim = claim_settled_paid_work(session)
    if claim is None:
        return None
    try:
        with session.begin_nested():
            return fulfill_claimed_paid_work(
                session, claim, acknowledge=acknowledge_settled_paid_work
            )
    except Exception as exc:  # noqa: BLE001 - unexpected poison must be persisted
        error_code, retryable, retry_delay = _failure_contract(exc)
        fail_settled_paid_work(
            session,
            claim,
            error_code=error_code,
            error_detail=str(exc),
            retryable=retryable,
            retry_delay_seconds=retry_delay,
        )
        return PaidWorkFailure(
            outbox_id=claim.outbox_id,
            intent_id=claim.intent_id,
            error_code=error_code,
            retryable=retryable,
        )


def payment_worker_database_url() -> str:
    raw = (os.getenv(PAYMENT_WORKER_DATABASE_URL_ENV) or "").strip()
    if not raw:
        raise PaidWorkError(f"{PAYMENT_WORKER_DATABASE_URL_ENV} is required")
    url = make_url(raw)
    required_role = _payment_worker_role()
    if url.get_backend_name() != "postgresql" or url.username != required_role:
        raise PaidWorkError(
            f"{PAYMENT_WORKER_DATABASE_URL_ENV} must use PostgreSQL and the exact {required_role} role"
        )
    return raw


def _assert_worker_connection(session: Session) -> None:
    row = session.execute(
        text(
            "SELECT current_user AS role_name, role.rolcanlogin, role.rolsuper, "
            "role.rolcreatedb, role.rolcreaterole, role.rolinherit, "
            "role.rolreplication, role.rolbypassrls, "
            "EXISTS (SELECT 1 FROM pg_auth_members m "
            "WHERE m.member=role.oid) AS member_of_role, "
            "EXISTS (SELECT 1 FROM pg_auth_members m "
            "WHERE m.roleid=role.oid) AS granted_to_member, "
            "EXISTS (SELECT 1 FROM pg_shdepend d "
            "WHERE d.refclassid='pg_authid'::regclass AND d.refobjid=role.oid "
            "AND d.deptype='o') AS owns_objects "
            "FROM pg_roles role WHERE role.rolname=current_user"
        )
    ).mappings().one()
    required_role = _payment_worker_role()
    if (
        row["role_name"] != required_role
        or not row["rolcanlogin"]
        or row["rolsuper"]
        or row["rolcreatedb"]
        or row["rolcreaterole"]
        or row["rolinherit"]
        or row["rolreplication"]
        or row["rolbypassrls"]
        or row["member_of_role"]
        or row["granted_to_member"]
        or row["owns_objects"]
    ):
        raise PaidWorkError("payment worker database role capabilities are unsafe")

    boundary = session.execute(
        text(
            "SELECT "
            "has_database_privilege(current_user,current_database(),'CONNECT') AS db_connect, "
            "has_database_privilege(current_user,current_database(),'CREATE') AS db_create, "
            "has_database_privilege(current_user,current_database(),'TEMPORARY') AS db_temp, "
            "has_schema_privilege(current_user,'public','USAGE') AS schema_usage, "
            "has_schema_privilege(current_user,'public','CREATE') AS schema_create"
        )
    ).mappings().one()
    if dict(boundary) != {
        "db_connect": True,
        "db_create": False,
        "db_temp": False,
        "schema_usage": True,
        "schema_create": False,
    }:
        raise PaidWorkError("payment worker database or schema privileges are unsafe")

    function_rows = session.execute(
        text(
            "SELECT proc.oid::regprocedure::text AS signature, "
            "proc.proname, proc.prosecdef, pg_get_userbyid(proc.proowner) AS owner, "
            "proc.proconfig, "
            "has_function_privilege(current_user, proc.oid, 'EXECUTE') AS worker_execute, "
            "has_function_privilege('public', proc.oid, 'EXECUTE') AS public_execute, "
            "EXISTS (SELECT 1 FROM aclexplode(COALESCE(proc.proacl, "
            "acldefault('f', proc.proowner))) acl "
            "WHERE acl.privilege_type='EXECUTE' "
            "AND acl.grantee NOT IN (proc.proowner, "
            "(SELECT oid FROM pg_roles WHERE rolname=current_user))) AS unexpected_acl "
            "FROM pg_proc proc JOIN pg_namespace ns ON ns.oid=proc.pronamespace "
            "WHERE ns.nspname='public' AND proc.proname IN ("
            "'icmfyi_claim_settled_paid_work','icmfyi_ack_settled_paid_work',"
            "'icmfyi_fail_settled_paid_work')"
        )
    ).mappings().all()
    observed_functions = {
        str(item["signature"]).removeprefix("public.").replace(" ", "")
        for item in function_rows
    }
    owners = {str(item["owner"]) for item in function_rows}
    if (
        observed_functions != _EXPECTED_FUNCTIONS
        or len(owners) != 1
        or required_role in owners
        or any(
            not item["prosecdef"]
            or item["proconfig"] != ["search_path=pg_catalog, public"]
            or not item["worker_execute"]
            or item["public_execute"]
            or item["unexpected_acl"]
            for item in function_rows
        )
    ):
        raise PaidWorkError("payment worker function ownership or ACL is unsafe")

    table_rows = session.execute(
        text(
            "SELECT rel.relname, "
            + ", ".join(
                f"has_table_privilege(current_user, rel.oid, '{privilege}') "
                f"AS {privilege.lower()}_privilege"
                for privilege in _TABLE_PRIVILEGES
            )
            + " FROM pg_class rel JOIN pg_namespace ns ON ns.oid=rel.relnamespace "
            "WHERE ns.nspname='public' AND rel.relkind IN ('r','p','v','m')"
        )
    ).mappings().all()
    for table in table_rows:
        actual = {
            privilege
            for privilege in _TABLE_PRIVILEGES
            if table[f"{privilege.lower()}_privilege"]
        }
        if actual != _EXPECTED_TABLE_PRIVILEGES.get(str(table["relname"]), set()):
            raise PaidWorkError("payment worker table privilege contract is unsafe")
    if not _EXPECTED_TABLE_PRIVILEGES.keys() <= {
        str(table["relname"]) for table in table_rows
    }:
        raise PaidWorkError("payment worker required table is missing")

    column_acl = session.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM pg_attribute attr "
            "JOIN pg_class rel ON rel.oid=attr.attrelid "
            "JOIN pg_namespace ns ON ns.oid=rel.relnamespace, "
            "LATERAL aclexplode(attr.attacl) acl "
            "WHERE ns.nspname='public' AND attr.attacl IS NOT NULL "
            "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=current_user))"
        )
    ).scalar_one()
    if column_acl:
        raise PaidWorkError("payment worker has unexpected column privileges")

    sequence_rows = session.execute(
        text(
            "SELECT rel.relname, "
            "has_sequence_privilege(current_user, rel.oid, 'USAGE') AS usage, "
            "has_sequence_privilege(current_user, rel.oid, 'SELECT') AS selected, "
            "has_sequence_privilege(current_user, rel.oid, 'UPDATE') AS updated "
            "FROM pg_class rel JOIN pg_namespace ns ON ns.oid=rel.relnamespace "
            "WHERE ns.nspname='public' AND rel.relkind='S'"
        )
    ).mappings().all()
    for sequence in sequence_rows:
        expected_usage = sequence["relname"] == "pack_videos_id_seq"
        if (
            bool(sequence["usage"]) != expected_usage
            or sequence["selected"]
            or sequence["updated"]
        ):
            raise PaidWorkError("payment worker sequence privilege contract is unsafe")


def _worker_id() -> str:
    host = (os.getenv("HOSTNAME") or socket.gethostname() or "host").strip()
    return f"{host}-p{os.getpid()}"


def run_worker() -> None:
    logging.basicConfig(
        level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    engine = create_engine(
        payment_worker_database_url(),
        future=True,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
    )
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    poll_seconds = max(1.0, float(os.getenv("ICMFYI_PAID_WORK_POLL_SECONDS") or "2"))
    once = (os.getenv("ICMFYI_PAID_WORK_ONCE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    worker_id = _worker_id()
    try:
        with factory.begin() as session:
            _assert_worker_connection(session)
        LOG.info("paid-work worker ready worker_id=%s", worker_id)
        while True:
            try:
                with factory.begin() as session:
                    result = consume_one_settled_paid_work(session)
                if isinstance(result, PaidWorkResult):
                    LOG.info(
                        "paid-work committed worker_id=%s outbox_id=%s order_id=%s created=%s",
                        worker_id,
                        result.outbox_id,
                        result.order_id,
                        result.created,
                    )
                elif isinstance(result, PaidWorkFailure):
                    LOG.error(
                        "paid-work failure recorded worker_id=%s outbox_id=%s "
                        "error_code=%s retryable=%s",
                        worker_id,
                        result.outbox_id,
                        result.error_code,
                        result.retryable,
                    )
                elif once:
                    return
                else:
                    time.sleep(poll_seconds)
            except Exception:
                LOG.exception(
                    "paid-work transaction rolled back worker_id=%s", worker_id
                )
                if once:
                    raise
                time.sleep(max(poll_seconds, 5.0))
            if once:
                return
    finally:
        engine.dispose()


if __name__ == "__main__":
    run_worker()
