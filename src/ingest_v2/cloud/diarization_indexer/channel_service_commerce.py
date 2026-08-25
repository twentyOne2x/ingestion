from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime

from sqlalchemy import select

from .channel_service_acp import get_acp_offering
from .channel_service_store import (
    COMMERCE_AUTHORITY_GATEWAY,
    ChannelOrder,
    ChannelQuote,
    CommerceScope,
    commerce_record_matches_scope,
    commerce_scope_predicates,
    utcnow,
)

COMMERCE_SCHEMA = "icmfyi.authoritative-commerce-quote.v1"
WORK_SCHEMA = "icmfyi.channel-pack-work.v1"
WORK_OPERATION = "create_settled_channel_pack_order"
YOUTUBE_INGEST_TOOL = "icmfyi.ingest.youtube"
_ASSET_PATTERN = re.compile(r"[\x21-\x7e]{1,255}\Z")
_IDEMPOTENCY_PATTERN = re.compile(r"[\x21-\x2b\x2d-\x7e]{1,255}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PAYMENT_IDEMPOTENCY_KEY = "paymentIdempotencyKey"
_OUTPUT_KEYS = (
    "provider",
    "quoteId",
    "offeringId",
    "quoteHash",
    "requestHash",
    "toolName",
    "asset",
    "amountAtomic",
    "expiresAt",
    "workPayload",
)


class CommerceConfigurationError(RuntimeError):
    """The x402 quote projection is not configured safely."""


class CommerceResolutionError(RuntimeError):
    """The requested quote is not exact, enabled, or authorized for payment."""


def bind_gateway_commerce_quote(quote: ChannelQuote, scope: CommerceScope) -> None:
    """Freeze the exact payment/work projection on a newly persisted gateway quote."""
    if scope.authority_kind != COMMERCE_AUTHORITY_GATEWAY:
        raise CommerceConfigurationError("only gateway quotes can be payment-enabled")
    if not commerce_record_matches_scope(quote, scope):
        raise CommerceConfigurationError("quote ownership does not match gateway scope")

    disabled = {
        "schema": COMMERCE_SCHEMA,
        "enabled": False,
        "reason": "x402_commerce_disabled",
    }
    if not _env_enabled("CHANNEL_SERVICE_X402_COMMERCE_ENABLED"):
        quote.commerce_json = disabled
        return

    asset, atomic_units_per_cent = _payment_asset_contract()

    offering_id = _offering_id_for_quote(quote)
    if offering_id is None:
        quote.commerce_json = {
            **disabled,
            "reason": "quote_has_no_exact_x402_offering",
        }
        return
    offering = get_acp_offering(offering_id)
    work_payload = {
        "schema": WORK_SCHEMA,
        "operation": WORK_OPERATION,
        "quoteId": quote.id,
        "packId": (quote.request_json or {}).get("pack_id"),
    }
    request_hash = _sha256_json(work_payload)
    output_without_hash = {
        "provider": "icmfyi-acp",
        "quoteId": quote.id,
        "offeringId": offering.offering_id,
        "requestHash": request_hash,
        "toolName": YOUTUBE_INGEST_TOOL,
        "asset": asset,
        "amountAtomic": str(offering.fixed_price_cents * atomic_units_per_cent),
        "expiresAt": quote.expires_at.isoformat(),
        "workPayload": work_payload,
    }
    quote_hash = _sha256_json(
        {
            **output_without_hash,
            "tenantId": scope.tenant_id,
            "principalId": scope.principal_user_id,
        }
    )
    quote.commerce_json = {
        "schema": COMMERCE_SCHEMA,
        "enabled": True,
        "tenantId": scope.tenant_id,
        "principalId": scope.principal_user_id,
        **output_without_hash,
        "quoteHash": quote_hash,
    }


def validate_x402_commerce_runtime() -> None:
    if _env_enabled("CHANNEL_SERVICE_X402_COMMERCE_ENABLED"):
        if not _env_enabled("CHANNEL_SERVICE_REQUIRE_PAYMENT"):
            raise CommerceConfigurationError(
                "x402 commerce requires CHANNEL_SERVICE_REQUIRE_PAYMENT=true"
            )
        _payment_asset_contract()


def _payment_asset_contract() -> tuple[str, int]:
    asset = (os.getenv("CHANNEL_SERVICE_X402_ASSET") or "").strip()
    if not _ASSET_PATTERN.fullmatch(asset):
        raise CommerceConfigurationError(
            "x402 commerce requires CHANNEL_SERVICE_X402_ASSET as one printable atom"
        )
    try:
        atomic_units_per_cent = int(
            os.getenv("CHANNEL_SERVICE_X402_ATOMIC_UNITS_PER_CENT") or ""
        )
    except ValueError as exc:
        raise CommerceConfigurationError(
            "CHANNEL_SERVICE_X402_ATOMIC_UNITS_PER_CENT must be a positive integer"
        ) from exc
    if atomic_units_per_cent < 1 or atomic_units_per_cent > 10**18:
        raise CommerceConfigurationError(
            "CHANNEL_SERVICE_X402_ATOMIC_UNITS_PER_CENT must be between 1 and 10^18"
        )

    return asset, atomic_units_per_cent


def public_commerce_projection(quote: ChannelQuote) -> dict:
    stored = dict(quote.commerce_json or {})
    if not stored.get("enabled"):
        return {
            "enabled": False,
            "reason": str(stored.get("reason") or "x402_commerce_disabled"),
        }
    return {"enabled": True, **{key: stored.get(key) for key in _OUTPUT_KEYS}}


def resolve_authoritative_commerce_quote(
    *,
    session,
    scope: CommerceScope,
    quote_id: str,
    tool_name: str,
    idempotency_key: str,
    request_hash: str,
    now: datetime | None = None,
) -> dict:
    """Resolve one exact enabled quote for the authenticated gateway pair.

    The first payable resolution atomically binds the adapter idempotency key.
    A settled replay remains resolvable after expiry only when the exact scoped
    quote has that binding and an x402-settled order. No payment or provider
    effect occurs here.
    """
    if scope.authority_kind != COMMERCE_AUTHORITY_GATEWAY:
        raise CommerceResolutionError("payment quote resolution requires gateway scope")
    if not isinstance(idempotency_key, str) or not _IDEMPOTENCY_PATTERN.fullmatch(
        idempotency_key
    ):
        raise CommerceResolutionError("commerce idempotency key is invalid")
    quote = session.execute(
        select(ChannelQuote)
        .where(
            ChannelQuote.id == str(quote_id),
            *commerce_scope_predicates(ChannelQuote, scope),
        )
        .with_for_update()
    ).scalar_one_or_none()
    if quote is None:
        raise CommerceResolutionError("commerce quote was not found")

    stored = dict(quote.commerce_json or {})
    if stored.get("schema") != COMMERCE_SCHEMA or stored.get("enabled") is not True:
        raise CommerceResolutionError("commerce quote is not payment-enabled")
    if (
        stored.get("tenantId") != scope.tenant_id
        or stored.get("principalId") != scope.principal_user_id
    ):
        raise CommerceResolutionError("commerce quote owner does not match")
    if stored.get("quoteId") != quote.id or stored.get("toolName") != tool_name:
        raise CommerceResolutionError(
            "commerce quote does not match the requested tool"
        )
    if not _SHA256_PATTERN.fullmatch(
        str(request_hash or "")
    ) or not hmac.compare_digest(
        str(stored.get("requestHash") or ""), str(request_hash)
    ):
        raise CommerceResolutionError("commerce request hash does not match")
    validate_stored_commerce_projection(stored)

    orders = (
        session.execute(
            select(ChannelOrder).where(
                ChannelOrder.quote_id == quote.id,
                *commerce_scope_predicates(ChannelOrder, scope),
            )
        )
        .scalars()
        .all()
    )
    settled_replay = any(
        order.payment_provider == "x402"
        and order.payment_status == "settled_x402"
        and order.status not in {"failed", "cancelled"}
        for order in orders
    )
    bound_idempotency_key = stored.get(PAYMENT_IDEMPOTENCY_KEY)
    if bound_idempotency_key is None:
        if orders:
            raise CommerceResolutionError(
                "commerce quote with order history lacks an idempotency binding"
            )
        stored[PAYMENT_IDEMPOTENCY_KEY] = idempotency_key
        quote.commerce_json = stored
    elif not isinstance(bound_idempotency_key, str) or not hmac.compare_digest(
        bound_idempotency_key, idempotency_key
    ):
        raise CommerceResolutionError(
            "commerce quote is bound to a different idempotency key"
        )
    if orders and not settled_replay:
        raise CommerceResolutionError("commerce quote already has a non-settled order")
    current_time = now or utcnow()
    expires_at = datetime.fromisoformat(str(stored["expiresAt"]).replace("Z", "+00:00"))
    if not settled_replay:
        if quote.status != "open" or int(quote.current_batch_video_count or 0) <= 0:
            raise CommerceResolutionError("commerce quote is not currently payable")
        if expires_at <= current_time:
            raise CommerceResolutionError("commerce quote has expired")

    return {key: stored[key] for key in _OUTPUT_KEYS}


def validate_stored_commerce_projection(stored: dict) -> None:
    expected_keys = {
        "schema",
        "enabled",
        "tenantId",
        "principalId",
        *_OUTPUT_KEYS,
    }
    if set(stored) not in (expected_keys, expected_keys | {PAYMENT_IDEMPOTENCY_KEY}):
        raise CommerceResolutionError("commerce quote projection shape is invalid")
    if (
        stored.get("schema") != COMMERCE_SCHEMA
        or stored.get("enabled") is not True
        or stored.get("provider") != "icmfyi-acp"
        or stored.get("toolName") != YOUTUBE_INGEST_TOOL
    ):
        raise CommerceResolutionError("commerce quote projection identity is invalid")
    try:
        offering = get_acp_offering(str(stored.get("offeringId") or ""))
    except ValueError as exc:
        raise CommerceResolutionError("commerce offering is invalid") from exc
    work_payload = stored.get("workPayload")
    if (
        not isinstance(work_payload, dict)
        or set(work_payload) != {"schema", "operation", "quoteId", "packId"}
        or work_payload.get("schema") != WORK_SCHEMA
        or work_payload.get("operation") != WORK_OPERATION
    ):
        raise CommerceResolutionError("commerce work payload is invalid")
    if work_payload.get("quoteId") != stored.get("quoteId"):
        raise CommerceResolutionError("commerce work payload quote does not match")
    if _sha256_json(work_payload) != stored.get("requestHash"):
        raise CommerceResolutionError("commerce work payload hash does not match")
    if not _SHA256_PATTERN.fullmatch(str(stored.get("quoteHash") or "")):
        raise CommerceResolutionError("commerce quote hash is invalid")
    expected_quote_hash = _sha256_json(
        {
            **{key: stored.get(key) for key in _OUTPUT_KEYS if key != "quoteHash"},
            "tenantId": stored.get("tenantId"),
            "principalId": stored.get("principalId"),
        }
    )
    if not hmac.compare_digest(str(stored["quoteHash"]), expected_quote_hash):
        raise CommerceResolutionError("commerce quote projection hash does not match")
    if PAYMENT_IDEMPOTENCY_KEY in stored:
        binding = stored.get(PAYMENT_IDEMPOTENCY_KEY)
        if not isinstance(binding, str) or not _IDEMPOTENCY_PATTERN.fullmatch(binding):
            raise CommerceResolutionError("commerce idempotency binding is invalid")
    try:
        amount = int(str(stored.get("amountAtomic") or ""))
        expires_at = datetime.fromisoformat(
            str(stored.get("expiresAt") or "").replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise CommerceResolutionError("commerce amount or expiry is invalid") from exc
    configured_asset, atomic_units_per_cent = _payment_asset_contract()
    if (
        amount < 1
        or expires_at.tzinfo is None
        or not _ASSET_PATTERN.fullmatch(str(stored.get("asset") or ""))
        or stored.get("asset") != configured_asset
        or amount != offering.fixed_price_cents * atomic_units_per_cent
        or not stored.get("quoteId")
        or offering.offering_id != stored.get("offeringId")
    ):
        raise CommerceResolutionError("commerce amount or asset is invalid")


def _offering_id_for_quote(quote: ChannelQuote) -> str | None:
    request_json = dict(quote.request_json or {})
    offering_id = (
        "transcript_pack_expansion_25"
        if request_json.get("pack_id")
        else "transcript_pack_starter_10"
    )
    offering = get_acp_offering(offering_id)
    if quote.mode != offering.mode:
        return None
    if int(quote.requested_max_videos or 0) > offering.max_videos:
        return None
    if offering.requires_pack_id != bool(request_json.get("pack_id")):
        return None
    return offering_id


def _env_enabled(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_json(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
