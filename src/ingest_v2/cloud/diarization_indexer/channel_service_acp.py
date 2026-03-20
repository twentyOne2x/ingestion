from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time
import os
from typing import Any, Dict, Optional

_ACP_QUOTE_TTL_HOURS = 6


@dataclass(frozen=True)
class AcpOffering:
    offering_id: str
    name: str
    headline: str
    description: str
    fixed_price_cents: int
    max_videos: int
    mode: str
    requires_pack_id: bool = False
    default_namespace: str = "videos"
    default_language: str = "en"
    prefer_auto: bool = True
    sla_minutes: int = 120


ACP_OFFERINGS: tuple[AcpOffering, ...] = (
    AcpOffering(
        offering_id="transcript_pack_starter_10",
        name="Starter Pack 10",
        headline="Get up to 10 transcript-ready videos from one YouTube channel as a normalized export pack",
        description=(
            "Builds a starter transcript pack for one YouTube channel. Returns manifest, videos, links, "
            "transcripts, and a bundle archive. Includes up to 10 transcript-ready videos."
        ),
        fixed_price_cents=1,
        max_videos=10,
        mode="recent_pack",
        requires_pack_id=False,
        sla_minutes=90,
    ),
    AcpOffering(
        offering_id="transcript_pack_expansion_25",
        name="Expansion Pack 25",
        headline="Add up to 25 more transcript-ready videos to an existing transcript pack",
        description=(
            "Extends an existing transcript pack with the next batch of transcript-ready channel videos. "
            "Returns refreshed manifest, videos, links, transcripts, and a bundle archive."
        ),
        fixed_price_cents=500,
        max_videos=25,
        mode="recent_pack",
        requires_pack_id=True,
        sla_minutes=180,
    ),
)

_ACP_OFFERINGS_BY_ID = {offering.offering_id: offering for offering in ACP_OFFERINGS}


def get_acp_offering(offering_id: str) -> AcpOffering:
    offering = _ACP_OFFERINGS_BY_ID.get(str(offering_id or "").strip())
    if offering is None:
        raise ValueError(f"Unsupported ACP offering id: {offering_id}")
    return offering


def list_acp_offerings(
    *,
    base_url: str,
    visible_offering_ids: Optional[list[str]] = None,
    readiness: Optional[dict] = None,
) -> dict:
    if visible_offering_ids is None:
        visible = {offering.offering_id for offering in ACP_OFFERINGS}
    else:
        visible = set(visible_offering_ids)
    return {
        "ok": True,
        "publication_state": readiness.get("publication_state") if readiness else None,
        "acceptance_scope": readiness.get("acceptance_scope") if readiness else None,
        "reason_codes": list(readiness.get("reason_codes") or []) if readiness else [],
        "offerings": [
            serialize_acp_offering(offering=offering, base_url=base_url)
            for offering in ACP_OFFERINGS
            if offering.offering_id in visible
        ],
    }


def serialize_acp_offering(*, offering: AcpOffering, base_url: str) -> dict:
    channel_catalog_url = _public_catalog_url(base_url)
    properties = {
        "channel_handle": {
            "type": "string",
            "description": "YouTube channel handle such as @OpenAI",
        },
        "max_videos": {
            "type": "integer",
            "minimum": 1,
            "maximum": offering.max_videos,
            "default": offering.max_videos,
            "description": f"Requested maximum videos for this fixed-price offer. Cap: {offering.max_videos}.",
        },
        "namespace": {
            "type": "string",
            "default": offering.default_namespace,
            "description": "Corpus namespace to fulfill against. Default is videos.",
        },
        "language": {
            "type": "string",
            "default": offering.default_language,
            "description": "Transcript language. Default is en.",
        },
        "prefer_auto": {
            "type": "boolean",
            "default": offering.prefer_auto,
            "description": "Prefer automatic YouTube subtitles when manual captions are unavailable.",
        },
        "published_after": {
            "type": "string",
            "description": "Optional inclusive lower bound in YYYY-MM-DD.",
        },
        "published_before": {
            "type": "string",
            "description": "Optional inclusive upper bound in YYYY-MM-DD.",
        },
    }
    required = ["channel_handle"]
    if offering.requires_pack_id:
        properties["pack_id"] = {
            "type": "string",
            "description": "Existing transcript pack id to expand.",
        }
        required.append("pack_id")
    return {
        "offering_id": offering.offering_id,
        "name": offering.name,
        "headline": offering.headline,
        "description": offering.description,
        "pricing": {
            "currency": "USD",
            "amount_cents": offering.fixed_price_cents,
            "display": f"${offering.fixed_price_cents / 100:.2f}",
        },
        "defaults": {
            "mode": offering.mode,
            "max_videos": offering.max_videos,
            "namespace": offering.default_namespace,
            "language": offering.default_language,
            "prefer_auto": offering.prefer_auto,
        },
        "delivery": {
            "format": ["manifest.json", "videos.ndjson", "links.ndjson", "transcripts.ndjson", "bundle.zip"],
            "catalog_channels_url": channel_catalog_url,
            "order_status_url_template": _public_acp_job_url_template(base_url),
        },
        "sla": {
            "estimated_minutes": offering.sla_minutes,
        },
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
    }


def create_or_sync_acp_job(*, session, payload: dict, base_url: str) -> AcpJobBridge:
    from .channel_service_logic import persist_quote, plan_quote
    from .channel_service_store import AcpJobBridge, utcnow

    acp_job_id = str(payload.get("acp_job_id") or "").strip()
    if not acp_job_id:
        raise ValueError("acp_job_id is required")
    offering = get_acp_offering(str(payload.get("offering_id") or "").strip())
    normalized = normalize_acp_job_payload(payload=payload, offering=offering)

    bridge = session.get(AcpJobBridge, acp_job_id)
    if bridge is None:
        plan_started = time.perf_counter()
        plan = plan_quote(
            session=session,
            channel_handle=normalized["channel_handle"],
            namespace=normalized["namespace"],
            mode=offering.mode,
            max_videos=normalized["max_videos"],
            language=normalized["language"],
            prefer_auto=normalized["prefer_auto"],
            pack_id=normalized.get("pack_id"),
            published_after=normalized.get("published_after"),
            published_before=normalized.get("published_before"),
        )
        planning_latency_ms = int((time.perf_counter() - plan_started) * 1000)
        quote = persist_quote(
            session=session,
            request_payload={
                "channel_handle": normalized["channel_handle"],
                "max_videos": normalized["max_videos"],
                "namespace": normalized["namespace"],
                "mode": offering.mode,
                "language": normalized["language"],
                "prefer_auto": normalized["prefer_auto"],
                "pack_id": normalized.get("pack_id"),
                "published_after": normalized.get("published_after"),
                "published_before": normalized.get("published_before"),
                "source": "acp",
                "acp_job_id": acp_job_id,
                "offering_id": offering.offering_id,
            },
            plan=plan,
            planning_latency_ms=planning_latency_ms,
        )
        quote.expires_at = utcnow() + timedelta(hours=_ACP_QUOTE_TTL_HOURS)
        bridge = AcpJobBridge(
            acp_job_id=acp_job_id,
            offering_id=offering.offering_id,
            status="received",
            quote_id=quote.id,
            fixed_price_cents=offering.fixed_price_cents,
            currency="USD",
            payment_provider=normalized["payment_provider"],
            payment_status=normalized["payment_status"],
            buyer_subject_type=normalized.get("buyer_subject_type"),
            buyer_subject_id=normalized.get("buyer_subject_id"),
            request_json=normalized,
            delivery_json={},
            error_detail=None,
        )
        session.add(bridge)
        session.flush()
    else:
        _ensure_matching_acp_request(bridge=bridge, normalized=normalized, offering=offering)
        bridge.payment_provider = normalized["payment_provider"]
        bridge.payment_status = normalized["payment_status"]
        bridge.buyer_subject_type = normalized.get("buyer_subject_type")
        bridge.buyer_subject_id = normalized.get("buyer_subject_id")
        bridge.request_json = {
            **dict(bridge.request_json or {}),
            **normalized,
        }

    refresh_acp_job(session=session, bridge=bridge, base_url=base_url)
    return bridge


def refresh_acp_job(*, session, bridge: AcpJobBridge, base_url: str) -> AcpJobBridge:
    from sqlalchemy import select

    from .channel_service_logic import create_checkout_session_with_payment, create_order_from_quote, refresh_quote_state
    from .channel_service_store import ChannelOrder, ChannelPack, ChannelQuote, CheckoutSessionRecord, PackBatch, utcnow

    quote = session.get(ChannelQuote, bridge.quote_id) if bridge.quote_id else None
    if quote is None:
        bridge.status = "failed"
        bridge.error_detail = f"quote {bridge.quote_id} was not found"
        bridge.delivery_json = {}
        session.flush()
        return bridge

    refresh_quote_state(session=session, quote=quote, enqueue_missing=True)
    quote.expires_at = utcnow() + timedelta(hours=_ACP_QUOTE_TTL_HOURS)
    request_json = dict(bridge.request_json or {})
    target_video_count = int(request_json.get("max_videos") or 0)
    pending_count = _quote_pending_video_count(quote)
    payment_is_settled = _payment_status_is_settled(bridge.payment_status)

    checkout = session.get(CheckoutSessionRecord, bridge.checkout_session_id) if bridge.checkout_session_id else None
    order = session.get(ChannelOrder, bridge.order_id) if bridge.order_id else None
    batch = session.get(PackBatch, order.batch_id) if order is not None else None
    pack = session.get(ChannelPack, order.pack_id) if order is not None else None

    if order is None and payment_is_settled and _quote_ready_for_checkout(quote=quote, target_video_count=target_video_count):
        if checkout is None:
            checkout = create_checkout_session_with_payment(
                session=session,
                quote_ids=[quote.id],
                idempotency_key=f"acp:{bridge.acp_job_id}",
                payment_provider=bridge.payment_provider,
                payment_status=bridge.payment_status,
                line_item_amount_overrides={quote.id: int(bridge.fixed_price_cents or 0)},
            )
            bridge.checkout_session_id = checkout.id

        existing_order = session.execute(
            select(ChannelOrder).where(
                ChannelOrder.checkout_session_id == checkout.id,
                ChannelOrder.quote_id == quote.id,
            )
        ).scalar_one_or_none()
        if existing_order is not None:
            order = existing_order
            batch = session.get(PackBatch, order.batch_id)
            pack = session.get(ChannelPack, order.pack_id)
        else:
            pack, batch, order = create_order_from_quote(
                session=session,
                quote=quote,
                checkout=checkout,
                pack_id=request_json.get("pack_id"),
                buyer_subject_type=bridge.buyer_subject_type,
                buyer_subject_id=bridge.buyer_subject_id,
                external_payment={
                    "provider": bridge.payment_provider,
                    "payment_status": bridge.payment_status,
                    "amount_cents": int(bridge.fixed_price_cents or 0),
                    "receipt_status": "settled",
                    "receipt_json": {
                        "provider": bridge.payment_provider,
                        "acp_job_id": bridge.acp_job_id,
                        "offering_id": bridge.offering_id,
                        "fixed_price_cents": int(bridge.fixed_price_cents or 0),
                    },
                },
            )
        bridge.order_id = order.id if order is not None else bridge.order_id
        bridge.pack_id = pack.id if pack is not None else bridge.pack_id

    if order is not None and batch is not None and pack is not None:
        bridge.status = _bridge_status_from_order(order=order, batch=batch, pack=pack)
        bridge.error_detail = None
        bridge.delivery_json = build_acp_delivery_payload(
            bridge=bridge,
            quote=quote,
            order=order,
            batch=batch,
            pack=pack,
            base_url=base_url,
        )
    else:
        if quote.status == "unavailable" and pending_count == 0:
            bridge.status = "unavailable"
            bridge.error_detail = "No transcript-ready videos are currently available for this ACP offer."
            bridge.delivery_json = {}
        elif _quote_ready_for_checkout(quote=quote, target_video_count=target_video_count) and not payment_is_settled:
            bridge.status = "awaiting_payment"
            bridge.error_detail = None
            bridge.delivery_json = {
                "quote_id": quote.id,
                "state": "awaiting_payment",
                "included_video_count": int(quote.included_video_count or 0),
                "pending_video_count": pending_count,
                "target_video_count": target_video_count,
                "next_poll_after_seconds": poll_after_seconds_for_status("awaiting_payment"),
            }
        else:
            bridge.status = "acquiring"
            bridge.error_detail = None
            bridge.delivery_json = {
                "quote_id": quote.id,
                "state": "acquiring",
                "included_video_count": int(quote.included_video_count or 0),
                "pending_video_count": pending_count,
                "target_video_count": target_video_count,
                "next_poll_after_seconds": poll_after_seconds_for_status("acquiring"),
            }

    session.flush()
    return bridge


def serialize_acp_job(*, session, bridge: AcpJobBridge, base_url: str) -> dict:
    from .channel_service_logic import serialize_order
    from .channel_service_store import ChannelOrder, ChannelPack, ChannelQuote, CheckoutSessionRecord, PackBatch

    quote = session.get(ChannelQuote, bridge.quote_id) if bridge.quote_id else None
    checkout = session.get(CheckoutSessionRecord, bridge.checkout_session_id) if bridge.checkout_session_id else None
    order = session.get(ChannelOrder, bridge.order_id) if bridge.order_id else None
    batch = session.get(PackBatch, order.batch_id) if order is not None else None
    pack = session.get(ChannelPack, order.pack_id) if order is not None else None

    quote_summary = None
    if quote is not None:
        pending_count = _quote_pending_video_count(quote)
        quote_summary = {
            "quote_id": quote.id,
            "status": quote.status,
            "included_video_count": int(quote.included_video_count or 0),
            "pending_video_count": pending_count,
            "excluded_video_count": int(quote.excluded_video_count or 0),
            "current_batch_index": int(quote.current_batch_index or 0),
            "current_batch_video_count": int(quote.current_batch_video_count or 0),
            "expires_at": quote.expires_at.isoformat(),
        }

    return {
        "ok": True,
        "acp_job_id": bridge.acp_job_id,
        "offering_id": bridge.offering_id,
        "status": bridge.status,
        "pricing": {
            "currency": bridge.currency,
            "amount_cents": int(bridge.fixed_price_cents or 0),
        },
        "buyer": {
            "subject_type": bridge.buyer_subject_type,
            "subject_id": bridge.buyer_subject_id,
        },
        "input": bridge.request_json or {},
        "quote": quote_summary,
        "checkout_session": {
            "checkout_session_id": checkout.id,
            "status": checkout.status,
            "payment_provider": checkout.payment_provider,
            "payment_status": checkout.payment_status,
        }
        if checkout is not None
        else None,
        "order": serialize_order(order, batch, pack) if order is not None and batch is not None and pack is not None else None,
        "delivery": dict(bridge.delivery_json or {}),
        "next_poll_after_seconds": poll_after_seconds_for_status(bridge.status),
        "error_detail": bridge.error_detail,
        "offerings_url": _join_url(base_url, "/v1/channel-packs/acp/offerings"),
        "catalog_channels_url": _join_url(base_url, "/v1/channel-packs/catalog/channels"),
    }


def normalize_acp_job_payload(*, payload: dict, offering: AcpOffering) -> dict:
    request_input = dict(payload.get("input") or {})
    channel_handle = str(request_input.get("channel_handle") or "").strip()
    if not channel_handle:
        raise ValueError("input.channel_handle is required")

    pack_id = str(request_input.get("pack_id") or "").strip() or None
    if offering.requires_pack_id and not pack_id:
        raise ValueError(f"input.pack_id is required for offering {offering.offering_id}")
    if not offering.requires_pack_id and pack_id:
        raise ValueError(f"input.pack_id is not supported for offering {offering.offering_id}")

    max_videos = int(request_input.get("max_videos") or offering.max_videos)
    if max_videos <= 0 or max_videos > offering.max_videos:
        raise ValueError(f"input.max_videos must be between 1 and {offering.max_videos}")

    namespace = str(request_input.get("namespace") or offering.default_namespace).strip() or offering.default_namespace
    language = str(request_input.get("language") or offering.default_language).strip() or offering.default_language
    prefer_auto = bool(request_input.get("prefer_auto", offering.prefer_auto))

    buyer = dict(payload.get("buyer") or {})
    payment = dict(payload.get("payment") or {})

    return {
        "acp_job_id": str(payload.get("acp_job_id") or "").strip(),
        "offering_id": offering.offering_id,
        "channel_handle": channel_handle,
        "pack_id": pack_id,
        "mode": offering.mode,
        "max_videos": max_videos,
        "namespace": namespace,
        "language": language,
        "prefer_auto": prefer_auto,
        "published_after": str(request_input.get("published_after") or "").strip() or None,
        "published_before": str(request_input.get("published_before") or "").strip() or None,
        "buyer_subject_type": str(buyer.get("subject_type") or "acp_buyer").strip() or "acp_buyer",
        "buyer_subject_id": str(buyer.get("subject_id") or "").strip() or None,
        "payment_provider": str(payment.get("provider") or "acp").strip() or "acp",
        "payment_status": str(payment.get("status") or "settled_acp").strip() or "settled_acp",
        "meta": dict(payload.get("meta") or {}),
    }


def build_acp_delivery_payload(
    *,
    bridge: AcpJobBridge,
    quote: ChannelQuote,
    order: ChannelOrder,
    batch: PackBatch,
    pack: ChannelPack,
    base_url: str,
) -> dict:
    exports = {
        "manifest_url": _public_pack_manifest_url(base_url, pack.id),
        "manifest_export_url": _public_pack_export_url(base_url, pack.id, "manifest"),
        "videos_url": _public_pack_export_url(base_url, pack.id, "videos"),
        "links_url": _public_pack_export_url(base_url, pack.id, "links"),
        "transcripts_url": _public_pack_export_url(base_url, pack.id, "transcripts"),
        "archive_url": _public_pack_export_url(base_url, pack.id, "archive"),
    }
    return {
        "state": "ready_for_delivery" if batch.ready_video_count else "processing",
        "acp_job_id": bridge.acp_job_id,
        "pack_id": pack.id,
        "order_id": order.id,
        "batch_id": batch.id,
        "channel_handle": pack.channel_handle,
        "channel_name": pack.resolved_channel_name,
        "billable_video_count": int(batch.billable_video_count or 0),
        "ready_video_count": int(batch.ready_video_count or 0),
        "requested_video_count": int((bridge.request_json or {}).get("max_videos") or 0),
        "quote_id": quote.id,
        "exports": exports,
    }


def poll_after_seconds_for_status(status: str) -> int:
    normalized = str(status or "").strip().lower()
    if normalized == "acquiring":
        return 10
    if normalized == "processing":
        return 15
    if normalized == "awaiting_payment":
        return 0
    return 0


def _ensure_matching_acp_request(*, bridge: AcpJobBridge, normalized: dict, offering: AcpOffering) -> None:
    stored = dict(bridge.request_json or {})
    comparable_keys = (
        "acp_job_id",
        "offering_id",
        "channel_handle",
        "pack_id",
        "mode",
        "max_videos",
        "namespace",
        "language",
        "prefer_auto",
        "published_after",
        "published_before",
    )
    for key in comparable_keys:
        if stored.get(key) != normalized.get(key):
            raise ValueError(f"acp job {bridge.acp_job_id} was already created with different {key}")
    if bridge.offering_id != offering.offering_id:
        raise ValueError(f"acp job {bridge.acp_job_id} was already created for a different offering")


def _bridge_status_from_order(*, order: ChannelOrder, batch: PackBatch, pack: ChannelPack) -> str:
    if order.payment_status == "requires_payment" or order.status == "awaiting_payment":
        return "awaiting_payment"
    if pack.export_paths_json and int(batch.ready_video_count or 0) > 0:
        return "ready_for_delivery"
    if order.status in {"queued", "partial", "ready"} or batch.status in {"queued", "partial", "ready"}:
        return "processing"
    return str(order.status or "processing")


def _quote_pending_video_count(quote: ChannelQuote) -> int:
    return sum(1 for row in quote.videos if row.status == "pending_acquisition")


def _quote_ready_for_checkout(*, quote: ChannelQuote, target_video_count: int) -> bool:
    pending_count = _quote_pending_video_count(quote)
    current_batch_video_count = int(quote.current_batch_video_count or 0)
    if current_batch_video_count <= 0:
        return False
    return current_batch_video_count >= max(1, target_video_count) or pending_count == 0


def _payment_status_is_settled(payment_status: Optional[str]) -> bool:
    normalized = str(payment_status or "").strip().lower()
    if not normalized:
        return False
    return normalized in {
        "paid",
        "payment_confirmed",
        "settled",
        "settled_acp",
        "settled_external",
        "succeeded",
    }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _public_proxy_base_url(base_url: str) -> str:
    configured = (os.getenv("CHANNEL_SERVICE_PUBLIC_PROXY_BASE_URL") or "").strip()
    if configured:
        return configured.rstrip("/")
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/api/service"):
        return trimmed
    if trimmed.startswith("https://www.icm.fyi") or trimmed.startswith("https://icm.fyi"):
        return f"{trimmed}/api/service"
    return trimmed


def _public_catalog_url(base_url: str) -> str:
    proxy_base = _public_proxy_base_url(base_url)
    if proxy_base != base_url.rstrip("/"):
        return _join_url(proxy_base, "/catalog/channels?namespace=videos")
    return _join_url(base_url, "/v1/channel-packs/catalog/channels")


def _public_acp_job_url_template(base_url: str) -> str:
    proxy_base = _public_proxy_base_url(base_url)
    if proxy_base != base_url.rstrip("/"):
        return _join_url(proxy_base, "/acp/jobs/{acp_job_id}")
    return _join_url(base_url, "/v1/channel-packs/acp/jobs/{acp_job_id}")


def _public_pack_manifest_url(base_url: str, pack_id: str) -> str:
    proxy_base = _public_proxy_base_url(base_url)
    if proxy_base != base_url.rstrip("/"):
        return _join_url(proxy_base, f"/packs/{pack_id}/manifest")
    return _join_url(base_url, f"/v1/channel-packs/{pack_id}/manifest")


def _public_pack_export_url(base_url: str, pack_id: str, export_name: str) -> str:
    proxy_base = _public_proxy_base_url(base_url)
    if proxy_base != base_url.rstrip("/"):
        return _join_url(proxy_base, f"/packs/{pack_id}/exports/{export_name}")
    return _join_url(base_url, f"/v1/channel-packs/{pack_id}/exports/{export_name}")
