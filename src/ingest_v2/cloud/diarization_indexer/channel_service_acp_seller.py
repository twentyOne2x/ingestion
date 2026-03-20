from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


logging.basicConfig(
    level=os.getenv("CHANNEL_SERVICE_ACP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("channel_service_acp_seller")


@dataclass(frozen=True)
class SellerRuntimeConfig:
    bridge_base_url: str
    bridge_secret: str
    poll_interval_s: float
    delivery_timeout_s: int
    request_timeout_s: float
    acp_network: str
    contract_version: str
    default_offering_id: str
    skip_socket_connection: bool


class BridgeError(RuntimeError):
    pass


class AcpBridgeClient:
    def __init__(self, *, base_url: str, shared_secret: str, request_timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-acp-shared-secret": shared_secret,
                "content-type": "application/json",
            }
        )

    def sync_job(self, payload: dict) -> dict:
        return self._request("POST", "/v1/channel-packs/acp/jobs", json_body=payload)

    def get_job(self, acp_job_id: str) -> dict:
        return self._request("GET", f"/v1/channel-packs/acp/jobs/{acp_job_id}")

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, json=json_body, timeout=self.timeout)
        if response.status_code >= 400:
            detail = None
            try:
                body = response.json()
                detail = body.get("detail") or body.get("error")
            except Exception:
                detail = response.text.strip() or None
            raise BridgeError(f"{method} {path} failed with {response.status_code}: {detail or 'unknown error'}")
        try:
            return response.json()
        except Exception as exc:
            raise BridgeError(f"{method} {path} returned non-JSON response") from exc


class DeliveryPollRegistry:
    def __init__(self):
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def ensure(self, job_id: str, target) -> bool:
        with self._lock:
            thread = self._threads.get(job_id)
            if thread is not None and thread.is_alive():
                return False
            thread = threading.Thread(target=target, name=f"acp-delivery-{job_id}", daemon=True)
            self._threads[job_id] = thread
            thread.start()
            return True


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def load_runtime_config() -> SellerRuntimeConfig:
    bridge_base_url = (
        (os.getenv("CHANNEL_SERVICE_ACP_BRIDGE_URL") or "").strip()
        or (os.getenv("INGESTION_SERVICE_URL") or "").strip()
        or "http://ingestion-api:8080"
    )
    bridge_secret = (os.getenv("ACP_SHARED_SECRET") or "").strip()
    if not bridge_secret:
        raise RuntimeError("ACP_SHARED_SECRET is required for ACP seller runtime")
    default_offering_id = (os.getenv("CHANNEL_SERVICE_ACP_DEFAULT_OFFERING_ID") or "").strip() or "transcript_pack_starter_10"
    return SellerRuntimeConfig(
        bridge_base_url=bridge_base_url.rstrip("/"),
        bridge_secret=bridge_secret,
        poll_interval_s=max(1.0, float(os.getenv("CHANNEL_SERVICE_ACP_POLL_INTERVAL_S") or "10")),
        delivery_timeout_s=max(60, int(os.getenv("CHANNEL_SERVICE_ACP_DELIVERY_TIMEOUT_S") or "5400")),
        request_timeout_s=max(1.0, float(os.getenv("CHANNEL_SERVICE_ACP_REQUEST_TIMEOUT_S") or "30")),
        acp_network=(os.getenv("CHANNEL_SERVICE_ACP_NETWORK") or "base").strip().lower(),
        contract_version=(os.getenv("CHANNEL_SERVICE_ACP_CONTRACT_VERSION") or "v2").strip().lower(),
        default_offering_id=default_offering_id,
        skip_socket_connection=_env_bool("CHANNEL_SERVICE_ACP_SKIP_SOCKET_CONNECTION", False),
    )


def _normalize_json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return decoded
    return {}


def _offering_aliases() -> dict[str, str]:
    aliases = {
        "transcript_pack_starter_10": "transcript_pack_starter_10",
        "starter pack 10": "transcript_pack_starter_10",
        "youtube transcript pack starter 10": "transcript_pack_starter_10",
        "youtubetranscriptpackstarter10": "transcript_pack_starter_10",
        "transcript_pack_expansion_25": "transcript_pack_expansion_25",
        "expansion pack 25": "transcript_pack_expansion_25",
        "youtube transcript pack expansion 25": "transcript_pack_expansion_25",
        "youtubetranscriptpackexpansion25": "transcript_pack_expansion_25",
    }
    raw = (os.getenv("CHANNEL_SERVICE_ACP_JOB_NAME_MAP_JSON") or "").strip()
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                for key, value in decoded.items():
                    aliases[str(key).strip().lower()] = str(value).strip()
        except json.JSONDecodeError:
            logger.warning("CHANNEL_SERVICE_ACP_JOB_NAME_MAP_JSON is invalid JSON; using built-in aliases only")
    return aliases


def _resolve_offering_id(*, job_name: Optional[str], requirement: dict, default_offering_id: str) -> str:
    explicit = str(requirement.get("offering_id") or "").strip()
    if explicit:
        return explicit
    aliases = _offering_aliases()
    normalized_job_name = str(job_name or "").strip().lower()
    if normalized_job_name and normalized_job_name in aliases:
        return aliases[normalized_job_name]
    return default_offering_id


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_job_input(raw_requirement: Any) -> dict:
    requirement = _normalize_json_object(raw_requirement)
    if not requirement:
        raise ValueError("ACP job requirement must be a JSON object")
    input_payload = _normalize_json_object(requirement.get("input"))
    if input_payload:
        requirement = {**requirement, **input_payload}
    channel_handle = str(requirement.get("channel_handle") or "").strip()
    if not channel_handle:
        raise ValueError("ACP job requirement must include channel_handle")

    normalized: dict[str, Any] = {"channel_handle": channel_handle}
    for key in ("max_videos", "namespace", "language", "pack_id", "published_after", "published_before"):
        if requirement.get(key) is not None:
            normalized[key] = requirement.get(key)
    if "prefer_auto" in requirement:
        normalized["prefer_auto"] = _normalize_bool(requirement.get("prefer_auto"))
    return normalized


def _phase_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name).strip().lower()
    return str(value).strip().lower()


def _build_bridge_payload(*, job, offering_id: str, input_payload: dict, payment_status: str) -> dict:
    return {
        "acp_job_id": str(job.id),
        "offering_id": offering_id,
        "input": input_payload,
        "buyer": {
            "subject_type": "acp_client",
            "subject_id": str(job.client_address or "").strip() or None,
        },
        "payment": {
            "provider": "acp",
            "status": payment_status,
        },
        "meta": {
            "job_phase": str(job.phase),
            "job_name": getattr(job, "name", None),
        },
    }


def _requirement_message(bridge_job: dict) -> str:
    status = str(bridge_job.get("status") or "").strip()
    payload = dict(bridge_job.get("delivery") or {})
    input_payload = dict(bridge_job.get("input") or {})
    channel_handle = str(input_payload.get("channel_handle") or "").strip()
    target_video_count = int(payload.get("target_video_count") or input_payload.get("max_videos") or 0)
    if status == "awaiting_payment":
        return (
            f"Accepted transcript pack request for {channel_handle}. "
            f"After ACP payment settles, I will deliver up to {target_video_count} transcript-ready videos "
            "with manifest, metadata, links, transcripts, and bundle archive."
        )
    if status == "acquiring":
        return (
            f"Accepted transcript pack request for {channel_handle}. "
            "Channel discovery is in progress; once ACP payment settles I will deliver as soon as the starter batch is ready."
        )
    if status == "ready_for_delivery":
        return (
            f"Accepted transcript pack request for {channel_handle}. "
            "The transcript pack is already ready and will be delivered immediately after payment."
        )
    return "Accepted transcript pack request. After ACP payment settles, I will deliver the transcript pack result."


def _delivery_payload(bridge_job: dict) -> dict:
    delivery = dict(bridge_job.get("delivery") or {})
    exports = dict(delivery.get("exports") or {})
    return {
        "type": "transcript_pack",
        "pack_id": delivery.get("pack_id"),
        "ready_video_count": delivery.get("ready_video_count"),
        "manifest_url": exports.get("manifest_url"),
        "archive_url": exports.get("archive_url"),
        "exports": exports,
    }


def _load_acp_sdk(config: SellerRuntimeConfig):
    try:
        from virtuals_acp.client import VirtualsACP
        from virtuals_acp.configs.configs import BASE_MAINNET_CONFIG_V2, BASE_SEPOLIA_CONFIG_V2
        from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2
    except ImportError as exc:
        raise RuntimeError(
            "virtuals_acp is not installed. Rebuild the ingestion image after updating requirements."
        ) from exc

    if config.contract_version != "v2":
        raise RuntimeError(f"Unsupported ACP contract version: {config.contract_version}")
    if config.acp_network == "base-sepolia":
        contract_config = BASE_SEPOLIA_CONFIG_V2
    elif config.acp_network == "base":
        contract_config = BASE_MAINNET_CONFIG_V2
    else:
        raise RuntimeError(f"Unsupported CHANNEL_SERVICE_ACP_NETWORK: {config.acp_network}")
    return VirtualsACP, ACPContractClientV2, contract_config


class AcpSellerRuntime:
    def __init__(self, config: SellerRuntimeConfig):
        self.config = config
        self.bridge = AcpBridgeClient(
            base_url=config.bridge_base_url,
            shared_secret=config.bridge_secret,
            request_timeout_s=config.request_timeout_s,
        )
        self.poll_registry = DeliveryPollRegistry()

    def on_new_task(self, job, memo_to_sign=None) -> None:
        phase = _phase_name(getattr(job, "phase", ""))
        next_phase = _phase_name(getattr(memo_to_sign, "next_phase", "")) if memo_to_sign is not None else None
        logger.info("received ACP job id=%s phase=%s next_phase=%s name=%s", job.id, phase, next_phase, getattr(job, "name", None))
        try:
            if phase == "request" and next_phase == "negotiation":
                self._handle_request_phase(job)
                return
            if phase == "transaction" and next_phase == "evaluation":
                self._handle_transaction_phase(job)
                return
            if phase == "completed":
                logger.info("ACP job %s completed", job.id)
                return
            if phase == "rejected":
                logger.info("ACP job %s rejected: %s", job.id, getattr(job, "rejection_reason", None))
                return
            logger.info("ACP job %s ignored for unsupported phase transition phase=%s next=%s", job.id, phase, next_phase)
        except Exception as exc:
            logger.exception("ACP job %s handler failed: %s", job.id, exc)
            try:
                if phase in {"request", "transaction"}:
                    job.reject(f"Transcript-pack backend failed: {exc}")
            except Exception:
                logger.exception("ACP job %s reject fallback also failed", job.id)

    def _handle_request_phase(self, job) -> None:
        input_payload = _normalize_job_input(getattr(job, "requirement", None))
        offering_id = _resolve_offering_id(
            job_name=getattr(job, "name", None),
            requirement=input_payload,
            default_offering_id=self.config.default_offering_id,
        )
        bridge_job = self.bridge.sync_job(
            _build_bridge_payload(
                job=job,
                offering_id=offering_id,
                input_payload=input_payload,
                payment_status="awaiting_acp_payment",
            )
        )
        status = str(bridge_job.get("status") or "").strip()
        if status in {"failed", "unavailable"}:
            error_detail = bridge_job.get("error_detail") or f"ACP job {job.id} is not fulfillable"
            logger.warning("rejecting ACP request job=%s status=%s detail=%s", job.id, status, error_detail)
            job.reject(str(error_detail))
            return
        logger.info("accepting ACP request job=%s bridge_status=%s", job.id, status)
        job.accept(_requirement_message(bridge_job))
        job.create_requirement(_requirement_message(bridge_job))

    def _handle_transaction_phase(self, job) -> None:
        input_payload = _normalize_job_input(getattr(job, "requirement", None))
        offering_id = _resolve_offering_id(
            job_name=getattr(job, "name", None),
            requirement=input_payload,
            default_offering_id=self.config.default_offering_id,
        )
        bridge_job = self.bridge.sync_job(
            _build_bridge_payload(
                job=job,
                offering_id=offering_id,
                input_payload=input_payload,
                payment_status="settled_acp",
            )
        )
        status = str(bridge_job.get("status") or "").strip()
        if status == "ready_for_delivery":
            logger.info("delivering ACP job=%s immediately", job.id)
            job.deliver(_delivery_payload(bridge_job))
            return
        if status in {"failed", "unavailable"}:
            error_detail = bridge_job.get("error_detail") or f"ACP job {job.id} failed before delivery"
            logger.warning("rejecting ACP transaction job=%s status=%s detail=%s", job.id, status, error_detail)
            job.reject(str(error_detail))
            return

        def wait_and_deliver() -> None:
            deadline = time.time() + self.config.delivery_timeout_s
            while time.time() < deadline:
                current = self.bridge.get_job(str(job.id))
                current_status = str(current.get("status") or "").strip()
                if current_status == "ready_for_delivery":
                    logger.info("delivering ACP job=%s after polling", job.id)
                    job.deliver(_delivery_payload(current))
                    return
                if current_status in {"failed", "unavailable"}:
                    error_detail = current.get("error_detail") or f"ACP job {job.id} failed during fulfillment"
                    logger.warning("rejecting ACP job=%s after polling failure=%s", job.id, error_detail)
                    job.reject(str(error_detail))
                    return
                next_poll = float(current.get("next_poll_after_seconds") or self.config.poll_interval_s)
                time.sleep(max(self.config.poll_interval_s, next_poll))
            timeout_reason = f"Transcript pack delivery timed out after {self.config.delivery_timeout_s} seconds"
            logger.error("ACP job=%s timed out waiting for delivery", job.id)
            job.reject(timeout_reason)

        started = self.poll_registry.ensure(str(job.id), wait_and_deliver)
        if started:
            logger.info("started ACP delivery poll loop for job=%s", job.id)
        else:
            logger.info("ACP delivery poll loop already active for job=%s", job.id)


def main() -> None:
    config = load_runtime_config()
    logger.info(
        "starting ACP seller runtime network=%s bridge=%s skip_socket=%s",
        config.acp_network,
        config.bridge_base_url,
        config.skip_socket_connection,
    )
    VirtualsACP, ACPContractClientV2, contract_config = _load_acp_sdk(config)
    runtime = AcpSellerRuntime(config)
    contract_client = ACPContractClientV2(
        wallet_private_key=(os.getenv("WHITELISTED_WALLET_PRIVATE_KEY") or "").strip(),
        agent_wallet_address=(os.getenv("SELLER_AGENT_WALLET_ADDRESS") or "").strip(),
        entity_id=int(str(os.getenv("SELLER_ENTITY_ID") or "0").strip()),
        config=contract_config,
    )
    VirtualsACP(
        acp_contract_clients=contract_client,
        on_new_task=runtime.on_new_task,
        skip_socket_connection=config.skip_socket_connection,
    )
    logger.info("ACP seller runtime connected and waiting for tasks")
    threading.Event().wait()


if __name__ == "__main__":
    main()
