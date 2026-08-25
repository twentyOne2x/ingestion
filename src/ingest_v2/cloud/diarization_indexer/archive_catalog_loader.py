from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .archive_receipts import (
    ArchiveProtocolError,
    acquire_transaction_lock,
    canonical_json_bytes,
    prepare_immutable_receipt_dir,
    read_pinned_regular_file,
    sha256_json,
    write_immutable_json_receipt,
)
from .canonical_media import canonical_source_channel_id, canonical_source_video_id
from .channel_service_store import (
    ArchiveCatalogImport,
    MediaLocation,
    MediaObject,
    SourceChannel,
    SourceVideo,
    VideoMediaRef,
    utcnow,
)

ARCHIVE_CATALOG_SCHEMA = "icmfyi.archive-catalog-import.v1"
ARCHIVE_RECEIPT_SCHEMA = "icmfyi.archive-catalog-import-receipt.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SIDECAR_PATTERN = re.compile(r"([0-9a-f]{64})  ([^/\n]+)\n\Z")
_CATALOG_ACQUISITION_STATES = {
    "pending_discovery": 0,
    "blocked_public_age_gate": 1,
    "partial_only": 2,
    "retained_remote_verified": 3,
}
_ARCHIVE_STATE_RANK = {**_CATALOG_ACQUISITION_STATES, "retained_hot_verified": 4}


class ArchiveCatalogError(ArchiveProtocolError):
    """The immutable archive import packet failed validation or reconciliation."""


@dataclass
class _ImportState:
    source_ids_by_key: dict[str, str] = field(default_factory=dict)
    source_ids: set[str] = field(default_factory=set)
    video_ids: set[str] = field(default_factory=set)
    media_sha256s: set[str] = field(default_factory=set)
    contract_records: list[dict[str, Any]] = field(default_factory=list)
    inventory_records: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "contracts": 0,
            "inventory_summaries": 0,
            "items_created": 0,
            "items_updated": 0,
            "items_unchanged": 0,
            "locations_created": 0,
            "locations_unchanged": 0,
            "locations_updated": 0,
            "media_created": 0,
            "media_unchanged": 0,
            "media_updated": 0,
            "media_variants": 0,
            "records": 0,
            "refs_created": 0,
            "refs_unchanged": 0,
            "sources_created": 0,
            "sources_updated": 0,
            "sources_unchanged": 0,
        }
    )


@dataclass(frozen=True)
class PreparedArchiveCatalog:
    jsonl_path: Path
    sidecar_path: Path
    jsonl_sha256: str
    sidecar_sha256: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ArchiveCatalogApplyResult:
    receipt: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    reconciled: bool


def load_archive_catalog(
    session: Session,
    *,
    jsonl_path: Path,
    sidecar_path: Path,
    expected_jsonl_sha256: str,
    max_records: int = 1_000_000,
    max_line_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Validate and stage one catalog in the caller's transaction.

    Admin callers must use :func:`apply_archive_catalog` so the immutable receipt
    is durable before their transaction can commit. This lower-level function is
    retained for tests and composition inside an already-managed transaction.
    """
    prepared = prepare_archive_catalog(
        jsonl_path=jsonl_path,
        sidecar_path=sidecar_path,
        expected_jsonl_sha256=expected_jsonl_sha256,
        max_records=max_records,
        max_line_bytes=max_line_bytes,
    )
    return _load_prepared_archive_catalog(session, prepared)


def prepare_archive_catalog(
    *,
    jsonl_path: Path,
    sidecar_path: Path,
    expected_jsonl_sha256: str,
    max_records: int = 1_000_000,
    max_line_bytes: int = 4 * 1024 * 1024,
) -> PreparedArchiveCatalog:
    """Read, hash, and parse all input bytes without touching the database."""
    expected = _validated_sha256(expected_jsonl_sha256, "expected_jsonl_sha256")
    try:
        jsonl_file = read_pinned_regular_file(
            jsonl_path,
            expected_sha256=expected,
            max_bytes=max_records * min(max_line_bytes, 64 * 1024),
        )
        sidecar_file = read_pinned_regular_file(sidecar_path, max_bytes=4096)
    except ArchiveProtocolError as exc:
        raise ArchiveCatalogError(str(exc)) from exc
    _validate_sidecar_bytes(
        sidecar_file.payload, jsonl_file.path.name, jsonl_file.sha256
    )
    records = _read_records_bytes(
        jsonl_file.payload,
        max_records=max_records,
        max_line_bytes=max_line_bytes,
    )
    return PreparedArchiveCatalog(
        jsonl_path=jsonl_file.path,
        sidecar_path=sidecar_file.path,
        jsonl_sha256=jsonl_file.sha256,
        sidecar_sha256=sidecar_file.sha256,
        records=tuple(records),
    )


def apply_archive_catalog(
    session: Session,
    *,
    jsonl_path: Path,
    sidecar_path: Path,
    expected_jsonl_sha256: str,
    receipt_dir: Path,
    max_records: int = 1_000_000,
    max_line_bytes: int = 4 * 1024 * 1024,
) -> ArchiveCatalogApplyResult:
    """Apply one packet with a receipt-before-commit, replay-safe protocol."""
    prepared = prepare_archive_catalog(
        jsonl_path=jsonl_path,
        sidecar_path=sidecar_path,
        expected_jsonl_sha256=expected_jsonl_sha256,
        max_records=max_records,
        max_line_bytes=max_line_bytes,
    )
    try:
        receipt_dir = prepare_immutable_receipt_dir(receipt_dir)
        acquire_transaction_lock(session, f"archive-catalog:{prepared.jsonl_sha256}")
    except ArchiveProtocolError as exc:
        raise ArchiveCatalogError(str(exc)) from exc

    existing = session.execute(
        select(ArchiveCatalogImport).where(
            ArchiveCatalogImport.jsonl_sha256 == prepared.jsonl_sha256
        )
    ).scalar_one_or_none()
    if existing is not None:
        _validate_existing_import(session, existing, prepared)
        receipt_path, receipt_sha256 = write_archive_catalog_receipt(
            existing.receipt_json,
            receipt_dir=receipt_dir,
        )
        if receipt_sha256 != existing.receipt_sha256:
            raise ArchiveCatalogError("catalog ledger receipt digest is inconsistent")
        return ArchiveCatalogApplyResult(
            receipt=existing.receipt_json,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            reconciled=True,
        )

    try:
        receipt, state = _load_prepared_archive_catalog(
            session, prepared, return_state=True
        )
        receipt_path, receipt_sha256 = write_archive_catalog_receipt(
            receipt,
            receipt_dir=receipt_dir,
        )
        session.add(
            ArchiveCatalogImport(
                id=_stable_id("aci", prepared.jsonl_sha256),
                jsonl_sha256=prepared.jsonl_sha256,
                sidecar_sha256=prepared.sidecar_sha256,
                input_filename=prepared.jsonl_path.name,
                receipt_sha256=receipt_sha256,
                receipt_json=receipt,
                source_keys_json=dict(sorted(state.source_ids_by_key.items())),
                source_ids_json=sorted(state.source_ids),
                video_ids_json=sorted(state.video_ids),
                media_sha256s_json=sorted(state.media_sha256s),
                status="applied",
            )
        )
        session.flush()
        return ArchiveCatalogApplyResult(
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            reconciled=False,
        )
    except Exception:
        session.rollback()
        raise


def _load_prepared_archive_catalog(
    session: Session,
    prepared: PreparedArchiveCatalog,
    *,
    return_state: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], _ImportState]:
    records = prepared.records
    state = _ImportState()
    for record in records:
        state.counts["records"] += 1
        record_type = _required_text(record.get("record_type"), "record_type", 64)
        if record_type == "import_contract":
            _validate_contract(record)
            state.contract_records.append(record)
            state.counts["contracts"] += 1
        elif record_type == "source":
            _load_source(session, record, state)
        elif record_type not in {"item", "inventory_summary"}:
            raise ArchiveCatalogError(f"unsupported record_type: {record_type}")

    for record in records:
        record_type = record["record_type"]
        if record_type == "item":
            _load_item(session, record, state)
        elif record_type == "inventory_summary":
            _validate_inventory_summary(record, state)
            state.inventory_records.append(record)
            state.counts["inventory_summaries"] += 1

    if len(state.contract_records) != 1:
        raise ArchiveCatalogError(
            "catalog must contain exactly one import_contract record"
        )
    if state.counts["records"] != len(records):
        raise ArchiveCatalogError("catalog record count changed during validation")
    session.flush()

    receipt = {
        "schema": ARCHIVE_RECEIPT_SCHEMA,
        "input": {
            "filename": prepared.jsonl_path.name,
            "jsonl_sha256": prepared.jsonl_sha256,
            "sidecar_sha256": prepared.sidecar_sha256,
        },
        "counts": dict(sorted(state.counts.items())),
        "readback": {
            "contract_sha256": _sha256_json(state.contract_records[0]),
            "inventory_summaries_sha256": _sha256_json(state.inventory_records),
            "media_identities_sha256": _sha256_json(sorted(state.media_sha256s)),
            "source_identities_sha256": _sha256_json(sorted(state.source_ids)),
            "video_identities_sha256": _sha256_json(sorted(state.video_ids)),
        },
    }
    if return_state:
        return receipt, state
    return receipt


def _validate_existing_import(
    session: Session,
    existing: ArchiveCatalogImport,
    prepared: PreparedArchiveCatalog,
) -> None:
    expected_id = _stable_id("aci", prepared.jsonl_sha256)
    if (
        existing.id != expected_id
        or existing.status != "applied"
        or existing.jsonl_sha256 != prepared.jsonl_sha256
        or existing.sidecar_sha256 != prepared.sidecar_sha256
        or existing.input_filename != prepared.jsonl_path.name
        or not isinstance(existing.receipt_json, dict)
        or _sha256_json(existing.receipt_json) != existing.receipt_sha256
    ):
        raise ArchiveCatalogError("existing archive import ledger is inconsistent")
    receipt_input = existing.receipt_json.get("input")
    if receipt_input != {
        "filename": prepared.jsonl_path.name,
        "jsonl_sha256": prepared.jsonl_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
    }:
        raise ArchiveCatalogError("existing archive import input identity collision")
    source_ids = _validated_identity_list(existing.source_ids_json, "source_ids_json")
    video_ids = _validated_identity_list(existing.video_ids_json, "video_ids_json")
    media_sha256s = _validated_identity_list(
        existing.media_sha256s_json, "media_sha256s_json", sha256=True
    )
    source_keys = existing.source_keys_json
    if (
        not isinstance(source_keys, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in source_keys.items()
        )
        or set(source_keys.values()) != set(source_ids)
    ):
        raise ArchiveCatalogError("existing archive import source map is inconsistent")
    readback = existing.receipt_json.get("readback")
    expected_readback = {
        "media_identities_sha256": _sha256_json(media_sha256s),
        "source_identities_sha256": _sha256_json(source_ids),
        "video_identities_sha256": _sha256_json(video_ids),
    }
    if not isinstance(readback, dict) or any(
        readback.get(key) != value for key, value in expected_readback.items()
    ):
        raise ArchiveCatalogError(
            "existing archive import identity digest is inconsistent"
        )
    for model, identities, identity_column in (
        (SourceChannel, source_ids, SourceChannel.id),
        (SourceVideo, video_ids, SourceVideo.id),
        (MediaObject, media_sha256s, MediaObject.sha256),
    ):
        if not identities:
            continue
        found = session.scalar(
            select(func.count())
            .select_from(model)
            .where(identity_column.in_(identities))
        )
        if found != len(identities):
            raise ArchiveCatalogError(
                "existing archive import database readback is incomplete"
            )


def _validated_identity_list(
    value: Any, field_name: str, *, sha256: bool = False
) -> list[str]:
    if (
        not isinstance(value, list)
        or value != sorted(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ArchiveCatalogError(f"{field_name} is not a canonical identity list")
    if sha256 and any(not _SHA256_PATTERN.fullmatch(item) for item in value):
        raise ArchiveCatalogError(f"{field_name} contains a non-SHA-256 identity")
    return value


def write_archive_catalog_receipt(
    receipt: dict[str, Any], *, receipt_dir: Path
) -> tuple[Path, str]:
    try:
        return write_immutable_json_receipt(
            receipt,
            receipt_dir=receipt_dir,
            schema=ARCHIVE_RECEIPT_SCHEMA,
        )
    except ArchiveProtocolError as exc:
        raise ArchiveCatalogError(str(exc)) from exc


def _read_records_bytes(
    payload: bytes,
    *,
    max_records: int,
    max_line_bytes: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if len(raw_line) > max_line_bytes:
            raise ArchiveCatalogError(f"line {line_number} exceeds max_line_bytes")
        if not raw_line.strip():
            raise ArchiveCatalogError(f"line {line_number} is blank")
        if len(records) >= max_records:
            raise ArchiveCatalogError("catalog exceeds max_records")
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArchiveCatalogError(
                f"line {line_number} is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ArchiveCatalogError(f"line {line_number} must be a JSON object")
        if record.get("schema") != ARCHIVE_CATALOG_SCHEMA:
            raise ArchiveCatalogError(f"line {line_number} has the wrong schema")
        records.append(record)
    if payload and not payload.endswith(b"\n"):
        raise ArchiveCatalogError("catalog must end with a newline")
    return records


def _validate_sidecar_bytes(
    payload: bytes, filename: str, expected_sha256: str
) -> None:
    try:
        sidecar = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ArchiveCatalogError("sidecar must be ASCII") from exc
    match = _SIDECAR_PATTERN.fullmatch(sidecar)
    if match is None:
        raise ArchiveCatalogError(
            "sidecar must be exact '<sha256>  <filename>\\n' text"
        )
    if match.group(2) != filename:
        raise ArchiveCatalogError("sidecar filename does not match JSONL basename")
    if match.group(1) != expected_sha256:
        raise ArchiveCatalogError("sidecar SHA-256 does not match JSONL bytes")


def _validate_contract(record: dict[str, Any]) -> None:
    if record.get("complete") is not True:
        raise ArchiveCatalogError("import_contract.complete must be true")
    if record.get("clip_ready_asserted") is not False:
        raise ArchiveCatalogError("import contract must not assert clip-ready")
    if record.get("twitch_discovery_item_IDs_available") is not False:
        raise ArchiveCatalogError("aggregate Twitch discovery must not assert item IDs")
    inputs = _required_dict(record.get("inputs"), "import_contract.inputs")
    if not inputs:
        raise ArchiveCatalogError("import_contract.inputs must not be empty")
    for name, digest in inputs.items():
        _required_text(name, "import input name", 255)
        _validated_sha256(digest, f"import_contract.inputs.{name}")
    identities = _required_dict(
        record.get("item_identity"), "import_contract.item_identity"
    )
    for platform in ("youtube", "twitch", "x"):
        _required_text(identities.get(platform), f"item_identity.{platform}", 255)
    _required_text(record.get("pumpfun_scope"), "pumpfun_scope", 1000)


def _load_source(session: Session, record: dict[str, Any], state: _ImportState) -> None:
    source_key = _required_text(record.get("source_key"), "source.source_key", 255)
    if source_key in state.source_ids_by_key:
        raise ArchiveCatalogError(f"duplicate source_key: {source_key}")
    platform = _required_text(record.get("platform"), "source.platform", 32).lower()
    handle = _optional_text(record.get("handle"), "source.handle", 255)
    identity_state = _required_text(
        record.get("identity_state"), "source.identity_state", 255
    )
    raw_external_id = record.get("platform_entity_id")
    if raw_external_id is None:
        if identity_state != "verified_handle_only" or handle is None:
            raise ArchiveCatalogError(
                "source without platform_entity_id must be verified_handle_only"
            )
        external_id = f"handle:{handle.casefold()}"
    else:
        external_id = _required_text(raw_external_id, "source.platform_entity_id", 255)
    evidence_ceilings = record.get("evidence_ceilings")
    if not isinstance(evidence_ceilings, list) or not all(
        isinstance(value, str) and value for value in evidence_ceilings
    ):
        raise ArchiveCatalogError(
            "source.evidence_ceilings must be a non-empty string list"
        )
    channel_id = canonical_source_channel_id(platform, external_id)
    state.source_ids_by_key[source_key] = channel_id
    state.source_ids.add(channel_id)
    archive_metadata = {
        "archive_import": {
            "source_key": source_key,
            "identity_state": identity_state,
            "evidence_ceilings": evidence_ceilings,
        }
    }
    channel = session.get(SourceChannel, channel_id)
    if channel is None:
        channel = SourceChannel(
            id=channel_id,
            platform=platform,
            external_id=external_id,
            handle=handle,
            metadata_json=archive_metadata,
        )
        session.add(channel)
        state.counts["sources_created"] += 1
    else:
        if channel.platform != platform or channel.external_id != external_id:
            raise ArchiveCatalogError(f"source identity collision: {source_key}")
        metadata = {**(channel.metadata_json or {}), **archive_metadata}
        changed = (
            channel.handle != handle
            or channel.metadata_json != metadata
            or channel.status != "active"
        )
        if changed:
            channel.handle = handle
            channel.metadata_json = metadata
            channel.status = "active"
            channel.updated_at = utcnow()
        state.counts["sources_updated" if changed else "sources_unchanged"] += 1
    session.flush()


def _load_item(session: Session, record: dict[str, Any], state: _ImportState) -> None:
    catalog_key = _required_text(record.get("catalog_key"), "item.catalog_key", 255)
    platform = _required_text(record.get("platform"), "item.platform", 32).lower()
    provider_item_id = _required_text(
        record.get("provider_item_id"), "item.provider_item_id", 255
    )
    provider_media_id = _optional_text(
        record.get("provider_media_id"), "item.provider_media_id", 255
    )
    provider_external_id = provider_item_id
    if platform == "x":
        if provider_media_id is None:
            raise ArchiveCatalogError("X item identity requires provider_media_id")
        provider_external_id = f"{provider_item_id}:{provider_media_id}"
    source_key = _required_text(record.get("source_key"), "item.source_key", 255)
    channel_id = state.source_ids_by_key.get(source_key)
    if channel_id is None:
        raise ArchiveCatalogError(
            f"item references a source not present earlier: {source_key}"
        )
    acquisition_state = _required_text(
        record.get("acquisition_state"), "item.acquisition_state", 64
    )
    if acquisition_state not in _CATALOG_ACQUISITION_STATES:
        raise ArchiveCatalogError(f"unsupported acquisition_state: {acquisition_state}")
    if not isinstance(record.get("retained"), bool):
        raise ArchiveCatalogError("item.retained must be boolean")
    if not isinstance(record.get("clip_candidate"), bool):
        raise ArchiveCatalogError("item.clip_candidate must be boolean")
    if record.get("clip_ready") is not False:
        raise ArchiveCatalogError("archive import must never assert clip_ready=true")
    media_variants = record.get("media_variants")
    if not isinstance(media_variants, list):
        raise ArchiveCatalogError("item.media_variants must be a list")
    topic_assertions = record.get("topic_assertions")
    if not isinstance(topic_assertions, list):
        raise ArchiveCatalogError("item.topic_assertions must be a list")

    video_id = canonical_source_video_id(platform, provider_external_id)
    state.video_ids.add(video_id)
    view_count = record.get("view_count_at_catalog_time")
    if view_count is not None:
        view_count = _required_nonnegative_int(
            view_count, "item.view_count_at_catalog_time"
        )
    canonical_url = _optional_text(
        record.get("canonical_url"), "item.canonical_url", 4000
    )
    title = _optional_text(record.get("title"), "item.title", 4000)
    archive_metadata = {
        "archive_import": {
            "catalog_key": catalog_key,
            "provider_media_id": provider_media_id,
            "provider_external_id": provider_external_id,
            "source_key": source_key,
            "retained": record["retained"],
            "clip_state": _required_text(
                record.get("clip_state"), "item.clip_state", 255
            ),
            "topic_assertions": topic_assertions,
            "evidence_ceiling": _required_text(
                record.get("evidence_ceiling"), "item.evidence_ceiling", 1000
            ),
            "status_url": _optional_text(
                record.get("status_url"), "item.status_url", 4000
            ),
            "media_kind": _optional_text(
                record.get("media_kind"), "item.media_kind", 64
            ),
            "canonical_url": canonical_url,
            "title": title,
            "source_tab": _optional_text(
                record.get("source_tab"), "item.source_tab", 64
            ),
            "view_count_at_catalog_time": view_count,
            "blocked_reason": _optional_text(
                record.get("blocked_reason"), "item.blocked_reason", 255
            ),
        }
    }
    video = session.get(SourceVideo, video_id)
    if video is None:
        video = SourceVideo(
            id=video_id,
            channel_id=channel_id,
            platform=platform,
            external_id=provider_external_id,
            canonical_url=canonical_url
            or archive_metadata["archive_import"]["status_url"],
            title=title,
            archive_state=acquisition_state,
            clip_candidate=record["clip_candidate"],
            clip_ready=False,
            metadata_json=archive_metadata,
        )
        session.add(video)
        state.counts["items_created"] += 1
    else:
        if (
            video.platform != platform
            or video.external_id != provider_external_id
            or video.channel_id != channel_id
        ):
            raise ArchiveCatalogError(f"item identity collision: {catalog_key}")
        next_state = max(
            (video.archive_state, acquisition_state),
            key=lambda value: _ARCHIVE_STATE_RANK[value],
        )
        changed = (
            video.archive_state != next_state
            or video.clip_candidate
            != (video.clip_candidate or record["clip_candidate"])
            or video.metadata_json
            != {**(video.metadata_json or {}), **archive_metadata}
            or (title is not None and video.title != title)
            or (canonical_url is not None and video.canonical_url != canonical_url)
            or video.status != "active"
        )
        if changed:
            video.archive_state = next_state
            video.clip_candidate = video.clip_candidate or record["clip_candidate"]
            if title is not None:
                video.title = title
            if canonical_url is not None:
                video.canonical_url = canonical_url
            video.metadata_json = {**(video.metadata_json or {}), **archive_metadata}
            video.status = "active"
            video.updated_at = utcnow()
        state.counts["items_updated" if changed else "items_unchanged"] += 1
    session.flush()

    for variant in media_variants:
        _load_media_variant(session, video=video, variant=variant, state=state)


def _load_media_variant(
    session: Session,
    *,
    video: SourceVideo,
    variant: Any,
    state: _ImportState,
) -> None:
    if not isinstance(variant, dict):
        raise ArchiveCatalogError("media_variants entries must be objects")
    state.counts["media_variants"] += 1
    digest = _validated_sha256(variant.get("sha256"), "media_variant.sha256")
    size_bytes = _required_nonnegative_int(variant.get("bytes"), "media_variant.bytes")
    media_kind = _required_text(
        variant.get("media_kind"), "media_variant.media_kind", 64
    )
    suffix = _required_text(
        variant.get("container_suffix"), "media_variant.container_suffix", 32
    )
    complete_media = variant.get("complete_media")
    remote_verified = variant.get("remote_sha256_verified")
    if not isinstance(complete_media, bool) or not isinstance(remote_verified, bool):
        raise ArchiveCatalogError(
            "media completeness and verification fields must be boolean"
        )
    evidence = {
        "dataset": _required_text(variant.get("dataset"), "media_variant.dataset", 255),
        "source_manifest_sha256": _validated_sha256(
            variant.get("source_manifest_sha256"),
            "media_variant.source_manifest_sha256",
        ),
        "source_receipt_sha256": _validated_sha256(
            variant.get("source_receipt_sha256"), "media_variant.source_receipt_sha256"
        ),
        "row_id": _required_text(variant.get("row_id"), "media_variant.row_id", 255),
        "relative_path": _relative_posix_path(
            variant.get("relative_path"), "media_variant.relative_path", 4000
        ),
        "remote_path": _relative_posix_path(
            variant.get("remote_path"), "media_variant.remote_path", 8000
        ),
        "media_kind": media_kind,
        "container_suffix": suffix,
        "complete_media": complete_media,
        "remote_sha256_verified": remote_verified,
    }
    mime_type = _mime_type(media_kind, suffix)
    media = session.get(MediaObject, digest)
    if media is None:
        media = MediaObject(
            sha256=digest,
            size_bytes=size_bytes,
            mime_type=mime_type,
            status="active",
            metadata_json={"archive_evidence": [evidence]},
        )
        session.add(media)
        state.counts["media_created"] += 1
    else:
        if media.size_bytes != size_bytes or media.mime_type != mime_type:
            raise ArchiveCatalogError(f"media fact collision for SHA-256 {digest}")
        existing_evidence = list(
            (media.metadata_json or {}).get("archive_evidence") or []
        )
        if evidence not in existing_evidence:
            existing_evidence.append(evidence)
            existing_evidence.sort(key=_canonical_sort_key)
            media.metadata_json = {
                **(media.metadata_json or {}),
                "archive_evidence": existing_evidence,
            }
            state.counts["media_updated"] += 1
        else:
            state.counts["media_unchanged"] += 1
    state.media_sha256s.add(digest)
    session.flush()

    role = _media_role(media_kind, complete_media)
    ref_id = _stable_id("vmr", f"{video.id}:{role}:{digest}")
    reference = session.get(VideoMediaRef, ref_id)
    if reference is None:
        session.add(
            VideoMediaRef(
                id=ref_id,
                video_id=video.id,
                media_sha256=digest,
                role=role,
                status="active",
            )
        )
        state.counts["refs_created"] += 1
    else:
        if (
            reference.video_id != video.id
            or reference.media_sha256 != digest
            or reference.role != role
        ):
            raise ArchiveCatalogError(f"media reference collision: {ref_id}")
        state.counts["refs_unchanged"] += 1

    location_key = evidence["remote_path"]
    location_id = _stable_id("loc", f"storagebox:{location_key}")
    location = session.get(MediaLocation, location_id)
    location_status = "active" if remote_verified else "pending"
    if location is None:
        session.add(
            MediaLocation(
                id=location_id,
                media_sha256=digest,
                backend="storagebox",
                location_key=location_key,
                status=location_status,
                bytes=size_bytes,
                verified_at=utcnow() if remote_verified else None,
            )
        )
        state.counts["locations_created"] += 1
    else:
        if (
            location.media_sha256 != digest
            or location.backend != "storagebox"
            or location.location_key != location_key
            or location.bytes != size_bytes
        ):
            raise ArchiveCatalogError(f"media location collision: {location_id}")
        changed = remote_verified and location.status != "active"
        if remote_verified:
            location.status = "active"
            location.verified_at = location.verified_at or utcnow()
        state.counts["locations_updated" if changed else "locations_unchanged"] += 1
    session.flush()


def _validate_inventory_summary(record: dict[str, Any], state: _ImportState) -> None:
    _required_text(record.get("catalog_key"), "inventory_summary.catalog_key", 255)
    source_key = _required_text(
        record.get("source_key"), "inventory_summary.source_key", 255
    )
    if source_key not in state.source_ids_by_key:
        raise ArchiveCatalogError(
            f"inventory summary references an unknown source: {source_key}"
        )
    _required_text(record.get("platform"), "inventory_summary.platform", 32)
    state = _required_text(
        record.get("acquisition_state"), "inventory_summary.acquisition_state", 64
    )
    if state != "pending_discovery":
        raise ArchiveCatalogError("inventory_summary must remain pending_discovery")
    if not isinstance(record.get("auto_download_requested"), bool):
        raise ArchiveCatalogError(
            "inventory_summary.auto_download_requested must be boolean"
        )
    counts = _required_dict(
        record.get("observed_feed_counts"), "inventory_summary.observed_feed_counts"
    )
    for key, value in counts.items():
        _required_text(key, "observed feed name", 64)
        _required_nonnegative_int(value, f"observed_feed_counts.{key}")
    _required_nonnegative_int(record.get("observed_item_count"), "observed_item_count")
    if record.get("retained_item_ids_emitted") != 0:
        raise ArchiveCatalogError(
            "inventory summary must not fabricate retained item IDs"
        )
    _required_text(
        record.get("evidence_ceiling"), "inventory_summary.evidence_ceiling", 1000
    )


def _media_role(media_kind: str, complete_media: bool) -> str:
    normalized = media_kind.strip().lower()
    if "audio" in normalized:
        return "audio"
    if "thumb" in normalized or "image" in normalized:
        return "thumbnail"
    if "video" in normalized or "stream" in normalized:
        return "source_video" if complete_media else "proxy"
    return "proxy"


def _mime_type(media_kind: str, suffix: str) -> str:
    normalized = media_kind.strip().lower()
    suffix = suffix.strip().lower()
    if "audio" in normalized:
        return {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav"}.get(
            suffix, "application/octet-stream"
        )
    if "image" in normalized or "thumb" in normalized:
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
            suffix, "application/octet-stream"
        )
    return {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska"}.get(
        suffix, "application/octet-stream"
    )


def _required_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchiveCatalogError(f"{field_name} must be an object")
    return value


def _required_text(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ArchiveCatalogError(f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > max_length
        or any(ord(char) < 32 for char in normalized)
    ):
        raise ArchiveCatalogError(f"{field_name} is missing or invalid")
    return normalized


def _optional_text(value: Any, field_name: str, max_length: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name, max_length)


def _relative_posix_path(value: Any, field_name: str, max_length: int) -> str:
    path = _required_text(value, field_name, max_length)
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ArchiveCatalogError(
            f"{field_name} must be a traversal-free relative path"
        )
    return path


def _required_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArchiveCatalogError(f"{field_name} must be a non-negative integer")
    return value


def _validated_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ArchiveCatalogError(f"{field_name} must be lowercase SHA-256")
    return value


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:40]}"


def _canonical_sort_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_json_bytes(payload: Any) -> bytes:
    return canonical_json_bytes(payload)


def _sha256_json(payload: Any) -> str:
    return sha256_json(payload)
