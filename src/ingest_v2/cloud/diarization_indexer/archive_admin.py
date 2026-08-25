from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .archive_receipts import (
    ArchiveProtocolError,
    acquire_transaction_lock,
    hash_pinned_regular_file,
    prepare_immutable_receipt_dir,
    read_pinned_regular_file,
    safe_absolute_path,
    sha256_json,
    write_immutable_json_receipt,
)
from .channel_service_config import (
    ChannelServiceConfigurationError,
    validate_tenant_id,
    validate_user_id,
)
from .channel_service_store import (
    ArchiveCatalogImport,
    ArchiveHydrationRegistration,
    ArchiveTenantClaim,
    MediaLocation,
    MediaObject,
    SourceVideo,
    Tenant,
    TenantChannelEntitlement,
    TenantMembership,
    UserAccount,
    VideoMediaRef,
    set_tenant_scope,
    utcnow,
)

ARCHIVE_TENANT_CLAIM_RECEIPT_SCHEMA = "icmfyi.archive-tenant-claim-receipt.v1"
HOT_MEDIA_HYDRATION_SOURCE_SCHEMA = "icmfyi.hot-media-hydration-source.v1"
HOT_MEDIA_HYDRATION_RECEIPT_SCHEMA = "icmfyi.hot-media-hydration-receipt.v1"
FFPROBE_VIDEO_PROOF_SCHEMA = "icmfyi.ffprobe-video-proof.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_SUFFIXES = {".mkv", ".mp4", ".webm"}


class ArchiveAdminError(ArchiveProtocolError):
    """An internal-only archive admin operation failed closed."""


@dataclass(frozen=True)
class ArchiveAdminApplyResult:
    receipt: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    reconciled: bool


def claim_archive_sources(
    session: Session,
    *,
    catalog_jsonl_sha256: str,
    tenant_id: str,
    admin_user_id: str,
    idempotency_key: str,
    source_keys: list[str] | tuple[str, ...],
    receipt_dir: Path,
) -> ArchiveAdminApplyResult:
    """Grant an exact imported source set through an admin-only database session."""
    digest = _validated_sha256(catalog_jsonl_sha256, "catalog_jsonl_sha256")
    tenant_id = _validated_tenant_id(tenant_id)
    admin_user_id = _validated_user_id(admin_user_id)
    key = _required_text(idempotency_key, "idempotency_key", 255)
    canonical_source_keys = _canonical_text_set(source_keys, "source_keys", 10_000)
    request_fingerprint = sha256_json(
        {
            "access_level": "query",
            "admin_user_id": admin_user_id,
            "catalog_jsonl_sha256": digest,
            "source_keys": canonical_source_keys,
            "tenant_id": tenant_id,
        }
    )
    try:
        directory = prepare_immutable_receipt_dir(receipt_dir)
        acquire_transaction_lock(session, f"archive-tenant-claim:{tenant_id}")
        existing = session.execute(
            select(ArchiveTenantClaim).where(
                ArchiveTenantClaim.tenant_id == tenant_id,
                ArchiveTenantClaim.idempotency_key == key,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return _reconcile_claim(
                session,
                existing=existing,
                request_fingerprint=request_fingerprint,
                receipt_dir=directory,
            )

        catalog = session.execute(
            select(ArchiveCatalogImport).where(
                ArchiveCatalogImport.jsonl_sha256 == digest,
                ArchiveCatalogImport.status == "applied",
            )
        ).scalar_one_or_none()
        if catalog is None:
            raise ArchiveAdminError("the pinned archive catalog has not been applied")
        _require_tenant_admin(session, tenant_id=tenant_id, admin_user_id=admin_user_id)
        source_map = catalog.source_keys_json
        if not isinstance(source_map, dict):
            raise ArchiveAdminError("archive catalog source map is invalid")
        missing = [name for name in canonical_source_keys if name not in source_map]
        if missing:
            raise ArchiveAdminError(
                f"source key is not part of the pinned catalog: {missing[0]}"
            )
        source_ids = sorted({str(source_map[name]) for name in canonical_source_keys})
        if len(source_ids) != len(canonical_source_keys):
            raise ArchiveAdminError("source selection aliases the same channel twice")

        set_tenant_scope(session, tenant_id)
        created = 0
        unchanged = 0
        entitlement_ids: list[str] = []
        for channel_id in source_ids:
            entitlement_id = _stable_id("tce", f"{tenant_id}:{channel_id}")
            entitlement = session.execute(
                select(TenantChannelEntitlement).where(
                    TenantChannelEntitlement.tenant_id == tenant_id,
                    TenantChannelEntitlement.channel_id == channel_id,
                )
            ).scalar_one_or_none()
            if entitlement is None:
                session.add(
                    TenantChannelEntitlement(
                        id=entitlement_id,
                        tenant_id=tenant_id,
                        channel_id=channel_id,
                        granted_by_user_id=admin_user_id,
                        access_level="query",
                        status="active",
                    )
                )
                created += 1
            else:
                if (
                    entitlement.id != entitlement_id
                    or entitlement.tenant_id != tenant_id
                    or entitlement.channel_id != channel_id
                    or entitlement.access_level != "query"
                    or entitlement.status != "active"
                ):
                    raise ArchiveAdminError("existing channel entitlement conflicts")
                unchanged += 1
            entitlement_ids.append(entitlement_id)
        session.flush()

        receipt = {
            "schema": ARCHIVE_TENANT_CLAIM_RECEIPT_SCHEMA,
            "operation": {
                "idempotency_key": key,
                "request_fingerprint": request_fingerprint,
            },
            "actor": {"admin_user_id": admin_user_id},
            "catalog": {
                "jsonl_sha256": digest,
                "receipt_sha256": catalog.receipt_sha256,
            },
            "target": {
                "access_level": "query",
                "source_ids": source_ids,
                "source_keys": canonical_source_keys,
                "tenant_id": tenant_id,
            },
            "counts": {"created": created, "unchanged": unchanged},
            "readback": {
                "entitlement_ids_sha256": sha256_json(sorted(entitlement_ids))
            },
            "protocol": {"database_commit_required": True},
        }
        receipt_path, receipt_sha256 = write_immutable_json_receipt(
            receipt,
            receipt_dir=directory,
            schema=ARCHIVE_TENANT_CLAIM_RECEIPT_SCHEMA,
        )
        session.add(
            ArchiveTenantClaim(
                id=_stable_id("atc", f"{tenant_id}:{key}"),
                tenant_id=tenant_id,
                admin_user_id=admin_user_id,
                catalog_import_id=catalog.id,
                idempotency_key=key,
                request_fingerprint=request_fingerprint,
                source_ids_json=source_ids,
                receipt_sha256=receipt_sha256,
                receipt_json=receipt,
                status="applied",
            )
        )
        session.flush()
        return ArchiveAdminApplyResult(
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            reconciled=False,
        )
    except Exception:
        session.rollback()
        raise


def register_hot_media_hydration(
    session: Session,
    *,
    source_receipt_path: Path,
    expected_source_receipt_sha256: str,
    hot_media_root: Path,
    receipt_dir: Path,
    ffprobe_bin: Path,
) -> ArchiveAdminApplyResult:
    """Register one independently verified archive video in the hot-media CAS."""
    expected_digest = _validated_sha256(
        expected_source_receipt_sha256, "expected_source_receipt_sha256"
    )
    prepared = _prepare_hydration_source(
        source_receipt_path=source_receipt_path,
        expected_source_receipt_sha256=expected_digest,
        hot_media_root=hot_media_root,
        ffprobe_bin=ffprobe_bin,
    )
    try:
        directory = prepare_immutable_receipt_dir(receipt_dir)
        acquire_transaction_lock(
            session, f"archive-hydration:{prepared['media_sha256']}"
        )
        existing = session.execute(
            select(ArchiveHydrationRegistration).where(
                ArchiveHydrationRegistration.input_receipt_sha256 == expected_digest
            )
        ).scalar_one_or_none()
        if existing is not None:
            return _reconcile_hydration(
                session,
                existing=existing,
                prepared=prepared,
                receipt_dir=directory,
            )

        media = session.get(MediaObject, prepared["media_sha256"])
        if media is None:
            raise ArchiveAdminError("hydrated media is not in the archive catalog")
        if (
            media.size_bytes != prepared["size_bytes"]
            or media.mime_type != prepared["mime_type"]
            or not media.mime_type.startswith("video/")
        ):
            raise ArchiveAdminError("hydrated media conflicts with canonical facts")
        references = list(
            session.execute(
                select(VideoMediaRef).where(
                    VideoMediaRef.media_sha256 == prepared["media_sha256"],
                    VideoMediaRef.role == "source_video",
                    VideoMediaRef.status == "active",
                )
            ).scalars()
        )
        if not references:
            raise ArchiveAdminError("hydrated media is not a complete source video")

        path = prepared["cas_path"]
        location_id = _stable_id("loc", f"hot_local:{path}")
        active_location = session.execute(
            select(MediaLocation).where(
                MediaLocation.media_sha256 == prepared["media_sha256"],
                MediaLocation.backend == "hot_local",
                MediaLocation.status == "active",
            )
        ).scalar_one_or_none()
        if active_location is not None and active_location.location_key != path:
            raise ArchiveAdminError("media already has another active hot-local path")
        location = session.get(MediaLocation, location_id)
        location_created = location is None
        now = utcnow()
        if location is None:
            location = MediaLocation(
                id=location_id,
                media_sha256=prepared["media_sha256"],
                backend="hot_local",
                location_key=path,
                status="active",
                bytes=prepared["size_bytes"],
                verified_at=now,
            )
            session.add(location)
        elif (
            location.media_sha256 != prepared["media_sha256"]
            or location.backend != "hot_local"
            or location.location_key != path
            or location.bytes != prepared["size_bytes"]
        ):
            raise ArchiveAdminError("hot-local media location identity collision")
        else:
            location.status = "active"
            location.verified_at = now
            location.updated_at = now

        video_ids = sorted({reference.video_id for reference in references})
        for video_id in video_ids:
            video = session.get(SourceVideo, video_id)
            if video is None:
                raise ArchiveAdminError("source-video reference target is missing")
            video.clip_candidate = True
            video.clip_ready = True
            video.archive_state = "retained_hot_verified"
            video.status = "active"
            video.updated_at = now
        session.flush()

        receipt = {
            "schema": HOT_MEDIA_HYDRATION_RECEIPT_SCHEMA,
            "input": {
                "filename": prepared["input_filename"],
                "source_receipt_sha256": expected_digest,
            },
            "media": {
                "cas_path": path,
                "ffprobe": prepared["ffprobe"],
                "media_sha256": prepared["media_sha256"],
                "mime_type": prepared["mime_type"],
                "size_bytes": prepared["size_bytes"],
            },
            "counts": {
                "locations_created": int(location_created),
                "locations_unchanged": int(not location_created),
                "videos_marked_clip_ready": len(video_ids),
            },
            "readback": {
                "location_id": location_id,
                "video_ids": video_ids,
                "video_ids_sha256": sha256_json(video_ids),
            },
            "protocol": {"database_commit_required": True},
        }
        receipt_path, receipt_sha256 = write_immutable_json_receipt(
            receipt,
            receipt_dir=directory,
            schema=HOT_MEDIA_HYDRATION_RECEIPT_SCHEMA,
        )
        session.add(
            ArchiveHydrationRegistration(
                id=_stable_id("ahr", expected_digest),
                input_receipt_sha256=expected_digest,
                media_sha256=prepared["media_sha256"],
                location_id=location_id,
                receipt_sha256=receipt_sha256,
                receipt_json=receipt,
                status="applied",
            )
        )
        session.flush()
        return ArchiveAdminApplyResult(
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            reconciled=False,
        )
    except Exception:
        session.rollback()
        raise


def _reconcile_claim(
    session: Session,
    *,
    existing: ArchiveTenantClaim,
    request_fingerprint: str,
    receipt_dir: Path,
) -> ArchiveAdminApplyResult:
    if (
        existing.status != "applied"
        or existing.request_fingerprint != request_fingerprint
        or not isinstance(existing.receipt_json, dict)
        or sha256_json(existing.receipt_json) != existing.receipt_sha256
    ):
        raise ArchiveAdminError("tenant claim idempotency collision")
    set_tenant_scope(session, existing.tenant_id)
    source_ids = _canonical_text_set(
        existing.source_ids_json, "existing.source_ids_json", 10_000
    )
    entitlements = list(
        session.execute(
            select(TenantChannelEntitlement).where(
                TenantChannelEntitlement.tenant_id == existing.tenant_id,
                TenantChannelEntitlement.channel_id.in_(source_ids),
                TenantChannelEntitlement.status == "active",
                TenantChannelEntitlement.access_level == "query",
            )
        ).scalars()
    )
    if len(entitlements) != len(source_ids):
        raise ArchiveAdminError("tenant claim database readback is incomplete")
    receipt_path, receipt_sha256 = write_immutable_json_receipt(
        existing.receipt_json,
        receipt_dir=receipt_dir,
        schema=ARCHIVE_TENANT_CLAIM_RECEIPT_SCHEMA,
    )
    if receipt_sha256 != existing.receipt_sha256:
        raise ArchiveAdminError("tenant claim receipt digest is inconsistent")
    return ArchiveAdminApplyResult(
        receipt=existing.receipt_json,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        reconciled=True,
    )


def _reconcile_hydration(
    session: Session,
    *,
    existing: ArchiveHydrationRegistration,
    prepared: dict[str, Any],
    receipt_dir: Path,
) -> ArchiveAdminApplyResult:
    receipt = existing.receipt_json
    if (
        existing.status != "applied"
        or existing.media_sha256 != prepared["media_sha256"]
        or not isinstance(receipt, dict)
        or sha256_json(receipt) != existing.receipt_sha256
    ):
        raise ArchiveAdminError("hydration registration identity collision")

    expected_location_id = _stable_id("loc", f"hot_local:{prepared['cas_path']}")
    media = session.execute(
        select(MediaObject)
        .where(MediaObject.sha256 == prepared["media_sha256"])
        .with_for_update()
    ).scalar_one_or_none()
    if (
        media is None
        or media.size_bytes != prepared["size_bytes"]
        or media.mime_type != prepared["mime_type"]
        or media.status != "active"
    ):
        raise ArchiveAdminError("hydration registration media readback is incomplete")

    location = session.execute(
        select(MediaLocation)
        .where(MediaLocation.id == existing.location_id)
        .with_for_update()
    ).scalar_one_or_none()
    if (
        location is None
        or existing.location_id != expected_location_id
        or location.media_sha256 != prepared["media_sha256"]
        or location.backend != "hot_local"
        or location.location_key != prepared["cas_path"]
        or location.status != "active"
        or location.bytes != prepared["size_bytes"]
        or location.verified_at is None
    ):
        raise ArchiveAdminError(
            "hydration registration database readback is incomplete"
        )

    references = list(
        session.execute(
            select(VideoMediaRef)
            .where(
                VideoMediaRef.media_sha256 == prepared["media_sha256"],
                VideoMediaRef.role == "source_video",
                VideoMediaRef.status == "active",
            )
            .with_for_update()
        ).scalars()
    )
    video_ids = sorted({reference.video_id for reference in references})
    if not video_ids or len(video_ids) != len(references):
        raise ArchiveAdminError(
            "hydration registration video references are incomplete"
        )
    videos = list(
        session.execute(
            select(SourceVideo).where(SourceVideo.id.in_(video_ids)).with_for_update()
        ).scalars()
    )
    if len(videos) != len(video_ids) or any(
        video.id not in video_ids
        or video.clip_candidate is not True
        or video.clip_ready is not True
        or video.archive_state != "retained_hot_verified"
        or video.status != "active"
        for video in videos
    ):
        raise ArchiveAdminError("hydration registration video readback is incomplete")

    counts = receipt.get("counts")
    expected_readback = {
        "location_id": expected_location_id,
        "video_ids": video_ids,
        "video_ids_sha256": sha256_json(video_ids),
    }
    if (
        set(receipt) != {"schema", "input", "media", "counts", "readback", "protocol"}
        or receipt.get("schema") != HOT_MEDIA_HYDRATION_RECEIPT_SCHEMA
        or receipt.get("input")
        != {
            "filename": prepared["input_filename"],
            "source_receipt_sha256": existing.input_receipt_sha256,
        }
        or receipt.get("media")
        != {
            "cas_path": prepared["cas_path"],
            "ffprobe": prepared["ffprobe"],
            "media_sha256": prepared["media_sha256"],
            "mime_type": prepared["mime_type"],
            "size_bytes": prepared["size_bytes"],
        }
        or not isinstance(counts, dict)
        or set(counts)
        != {
            "locations_created",
            "locations_unchanged",
            "videos_marked_clip_ready",
        }
        or type(counts.get("locations_created")) is not int
        or type(counts.get("locations_unchanged")) is not int
        or type(counts.get("videos_marked_clip_ready")) is not int
        or (counts["locations_created"], counts["locations_unchanged"])
        not in {(1, 0), (0, 1)}
        or counts.get("videos_marked_clip_ready") != len(video_ids)
        or receipt.get("readback") != expected_readback
        or receipt.get("protocol") != {"database_commit_required": True}
    ):
        raise ArchiveAdminError(
            "hydration registration receipt readback is inconsistent"
        )
    receipt_path, receipt_sha256 = write_immutable_json_receipt(
        receipt,
        receipt_dir=receipt_dir,
        schema=HOT_MEDIA_HYDRATION_RECEIPT_SCHEMA,
    )
    if receipt_sha256 != existing.receipt_sha256:
        raise ArchiveAdminError("hydration receipt digest is inconsistent")
    return ArchiveAdminApplyResult(
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        reconciled=True,
    )


def _prepare_hydration_source(
    *,
    source_receipt_path: Path,
    expected_source_receipt_sha256: str,
    hot_media_root: Path,
    ffprobe_bin: Path,
) -> dict[str, Any]:
    try:
        source = read_pinned_regular_file(
            source_receipt_path,
            expected_sha256=expected_source_receipt_sha256,
            max_bytes=64 * 1024,
        )
    except ArchiveProtocolError as exc:
        raise ArchiveAdminError(str(exc)) from exc
    try:
        payload = json.loads(source.payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveAdminError("hydration source receipt must be ASCII JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != HOT_MEDIA_HYDRATION_SOURCE_SCHEMA
    ):
        raise ArchiveAdminError("hydration source receipt has the wrong schema")
    digest = _validated_sha256(payload.get("media_sha256"), "media_sha256")
    size_bytes = _positive_int(payload.get("size_bytes"), "size_bytes")
    mime_type = _required_text(payload.get("mime_type"), "mime_type", 255).lower()
    if not mime_type.startswith("video/"):
        raise ArchiveAdminError("hydration source must be video media")
    root = safe_absolute_path(hot_media_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ArchiveAdminError(
            "hot_media_root must be an existing non-symlink directory"
        )
    cas_path = safe_absolute_path(
        Path(_required_text(payload.get("cas_path"), "cas_path", 8000))
    )
    try:
        relative = cas_path.relative_to(root)
    except ValueError as exc:
        raise ArchiveAdminError("CAS path is outside hot_media_root") from exc
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "sha256"
        or relative.parts[1] != digest[:2]
        or Path(relative.parts[2]).stem != digest
        or Path(relative.parts[2]).suffix.lower() not in _SAFE_SUFFIXES
    ):
        raise ArchiveAdminError("CAS path is not the canonical SHA-256 object path")
    try:
        retained = hash_pinned_regular_file(
            cas_path,
            expected_sha256=digest,
            expected_size_bytes=size_bytes,
        )
    except ArchiveProtocolError as exc:
        raise ArchiveAdminError(str(exc)) from exc
    if retained.mode & 0o222:
        raise ArchiveAdminError("hydrated CAS object must not be writable")
    proof = _run_ffprobe(retained.path, ffprobe_bin=ffprobe_bin)
    if payload.get("ffprobe") != proof:
        raise ArchiveAdminError("ffprobe proof does not match independent readback")
    after = retained.path.lstat()
    if (
        after.st_dev != retained.device
        or after.st_ino != retained.inode
        or after.st_size != retained.size_bytes
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ArchiveAdminError("hydrated CAS object changed during ffprobe")
    return {
        "cas_path": str(retained.path),
        "ffprobe": proof,
        "input_filename": source.path.name,
        "media_sha256": digest,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
    }


def _run_ffprobe(path: Path, *, ffprobe_bin: Path) -> dict[str, Any]:
    executable = safe_absolute_path(ffprobe_bin)
    if not executable.is_absolute():
        raise ArchiveAdminError("ffprobe_bin must be absolute")
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height:format=format_name,duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"LANG": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArchiveAdminError("ffprobe failed for hydrated media") from exc
    if completed.returncode != 0:
        raise ArchiveAdminError("ffprobe rejected hydrated media")
    try:
        raw = json.loads(completed.stdout)
        streams = [
            stream
            for stream in raw.get("streams", [])
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ]
        duration_ms = int(Decimal(str(raw["format"]["duration"])) * 1000)
        codec_names = sorted(
            {
                _required_text(stream.get("codec_name"), "codec_name", 64)
                for stream in streams
            }
        )
        widths = [_positive_int(stream.get("width"), "width") for stream in streams]
        heights = [_positive_int(stream.get("height"), "height") for stream in streams]
        format_names = sorted(
            {
                name.strip()
                for name in _required_text(
                    raw["format"].get("format_name"), "format_name", 255
                ).split(",")
                if name.strip()
            }
        )
    except (
        ArchiveAdminError,
        InvalidOperation,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ArchiveAdminError("ffprobe returned invalid video metadata") from exc
    if not streams or duration_ms <= 0 or not format_names:
        raise ArchiveAdminError("ffprobe did not prove a decodable video")
    return {
        "schema": FFPROBE_VIDEO_PROOF_SCHEMA,
        "codec_names": codec_names,
        "duration_ms": duration_ms,
        "format_names": format_names,
        "max_height": max(heights),
        "max_width": max(widths),
        "video_stream_count": len(streams),
    }


def _require_tenant_admin(
    session: Session, *, tenant_id: str, admin_user_id: str
) -> None:
    user = session.get(UserAccount, admin_user_id)
    tenant = session.get(Tenant, tenant_id)
    membership = session.execute(
        select(TenantMembership).where(
            TenantMembership.tenant_id == tenant_id,
            TenantMembership.user_id == admin_user_id,
        )
    ).scalar_one_or_none()
    if user is None or user.status != "active":
        raise ArchiveAdminError("archive claim admin user is not active")
    if tenant is None or tenant.status != "active":
        raise ArchiveAdminError("archive claim tenant is not active")
    if (
        membership is None
        or membership.status != "active"
        or membership.role not in {"admin", "owner"}
    ):
        raise ArchiveAdminError("archive claim requires an active tenant admin")


def _validated_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ArchiveAdminError(f"{field_name} must be lowercase SHA-256")
    return value


def _validated_tenant_id(value: str) -> str:
    try:
        return validate_tenant_id(value)
    except ChannelServiceConfigurationError as exc:
        raise ArchiveAdminError("tenant_id is invalid") from exc


def _validated_user_id(value: str) -> str:
    try:
        return validate_user_id(value)
    except ChannelServiceConfigurationError as exc:
        raise ArchiveAdminError("admin_user_id is invalid") from exc


def _required_text(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ArchiveAdminError(f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > max_length
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ArchiveAdminError(f"{field_name} is missing or invalid")
    return normalized


def _canonical_text_set(values: Any, field_name: str, maximum: int) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ArchiveAdminError(f"{field_name} must be a non-empty list")
    if len(values) > maximum:
        raise ArchiveAdminError(f"{field_name} exceeds {maximum} entries")
    normalized = sorted(
        {_required_text(value, f"{field_name} entry", 255) for value in values}
    )
    if len(normalized) != len(values):
        raise ArchiveAdminError(f"{field_name} must not contain duplicates")
    return normalized


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArchiveAdminError(f"{field_name} must be a positive integer")
    return value


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:40]}"
