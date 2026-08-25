from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .channel_service_config import InternalRequestIdentity
from .channel_service_jobs import ensure_channel_entitlement, ensure_source_channel
from .channel_service_store import (
    MediaLocation,
    MediaObject,
    SourceVideo,
    TranscriptRevision,
    TranscriptSegment,
    VideoMediaRef,
    utcnow,
)
from .tenant_export import ensure_gateway_principals


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class CanonicalPublishError(RuntimeError):
    """A successful ingestion could not be represented as canonical durable facts."""


@dataclass(frozen=True)
class HotMediaSpec:
    path: Path
    sha256: str
    size_bytes: int
    mime_type: str


@dataclass(frozen=True)
class CanonicalPublishResult:
    media_id: str
    channel_id: str
    transcript_revision_id: str
    transcript_segment_count: int
    clip_ready: bool
    hot_media_sha256: str | None


def canonical_source_channel_id(platform: str, external_id: str) -> str:
    return _stable_id(
        "chn",
        f"{_provider(platform)}:{_required(external_id, 'channel_external_id', 255)}",
    )


def canonical_source_video_id(platform: str, external_id: str) -> str:
    return _stable_id(
        "vid",
        f"{_provider(platform)}:{_required(external_id, 'provider_video_id', 255)}",
    )


def publish_canonical_ingestion(
    session: Session,
    *,
    identity: InternalRequestIdentity,
    platform: str,
    provider_video_id: str,
    channel_external_id: str,
    channel_handle: str | None,
    channel_name: str | None,
    canonical_url: str | None,
    title: str | None,
    description: str | None,
    published_at: str | datetime | None,
    duration_ms: int | None,
    language: str,
    transcript_provider: str,
    transcript_segments: Iterable[dict[str, Any]],
    hot_media: HotMediaSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> CanonicalPublishResult:
    """Publish one provider item, entitlement, transcript, and optional verified hot media."""
    provider = _provider(platform)
    provider_video_id = _required(provider_video_id, "provider_video_id", 255)
    channel_external_id = _required(channel_external_id, "channel_external_id", 255)
    language = _required(language, "language", 32)
    transcript_provider = _required(transcript_provider, "transcript_provider", 64)
    normalized_segments = _normalize_transcript_segments(transcript_segments)
    content_sha256 = _sha256_json(normalized_segments)
    video_id = canonical_source_video_id(provider, provider_video_id)
    channel_id = canonical_source_channel_id(provider, channel_external_id)
    revision_id = _stable_id(
        "trv", f"{video_id}:{transcript_provider}:{content_sha256}"
    )
    now = utcnow()

    verified_hot_media = _verify_hot_media(hot_media) if hot_media is not None else None
    ensure_gateway_principals(session, identity)
    channel = ensure_source_channel(
        session,
        platform=provider,
        external_id=channel_external_id,
        handle=channel_handle,
        display_name=channel_name,
        canonical_url=None,
        metadata={"canonical_publisher": {"latest_video_id": video_id}},
    )
    if channel.id != channel_id:
        raise CanonicalPublishError("canonical source channel identity is inconsistent")
    channel.status = "active"
    if channel_handle:
        channel.handle = channel_handle
    if channel_name:
        channel.display_name = channel_name
    channel.updated_at = now
    ensure_channel_entitlement(
        session,
        tenant_id=identity.tenant_id,
        channel_id=channel.id,
        granted_by_user_id=identity.user_id,
        access_level="query",
    )

    video = session.execute(
        select(SourceVideo).where(
            SourceVideo.platform == provider,
            SourceVideo.external_id == provider_video_id,
        )
    ).scalar_one_or_none()
    video_metadata = {
        **(metadata or {}),
        "canonical_publisher": {
            "transcript_content_sha256": content_sha256,
            "transcript_provider": transcript_provider,
        },
    }
    parsed_published_at = _parse_timestamp(published_at)
    normalized_duration_ms = _duration_ms(duration_ms)
    if video is None:
        video = SourceVideo(
            id=video_id,
            channel_id=channel.id,
            platform=provider,
            external_id=provider_video_id,
            canonical_url=_optional(canonical_url, 8000),
            title=_optional(title, 100_000),
            description=_optional_content(description, 1_000_000),
            published_at=parsed_published_at,
            duration_ms=normalized_duration_ms,
            archive_state=(
                "retained_hot_verified"
                if verified_hot_media is not None
                else "partial_only"
            ),
            clip_candidate=verified_hot_media is not None,
            clip_ready=verified_hot_media is not None,
            status="active",
            metadata_json=video_metadata,
        )
        session.add(video)
    else:
        if video.id != video_id or video.channel_id != channel.id:
            raise CanonicalPublishError("provider video identity collision")
        video.canonical_url = _optional(canonical_url, 8000) or video.canonical_url
        video.title = _optional(title, 100_000) or video.title
        video.description = (
            _optional_content(description, 1_000_000) or video.description
        )
        video.published_at = parsed_published_at or video.published_at
        video.duration_ms = normalized_duration_ms or video.duration_ms
        video.archive_state = (
            "retained_hot_verified"
            if verified_hot_media is not None
            else video.archive_state
        )
        video.clip_candidate = video.clip_candidate or verified_hot_media is not None
        video.clip_ready = video.clip_ready or verified_hot_media is not None
        video.status = "active"
        video.metadata_json = {**(video.metadata_json or {}), **video_metadata}
        video.updated_at = now
    session.flush()

    revision = session.get(TranscriptRevision, revision_id)
    if revision is None:
        session.execute(
            update(TranscriptRevision)
            .where(
                TranscriptRevision.video_id == video_id,
                TranscriptRevision.is_current.is_(True),
            )
            .values(is_current=False)
        )
        revision = TranscriptRevision(
            id=revision_id,
            video_id=video_id,
            provider=transcript_provider,
            provider_revision_id=f"{video_id}:{content_sha256}",
            language=language,
            content_sha256=content_sha256,
            is_current=True,
            status="active",
            captured_at=now,
            metadata_json={"canonical_segment_count": len(normalized_segments)},
        )
        session.add(revision)
        session.flush()
        session.add_all(
            [
                TranscriptSegment(
                    id=_stable_id("tsg", f"{revision_id}:{segment['ordinal']}"),
                    revision_id=revision_id,
                    ordinal=segment["ordinal"],
                    start_ms=segment["start_ms"],
                    end_ms=segment["end_ms"],
                    speaker_label=segment["speaker_label"],
                    text=segment["text"],
                    status="active",
                )
                for segment in normalized_segments
            ]
        )
    else:
        if (
            revision.video_id != video_id
            or revision.provider != transcript_provider
            or revision.content_sha256 != content_sha256
        ):
            raise CanonicalPublishError("transcript revision identity collision")
        stored_segments = [
            {
                "ordinal": row.ordinal,
                "start_ms": row.start_ms,
                "end_ms": row.end_ms,
                "speaker_label": row.speaker_label,
                "text": row.text,
            }
            for row in session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.revision_id == revision_id)
                .order_by(TranscriptSegment.ordinal.asc())
            ).scalars()
        ]
        if stored_segments != normalized_segments:
            raise CanonicalPublishError(
                "transcript revision segments do not match its digest"
            )
        revision.is_current = True
        revision.status = "active"

    if verified_hot_media is not None:
        _attach_verified_hot_media(
            session,
            video=video,
            media=verified_hot_media,
            now=now,
        )
    session.flush()
    return CanonicalPublishResult(
        media_id=video_id,
        channel_id=channel_id,
        transcript_revision_id=revision_id,
        transcript_segment_count=len(normalized_segments),
        clip_ready=video.clip_ready,
        hot_media_sha256=(verified_hot_media.sha256 if verified_hot_media else None),
    )


def _attach_verified_hot_media(
    session: Session,
    *,
    video: SourceVideo,
    media: HotMediaSpec,
    now: datetime,
) -> None:
    media_object = session.get(MediaObject, media.sha256)
    if media_object is None:
        media_object = MediaObject(
            sha256=media.sha256,
            size_bytes=media.size_bytes,
            mime_type=media.mime_type,
            status="active",
            metadata_json={"verification": "sha256_size_ffprobe_video"},
        )
        session.add(media_object)
    elif (
        media_object.size_bytes != media.size_bytes
        or media_object.mime_type != media.mime_type
    ):
        raise CanonicalPublishError(
            "hot media facts conflict with the canonical digest"
        )
    else:
        media_object.status = "active"
    session.flush()

    ref_id = _stable_id("vmr", f"{video.id}:source_video:{media.sha256}")
    reference = session.get(VideoMediaRef, ref_id)
    if reference is None:
        session.add(
            VideoMediaRef(
                id=ref_id,
                video_id=video.id,
                media_sha256=media.sha256,
                role="source_video",
                status="active",
            )
        )
    elif (
        reference.video_id != video.id
        or reference.media_sha256 != media.sha256
        or reference.role != "source_video"
    ):
        raise CanonicalPublishError("hot media reference identity collision")
    else:
        reference.status = "active"

    location_key = str(media.path.resolve())
    active_location = session.execute(
        select(MediaLocation).where(
            MediaLocation.media_sha256 == media.sha256,
            MediaLocation.backend == "hot_local",
            MediaLocation.status == "active",
        )
    ).scalar_one_or_none()
    if active_location is not None and active_location.location_key != location_key:
        raise CanonicalPublishError(
            "canonical media already has another active hot-local path"
        )
    location_id = _stable_id("loc", f"hot_local:{location_key}")
    location = session.get(MediaLocation, location_id)
    if location is None:
        session.add(
            MediaLocation(
                id=location_id,
                media_sha256=media.sha256,
                backend="hot_local",
                location_key=location_key,
                status="active",
                bytes=media.size_bytes,
                verified_at=now,
            )
        )
    elif (
        location.media_sha256 != media.sha256
        or location.backend != "hot_local"
        or location.location_key != location_key
        or location.bytes != media.size_bytes
    ):
        raise CanonicalPublishError("hot media location identity collision")
    else:
        location.status = "active"
        location.verified_at = now
        location.updated_at = now


def verify_hot_media(spec: HotMediaSpec) -> HotMediaSpec:
    """Verify a retained file without mutating it or trusting caller media claims."""
    return _verify_hot_media(spec)


def _verify_hot_media(spec: HotMediaSpec) -> HotMediaSpec:
    digest = str(spec.sha256 or "").strip()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise CanonicalPublishError("hot media SHA-256 must be lowercase hex")
    if isinstance(spec.size_bytes, bool) or int(spec.size_bytes) <= 0:
        raise CanonicalPublishError("hot media size must be positive")
    mime_type = _required(spec.mime_type, "hot media MIME type", 255).lower()
    if not mime_type.startswith("video/"):
        raise CanonicalPublishError(
            "clip-ready source media must have a video MIME type"
        )

    root_value = (
        os.getenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT") or "/data/hot-media"
    ).strip()
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise CanonicalPublishError("CHANNEL_SERVICE_HOT_MEDIA_ROOT must be absolute")
    root = root.resolve()
    path = Path(spec.path).expanduser()
    if not path.is_absolute():
        raise CanonicalPublishError("hot media path must be absolute")
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CanonicalPublishError(
            "hot media path is outside CHANNEL_SERVICE_HOT_MEDIA_ROOT"
        ) from exc
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise CanonicalPublishError("hot media path must be a regular file")
    if file_stat.st_size != int(spec.size_bytes):
        raise CanonicalPublishError("hot media size does not match the retained file")
    if _sha256_file(path) != digest:
        raise CanonicalPublishError(
            "hot media SHA-256 does not match the retained file"
        )
    _probe_video(path)
    return HotMediaSpec(
        path=path,
        sha256=digest,
        size_bytes=int(spec.size_bytes),
        mime_type=mime_type,
    )


def _probe_video(path: Path) -> None:
    executable = (
        os.getenv("CHANNEL_SERVICE_FFPROBE_BIN") or "/usr/local/bin/ffprobe"
    ).strip()
    if not executable or not Path(executable).is_absolute():
        raise CanonicalPublishError(
            "CHANNEL_SERVICE_FFPROBE_BIN must be an absolute path"
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type:format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CanonicalPublishError(
            "ffprobe could not verify retained hot media"
        ) from exc
    if completed.returncode != 0:
        raise CanonicalPublishError(
            "retained hot media failed ffprobe decode validation"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CanonicalPublishError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not any(
        isinstance(stream, dict) and stream.get("codec_type") == "video"
        for stream in streams
    ):
        raise CanonicalPublishError("retained hot media has no decodable video stream")


def _normalize_transcript_segments(
    segments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(segments):
        if ordinal >= 500_000 or not isinstance(segment, dict):
            raise CanonicalPublishError(
                "transcript segment collection is invalid or too large"
            )
        text = _content_text(segment.get("text"), "transcript segment text", 1_000_000)
        if "start_ms" in segment:
            start_ms = _milliseconds(
                segment.get("start_ms"), "start_ms", milliseconds=True
            )
        else:
            start_ms = _milliseconds(segment.get("start"), "start", milliseconds=False)
        if "end_ms" in segment:
            end_ms = _milliseconds(segment.get("end_ms"), "end_ms", milliseconds=True)
        else:
            end_ms = _milliseconds(segment.get("end"), "end", milliseconds=False)
        if end_ms < start_ms:
            raise CanonicalPublishError("transcript segment end precedes start")
        speaker = segment.get("speaker_label", segment.get("speaker"))
        normalized.append(
            {
                "ordinal": ordinal,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_label": _optional(speaker, 255),
                "text": text,
            }
        )
    return normalized


def _milliseconds(value: Any, field_name: str, *, milliseconds: bool) -> int:
    if value is None or isinstance(value, bool):
        raise CanonicalPublishError(f"transcript segment {field_name} is required")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalPublishError(
            f"transcript segment {field_name} is invalid"
        ) from exc
    if numeric < 0 or numeric > 10_000_000_000:
        raise CanonicalPublishError(f"transcript segment {field_name} is out of range")
    if milliseconds:
        return int(round(numeric))
    return int(round(numeric * 1000))


def _duration_ms(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) < 0:
        raise CanonicalPublishError("duration_ms must be non-negative")
    return int(value)


def _parse_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if re.fullmatch(r"[0-9]{8}", raw):
            parsed = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CanonicalPublishError(
                    "published_at must be ISO-8601 or YYYYMMDD"
                ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _provider(value: str) -> str:
    normalized = _required(value, "platform", 32).lower()
    return {
        "yt": "youtube",
        "twitch_vod": "twitch",
        "pump.fun": "pumpfun",
        "pump_fun": "pumpfun",
    }.get(normalized, normalized)


def _required(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise CanonicalPublishError(f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > max_length
        or any(ord(char) < 32 for char in normalized)
    ):
        raise CanonicalPublishError(f"{field_name} is missing or invalid")
    return normalized


def _optional(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    return _required(value, "optional canonical field", max_length)


def _content_text(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise CanonicalPublishError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or "\x00" in normalized:
        raise CanonicalPublishError(f"{field_name} is missing or invalid")
    return normalized


def _optional_content(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    return _content_text(value, "optional canonical content", max_length)


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:40]}"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
