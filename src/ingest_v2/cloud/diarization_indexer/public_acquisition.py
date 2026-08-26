from __future__ import annotations

import hashlib
import html
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from src.ingest_v2.pipelines.index_youtube_captions import _require_ytdlp
from src.ingest_v2.pipelines.youtube_ytdlp_options import (
    build_youtube_ytdlp_options,
    safe_ytdlp_error_message,
)

from .canonical_media import HotMediaSpec, verify_hot_media
from .public_platforms import CanonicalPublicTarget, normalize_public_target


class PublicAcquisitionError(RuntimeError):
    """Public discovery or media acquisition failed closed."""


@dataclass(frozen=True)
class PublicItemDescriptor:
    platform: str
    external_id: str
    channel_external_id: str
    channel_handle: str | None
    canonical_url: str
    title: str | None = None
    description: str | None = None
    published_at: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "external_id": self.external_id,
            "channel_external_id": self.channel_external_id,
            "channel_handle": self.channel_handle,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "description": self.description,
            "published_at": self.published_at,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class AcquiredPublicItem:
    item: PublicItemDescriptor
    media: HotMediaSpec


def discover_public_items(
    target: CanonicalPublicTarget,
    *,
    max_items: int,
    ydl_factory: Callable[[dict[str, Any]], Any] | None = None,
    http: Any | None = None,
) -> tuple[PublicItemDescriptor, ...]:
    if target.target_kind != "channel":
        raise PublicAcquisitionError("discovery requires a canonical channel target")
    if max_items < 1 or max_items > 200:
        raise PublicAcquisitionError("max_items must be between 1 and 200")
    if target.platform in {"youtube", "twitch"}:
        return _discover_ytdlp(target, max_items=max_items, ydl_factory=ydl_factory)
    if target.platform == "pumpfun":
        return _discover_pumpfun(target, max_items=max_items, http=http)
    if target.platform == "x":
        return _discover_x(target, max_items=max_items, http=http)
    raise PublicAcquisitionError("unsupported public platform")


def acquire_public_item(
    item: PublicItemDescriptor,
    *,
    ydl_factory: Callable[[dict[str, Any]], Any] | None = None,
    http: Any | None = None,
) -> AcquiredPublicItem:
    if item.platform == "pumpfun":
        return _acquire_pumpfun(item, http=http)
    if item.platform in {"youtube", "twitch", "x"}:
        return _acquire_ytdlp(item, ydl_factory=ydl_factory)
    raise PublicAcquisitionError("unsupported public platform")


def descriptor_from_target(target: CanonicalPublicTarget) -> PublicItemDescriptor:
    if target.target_kind != "item":
        raise PublicAcquisitionError("an item target is required")
    channel_external_id = (
        target.channel_external_id or target.handle or "pending-provider-identity"
    )
    return PublicItemDescriptor(
        platform=target.platform,
        external_id=target.external_id,
        channel_external_id=channel_external_id,
        channel_handle=target.handle,
        canonical_url=target.canonical_url,
        metadata={
            "identity_pending_provider_readback": target.channel_external_id is None
        },
    )


def _discover_ytdlp(
    target: CanonicalPublicTarget,
    *,
    max_items: int,
    ydl_factory: Callable[[dict[str, Any]], Any] | None,
) -> tuple[PublicItemDescriptor, ...]:
    url = target.canonical_url
    if target.platform == "twitch":
        url = f"{url}/videos?filter=all&sort=time"
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": max_items,
        "ignoreerrors": False,
        "socket_timeout": _bounded_int(
            "CHANNEL_SERVICE_PUBLIC_SOCKET_TIMEOUT_SECONDS", 30, 5, 300
        ),
        "retries": _bounded_int("CHANNEL_SERVICE_PUBLIC_RETRIES", 3, 0, 20),
    }
    if target.platform == "youtube":
        opts = build_youtube_ytdlp_options(opts)
    factory = ydl_factory or _require_ytdlp()
    try:
        with factory(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        if target.platform != "youtube":
            raise PublicAcquisitionError(
                f"{target.platform} public discovery failed: {exc}"
            ) from exc
        raise PublicAcquisitionError(
            f"youtube public discovery failed: {safe_ytdlp_error_message(exc)}"
        ) from None
    if not isinstance(info, dict) or not isinstance(info.get("entries"), (list, tuple)):
        raise PublicAcquisitionError("provider returned no bounded channel item list")
    rows: list[PublicItemDescriptor] = []
    seen: set[str] = set()
    for raw in info["entries"]:
        if not isinstance(raw, dict):
            raise PublicAcquisitionError(
                "provider channel list contains an invalid item"
            )
        external_id = (
            str(raw.get("id") or "")
            .strip()
            .lstrip("v" if target.platform == "twitch" else "\0")
        )
        if target.platform == "twitch":
            if not re.fullmatch(r"[0-9]{6,22}", external_id):
                raise PublicAcquisitionError(
                    "Twitch discovery returned an unsafe VOD identity"
                )
            canonical_url = f"https://www.twitch.tv/videos/{external_id}"
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", external_id):
                raise PublicAcquisitionError(
                    "YouTube discovery returned an unsafe video identity"
                )
            canonical_url = f"https://www.youtube.com/watch?v={external_id}"
        if external_id in seen:
            continue
        seen.add(external_id)
        channel_external_id = target.channel_external_id or target.external_id
        uploader_id = str(raw.get("channel_id") or raw.get("uploader_id") or "").strip()
        if (
            uploader_id
            and target.platform == "twitch"
            and uploader_id.casefold() != str(target.handle).casefold()
        ):
            raise PublicAcquisitionError("Twitch VOD was rebound to another channel")
        duration = raw.get("duration")
        rows.append(
            PublicItemDescriptor(
                platform=target.platform,
                external_id=external_id,
                channel_external_id=channel_external_id,
                channel_handle=target.handle,
                canonical_url=canonical_url,
                title=_optional_text(raw.get("title")),
                description=_optional_text(raw.get("description")),
                published_at=_upload_date(raw.get("upload_date")),
                duration_ms=(
                    int(float(duration) * 1000)
                    if isinstance(duration, (int, float)) and duration >= 0
                    else None
                ),
                metadata={"discovery": "public_yt_dlp_flat"},
            )
        )
        if len(rows) >= max_items:
            break
    return tuple(rows)


def _discover_pumpfun(
    target: CanonicalPublicTarget, *, max_items: int, http: Any | None
) -> tuple[PublicItemDescriptor, ...]:
    session = http or requests.Session()
    room = target.external_id
    rows: list[PublicItemDescriptor] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while len(rows) < max_items:
        params = {"limit": str(min(50, max_items - len(rows)))}
        if cursor:
            params["lastEvaluatedKey"] = cursor
        response = session.get(
            f"https://livestream-api.pump.fun/clips/{room}",
            params=params,
            timeout=_bounded_int(
                "CHANNEL_SERVICE_PUBLIC_SOCKET_TIMEOUT_SECONDS", 30, 5, 300
            ),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
            raise PublicAcquisitionError("Pump.fun clips response is invalid")
        for raw in payload["clips"]:
            if not isinstance(raw, dict):
                raise PublicAcquisitionError(
                    "Pump.fun clips response contains an invalid item"
                )
            clip_id = str(raw.get("clipId") or "").strip()
            canonical = normalize_public_target(
                platform="pumpfun",
                target_kind="item",
                target=f"pumpfun:{room}:{clip_id}",
            )
            playlist = str(raw.get("playlistUrl") or "").strip()
            if playlist:
                _validate_pumpfun_playlist(playlist)
            rows.append(
                PublicItemDescriptor(
                    platform="pumpfun",
                    external_id=canonical.external_id,
                    channel_external_id=room,
                    channel_handle=None,
                    canonical_url=canonical.canonical_url,
                    title=_optional_text(raw.get("title")),
                    published_at=_optional_text(
                        raw.get("startTime") or raw.get("createdAt")
                    ),
                    duration_ms=_milliseconds(
                        raw.get("duration") or raw.get("durationSeconds")
                    ),
                    metadata={
                        "discovery": "pumpfun_public_clips_api",
                        "playlist_url": playlist or None,
                    },
                )
            )
            if len(rows) >= max_items:
                break
        if not payload.get("hasMore") or len(rows) >= max_items:
            break
        next_cursor = str(payload.get("lastEvaluatedKey") or "").strip()
        if not next_cursor or next_cursor in seen_cursors or len(next_cursor) > 4_096:
            raise PublicAcquisitionError(
                "Pump.fun pagination cursor is missing or cyclic"
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return tuple(rows)


_NEXT_DATA = re.compile(
    rb'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)


def _discover_x(
    target: CanonicalPublicTarget, *, max_items: int, http: Any | None
) -> tuple[PublicItemDescriptor, ...]:
    if not target.platform_entity_id or not target.handle:
        raise PublicAcquisitionError(
            "X profile discovery requires numeric and handle identity"
        )
    session = http or requests.Session()
    url = (
        "https://syndication.twitter.com/srv/timeline-profile/user-id/"
        f"{target.platform_entity_id}?lang=en&dnt=true"
    )
    response = session.get(
        url,
        timeout=_bounded_int(
            "CHANNEL_SERVICE_PUBLIC_SOCKET_TIMEOUT_SECONDS", 30, 5, 300
        ),
        headers={"User-Agent": "Mozilla/5.0 ICMFYI-public-ingestion/1"},
    )
    response.raise_for_status()
    body = bytes(response.content)
    if len(body) > _bounded_int(
        "CHANNEL_SERVICE_X_MAX_DISCOVERY_BYTES",
        16 * 1024 * 1024,
        1024,
        64 * 1024 * 1024,
    ):
        raise PublicAcquisitionError("X discovery response exceeded its byte bound")
    match = _NEXT_DATA.search(body)
    if match is None:
        raise PublicAcquisitionError("X syndication response lacks __NEXT_DATA__")
    try:
        document = json.loads(html.unescape(match.group(1).decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicAcquisitionError("X syndication response is invalid") from exc
    entries = (
        document.get("props", {})
        .get("pageProps", {})
        .get("timeline", {})
        .get("entries")
        if isinstance(document, dict)
        else None
    )
    if not isinstance(entries, list):
        raise PublicAcquisitionError("X syndication timeline is missing")
    rows: list[PublicItemDescriptor] = []
    seen: set[str] = set()
    for entry in entries:
        tweet = (
            entry.get("content", {}).get("tweet") if isinstance(entry, dict) else None
        )
        if not isinstance(tweet, dict):
            raise PublicAcquisitionError("X timeline contains a non-tweet entry")
        user = tweet.get("user")
        if not isinstance(user, dict):
            raise PublicAcquisitionError("X timeline tweet author is missing")
        if (
            str(user.get("id_str") or "") != target.platform_entity_id
            or str(user.get("screen_name") or "").casefold() != target.handle.casefold()
        ):
            raise PublicAcquisitionError("X numeric/handle identity mismatch")
        post_id = str(tweet.get("id_str") or "")
        if not re.fullmatch(r"[0-9]{6,22}", post_id):
            raise PublicAcquisitionError("X post identity is unsafe")
        media = tweet.get("extended_entities", {}).get("media")
        if media is None:
            media = tweet.get("mediaDetails") or []
        if not isinstance(media, list):
            raise PublicAcquisitionError("X media shape is invalid")
        videos = [
            value
            for value in media
            if isinstance(value, dict)
            and value.get("type") in {"video", "animated_gif"}
        ]
        if not videos or post_id in seen:
            continue
        seen.add(post_id)
        rows.append(
            PublicItemDescriptor(
                platform="x",
                external_id=post_id,
                channel_external_id=target.platform_entity_id,
                channel_handle=target.handle,
                canonical_url=f"https://x.com/{target.handle}/status/{post_id}",
                title=_optional_text(tweet.get("full_text") or tweet.get("text")),
                metadata={
                    "discovery": "x_public_syndication_profile",
                    "media_count": len(videos),
                    "lifetime_complete": False,
                    "snapshot_sha256": hashlib.sha256(body).hexdigest(),
                },
            )
        )
        if len(rows) >= max_items:
            break
    return tuple(rows)


def _acquire_ytdlp(
    item: PublicItemDescriptor, *, ydl_factory: Callable[[dict[str, Any]], Any] | None
) -> AcquiredPublicItem:
    staging = _staging_dir(item)
    retained = _retained_receipt(staging / "receipt.json", item=item)
    if retained is not None:
        return retained
    max_bytes = _bounded_int(
        "CHANNEL_SERVICE_MAX_HOT_MEDIA_BYTES",
        8 * 1024 * 1024 * 1024,
        1,
        64 * 1024 * 1024 * 1024,
    )
    opts: dict[str, Any] = {
        "noplaylist": True,
        "outtmpl": str(staging / "download.%(ext)s"),
        "max_filesize": max_bytes,
        "socket_timeout": _bounded_int(
            "CHANNEL_SERVICE_PUBLIC_SOCKET_TIMEOUT_SECONDS", 30, 5, 300
        ),
        "retries": _bounded_int("CHANNEL_SERVICE_PUBLIC_RETRIES", 3, 0, 20),
        "fragment_retries": _bounded_int("CHANNEL_SERVICE_PUBLIC_RETRIES", 3, 0, 20),
        "ffmpeg_location": (
            os.getenv("CHANNEL_SERVICE_FFMPEG_BIN") or "/usr/local/bin/ffmpeg"
        ).strip(),
    }
    if item.platform == "youtube":
        opts = build_youtube_ytdlp_options(opts, media=True)
    else:
        opts.update(
            {
                "quiet": True,
                "no_warnings": True,
                "format": "bestvideo*+bestaudio/best",
                "merge_output_format": "mp4",
            }
        )
    factory = ydl_factory or _require_ytdlp()
    try:
        with factory(opts) as ydl:
            info = ydl.extract_info(item.canonical_url, download=True)
    except Exception as exc:
        if item.platform != "youtube":
            raise PublicAcquisitionError(
                f"{item.platform} public media download failed: {exc}"
            ) from exc
        raise PublicAcquisitionError(
            f"youtube public media download failed: {safe_ytdlp_error_message(exc)}"
        ) from None
    if not isinstance(info, dict) or info.get("entries"):
        raise PublicAcquisitionError(
            "provider returned an ambiguous multi-item download"
        )
    if item.platform in {"youtube", "twitch"}:
        resolved = str(info.get("id") or "").lstrip(
            "v" if item.platform == "twitch" else "\0"
        )
        if resolved != item.external_id:
            raise PublicAcquisitionError(
                "download resolved to a different provider item"
            )
    if item.platform == "x":
        webpage = str(info.get("webpage_url") or item.canonical_url)
        if f"/status/{item.external_id}" not in webpage:
            raise PublicAcquisitionError("X download resolved to a different post")
    resolved_item = _resolved_ytdlp_descriptor(item, info)
    candidate = _single_video_candidate(staging)
    return _publish_to_cas(item=resolved_item, source=candidate, max_bytes=max_bytes)


def _acquire_pumpfun(
    item: PublicItemDescriptor, *, http: Any | None
) -> AcquiredPublicItem:
    session = http or requests.Session()
    response = session.get(
        item.canonical_url,
        timeout=_bounded_int(
            "CHANNEL_SERVICE_PUBLIC_SOCKET_TIMEOUT_SECONDS", 30, 5, 300
        ),
    )
    response.raise_for_status()
    metadata = response.json()
    if (
        not isinstance(metadata, dict)
        or str(metadata.get("clipId") or "") != item.external_id
    ):
        raise PublicAcquisitionError("Pump.fun clip identity mismatch")
    playlist = str(metadata.get("playlistUrl") or "").strip()
    _validate_pumpfun_playlist(playlist)
    staging = _staging_dir(item)
    retained = _retained_receipt(staging / "receipt.json", item=item)
    if retained is not None:
        return retained
    source = staging / "download.mp4"
    if source.exists() or source.is_symlink():
        source.unlink(missing_ok=True)
    ffmpeg = (
        os.getenv("CHANNEL_SERVICE_FFMPEG_BIN") or "/usr/local/bin/ffmpeg"
    ).strip()
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            playlist,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            str(source),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_bounded_int(
            "CHANNEL_SERVICE_PUBLIC_DOWNLOAD_TIMEOUT_SECONDS", 3600, 30, 21_600
        ),
        check=False,
    )
    if completed.returncode != 0:
        source.unlink(missing_ok=True)
        raise PublicAcquisitionError(
            f"Pump.fun clip download failed: {completed.stderr[-1000:]}"
        )
    return _publish_to_cas(
        item=item,
        source=source,
        max_bytes=_bounded_int(
            "CHANNEL_SERVICE_MAX_HOT_MEDIA_BYTES",
            8 * 1024 * 1024 * 1024,
            1,
            64 * 1024 * 1024 * 1024,
        ),
    )


def _publish_to_cas(
    *, item: PublicItemDescriptor, source: Path, max_bytes: int
) -> AcquiredPublicItem:
    info = source.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
        raise PublicAcquisitionError("downloaded video size is invalid")
    digest = _sha256_file(source)
    root = _hot_root()
    destination_dir = root / "sha256" / digest[:2]
    destination_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    destination = destination_dir / f"{digest}.mp4"
    source.chmod(0o440)
    try:
        os.link(source, destination)
    except FileExistsError:
        pass
    spec = verify_hot_media(
        HotMediaSpec(
            path=destination,
            sha256=digest,
            size_bytes=int(info.st_size),
            mime_type="video/mp4",
        )
    )
    receipt = {
        "schema": "icmfyi.public-hot-media-receipt.v1",
        "platform": item.platform,
        "external_id": item.external_id,
        "canonical_url": item.canonical_url,
        "path": str(spec.path),
        "sha256": spec.sha256,
        "size_bytes": spec.size_bytes,
        "mime_type": spec.mime_type,
        "item": item.as_payload(),
    }
    receipt_path = _staging_dir(item) / "receipt.json"
    _atomic_json(receipt_path, receipt)
    source.unlink(missing_ok=True)
    return AcquiredPublicItem(item=item, media=spec)


def _retained_receipt(
    path: Path, *, item: PublicItemDescriptor
) -> AcquiredPublicItem | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise PublicAcquisitionError(
            "retained public-media receipt is invalid"
        ) from exc
    if (
        payload.get("schema") != "icmfyi.public-hot-media-receipt.v1"
        or payload.get("platform") != item.platform
        or payload.get("external_id") != item.external_id
        or payload.get("canonical_url") != item.canonical_url
    ):
        raise PublicAcquisitionError("retained public-media receipt identity mismatch")
    retained_item_payload = payload.get("item")
    if not isinstance(retained_item_payload, dict):
        raise PublicAcquisitionError(
            "retained public-media receipt lacks item identity"
        )
    retained_item = PublicItemDescriptor(
        platform=str(retained_item_payload.get("platform") or ""),
        external_id=str(retained_item_payload.get("external_id") or ""),
        channel_external_id=str(retained_item_payload.get("channel_external_id") or ""),
        channel_handle=(
            str(retained_item_payload["channel_handle"])
            if retained_item_payload.get("channel_handle")
            else None
        ),
        canonical_url=str(retained_item_payload.get("canonical_url") or ""),
        title=(
            str(retained_item_payload["title"])
            if retained_item_payload.get("title")
            else None
        ),
        description=(
            str(retained_item_payload["description"])
            if retained_item_payload.get("description")
            else None
        ),
        published_at=(
            str(retained_item_payload["published_at"])
            if retained_item_payload.get("published_at")
            else None
        ),
        duration_ms=(
            int(retained_item_payload["duration_ms"])
            if retained_item_payload.get("duration_ms") is not None
            else None
        ),
        metadata=(dict(retained_item_payload.get("metadata") or {})),
    )
    media = verify_hot_media(
        HotMediaSpec(
            path=Path(str(payload.get("path") or "")),
            sha256=str(payload.get("sha256") or ""),
            size_bytes=int(payload.get("size_bytes") or 0),
            mime_type=str(payload.get("mime_type") or ""),
        )
    )
    return AcquiredPublicItem(item=retained_item, media=media)


def _resolved_ytdlp_descriptor(
    item: PublicItemDescriptor, info: dict[str, Any]
) -> PublicItemDescriptor:
    channel_external_id = str(
        info.get("channel_id") or info.get("uploader_id") or item.channel_external_id
    ).strip()
    channel_handle = (
        str(
            info.get("channel") or info.get("uploader") or item.channel_handle or ""
        ).strip()
        or None
    )
    if not channel_external_id or len(channel_external_id) > 255:
        raise PublicAcquisitionError(
            "provider did not return a stable channel identity"
        )
    if item.channel_external_id not in {
        "pending-provider-identity",
        channel_external_id,
    }:
        expected = str(item.channel_handle or item.channel_external_id).casefold()
        observed = str(channel_handle or channel_external_id).casefold()
        if (
            expected != observed
            and str(item.channel_external_id) != channel_external_id
        ):
            raise PublicAcquisitionError(
                "downloaded item was rebound to another channel"
            )
    duration = info.get("duration")
    return PublicItemDescriptor(
        platform=item.platform,
        external_id=item.external_id,
        channel_external_id=channel_external_id,
        channel_handle=channel_handle,
        canonical_url=item.canonical_url,
        title=_optional_text(info.get("title")) or item.title,
        description=_optional_text(info.get("description")) or item.description,
        published_at=_upload_date(info.get("upload_date")) or item.published_at,
        duration_ms=(
            int(float(duration) * 1000)
            if isinstance(duration, (int, float)) and duration >= 0
            else item.duration_ms
        ),
        metadata={
            **(item.metadata or {}),
            "provider_extractor": _optional_text(info.get("extractor_key")),
        },
    )


def _staging_dir(item: PublicItemDescriptor) -> Path:
    root = _hot_root()
    digest = hashlib.sha256(
        f"{item.platform}:{item.channel_external_id}:{item.external_id}".encode("utf-8")
    ).hexdigest()
    path = root / ".staging" / "public" / item.platform / digest
    path.mkdir(mode=0o750, parents=True, exist_ok=True)
    if path.is_symlink():
        raise PublicAcquisitionError("public-media staging directory is a symlink")
    return path


def _hot_root() -> Path:
    root = Path(
        (os.getenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT") or "/data/hot-media").strip()
    ).expanduser()
    if not root.is_absolute():
        raise PublicAcquisitionError("CHANNEL_SERVICE_HOT_MEDIA_ROOT must be absolute")
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    return root.resolve()


def _single_video_candidate(staging: Path) -> Path:
    candidates = [
        path
        for path in staging.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]
    if len(candidates) != 1:
        raise PublicAcquisitionError("download did not produce exactly one video file")
    candidate = candidates[0]
    if candidate.suffix.lower() != ".mp4":
        converted = staging / "download.mp4"
        if converted.exists() and converted != candidate:
            converted.unlink()
        ffmpeg = (
            os.getenv("CHANNEL_SERVICE_FFMPEG_BIN") or "/usr/local/bin/ffmpeg"
        ).strip()
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(candidate),
                "-c",
                "copy",
                str(converted),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,
            check=False,
        )
        if completed.returncode != 0:
            raise PublicAcquisitionError("downloaded video could not be remuxed to MP4")
        candidate.unlink()
        candidate = converted
    return candidate


def _validate_pumpfun_playlist(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or host not in {"clips.pump.fun", "livestream-api.pump.fun"}
        or parsed.username
        or parsed.password
        or parsed.port
    ):
        raise PublicAcquisitionError("Pump.fun clip playlist host is not allowlisted")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    temp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with temp.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temp.chmod(0o440)
    os.replace(temp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text[:100_000] if text else None


def _upload_date(value: Any) -> str | None:
    raw = str(value or "")
    if not re.fullmatch(r"[0-9]{8}", raw):
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}T00:00:00Z"


def _milliseconds(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return int(round(float(value) * 1000))


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise PublicAcquisitionError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise PublicAcquisitionError(f"{name} must be between {minimum} and {maximum}")
    return value
