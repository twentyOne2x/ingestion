from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, unquote, urlparse


Platform = Literal["youtube", "twitch", "pumpfun", "x"]
TargetKind = Literal["channel", "item"]

_HANDLE = re.compile(r"[A-Za-z0-9_]{1,25}\Z")
_TWITCH_VIDEO_ID = re.compile(r"(?:v)?([0-9]{6,22})\Z")
_X_POST_ID = re.compile(r"[0-9]{6,22}\Z")
_X_USER_ID = re.compile(r"[0-9]{2,22}\Z")
_PUMPFUN_ROOM = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}\Z")
_PUMPFUN_CLIP = re.compile(r"[A-Za-z0-9][A-Za-z0-9:_-]{0,127}\Z")
_YOUTUBE_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


class PublicTargetError(ValueError):
    """A caller supplied an ambiguous or unsafe public-provider identity."""


@dataclass(frozen=True)
class CanonicalPublicTarget:
    platform: Platform
    target_kind: TargetKind
    external_id: str
    canonical_url: str
    handle: str | None = None
    channel_external_id: str | None = None
    platform_entity_id: str | None = None

    @property
    def source_key(self) -> str:
        if self.target_kind == "item" and self.channel_external_id:
            return f"{self.channel_external_id}:{self.external_id}"
        return self.external_id

    def as_payload(self) -> dict[str, str | None]:
        return {
            "platform": self.platform,
            "target_kind": self.target_kind,
            "external_id": self.external_id,
            "canonical_url": self.canonical_url,
            "handle": self.handle,
            "channel_external_id": self.channel_external_id,
            "platform_entity_id": self.platform_entity_id,
        }


def normalize_public_target(
    *,
    platform: str,
    target_kind: str,
    target: str,
    platform_entity_id: str | None = None,
) -> CanonicalPublicTarget:
    normalized_platform = str(platform or "").strip().lower()
    if normalized_platform == "twitter":
        normalized_platform = "x"
    if normalized_platform not in {"youtube", "twitch", "pumpfun", "x"}:
        raise PublicTargetError("platform must be youtube, twitch, pumpfun, or x")
    normalized_kind = str(target_kind or "").strip().lower()
    if normalized_kind not in {"channel", "item"}:
        raise PublicTargetError("target_kind must be channel or item")
    raw = str(target or "").strip()
    if not raw or len(raw) > 8_000 or any(ord(char) < 32 for char in raw):
        raise PublicTargetError("target is missing or unsafe")
    entity_id = str(platform_entity_id or "").strip() or None

    parser = {
        "youtube": _normalize_youtube,
        "twitch": _normalize_twitch,
        "pumpfun": _normalize_pumpfun,
        "x": _normalize_x,
    }[normalized_platform]
    return parser(normalized_kind, raw, entity_id)  # type: ignore[arg-type]


def _url(raw: str, *, allowed_hosts: set[str]):
    if "://" not in raw:
        return None
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or host not in allowed_hosts:
        raise PublicTargetError(
            "target URL must use the provider's canonical HTTPS host"
        )
    if parsed.username or parsed.password or parsed.port:
        raise PublicTargetError(
            "target URL must not contain credentials or a custom port"
        )
    return parsed


def _segments(parsed) -> list[str]:
    return [unquote(part) for part in parsed.path.split("/") if part]


def _normalize_twitch(kind: str, raw: str, _: str | None) -> CanonicalPublicTarget:
    parsed = _url(raw, allowed_hosts={"twitch.tv", "www.twitch.tv"})
    parts = _segments(parsed) if parsed else []
    if kind == "channel":
        handle = (parts[0] if parts else raw.lstrip("@")).lower()
        if not _HANDLE.fullmatch(handle) or handle in {
            "videos",
            "directory",
            "downloads",
        }:
            raise PublicTargetError("Twitch channel must be an exact public handle")
        return CanonicalPublicTarget(
            platform="twitch",
            target_kind="channel",
            external_id=handle,
            handle=handle,
            channel_external_id=handle,
            canonical_url=f"https://www.twitch.tv/{handle}",
        )
    candidate = parts[1] if len(parts) == 2 and parts[0].lower() == "videos" else raw
    match = _TWITCH_VIDEO_ID.fullmatch(candidate)
    if not match:
        raise PublicTargetError(
            "Twitch item must be a canonical /videos/<numeric-id> URL"
        )
    video_id = match.group(1)
    return CanonicalPublicTarget(
        platform="twitch",
        target_kind="item",
        external_id=video_id,
        canonical_url=f"https://www.twitch.tv/videos/{video_id}",
    )


def _normalize_pumpfun(kind: str, raw: str, _: str | None) -> CanonicalPublicTarget:
    parsed = _url(
        raw,
        allowed_hosts={
            "pump.fun",
            "www.pump.fun",
            "livestream-api.pump.fun",
        },
    )
    parts = _segments(parsed) if parsed else raw.split(":", 2)
    if kind == "channel":
        room = raw
        if parsed:
            if len(parts) == 2 and parts[0].lower() in {"coin", "board"}:
                room = parts[1]
            else:
                raise PublicTargetError(
                    "Pump.fun channel must be a canonical /coin/<mint> URL"
                )
        if not _PUMPFUN_ROOM.fullmatch(room):
            raise PublicTargetError(
                "Pump.fun channel must be an exact public room/coin mint"
            )
        return CanonicalPublicTarget(
            platform="pumpfun",
            target_kind="channel",
            external_id=room,
            channel_external_id=room,
            canonical_url=f"https://pump.fun/coin/{room}",
        )
    if parsed:
        if len(parts) != 3 or parts[0].lower() != "clips":
            raise PublicTargetError(
                "Pump.fun item must be livestream-api.pump.fun/clips/<room>/<clip-id>"
            )
        room, clip_id = parts[1], parts[2]
    else:
        if len(parts) != 3 or parts[0].lower() != "pumpfun":
            raise PublicTargetError(
                "Pump.fun item must include both room and clip identity"
            )
        room, clip_id = parts[1], parts[2]
    if not _PUMPFUN_ROOM.fullmatch(room) or not _PUMPFUN_CLIP.fullmatch(clip_id):
        raise PublicTargetError("Pump.fun room or clip identity is unsafe")
    return CanonicalPublicTarget(
        platform="pumpfun",
        target_kind="item",
        external_id=clip_id,
        channel_external_id=room,
        canonical_url=f"https://livestream-api.pump.fun/clips/{room}/{clip_id}",
    )


def _normalize_x(kind: str, raw: str, entity_id: str | None) -> CanonicalPublicTarget:
    parsed = _url(
        raw, allowed_hosts={"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
    )
    parts = _segments(parsed) if parsed else []
    if kind == "channel":
        handle = (parts[0] if parts else raw.lstrip("@")).lower()
        if not re.fullmatch(r"[a-z0-9_]{1,15}", handle):
            raise PublicTargetError("X channel must be an exact public handle")
        if not entity_id or not _X_USER_ID.fullmatch(entity_id):
            raise PublicTargetError(
                "X channel ingestion requires the profile's numeric platform_entity_id"
            )
        return CanonicalPublicTarget(
            platform="x",
            target_kind="channel",
            external_id=entity_id,
            platform_entity_id=entity_id,
            handle=handle,
            channel_external_id=entity_id,
            canonical_url=f"https://x.com/{handle}",
        )
    if len(parts) != 3 or parts[1].lower() != "status":
        raise PublicTargetError(
            "X item must be a canonical /<handle>/status/<post-id> URL"
        )
    handle, post_id = parts[0].lower(), parts[2]
    if not re.fullmatch(r"[a-z0-9_]{1,15}", handle) or not _X_POST_ID.fullmatch(
        post_id
    ):
        raise PublicTargetError("X item handle or post identity is unsafe")
    if not entity_id or not _X_USER_ID.fullmatch(entity_id):
        raise PublicTargetError(
            "X item ingestion requires the author's numeric platform_entity_id"
        )
    return CanonicalPublicTarget(
        platform="x",
        target_kind="item",
        external_id=post_id,
        handle=handle,
        channel_external_id=entity_id,
        platform_entity_id=entity_id,
        canonical_url=f"https://x.com/{handle}/status/{post_id}",
    )


def _normalize_youtube(
    kind: str, raw: str, entity_id: str | None
) -> CanonicalPublicTarget:
    parsed = _url(raw, allowed_hosts={"youtube.com", "www.youtube.com", "youtu.be"})
    if kind == "item":
        candidate = raw
        if parsed:
            if (parsed.hostname or "").lower() == "youtu.be":
                parts = _segments(parsed)
                candidate = parts[0] if len(parts) == 1 else ""
            else:
                candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        if not _YOUTUBE_VIDEO_ID.fullmatch(candidate):
            raise PublicTargetError("YouTube item must contain an exact video identity")
        return CanonicalPublicTarget(
            platform="youtube",
            target_kind="item",
            external_id=candidate,
            canonical_url=f"https://www.youtube.com/watch?v={candidate}",
        )
    if not parsed:
        raise PublicTargetError("YouTube channel must be a canonical HTTPS channel URL")
    parts = _segments(parsed)
    if (
        len(parts) != 2
        or parts[0] not in {"channel", "@"}
        and not parts[0].startswith("@")
    ):
        # @handle arrives as one path segment; /channel/UC... as two.
        if len(parts) != 1 or not parts[0].startswith("@"):
            raise PublicTargetError(
                "YouTube channel must use /channel/<id> or /@<handle>"
            )
    if len(parts) == 1:
        external_id = parts[0]
    else:
        external_id = parts[1]
    if entity_id:
        external_id = entity_id
    if not re.fullmatch(r"(?:UC[A-Za-z0-9_-]{22}|@[A-Za-z0-9_.-]{3,30})", external_id):
        raise PublicTargetError("YouTube channel identity is unsafe")
    return CanonicalPublicTarget(
        platform="youtube",
        target_kind="channel",
        external_id=external_id,
        handle=external_id if external_id.startswith("@") else None,
        channel_external_id=external_id,
        canonical_url=f"https://www.youtube.com/{external_id if external_id.startswith('@') else 'channel/' + external_id}",
    )
