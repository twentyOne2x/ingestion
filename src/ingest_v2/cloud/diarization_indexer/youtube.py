from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

try:  # pragma: no cover - optional dependency
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover
    build = None

try:  # pragma: no cover - optional dependency
    from isodate import parse_duration
except Exception:  # pragma: no cover
    def parse_duration(_duration: str):  # type: ignore
        class _Fallback:
            def total_seconds(self):
                return 0.0

        return _Fallback()

from .ingest import VideoMetadata

LOG = logging.getLogger(__name__)


@dataclass
class YouTubeClient:
    api_key: str
    _client: Optional[object] = field(default=None, init=False, repr=False)

    def _service(self):
        if self._client is None:
            if build is None:
                raise RuntimeError("googleapiclient is required to fetch YouTube metadata")
            self._client = build("youtube", "v3", developerKey=self.api_key)
        return self._client

    def fetch_video_metadata(self, video_id: str) -> VideoMetadata:
        service = self._service()
        response = (
            service.videos()
            .list(part="snippet,contentDetails", id=video_id)
            .execute()
        )
        items = response.get("items") or []
        if not items:
            raise ValueError(f"Video {video_id} not found")
        item = items[0]
        snippet = item.get("snippet", {})
        channel_id = snippet.get("channelId")
        duration_s = _parse_duration_seconds(item.get("contentDetails", {}).get("duration"))
        channel_handle = self._resolve_channel_handle(channel_id)
        return VideoMetadata(
            video_id=video_id,
            channel_id=channel_id or "",
            channel_title=snippet.get("channelTitle") or "",
            channel_handle=channel_handle,
            title=snippet.get("title") or "",
            description=snippet.get("description") or "",
            published_at=snippet.get("publishedAt"),
            duration_seconds=duration_s,
            thumbnail_url=_pick_thumbnail(snippet.get("thumbnails")),
        )

    def _resolve_channel_handle(self, channel_id: Optional[str]) -> Optional[str]:
        if not channel_id:
            return None
        response = (
            self._service()
            .channels()
            .list(part="snippet", id=channel_id)
            .execute()
        )
        items = response.get("items") or []
        if not items:
            return None
        snippet = items[0].get("snippet", {})
        handle = snippet.get("customUrl") or snippet.get("title")
        if handle:
            handle = handle.strip()
            if not handle.startswith("@"):
                handle = f"@{handle.lstrip('@')}"
        return handle


def _parse_duration_seconds(duration: Optional[str]) -> float:
    if not duration:
        return 0.0
    try:
        return parse_duration(duration).total_seconds()
    except Exception:
        LOG.debug("Could not parse duration %s", duration)
        return 0.0


def _pick_thumbnail(thumbnails: Optional[dict]) -> Optional[str]:
    if not thumbnails:
        return None
    for key in ("maxres", "high", "medium", "default"):
        entry = thumbnails.get(key)
        if entry and entry.get("url"):
            return entry["url"]
    return None
