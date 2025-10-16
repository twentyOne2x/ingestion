from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Set, Tuple

from .gcs import read_json_from_gcs

LOG = logging.getLogger(__name__)


@dataclass
class EventContext:
    source: str
    video_id: str
    mp3_uri: str
    pumpfun_room: Optional[str] = None
    pumpfun_clip: Optional[str] = None
    pumpfun_metadata: Optional[dict] = None

    def pumpfun_coin(self) -> dict:
        payload = self.pumpfun_metadata or {}
        coin = payload.get("coin") if isinstance(payload, dict) else None
        return coin if isinstance(coin, dict) else {}

    def pumpfun_clip_payload(self) -> dict:
        payload = self.pumpfun_metadata or {}
        clip = payload.get("clip") if isinstance(payload, dict) else None
        return clip if isinstance(clip, dict) else {}


def resolve_event_context(event) -> EventContext:
    mp3_uri = getattr(event, "mp3_uri", "") or ""
    if "pumpfun_streams" in mp3_uri:
        room, clip = _extract_pumpfun_path(mp3_uri)
        raw_video_id = getattr(event, "video_id", None)
        video_id = raw_video_id or _slugify(f"pumpfun_{room}_{clip}")
        context = EventContext(source="pumpfun", video_id=video_id, mp3_uri=mp3_uri, pumpfun_room=room, pumpfun_clip=clip)
        metadata_uri = getattr(event, "metadata_uri", None)
        if metadata_uri:
            try:
                context.pumpfun_metadata = read_json_from_gcs(metadata_uri)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to load Pump.fun metadata %s: %s", metadata_uri, exc)
                context.pumpfun_metadata = {}
        return context

    video_id = extract_video_id(getattr(event, "diarized_uri"))
    return EventContext(source="youtube", video_id=video_id, mp3_uri=mp3_uri)


def _extract_pumpfun_path(mp3_uri: str) -> Tuple[str, str]:
    without_scheme = mp3_uri[len("gs://") :] if mp3_uri.startswith("gs://") else mp3_uri
    segments = without_scheme.split("/")
    try:
        idx = segments.index("pumpfun_streams")
    except ValueError:
        return "pumpfun", _strip_extension(Path(mp3_uri).stem)
    room = segments[idx + 1] if idx + 1 < len(segments) else "pumpfun"
    clip = segments[idx + 2] if idx + 2 < len(segments) else Path(mp3_uri).stem
    return room, clip


def _slugify(value: str, max_length: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if len(slug) > max_length:
        slug = slug[:max_length]
    return slug or "pumpfun_clip"


def extract_video_id(diarized_uri: str) -> str:
    """
    Attempt to derive the YouTube video ID from a diarization artefact URI.

    We first look for ``youtube_diarized/<video_id>/`` in the path; if absent,
    fall back to the first 11-character YouTube-style token.
    """
    segments: Sequence[str]
    if diarized_uri.startswith("gs://"):
        without_scheme = diarized_uri[len("gs://") :]
        segments = without_scheme.split("/")
    elif diarized_uri.startswith("file://"):
        path = Path(diarized_uri[7:])
        segments = path.parts
    else:
        raise ValueError(f"Unsupported URI scheme: {diarized_uri}")

    for i, part in enumerate(segments):
        if part == "youtube_diarized" and i + 1 < len(segments):
            candidate = _strip_extension(segments[i + 1])
            if _looks_like_video_id(candidate):
                return candidate

    match = _find_first_video_id(segments)
    if match:
        return match
    raise ValueError(f"Could not determine video ID from {diarized_uri}")


def _looks_like_video_id(value: str) -> bool:
    return bool(value) and any(ch.isdigit() for ch in value) and len(value) >= 6


def _strip_extension(segment: str) -> str:
    return segment.split(".")[0]


_DATE_ID_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}_(?P<id>[A-Za-z0-9_-]{3,}?)(?:_|$)")
_UNDERSCORE_RE = re.compile(r"_(?P<id>[A-Za-z0-9_-]{3,}?)(?:_|$)")
_TOKEN_RE = re.compile(r"(?P<id>[A-Za-z0-9_-]{6,})")


def _find_first_video_id(segments: Sequence[str]) -> Optional[str]:
    for segment in segments:
        segment = _strip_extension(segment)
        if not segment or segment.lower().startswith("youtube"):
            continue
        m = _DATE_ID_RE.search(segment)
        if m and _looks_like_video_id(m.group("id")):
            return m.group("id")
        m = _UNDERSCORE_RE.search(segment)
        if m and _looks_like_video_id(m.group("id")):
            return m.group("id")
        m = _TOKEN_RE.search(segment)
        if m:
            candidate = m.group("id").split("_")[0]
            if _looks_like_video_id(candidate):
                return candidate
    return None


@dataclass
class VideoMetadata:
    video_id: str
    channel_id: str
    channel_title: str
    channel_handle: Optional[str]
    title: str
    description: str
    published_at: Optional[str]
    duration_seconds: float
    thumbnail_url: Optional[str]

    def preferred_channel_name(self) -> str:
        return self.channel_handle or self.channel_title


@dataclass
class DiarizationIngestService:
    namespace: str
    allowed_channels: Sequence[str]
    fetch_video: Callable[[str], VideoMetadata]
    load_artifacts: Callable[[object, VideoMetadata, EventContext], Tuple[dict, dict]]
    ingest_pipeline: Callable[[dict, dict, object], None]

    def handle_event(self, event) -> None:
        context = resolve_event_context(event)

        if context.source == "pumpfun":
            video_meta = build_pumpfun_metadata(event, context)
        else:
            video_meta = self.fetch_video(context.video_id)

        channel_name = video_meta.preferred_channel_name()
        allowed = {c.casefold() for c in self.allowed_channels if c}
        candidate_names: Set[str] = {
            (channel_name or "").casefold(),
            (video_meta.channel_title or "").casefold(),
        }
        if context.source == "pumpfun":
            clip_payload = context.pumpfun_clip_payload()
            coin_payload = context.pumpfun_coin()
            room = clip_payload.get("roomName") or context.pumpfun_room
            for value in [room, coin_payload.get("mint"), coin_payload.get("symbol"), coin_payload.get("name"), context.video_id]:
                if isinstance(value, str) and value:
                    candidate_names.add(value.casefold())
        if allowed and allowed.isdisjoint({name for name in candidate_names if name}):
            LOG.info(
                "Skipping video %s for namespace=%s channel=%s (not in %s)",
                context.video_id,
                self.namespace,
                channel_name,
                self.allowed_channels,
            )
            return

        meta, raw_segments = self.load_artifacts(event, video_meta, context)
        meta = dict(meta or {})
        meta.setdefault("video_id", context.video_id)
        meta.setdefault("channel_id", video_meta.channel_id)
        meta.setdefault("channel_name", channel_name)
        self.ingest_pipeline(meta, raw_segments, event)


def create_ingest_service(namespace: str, allowed_channels: Sequence[str]):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY env var is required for diarization ingestion")

    from .youtube import YouTubeClient
    from src.ingest_v2.pipelines.build_children import build_children_from_raw
    from src.ingest_v2.pipelines.build_parents import build_parent
    from src.ingest_v2.pipelines.upsert_parents import upsert_parents
    from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
    from src.ingest_v2.pipelines.run_all_components.assemblyai import convert_assemblyai_json_to_raw

    youtube_client = YouTubeClient(api_key=api_key)

    def fetch_video(video_id: str) -> VideoMetadata:
        return youtube_client.fetch_video_metadata(video_id)

    def load_artifacts(event, video_meta: VideoMetadata, context: EventContext) -> Tuple[dict, dict]:
        diarized_payload = read_json_from_gcs(event.diarized_uri)
        raw_segments = convert_assemblyai_json_to_raw(diarized_payload)
        entities_payload = None
        if event.entities_uri:
            entities_payload = read_json_from_gcs(event.entities_uri)
        elif isinstance(diarized_payload, dict) and diarized_payload.get("entities"):
            entities_payload = diarized_payload.get("entities")

        entities = _extract_entities(entities_payload)
        meta = {
            "video_id": video_meta.video_id,
            "title": video_meta.title,
            "description": video_meta.description,
            "channel_name": video_meta.preferred_channel_name(),
            "channel_id": video_meta.channel_id,
            "published_at": video_meta.published_at,
            "duration_s": _duration_from_meta(video_meta, raw_segments),
            "url": _resolve_video_url(context, video_meta),
            "thumbnail_url": video_meta.thumbnail_url,
            "language": "en",
            "source": context.source,
            "entities": entities,
        }
        if context.source == "pumpfun":
            clip_payload = context.pumpfun_clip_payload()
            coin_payload = context.pumpfun_coin()
            meta.update(
                {
                    "pumpfun_room": context.pumpfun_room or clip_payload.get("roomName"),
                    "pumpfun_clip_id": clip_payload.get("clipId"),
                    "pumpfun_coin_name": coin_payload.get("name"),
                    "pumpfun_coin_symbol": coin_payload.get("symbol"),
                    "pumpfun_start_time": clip_payload.get("startTime"),
                }
            )
        return meta, raw_segments

    def ingest_pipeline(meta: dict, raw_segments: dict, event) -> None:
        parent = build_parent(meta)
        parent_dict = parent.model_dump(mode="json")
        parent_payload = dict(parent_dict, parent_id=parent.parent_id)
        parent_payload.setdefault("video_id", parent.parent_id)
        upsert_parents([parent_payload])
        children = build_children_from_raw(parent_dict, raw_segments)
        if not children:
            LOG.info("No children produced for %s", meta["video_id"])
            return
        upsert_children(children)
        LOG.info(
            "Ingested diarization-ready video=%s children=%d",
            meta["video_id"],
            len(children),
        )

    return DiarizationIngestService(
        namespace=namespace,
        allowed_channels=tuple(allowed_channels),
        fetch_video=fetch_video,
        load_artifacts=load_artifacts,
        ingest_pipeline=ingest_pipeline,
    )


def build_pumpfun_metadata(event, context: EventContext) -> VideoMetadata:
    clip_payload = context.pumpfun_clip_payload()
    coin_payload = context.pumpfun_coin()

    room = clip_payload.get("roomName") if isinstance(clip_payload, dict) else None
    channel_token = coin_payload.get("symbol") if isinstance(coin_payload, dict) else None
    channel_name = coin_payload.get("name") if isinstance(coin_payload, dict) else None
    title_timestamp = clip_payload.get("startTime") if isinstance(clip_payload, dict) else None
    title = f"Pump.fun stream {title_timestamp}" if title_timestamp else "Pump.fun stream"
    description = coin_payload.get("description") if isinstance(coin_payload, dict) else ""
    duration = clip_payload.get("duration") if isinstance(clip_payload, dict) else 0
    thumbnail = clip_payload.get("thumbnailUrl") if isinstance(clip_payload, dict) else None

    def _as_float(value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    return VideoMetadata(
        video_id=context.video_id,
        channel_id=str(room or context.video_id),
        channel_title=str(channel_name or channel_token or room or "Pump.fun"),
        channel_handle=str(channel_token or channel_name or room or "pumpfun"),
        title=title,
        description=description or "",
        published_at=title_timestamp,
        duration_seconds=_as_float(duration),
        thumbnail_url=thumbnail,
    )


def _resolve_video_url(context: EventContext, video_meta: VideoMetadata) -> str:
    if context.source == "pumpfun":
        clip_payload = context.pumpfun_clip_payload()
        playlist_url = clip_payload.get("playlistUrl") if isinstance(clip_payload, dict) else None
        if isinstance(playlist_url, str) and playlist_url:
            return playlist_url
        return context.mp3_uri
    return f"https://www.youtube.com/watch?v={video_meta.video_id}"


def _extract_entities(payload) -> list[str]:
    texts = set()
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                text = (item.get("text") or item.get("entity"))
            else:
                text = item
            if text and isinstance(text, str):
                texts.add(text.strip())
    elif isinstance(payload, dict):
        for item in payload.get("entities", []):
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    texts.add(text.strip())
    return sorted(t for t in texts if t)


def _duration_from_meta(video_meta: VideoMetadata, raw_segments: dict) -> float:
    if video_meta.duration_seconds:
        return float(video_meta.duration_seconds)
    segments = raw_segments.get("segments") if isinstance(raw_segments, dict) else []
    max_end = 0.0
    for segment in segments or []:
        end = segment.get("end")
        if isinstance(end, (int, float)) and end > max_end:
            max_end = float(end)
    return max_end
