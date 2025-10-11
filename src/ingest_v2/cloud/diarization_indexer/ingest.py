from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)


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

    # gs://bucket/<prefix>/...
    # If we see youtube_diarized/<video_id>/..., use that segment.
    for i, part in enumerate(segments):
        if part == "youtube_diarized" and i + 1 < len(segments):
            candidate = segments[i + 1]
            candidate = _strip_extension(candidate)
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
        # prefer date prefix style: YYYY-MM-DD_<id>
        m = _DATE_ID_RE.search(segment)
        if m and _looks_like_video_id(m.group("id")):
            return m.group("id")
        m = _UNDERSCORE_RE.search(segment)
        if m and _looks_like_video_id(m.group("id")):
            return m.group("id")
        m = _TOKEN_RE.search(segment)
        if m:
            candidate = m.group("id")
            candidate = candidate.split("_")[0]
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
    load_artifacts: Callable[[object, VideoMetadata], Tuple[dict, dict]]
    ingest_pipeline: Callable[[dict, dict, object], None]

    def handle_event(self, event) -> None:
        video_id = extract_video_id(event.diarized_uri)
        video_meta = self.fetch_video(video_id)
        channel_name = video_meta.preferred_channel_name()
        allowed = {c.casefold() for c in self.allowed_channels}
        candidate_names = {
            (channel_name or "").casefold(),
            (video_meta.channel_title or "").casefold(),
        }
        if allowed and allowed.isdisjoint({name for name in candidate_names if name}):
            LOG.info(
                "Skipping video %s for namespace=%s channel=%s (not in %s)",
                video_id,
                self.namespace,
                channel_name,
                self.allowed_channels,
            )
            return

        meta, raw_segments = self.load_artifacts(event, video_meta)
        meta = dict(meta or {})
        meta.setdefault("video_id", video_meta.video_id)
        meta.setdefault("channel_id", video_meta.channel_id)
        meta.setdefault("channel_name", channel_name)
        self.ingest_pipeline(meta, raw_segments, event)


def create_ingest_service(namespace: str, allowed_channels: Sequence[str]):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY env var is required for diarization ingestion")

    from .gcs import read_json_from_gcs
    from .youtube import YouTubeClient
    from src.ingest_v2.pipelines.build_children import build_children_from_raw
    from src.ingest_v2.pipelines.build_parents import build_parent
    from src.ingest_v2.pipelines.upsert_parents import upsert_parents
    from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
    from src.ingest_v2.pipelines.run_all_components.assemblyai import convert_assemblyai_json_to_raw

    youtube_client = YouTubeClient(api_key=api_key)

    def fetch_video(video_id: str) -> VideoMetadata:
        return youtube_client.fetch_video_metadata(video_id)

    def load_artifacts(event, video_meta: VideoMetadata) -> Tuple[dict, dict]:
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
            "url": f"https://www.youtube.com/watch?v={video_meta.video_id}",
            "thumbnail_url": video_meta.thumbnail_url,
            "language": "en",
            "entities": entities,
        }
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
