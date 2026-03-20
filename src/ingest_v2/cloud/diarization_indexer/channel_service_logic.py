from __future__ import annotations

import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import select

from src.ingest_v2.pipelines.index_youtube_captions import (
    _fetch_transcript_api_cues,
    _coerce_watch_url,
    _normalize_channel_url,
    _resolve_channel_id_via_api,
    _require_ytdlp,
    _uploads_playlist_id,
    _yt_watch_url,
    _ytapi_get,
    _ytdlp_extra_opts,
    fetch_transcript_cues,
    index_youtube_video_captions,
)
from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels

from .channel_service_store import (
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    PackBatch,
    PackVideo,
    QuoteVideo,
    TranscriptProbe,
    utcnow,
)
from .youtube import _parse_duration_seconds

_PENDING_PROBE_STATUSES = {"queued", "running", "retry"}
_RESTRICTED_HANDLE_TERMS = ("porn", "nsfw", "adult", "xxx", "onlyfans", "sex")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def payment_required() -> bool:
    raw = (os.getenv("CHANNEL_SERVICE_REQUIRE_PAYMENT") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def inline_index_enabled() -> bool:
    raw = (os.getenv("CHANNEL_SERVICE_ENABLE_INLINE_INDEX") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def qdrant_collection_name(namespace: str) -> str:
    template = os.getenv("QDRANT_COLLECTION_TEMPLATE", "{index}__{namespace}")
    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    return template.format(index=index_name, namespace=namespace)


def qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def transcript_probe_key(video_id: str, language: str, prefer_auto: bool) -> str:
    mode = "auto" if prefer_auto else "manual"
    return f"{video_id}:{language}:{mode}"


def _probe_root() -> Path:
    raw = os.getenv("CHANNEL_SERVICE_PROBE_ROOT") or "/data/channel-service/probes"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _probe_artifact_path(*, video_id: str, language: str, prefer_auto: bool) -> Path:
    suffix = "auto" if prefer_auto else "manual"
    return _probe_root() / video_id / f"{language}.{suffix}.json"


def _write_probe_artifact(
    *,
    video_id: str,
    language: str,
    prefer_auto: bool,
    transcript_source: str,
    transcript_rows: List[dict],
) -> str:
    path = _probe_artifact_path(video_id=video_id, language=language, prefer_auto=prefer_auto)
    payload = {
        "video_id": video_id,
        "language": language,
        "prefer_auto": prefer_auto,
        "transcript_source": transcript_source,
        "transcript_rows": transcript_rows,
        "row_count": len(transcript_rows),
        "fetched_at": utcnow().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return str(path)


def load_probe_artifact_rows(probe: TranscriptProbe) -> List[dict]:
    path_value = (probe.artifact_path or "").strip()
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("transcript_rows")
    return list(rows) if isinstance(rows, list) else []


def probe_is_ready(probe: Optional[TranscriptProbe]) -> bool:
    if probe is None or probe.status != "ready":
        return False
    return bool(load_probe_artifact_rows(probe))


def load_transcript_probes(
    *,
    session,
    video_ids: List[str],
    language: str,
    prefer_auto: bool,
) -> Dict[str, TranscriptProbe]:
    keys = [transcript_probe_key(video_id, language, prefer_auto) for video_id in video_ids if video_id]
    if not keys:
        return {}
    rows = session.execute(select(TranscriptProbe).where(TranscriptProbe.key.in_(keys))).scalars().all()
    return {row.key: row for row in rows}


def ensure_transcript_probe(
    *,
    session,
    row: dict | QuoteVideo,
    language: str,
    prefer_auto: bool,
) -> TranscriptProbe:
    video_id = row.video_id if hasattr(row, "video_id") else str(row.get("video_id") or "").strip()
    if not video_id:
        raise ValueError("video_id is required for transcript acquisition")
    key = transcript_probe_key(video_id, language, prefer_auto)
    probe = session.get(TranscriptProbe, key)
    if probe is not None:
        if probe.status in {"unavailable", "failed"}:
            probe.status = "queued"
            probe.next_attempt_at = utcnow()
        return probe
    video_url = row.video_url if hasattr(row, "video_url") else row.get("video_url")
    channel_handle = row.channel_handle if hasattr(row, "channel_handle") else row.get("channel_handle")
    probe = TranscriptProbe(
        key=key,
        video_id=video_id,
        video_url=str(video_url or _yt_watch_url(video_id)),
        channel_handle=str(channel_handle or "").strip() or None,
        language=language,
        prefer_auto=prefer_auto,
        status="queued",
        next_attempt_at=utcnow(),
    )
    session.add(probe)
    session.flush()
    return probe


def supported_channels(namespace: str) -> dict:
    handles = [h.strip() for h in load_namespace_channels(namespace) if h and h.strip()]
    seen = set()
    rows = []
    for handle in handles:
        key = handle.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "handle": handle,
                "name": handle.lstrip("@"),
                "supported": True,
            }
        )
    rows.sort(key=lambda item: item["handle"].lower())
    defaults = [row["handle"] for row in rows]
    return {"scope": namespace, "channels": rows, "defaultSelected": defaults}


def _normalize_channel_handle(handle: Optional[str]) -> Optional[str]:
    value = str(handle or "").strip()
    if not value:
        return None
    return f"@{value.lstrip('@')}"


@lru_cache(maxsize=1)
def _channel_handle_to_id_map() -> Dict[str, str]:
    path_value = (os.getenv("YT_CHANNEL_MAPPING") or "").strip()
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows: Dict[str, str] = {}
    for raw_handle, raw_channel_id in payload.items():
        handle = _normalize_channel_handle(raw_handle)
        channel_id = str(raw_channel_id or "").strip()
        if handle and channel_id:
            rows[handle.lower()] = channel_id
    return rows


def _catalog_supports_handle(*, namespace: str, channel_handle: Optional[str]) -> bool:
    normalized = _normalize_channel_handle(channel_handle)
    if not normalized:
        return False
    supported = {_normalize_channel_handle(item) for item in load_namespace_channels(namespace)}
    return normalized in supported


def _channel_handle_is_restricted(channel_handle: Optional[str]) -> bool:
    normalized = (_normalize_channel_handle(channel_handle) or "").lower().lstrip("@")
    if not normalized:
        return False
    return any(term in normalized for term in _RESTRICTED_HANDLE_TERMS)


def _normalize_published_bound(value: Optional[str], *, field_name: str) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format") from exc


def _validated_published_bounds(
    *,
    published_after: Optional[str],
    published_before: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    after = _normalize_published_bound(published_after, field_name="published_after")
    before = _normalize_published_bound(published_before, field_name="published_before")
    if after and before and after > before:
        raise ValueError("published_after must be on or before published_before")
    return after, before


def _coerce_upload_date(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    else:
        raw = raw[:10]
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _extract_handle_from_channel_url(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if "/@" not in raw:
        return None
    handle = raw.split("/@", 1)[1].split("/", 1)[0].strip()
    if not handle:
        return None
    return _normalize_channel_handle(handle)


def _indexed_channel_candidates(
    *,
    namespace: str,
    channel_handle: str,
    target_count: int,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
) -> dict:
    normalized_handle = _normalize_channel_handle(channel_handle)
    if not normalized_handle:
        return {
            "channel_id": None,
            "channel_name": "",
            "channel_handle": channel_handle,
            "videos": [],
        }

    must_conditions = [qm.FieldCondition(key="node_type", match=qm.MatchValue(value="parent"))]
    channel_id = _channel_handle_to_id_map().get(normalized_handle.lower())
    if channel_id:
        must_conditions.append(qm.FieldCondition(key="channel_id", match=qm.MatchValue(value=channel_id)))
    else:
        channel_name = normalized_handle.lstrip("@")
        must_conditions.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="channel_handle", match=qm.MatchValue(value=normalized_handle)),
                    qm.FieldCondition(key="channel_name", match=qm.MatchValue(value=channel_name)),
                ]
            )
        )

    rows: List[dict] = []
    offset = None
    client = qdrant_client()
    collection_name = qdrant_collection_name(namespace)
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=qm.Filter(must=must_conditions),
            with_payload=True,
            with_vectors=False,
            limit=min(256, max(32, target_count * 4)),
            offset=offset,
        )
        if not batch:
            break
        for point in batch:
            payload = point.payload or {}
            video_id = str(payload.get("parent_id") or "").strip()
            if not video_id:
                continue
            published_value = str(payload.get("published_at") or "").strip()
            published_day = published_value[:10] or None
            if published_after and published_day and published_day < published_after:
                continue
            if published_before and published_day and published_day > published_before:
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "title": payload.get("title") or video_id,
                    "description": payload.get("description") or "",
                    "channel_name": payload.get("channel_name") or normalized_handle.lstrip("@"),
                    "channel_handle": payload.get("channel_handle") or normalized_handle,
                    "published_at": published_day,
                    "thumbnail_url": payload.get("thumbnail_url"),
                    "video_url": payload.get("url") or _yt_watch_url(video_id),
                    "duration_s": float(payload.get("duration_s") or 0.0),
                    "channel_id": payload.get("channel_id") or channel_id,
                }
            )
        if not offset:
            break

    if not rows:
        return {
            "channel_id": channel_id,
            "channel_name": normalized_handle.lstrip("@"),
            "channel_handle": normalized_handle,
            "videos": [],
        }

    rows.sort(
        key=lambda item: (
            item.get("published_at") or "",
            item.get("video_id") or "",
        ),
        reverse=True,
    )
    trimmed = rows[:target_count]
    first = trimmed[0]
    return {
        "channel_id": first.get("channel_id") or channel_id,
        "channel_name": first.get("channel_name") or normalized_handle.lstrip("@"),
        "channel_handle": first.get("channel_handle") or normalized_handle,
        "videos": trimmed,
    }


def resolve_channel_summary(channel_handle: str, api_key: str) -> dict:
    channel_id = _resolve_channel_id_via_api(channel_handle, api_key=api_key)
    if not channel_id:
        raise ValueError(f"Could not resolve channel {channel_handle}")
    resp = _ytapi_get("channels", api_key=api_key, params={"part": "snippet", "id": channel_id, "maxResults": 1})
    items = resp.get("items") or []
    if not items:
        raise ValueError(f"Could not load channel metadata for {channel_handle}")
    snippet = items[0].get("snippet") or {}
    custom = (snippet.get("customUrl") or "").strip()
    resolved_handle = f"@{custom.lstrip('@')}" if custom else channel_handle
    return {
        "channel_id": channel_id,
        "channel_name": snippet.get("title") or resolved_handle.lstrip("@"),
        "channel_handle": resolved_handle,
    }


def list_channel_candidates_via_ytdlp(
    *,
    channel_handle: str,
    target_count: int,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
) -> dict:
    normalized_handle = _normalize_channel_handle(channel_handle)
    if not normalized_handle:
        raise ValueError("input.channel_handle is required")

    YDL = _require_ytdlp()
    fetch_limit = max(25, min(200, max(1, int(target_count)) * 25))
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": fetch_limit,
    }
    ydl_opts.update(_ytdlp_extra_opts())

    with YDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(_normalize_channel_url(normalized_handle), download=False)
        except Exception as exc:
            raise ValueError(f"Could not resolve channel {normalized_handle}") from exc

    entries = info.get("entries") or []
    channel_id = str(info.get("channel_id") or info.get("id") or "").strip() or None
    channel_name = (
        str(info.get("channel") or info.get("uploader") or info.get("title") or "").strip()
        or normalized_handle.lstrip("@")
    )
    resolved_handle = normalized_handle
    channel_url = str(info.get("channel_url") or info.get("webpage_url") or "").strip()
    extracted_handle = _normalize_channel_handle(info.get("uploader_id")) or _extract_handle_from_channel_url(channel_url)
    if normalized_handle.startswith("@"):
        if not extracted_handle:
            raise ValueError(f"Could not resolve channel {normalized_handle}")
        if extracted_handle.lower() != normalized_handle.lower():
            raise ValueError(f"Could not resolve channel {normalized_handle}")
        resolved_handle = extracted_handle

    videos: List[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id") or "").strip()
        if not video_id:
            continue
        published_at = _coerce_upload_date(entry.get("upload_date"))
        if published_after and published_at and published_at < published_after:
            continue
        if published_before and published_at and published_at > published_before:
            continue
        value = entry.get("webpage_url") or entry.get("url") or entry.get("id")
        watch_url = _coerce_watch_url(str(value or ""))
        if not watch_url:
            continue
        thumbnail = entry.get("thumbnail")
        if not thumbnail:
            thumbs = entry.get("thumbnails") or []
            if isinstance(thumbs, list) and thumbs:
                thumbnail = (thumbs[-1] or {}).get("url")
        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title") or video_id,
                "description": entry.get("description") or "",
                "channel_name": entry.get("channel") or entry.get("uploader") or channel_name,
                "channel_handle": resolved_handle,
                "published_at": published_at,
                "thumbnail_url": thumbnail,
                "video_url": watch_url,
                "duration_s": float(entry.get("duration") or 0.0),
                "channel_id": str(entry.get("channel_id") or channel_id or "").strip() or None,
            }
        )
        if len(videos) >= int(target_count):
            break

    if not videos and not entries:
        raise ValueError(f"Could not resolve channel {normalized_handle}")

    return {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_handle": resolved_handle,
        "videos": videos[:target_count],
    }


def _chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def list_channel_candidates(
    *,
    api_key: str,
    namespace: str,
    channel_handle: str,
    target_count: int,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
) -> dict:
    published_after, published_before = _validated_published_bounds(
        published_after=published_after,
        published_before=published_before,
    )
    if _channel_handle_is_restricted(channel_handle):
        raise ValueError(f"Channel handle {channel_handle} is not allowed for this transcript-pack offer")
    indexed_fallback = _indexed_channel_candidates(
        namespace=namespace,
        channel_handle=channel_handle,
        target_count=target_count,
        published_after=published_after,
        published_before=published_before,
    )
    if indexed_fallback["videos"] and _catalog_supports_handle(namespace=namespace, channel_handle=channel_handle):
        return indexed_fallback

    ytdlp_candidates: Optional[dict] = None
    ytdlp_error: Optional[Exception] = None
    try:
        ytdlp_candidates = list_channel_candidates_via_ytdlp(
            channel_handle=channel_handle,
            target_count=target_count,
            published_after=published_after,
            published_before=published_before,
        )
    except Exception as exc:
        ytdlp_error = exc
    else:
        if ytdlp_candidates["videos"]:
            return ytdlp_candidates

    if indexed_fallback["videos"]:
        return indexed_fallback

    if api_key:
        try:
            summary = resolve_channel_summary(channel_handle, api_key)
            uploads = _uploads_playlist_id(summary["channel_id"], api_key=api_key)
            if not uploads:
                raise ValueError(f"Could not resolve uploads playlist for {channel_handle}")
        except Exception:
            if indexed_fallback["videos"]:
                return indexed_fallback
        else:
            items: List[dict] = []
            page_token: Optional[str] = None
            while len(items) < target_count:
                resp = _ytapi_get(
                    "playlistItems",
                    api_key=api_key,
                    params={
                        "part": "snippet",
                        "playlistId": uploads,
                        "maxResults": min(50, max(1, target_count - len(items))),
                        **({"pageToken": page_token} if page_token else {}),
                    },
                )
                batch = resp.get("items") or []
                if not batch:
                    break
                for item in batch:
                    snippet = item.get("snippet") or {}
                    resource = snippet.get("resourceId") or {}
                    video_id = str(resource.get("videoId") or "").strip()
                    published_at = str(snippet.get("publishedAt") or "").strip()[:10] or None
                    if published_after and published_at and published_at < published_after:
                        continue
                    if published_before and published_at and published_at > published_before:
                        continue
                    if not video_id:
                        continue
                    thumb = (snippet.get("thumbnails") or {}).get("high") or (snippet.get("thumbnails") or {}).get("default") or {}
                    items.append(
                        {
                            "video_id": video_id,
                            "title": snippet.get("title") or video_id,
                            "description": snippet.get("description") or "",
                            "channel_name": snippet.get("videoOwnerChannelTitle")
                            or snippet.get("channelTitle")
                            or summary["channel_name"],
                            "channel_handle": summary["channel_handle"],
                            "published_at": published_at,
                            "thumbnail_url": thumb.get("url"),
                            "video_url": _yt_watch_url(video_id),
                        }
                    )
                    if len(items) >= target_count:
                        break
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

            if items:
                for chunk in _chunked([row["video_id"] for row in items], 50):
                    resp = _ytapi_get(
                        "videos",
                        api_key=api_key,
                        params={"part": "contentDetails,snippet", "id": ",".join(chunk)},
                    )
                    by_id = {}
                    for item in resp.get("items") or []:
                        by_id[str(item.get("id") or "")] = item
                    for row in items:
                        item = by_id.get(row["video_id"])
                        if not item:
                            continue
                        snippet = item.get("snippet") or {}
                        row["published_at"] = (str(snippet.get("publishedAt") or "").strip()[:10] or row["published_at"])
                        row["channel_name"] = snippet.get("channelTitle") or row["channel_name"]
                        row["duration_s"] = float(
                            _parse_duration_seconds((item.get("contentDetails") or {}).get("duration"))
                        )
                        thumbs = snippet.get("thumbnails") or {}
                        row["thumbnail_url"] = (
                            (thumbs.get("high") or {}).get("url")
                            or (thumbs.get("medium") or {}).get("url")
                            or row["thumbnail_url"]
                        )
                return {**summary, "videos": items[:target_count]}

    if ytdlp_error is not None:
        raise ytdlp_error
    if ytdlp_candidates is not None:
        return ytdlp_candidates
    return indexed_fallback


def indexed_parent_rows(video_ids: List[str], namespace: str) -> Dict[str, dict]:
    backend = (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()
    if backend != "qdrant":
        return {}
    wanted = [str(v) for v in video_ids if v]
    if not wanted:
        return {}

    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="parent_id", match=qm.MatchAny(any=wanted)),
            qm.FieldCondition(key="node_type", match=qm.MatchValue(value="parent")),
        ]
    )

    rows: Dict[str, dict] = {}
    offset = None
    client = qdrant_client()
    try:
        while True:
            batch, offset = client.scroll(
                collection_name=qdrant_collection_name(namespace),
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=min(256, max(1, len(wanted))),
                offset=offset,
            )
            if not batch:
                break
            for point in batch:
                payload = dict(getattr(point, "payload", None) or {})
                parent_id = str(payload.get("parent_id") or payload.get("video_id") or "").strip()
                if not parent_id:
                    continue
                rows[parent_id] = payload
            if offset is None:
                break
    except Exception:
        return {}
    return rows


def child_segments_by_parent(parent_ids: List[str], namespace: str) -> Dict[str, List[dict]]:
    backend = (os.getenv("VECTOR_STORE", "pinecone") or "pinecone").strip().lower()
    if backend != "qdrant":
        return {}
    wanted = [str(v) for v in parent_ids if v]
    if not wanted:
        return {}
    flt = qm.Filter(
        must=[
            qm.FieldCondition(key="parent_id", match=qm.MatchAny(any=wanted)),
            qm.FieldCondition(key="node_type", match=qm.MatchValue(value="child")),
        ]
    )
    out: Dict[str, List[dict]] = {parent_id: [] for parent_id in wanted}
    offset = None
    client = qdrant_client()
    try:
        while True:
            batch, offset = client.scroll(
                collection_name=qdrant_collection_name(namespace),
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            if not batch:
                break
            for point in batch:
                payload = dict(getattr(point, "payload", None) or {})
                parent_id = str(payload.get("parent_id") or "").strip()
                if not parent_id:
                    continue
                out.setdefault(parent_id, []).append(payload)
            if offset is None:
                break
    except Exception:
        return {}
    for values in out.values():
        values.sort(key=lambda row: (float(row.get("start_s") or 0.0), str(row.get("segment_id") or "")))
    return out


def per_video_cents(mode: str, total_included_count: int) -> int:
    mode_norm = (mode or "recent_pack").strip().lower()
    if mode_norm == "full_channel_backfill":
        return 10
    return 15 if total_included_count >= 50 else 20


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def acquisition_queue_cap(existing_batch_count: int) -> int:
    if existing_batch_count <= 0:
        return max(1, _env_int("CHANNEL_SERVICE_ACQUIRE_QUEUE_CAP_INITIAL", 10))
    return max(1, _env_int("CHANNEL_SERVICE_ACQUIRE_QUEUE_CAP_LATER", 25))


def inline_probe_cap() -> int:
    return max(0, _env_int("CHANNEL_SERVICE_INLINE_PROBE_CAP", 12))


def candidate_discovery_target(*, max_videos: int, existing_video_count: int, catalog_supported: bool) -> int:
    baseline = max_videos + existing_video_count
    if catalog_supported:
        return min(200, max(1, baseline))
    return min(200, max(baseline, 40))


def probe_is_pending(probe: Optional[TranscriptProbe]) -> bool:
    return probe is not None and probe.status in _PENDING_PROBE_STATUSES


def maybe_inline_transcript_api_probe(
    *,
    session,
    row: dict,
    language: str,
    prefer_auto: bool,
) -> Optional[TranscriptProbe]:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        return None
    cues = _fetch_transcript_api_cues(video_id=video_id, language=language, prefer_auto=prefer_auto)
    if not cues:
        return None

    transcript_rows = _transcript_rows_from_cues(
        row=row,
        cues=cues,
        source="yt_transcript_api",
    )
    artifact_path = _write_probe_artifact(
        video_id=video_id,
        language=language,
        prefer_auto=prefer_auto,
        transcript_source="yt_transcript_api",
        transcript_rows=transcript_rows,
    )
    key = transcript_probe_key(video_id, language, prefer_auto)
    probe = session.get(TranscriptProbe, key)
    if probe is None:
        probe = TranscriptProbe(
            key=key,
            video_id=video_id,
            video_url=str(row.get("video_url") or _yt_watch_url(video_id)),
            channel_handle=str(row.get("channel_handle") or "").strip() or None,
            language=language,
            prefer_auto=prefer_auto,
        )
        session.add(probe)
    probe.status = "ready"
    probe.transcript_source = "yt_transcript_api"
    probe.artifact_path = artifact_path
    probe.error_detail = None
    probe.next_attempt_at = None
    probe.lease_owner = None
    probe.lease_expires_at = None
    session.flush()
    return probe


def pending_reason_for_probe(probe: Optional[TranscriptProbe]) -> str:
    detail = (probe.error_detail or "").lower() if probe is not None and probe.error_detail else ""
    if "rate_limited" in detail:
        return "rate_limited_retry_scheduled"
    if probe is not None and probe.status == "retry":
        return "retry_scheduled"
    return "queued_for_acquisition"


def _estimate_eta_minutes(rows: List[dict]) -> int:
    indexed = sum(1 for row in rows if row.get("indexed_parent_id"))
    probed = max(0, len(rows) - indexed)
    minutes = (indexed * 0.2) + (probed * 3.0)
    return max(1, int(math.ceil(minutes)))


def _eta_confidence(rows: List[dict]) -> str:
    indexed = sum(1 for row in rows if row.get("indexed_parent_id"))
    if indexed == len(rows):
        return "high"
    if indexed >= max(1, len(rows) // 2):
        return "medium"
    return "low"


def _batch_sizes(existing_batch_count: int) -> tuple[int, int]:
    return (10, 25) if existing_batch_count <= 0 else (25, 25)


def build_batch_plan(*, included_rows: List[dict], existing_batch_count: int, per_video: int) -> list:
    first_cap, later_cap = _batch_sizes(existing_batch_count)
    plan = []
    offset = 0
    batch_index = existing_batch_count + 1
    cap = first_cap
    while offset < len(included_rows):
        batch_rows = included_rows[offset : offset + cap]
        plan.append(
            {
                "batch_index": batch_index,
                "requested_video_count": len(batch_rows),
                "billable_video_count": len(batch_rows),
                "amount_cents": len(batch_rows) * per_video,
                "estimated_ready_minutes": _estimate_eta_minutes(batch_rows),
                "eta_confidence": _eta_confidence(batch_rows),
                "eligible_after_payment": batch_index == existing_batch_count + 1,
            }
        )
        offset += len(batch_rows)
        batch_index += 1
        cap = later_cap
    return plan


def quote_status_for_rows(*, included_rows: List[dict], pending_rows: List[dict]) -> str:
    if included_rows:
        return "open"
    if pending_rows:
        return "acquiring"
    return "unavailable"


@dataclass
class QuotePlan:
    channel_handle: str
    channel_name: str
    channel_id: Optional[str]
    namespace: str
    mode: str
    included_rows: List[dict]
    pending_rows: List[dict]
    excluded_rows: List[dict]
    batch_plan: List[dict]
    per_video_cents: int
    current_batch_index: int
    current_batch_amount_cents: int
    current_batch_video_count: int
    total_included_amount_cents: int
    estimated_ready_minutes: int
    eta_confidence: str
    recommended_starter_batch_size: int
    existing_pack_id: Optional[str]
    existing_batch_count: int


def plan_quote(
    *,
    session,
    channel_handle: str,
    namespace: str,
    mode: str,
    max_videos: int,
    language: str,
    prefer_auto: bool,
    pack_id: Optional[str] = None,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
) -> QuotePlan:
    api_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()

    existing_pack = None
    existing_video_ids: set[str] = set()
    existing_batch_count = 0
    if pack_id:
        existing_pack = session.get(ChannelPack, pack_id)
        if existing_pack is None:
            raise ValueError(f"Pack {pack_id} was not found")
        if existing_pack.status not in {"draft", "queued", "partial", "ready"}:
            raise ValueError(f"Pack {pack_id} is not available for expansion")
        existing_batch_count = int(existing_pack.batch_count or 0)
        rows = session.execute(select(PackVideo.video_id).where(PackVideo.pack_id == pack_id)).all()
        existing_video_ids = {str(video_id) for (video_id,) in rows if video_id}

    catalog_supported = _catalog_supports_handle(namespace=namespace, channel_handle=channel_handle)
    discovery_target = candidate_discovery_target(
        max_videos=max_videos,
        existing_video_count=len(existing_video_ids),
        catalog_supported=catalog_supported,
    )
    channel = list_channel_candidates(
        api_key=api_key,
        namespace=namespace,
        channel_handle=channel_handle,
        target_count=discovery_target,
        published_after=published_after,
        published_before=published_before,
    )
    discovery_rows = [row for row in channel["videos"] if row["video_id"] not in existing_video_ids]
    indexed_rows = indexed_parent_rows([row["video_id"] for row in discovery_rows], namespace=namespace)
    probes = load_transcript_probes(
        session=session,
        video_ids=[row["video_id"] for row in discovery_rows],
        language=language,
        prefer_auto=prefer_auto,
    )
    queue_cap = acquisition_queue_cap(existing_batch_count)
    active_probe_count = sum(1 for probe in probes.values() if probe_is_pending(probe))
    try_inline_probe = not catalog_supported
    inline_probe_budget = inline_probe_cap()

    included_rows: List[dict] = []
    pending_rows: List[dict] = []
    excluded_rows: List[dict] = []
    deferred_rows: List[tuple[dict, Optional[TranscriptProbe]]] = []
    for position, row in enumerate(discovery_rows, start=1):
        if len(included_rows) >= max_videos:
            break
        payload = dict(row)
        payload["position"] = position
        parent = indexed_rows.get(row["video_id"])
        if parent:
            payload["transcript_source"] = "indexed"
            payload["indexed_parent_id"] = row["video_id"]
            payload["status"] = "included"
            payload["batch_index"] = 0
            included_rows.append(payload)
            continue

        probe = probes.get(transcript_probe_key(row["video_id"], language, prefer_auto))
        if probe_is_ready(probe):
            payload["transcript_source"] = probe.transcript_source or "probe_cache"
            payload["indexed_parent_id"] = None
            payload["status"] = "included"
            payload["batch_index"] = 0
            included_rows.append(payload)
            continue

        if probe is not None and probe.status == "unavailable":
            payload["transcript_source"] = None
            payload["indexed_parent_id"] = None
            payload["status"] = "excluded"
            payload["reason"] = "transcript_unavailable"
            payload["detail"] = probe.error_detail
            excluded_rows.append(payload)
            continue

        if try_inline_probe and inline_probe_budget > 0 and probe is None:
            inline_probe = maybe_inline_transcript_api_probe(
                session=session,
                row=payload,
                language=language,
                prefer_auto=prefer_auto,
            )
            inline_probe_budget -= 1
            if probe_is_ready(inline_probe):
                probes[transcript_probe_key(row["video_id"], language, prefer_auto)] = inline_probe
                payload["transcript_source"] = inline_probe.transcript_source or "probe_cache"
                payload["indexed_parent_id"] = None
                payload["status"] = "included"
                payload["batch_index"] = 0
                included_rows.append(payload)
                continue
            probe = inline_probe

        deferred_rows.append((payload, probe))

    for payload, probe in deferred_rows:
        if len(included_rows) + len(pending_rows) >= max_videos:
            break
        if probe_is_pending(probe):
            payload["transcript_source"] = None
            payload["indexed_parent_id"] = None
            payload["status"] = "pending_acquisition"
            payload["reason"] = pending_reason_for_probe(probe)
            payload["detail"] = probe.error_detail
            pending_rows.append(payload)
            continue

        if active_probe_count < queue_cap:
            probe = ensure_transcript_probe(session=session, row=payload, language=language, prefer_auto=prefer_auto)
            probes[transcript_probe_key(payload["video_id"], language, prefer_auto)] = probe
            active_probe_count += 1
            reason = pending_reason_for_probe(probe)
            detail = probe.error_detail
        else:
            reason = "awaiting_queue_slot"
            detail = f"acquisition queue capped at {queue_cap} videos for this quote"
        payload["transcript_source"] = None
        payload["indexed_parent_id"] = None
        payload["status"] = "pending_acquisition"
        payload["reason"] = reason
        payload["detail"] = detail
        pending_rows.append(payload)

    per_video = per_video_cents(mode, len(included_rows))
    batch_plan = build_batch_plan(
        included_rows=included_rows,
        existing_batch_count=existing_batch_count,
        per_video=per_video,
    )
    batch_by_position: Dict[int, int] = {}
    offset = 0
    for batch in batch_plan:
        batch_index = int(batch["batch_index"])
        count = int(batch["billable_video_count"])
        for row in included_rows[offset : offset + count]:
            row["batch_index"] = batch_index
        offset += count

    current_batch_index = batch_plan[0]["batch_index"] if batch_plan else (existing_batch_count + 1)
    current_batch_amount_cents = batch_plan[0]["amount_cents"] if batch_plan else 0
    current_batch_video_count = batch_plan[0]["billable_video_count"] if batch_plan else 0
    estimated_ready_minutes = batch_plan[0]["estimated_ready_minutes"] if batch_plan else 0
    eta_confidence = batch_plan[0]["eta_confidence"] if batch_plan else "low"
    recommended_size = min(current_batch_video_count, 10 if existing_batch_count == 0 else 25)

    return QuotePlan(
        channel_handle=channel["channel_handle"],
        channel_name=channel["channel_name"],
        channel_id=channel["channel_id"],
        namespace=namespace,
        mode=mode,
        included_rows=included_rows,
        pending_rows=pending_rows,
        excluded_rows=excluded_rows,
        batch_plan=batch_plan,
        per_video_cents=per_video,
        current_batch_index=current_batch_index,
        current_batch_amount_cents=current_batch_amount_cents,
        current_batch_video_count=current_batch_video_count,
        total_included_amount_cents=len(included_rows) * per_video,
        estimated_ready_minutes=estimated_ready_minutes,
        eta_confidence=eta_confidence,
        recommended_starter_batch_size=recommended_size,
        existing_pack_id=existing_pack.id if existing_pack else None,
        existing_batch_count=existing_batch_count,
    )


def persist_quote(
    *,
    session,
    request_payload: dict,
    plan: QuotePlan,
    planning_latency_ms: int = 0,
) -> ChannelQuote:
    quote = ChannelQuote(
        id=new_id("quote"),
        status=quote_status_for_rows(included_rows=plan.included_rows, pending_rows=plan.pending_rows),
        mode=plan.mode,
        namespace=plan.namespace,
        channel_handle=plan.channel_handle,
        resolved_channel_id=plan.channel_id,
        resolved_channel_name=plan.channel_name,
        requested_max_videos=int(request_payload["max_videos"]),
        included_video_count=len(plan.included_rows),
        excluded_video_count=len(plan.excluded_rows),
        current_batch_index=plan.current_batch_index,
        current_batch_video_count=plan.current_batch_video_count,
        current_batch_amount_cents=plan.current_batch_amount_cents,
        total_included_amount_cents=plan.total_included_amount_cents,
        per_video_cents=plan.per_video_cents,
        estimated_ready_minutes=plan.estimated_ready_minutes,
        eta_confidence=plan.eta_confidence,
        recommended_starter_batch_size=plan.recommended_starter_batch_size,
        planning_latency_ms=max(0, int(planning_latency_ms or 0)),
        request_json=request_payload,
        batch_plan_json=plan.batch_plan,
        price_breakdown_json={
            "per_video_cents": plan.per_video_cents,
            "currency": "USD",
            "current_batch_amount_cents": plan.current_batch_amount_cents,
            "total_included_amount_cents": plan.total_included_amount_cents,
        },
        expires_at=utcnow() + timedelta(minutes=30),
    )
    session.add(quote)
    session.flush()

    all_rows = sorted(plan.included_rows + plan.pending_rows + plan.excluded_rows, key=lambda row: row["position"])
    for row in all_rows:
        session.add(
            QuoteVideo(
                quote_id=quote.id,
                position=int(row["position"]),
                batch_index=int(row.get("batch_index") or 0),
                included=1 if row["status"] == "included" else 0,
                video_id=row["video_id"],
                title=row.get("title"),
                description=row.get("description"),
                channel_name=row.get("channel_name"),
                channel_handle=row.get("channel_handle"),
                published_at=row.get("published_at"),
                duration_s=row.get("duration_s"),
                video_url=row.get("video_url"),
                thumbnail_url=row.get("thumbnail_url"),
                transcript_source=row.get("transcript_source"),
                indexed_parent_id=row.get("indexed_parent_id"),
                status=row["status"],
                reason=row.get("reason"),
                detail=row.get("detail"),
            )
        )
    session.flush()
    return quote


def serialize_quote(quote: ChannelQuote) -> dict:
    included_rows = []
    pending_rows = []
    excluded_rows = []
    queued_pending_count = 0
    deferred_pending_count = 0
    for row in quote.videos:
        payload = {
            "video_id": row.video_id,
            "title": row.title,
            "description": row.description,
            "channel_name": row.channel_name,
            "channel_handle": row.channel_handle,
            "published_at": row.published_at,
            "duration_s": row.duration_s,
            "video_url": row.video_url,
            "thumbnail_url": row.thumbnail_url,
            "transcript_source": row.transcript_source,
            "indexed_parent_id": row.indexed_parent_id,
            "status": row.status,
            "batch_index": row.batch_index,
            "reason": row.reason,
            "detail": row.detail,
        }
        if row.status == "included":
            included_rows.append(payload)
        elif row.status == "pending_acquisition":
            pending_rows.append(payload)
            if row.reason == "awaiting_queue_slot":
                deferred_pending_count += 1
            else:
                queued_pending_count += 1
        else:
            excluded_rows.append(payload)
    existing_batch_count = max(0, int(quote.current_batch_index or 1) - 1)
    return {
        "ok": True,
        "quote_id": quote.id,
        "status": quote.status,
        "channel": {
            "handle": quote.channel_handle,
            "channel_id": quote.resolved_channel_id,
            "channel_name": quote.resolved_channel_name,
        },
        "mode": quote.mode,
        "namespace": quote.namespace,
        "expires_at": quote.expires_at.isoformat(),
        "pricing": {
            "currency": "USD",
            "per_video_cents": quote.per_video_cents,
            "current_batch_amount_cents": quote.current_batch_amount_cents,
            "total_included_amount_cents": quote.total_included_amount_cents,
        },
        "current_batch": {
            "batch_index": quote.current_batch_index,
            "billable_video_count": quote.current_batch_video_count,
            "amount_cents": quote.current_batch_amount_cents,
            "estimated_ready_minutes": quote.estimated_ready_minutes,
            "eta_confidence": quote.eta_confidence,
        },
        "recommended_starter_batch_size": quote.recommended_starter_batch_size,
        "included_video_count": quote.included_video_count,
        "pending_video_count": len(pending_rows),
        "excluded_video_count": quote.excluded_video_count,
        "batch_plan": quote.batch_plan_json,
        "acquisition": {
            "state": "acquiring" if pending_rows else "ready",
            "pending_video_count": len(pending_rows),
            "queued_video_count": queued_pending_count,
            "awaiting_queue_slot_count": deferred_pending_count,
            "queue_cap": acquisition_queue_cap(existing_batch_count),
            "poll_after_seconds": 5 if pending_rows else 0,
        },
        "included_videos": included_rows,
        "pending_videos": pending_rows,
        "excluded_videos": excluded_rows,
    }


def refresh_quote_state(*, session, quote: ChannelQuote, enqueue_missing: bool) -> ChannelQuote:
    language = str((quote.request_json or {}).get("language") or "en")
    prefer_auto = bool((quote.request_json or {}).get("prefer_auto", True))
    existing_batch_count = max(0, int(quote.current_batch_index or 1) - 1)
    queue_cap = acquisition_queue_cap(existing_batch_count)
    probes = load_transcript_probes(
        session=session,
        video_ids=[row.video_id for row in quote.videos],
        language=language,
        prefer_auto=prefer_auto,
    )
    active_probe_count = sum(1 for probe in probes.values() if probe_is_pending(probe))
    included_rows: List[dict] = []
    pending_rows: List[dict] = []
    excluded_rows: List[dict] = []
    old_status = quote.status
    for row in sorted(quote.videos, key=lambda value: int(value.position or 0)):
        payload = {
            "position": row.position,
            "video_id": row.video_id,
            "title": row.title,
            "description": row.description,
            "channel_name": row.channel_name,
            "channel_handle": row.channel_handle,
            "published_at": row.published_at,
            "duration_s": row.duration_s,
            "video_url": row.video_url,
            "thumbnail_url": row.thumbnail_url,
            "transcript_source": row.transcript_source,
            "indexed_parent_id": row.indexed_parent_id,
        }
        if row.indexed_parent_id:
            row.included = True
            row.status = "included"
            row.reason = None
            row.transcript_source = row.transcript_source or "indexed"
            payload["transcript_source"] = row.transcript_source
            included_rows.append(payload)
            continue

        probe = probes.get(transcript_probe_key(row.video_id, language, prefer_auto))
        if probe_is_ready(probe):
            row.included = True
            row.status = "included"
            row.reason = None
            row.transcript_source = probe.transcript_source or row.transcript_source or "probe_cache"
            payload["transcript_source"] = row.transcript_source
            included_rows.append(payload)
            continue

        if probe is not None and probe.status == "unavailable":
            row.included = False
            row.status = "excluded"
            row.reason = "transcript_unavailable"
            row.detail = probe.error_detail
            excluded_rows.append(payload)
            continue

        if probe_is_pending(probe):
            row.included = False
            row.status = "pending_acquisition"
            row.reason = pending_reason_for_probe(probe)
            row.detail = probe.error_detail
            payload["detail"] = row.detail
            pending_rows.append(payload)
            continue

        if enqueue_missing and active_probe_count < queue_cap:
            probe = ensure_transcript_probe(session=session, row=row, language=language, prefer_auto=prefer_auto)
            probes[transcript_probe_key(row.video_id, language, prefer_auto)] = probe
            active_probe_count += 1
            row.reason = pending_reason_for_probe(probe)
            row.detail = probe.error_detail
        else:
            row.reason = "awaiting_queue_slot"
            row.detail = f"acquisition queue capped at {queue_cap} videos for this quote"
        row.included = False
        row.status = "pending_acquisition"
        payload["detail"] = row.detail
        pending_rows.append(payload)

    per_video = per_video_cents(quote.mode, len(included_rows))
    batch_plan = build_batch_plan(
        included_rows=included_rows,
        existing_batch_count=existing_batch_count,
        per_video=per_video,
    )
    offset = 0
    for row in included_rows:
        row["batch_index"] = 0
    for batch in batch_plan:
        count = int(batch["billable_video_count"])
        for payload in included_rows[offset : offset + count]:
            payload["batch_index"] = int(batch["batch_index"])
        offset += count
    included_by_position = {int(payload["position"]): payload for payload in included_rows}
    for row in quote.videos:
        row.batch_index = int(included_by_position.get(int(row.position), {}).get("batch_index") or 0)

    quote.status = quote_status_for_rows(included_rows=included_rows, pending_rows=pending_rows)
    quote.included_video_count = len(included_rows)
    quote.excluded_video_count = len(excluded_rows)
    quote.per_video_cents = per_video
    quote.batch_plan_json = batch_plan
    quote.total_included_amount_cents = len(included_rows) * per_video
    quote.current_batch_index = batch_plan[0]["batch_index"] if batch_plan else (existing_batch_count + 1)
    quote.current_batch_amount_cents = batch_plan[0]["amount_cents"] if batch_plan else 0
    quote.current_batch_video_count = batch_plan[0]["billable_video_count"] if batch_plan else 0
    quote.estimated_ready_minutes = batch_plan[0]["estimated_ready_minutes"] if batch_plan else 0
    quote.eta_confidence = batch_plan[0]["eta_confidence"] if batch_plan else "low"
    quote.recommended_starter_batch_size = (
        min(quote.current_batch_video_count, 10 if existing_batch_count == 0 else 25)
        if quote.current_batch_video_count
        else 0
    )
    quote.price_breakdown_json = {
        "per_video_cents": quote.per_video_cents,
        "currency": "USD",
        "current_batch_amount_cents": quote.current_batch_amount_cents,
        "total_included_amount_cents": quote.total_included_amount_cents,
    }
    if old_status != "open" and quote.status == "open":
        quote.expires_at = utcnow() + timedelta(minutes=30)
    session.flush()
    return quote


def create_checkout_session(*, session, quote_ids: List[str], idempotency_key: str) -> CheckoutSessionRecord:
    return create_checkout_session_with_payment(
        session=session,
        quote_ids=quote_ids,
        idempotency_key=idempotency_key,
    )


def create_checkout_session_with_payment(
    *,
    session,
    quote_ids: List[str],
    idempotency_key: str,
    payment_provider: str = "x402",
    payment_status: Optional[str] = None,
    line_item_amount_overrides: Optional[Dict[str, int]] = None,
) -> CheckoutSessionRecord:
    existing = session.execute(
        select(CheckoutSessionRecord).where(CheckoutSessionRecord.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    quotes = session.execute(
        select(ChannelQuote).where(ChannelQuote.id.in_(quote_ids))
    ).scalars().all()
    found_ids = {quote.id for quote in quotes}
    missing = [quote_id for quote_id in quote_ids if quote_id not in found_ids]
    if missing:
        raise ValueError(f"Unknown quote ids: {', '.join(missing)}")

    total = 0
    line_items = []
    line_item_amount_overrides = dict(line_item_amount_overrides or {})
    for quote in quotes:
        refresh_quote_state(session=session, quote=quote, enqueue_missing=False)
        if quote.status != "open" or int(quote.current_batch_video_count or 0) <= 0:
            raise ValueError(f"Quote {quote.id} does not have a billable starter batch yet")
        quoted_amount_cents = int(quote.current_batch_amount_cents or 0)
        charged_amount_cents = int(line_item_amount_overrides.get(quote.id, quoted_amount_cents))
        line_items.append(
            {
                "quote_id": quote.id,
                "channel_handle": quote.channel_handle,
                "mode": quote.mode,
                "batch_index": quote.current_batch_index,
                "billable_video_count": quote.current_batch_video_count,
                "amount_cents": charged_amount_cents,
                "quoted_amount_cents": quoted_amount_cents,
            }
        )
        total += charged_amount_cents

    record = CheckoutSessionRecord(
        id=new_id("checkout"),
        status="open",
        idempotency_key=idempotency_key,
        currency="USD",
        total_amount_cents=total,
        quote_ids_json=list(quote_ids),
        line_items_json=line_items,
        payment_provider=payment_provider,
        payment_status=payment_status or ("requires_payment" if payment_required() else "development_bypass"),
    )
    session.add(record)
    session.flush()
    return record


def serialize_checkout_session(record: CheckoutSessionRecord) -> dict:
    return {
        "ok": True,
        "checkout_session_id": record.id,
        "status": record.status,
        "currency": record.currency,
        "total_amount_cents": record.total_amount_cents,
        "line_items": record.line_items_json,
        "payment": {
            "provider": record.payment_provider,
            "status": record.payment_status,
            "requires_live_settlement": record.payment_status == "requires_payment",
        },
    }


def _export_root() -> Path:
    raw = os.getenv("CHANNEL_SERVICE_EXPORT_ROOT") or ".local-data/channel-packs"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _write_ndjson(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_bundle_archive(*, root: Path, pack_id: str) -> Path:
    archive_path = root / f"{pack_id}.bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("manifest.json", "videos.ndjson", "links.ndjson", "transcripts.ndjson"):
            file_path = root / name
            if file_path.exists():
                archive.write(file_path, arcname=name)
    return archive_path


def _transcript_rows_from_cues(*, row: PackVideo | QuoteVideo | dict, cues: List[Any], source: str) -> List[dict]:
    video_id = row.video_id if hasattr(row, "video_id") else row.get("video_id")
    transcripts = []
    for idx, cue in enumerate(cues, start=1):
        transcripts.append(
            {
                "video_id": video_id,
                "segment_id": f"{video_id}:{idx}",
                "start_s": float(cue.start_s),
                "end_s": float(cue.end_s),
                "speaker": None,
                "text": cue.text,
                "source": source,
            }
        )
    return transcripts


def _build_pack_artifacts(
    *,
    session,
    pack: ChannelPack,
    batch: PackBatch,
    language: str,
    prefer_auto: bool,
    transcript_rows_by_video: Optional[Dict[str, List[dict]]] = None,
) -> dict:
    pack_rows = session.execute(
        select(PackVideo).where(PackVideo.pack_id == pack.id).order_by(PackVideo.position)
    ).scalars().all()
    ready_rows = [row for row in pack_rows if row.status == "ready"]
    indexed_rows = [row for row in ready_rows if row.indexed_parent_id]
    parent_ids = [str(row.indexed_parent_id) for row in indexed_rows if row.indexed_parent_id]
    segments = child_segments_by_parent(parent_ids, namespace=pack.namespace) if indexed_rows else {}
    transcript_rows_by_video = dict(transcript_rows_by_video or {})

    videos_payload = [
        {
            "video_id": row.video_id,
            "title": row.title,
            "description": row.description,
            "channel_name": row.channel_name,
            "channel_handle": row.channel_handle,
            "published_at": row.published_at,
            "duration_s": row.duration_s,
            "video_url": row.video_url,
            "thumbnail_url": row.thumbnail_url,
            "transcript_source": row.transcript_source,
            "indexed_parent_id": row.indexed_parent_id,
        }
        for row in ready_rows
    ]
    links_payload = [
        {
            "video_id": row.video_id,
            "video_url": row.video_url,
            "thumbnail_url": row.thumbnail_url,
            "channel_url": f"https://www.youtube.com/{(row.channel_handle or '').lstrip('@')}" if row.channel_handle else None,
        }
        for row in ready_rows
    ]
    transcripts_payload = []
    for row in indexed_rows:
        for segment in segments.get(str(row.indexed_parent_id), []):
            transcripts_payload.append(
                {
                    "video_id": row.video_id,
                    "segment_id": segment.get("segment_id"),
                    "start_s": segment.get("start_s"),
                    "end_s": segment.get("end_s"),
                    "speaker": segment.get("speaker"),
                    "text": segment.get("text"),
                    "source": row.transcript_source,
                }
            )
    for row in ready_rows:
        if row.indexed_parent_id:
            continue
        rows = transcript_rows_by_video.get(row.video_id)
        if rows is None and row.transcript_source in {"youtube_transcript_api", "yt_captions"}:
            probe = session.get(TranscriptProbe, transcript_probe_key(row.video_id, language, prefer_auto))
            rows = load_probe_artifact_rows(probe) if probe is not None else []
        if rows:
            transcripts_payload.extend(rows)

    root = _export_root() / pack.id
    manifest = {
        "pack_id": pack.id,
        "status": pack.status,
        "channel_handle": pack.channel_handle,
        "channel_name": pack.resolved_channel_name,
        "channel_id": pack.resolved_channel_id,
        "namespace": pack.namespace,
        "mode": pack.mode,
        "batch_count": pack.batch_count,
        "ready_video_count": len(ready_rows),
        "total_purchased_video_count": pack.total_purchased_video_count,
        "latest_batch_index": batch.batch_index,
        "export_names": ["manifest.json", "videos.ndjson", "links.ndjson", "transcripts.ndjson", f"{pack.id}.bundle.zip"],
    }
    batch_manifest = {
        "pack_id": pack.id,
        "batch_id": batch.id,
        "batch_index": batch.batch_index,
        "status": batch.status,
        "billable_video_count": batch.billable_video_count,
        "ready_video_count": batch.ready_video_count,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / f"manifest.batch_{batch.batch_index}.json", batch_manifest)
    _write_ndjson(root / "videos.ndjson", videos_payload)
    _write_ndjson(root / "links.ndjson", links_payload)
    _write_ndjson(root / "transcripts.ndjson", transcripts_payload)
    archive_path = _write_bundle_archive(root=root, pack_id=pack.id)
    return {
        "manifest_path": str(root / "manifest.json"),
        "videos_path": str(root / "videos.ndjson"),
        "links_path": str(root / "links.ndjson"),
        "transcripts_path": str(root / "transcripts.ndjson"),
        "archive_path": str(archive_path),
    }


def create_or_attach_pack(*, session, quote: ChannelQuote, pack_id: Optional[str]) -> ChannelPack:
    if pack_id:
        pack = session.get(ChannelPack, pack_id)
        if pack is None:
            raise ValueError(f"Pack {pack_id} was not found")
        return pack

    pack = ChannelPack(
        id=new_id("pack"),
        status="draft",
        mode=quote.mode,
        namespace=quote.namespace,
        channel_handle=quote.channel_handle,
        resolved_channel_id=quote.resolved_channel_id,
        resolved_channel_name=quote.resolved_channel_name,
    )
    session.add(pack)
    session.flush()
    return pack


def create_order_from_quote(
    *,
    session,
    quote: ChannelQuote,
    checkout: CheckoutSessionRecord,
    pack_id: Optional[str],
    buyer_subject_type: Optional[str],
    buyer_subject_id: Optional[str],
    external_payment: Optional[dict] = None,
) -> tuple:
    refresh_quote_state(session=session, quote=quote, enqueue_missing=False)
    pack = create_or_attach_pack(session=session, quote=quote, pack_id=pack_id)
    existing_batch = session.execute(
        select(PackBatch).where(PackBatch.pack_id == pack.id, PackBatch.quote_id == quote.id)
    ).scalar_one_or_none()
    if existing_batch is not None:
        raise ValueError(f"Quote {quote.id} has already been ordered for pack {pack.id}")
    current_batch_rows = [
        row
        for row in quote.videos
        if row.status == "included" and int(row.batch_index or 0) == int(quote.current_batch_index)
    ]
    if not current_batch_rows:
        raise ValueError("Quote does not contain a payable batch")

    charge_amount_cents = int((external_payment or {}).get("amount_cents") or quote.current_batch_amount_cents or 0)
    externally_settled = external_payment is not None
    batch = PackBatch(
        id=new_id("batch"),
        pack_id=pack.id,
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        batch_index=int(quote.current_batch_index),
        status="awaiting_payment" if payment_required() and not externally_settled else "queued",
        billable_video_count=len(current_batch_rows),
        ready_video_count=0,
        amount_cents=charge_amount_cents,
        estimated_ready_minutes=int(quote.estimated_ready_minutes or 0),
        build_notes_json={},
    )
    session.add(batch)
    session.flush()

    order = ChannelOrder(
        id=new_id("order"),
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        pack_id=pack.id,
        batch_id=batch.id,
        status="awaiting_payment" if payment_required() and not externally_settled else "queued",
        payment_status=(
            str((external_payment or {}).get("payment_status") or "settled_external")
            if externally_settled
            else ("requires_payment" if payment_required() else "settled_development")
        ),
        payment_provider=(
            str((external_payment or {}).get("provider") or "external")
            if externally_settled
            else ("x402" if payment_required() else "development")
        ),
        amount_cents=charge_amount_cents,
        currency="USD",
        notes_json={},
    )
    session.add(order)
    session.flush()

    if payment_required() and not externally_settled:
        return pack, batch, order

    from .channel_service_store import Entitlement, PaymentReceipt

    receipt = (
        dict((external_payment or {}).get("receipt_json") or {})
        if externally_settled
        else {
            "provider": "development",
            "note": "development bypass used because live x402 settlement is not implemented in this slice",
        }
    )
    receipt.setdefault("provider", order.payment_provider)

    session.add(
        PaymentReceipt(
            id=new_id("receipt"),
            checkout_session_id=checkout.id,
            order_id=order.id,
            status=str((external_payment or {}).get("receipt_status") or "settled"),
            provider=order.payment_provider,
            amount_cents=int(order.amount_cents or 0),
            currency=order.currency,
            receipt_json=receipt,
        )
    )

    ready_count = 0
    pending_count = 0
    language = str((quote.request_json or {}).get("language") or "en")
    prefer_auto = bool((quote.request_json or {}).get("prefer_auto", True))
    transcript_rows_by_video: Dict[str, List[dict]] = {}
    for row in current_batch_rows:
        indexed_parent_id = row.indexed_parent_id
        status = "ready" if indexed_parent_id else "queued"

        if not indexed_parent_id:
            probe = session.get(TranscriptProbe, transcript_probe_key(row.video_id, language, prefer_auto))
            rows = load_probe_artifact_rows(probe) if probe is not None else []
            if rows:
                row.transcript_source = (probe.transcript_source if probe is not None else None) or row.transcript_source
                transcript_rows_by_video[row.video_id] = rows
                status = "ready"
            else:
                batch.build_notes_json = {
                    **(batch.build_notes_json or {}),
                    row.video_id: "transcript_acquisition_not_ready",
                }

        if status != "ready" and not indexed_parent_id and inline_index_enabled() and row.video_url:
            try:
                index_youtube_video_captions(
                    video_url=row.video_url,
                    namespace=quote.namespace,
                    language=language,
                    prefer_auto=prefer_auto,
                )
                indexed_parent_id = row.video_id
                status = "ready"
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                status = "queued"
                batch.build_notes_json = {
                    **(batch.build_notes_json or {}),
                    row.video_id: f"inline_index_failed: {exc}",
                }

        if status == "ready":
            ready_count += 1
        else:
            pending_count += 1

        session.add(
            PackVideo(
                pack_id=pack.id,
                batch_id=batch.id,
                quote_id=quote.id,
                position=row.position,
                video_id=row.video_id,
                title=row.title,
                description=row.description,
                channel_name=row.channel_name,
                channel_handle=row.channel_handle,
                published_at=row.published_at,
                duration_s=row.duration_s,
                video_url=row.video_url,
                thumbnail_url=row.thumbnail_url,
                transcript_source=row.transcript_source,
                indexed_parent_id=indexed_parent_id,
                status=status,
            )
        )

    batch.ready_video_count = ready_count
    batch.status = "ready" if pending_count == 0 else ("partial" if ready_count else "queued")
    order.status = batch.status
    pack.batch_count = max(int(pack.batch_count or 0), int(batch.batch_index))
    pack.total_purchased_video_count = int(pack.total_purchased_video_count or 0) + len(current_batch_rows)
    pack.ready_video_count = int(pack.ready_video_count or 0) + ready_count
    pack.status = batch.status
    if buyer_subject_type and buyer_subject_id:
        session.add(
            Entitlement(
                id=new_id("entitlement"),
                pack_id=pack.id,
                subject_type=buyer_subject_type,
                subject_id=buyer_subject_id,
                status="active",
            )
        )

    session.flush()

    export_paths = (
        _build_pack_artifacts(
            session=session,
            pack=pack,
            batch=batch,
            language=language,
            prefer_auto=prefer_auto,
            transcript_rows_by_video=transcript_rows_by_video,
        )
        if ready_count
        else {}
    )
    if export_paths:
        pack.export_paths_json = export_paths
        pack.manifest_json = json.loads(Path(export_paths["manifest_path"]).read_text(encoding="utf-8"))
        batch.manifest_json = {
            "pack_id": pack.id,
            "batch_index": batch.batch_index,
            "ready_video_count": ready_count,
            "billable_video_count": len(current_batch_rows),
        }
    session.flush()
    return pack, batch, order


def serialize_order(order, batch, pack) -> dict:
    return {
        "ok": True,
        "order_id": order.id,
        "pack_id": pack.id,
        "batch_id": batch.id,
        "status": order.status,
        "payment_status": order.payment_status,
        "payment_provider": order.payment_provider,
        "currency": order.currency,
        "amount_cents": order.amount_cents,
        "channel": {
            "handle": pack.channel_handle,
            "channel_id": pack.resolved_channel_id,
            "channel_name": pack.resolved_channel_name,
        },
        "batch": {
            "batch_index": batch.batch_index,
            "status": batch.status,
            "billable_video_count": batch.billable_video_count,
            "ready_video_count": batch.ready_video_count,
            "estimated_ready_minutes": batch.estimated_ready_minutes,
        },
        "exports": pack.export_paths_json or {},
        "payment": {
            "live_settlement_required": order.payment_status == "requires_payment",
            "development_bypass": order.payment_provider == "development",
            "externally_settled": order.payment_provider not in {"x402", "development"},
        },
    }


def serialize_batch(batch: PackBatch) -> dict:
    return {
        "batch_id": batch.id,
        "batch_index": batch.batch_index,
        "status": batch.status,
        "billable_video_count": batch.billable_video_count,
        "ready_video_count": batch.ready_video_count,
        "amount_cents": batch.amount_cents,
        "estimated_ready_minutes": batch.estimated_ready_minutes,
        "build_notes": batch.build_notes_json or {},
    }
