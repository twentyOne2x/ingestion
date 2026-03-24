from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels

from .channel_service_acp import create_or_sync_acp_job, list_acp_offerings, refresh_acp_job, serialize_acp_job
from .channel_service_logic import (
    create_checkout_session,
    create_order_from_quote,
    persist_quote,
    plan_quote,
    refresh_quote_state,
    serialize_batch,
    serialize_checkout_session,
    serialize_order,
    serialize_quote,
    supported_channels,
)
from .channel_service_readiness import (
    compute_readiness,
    create_readiness_override,
    enforce_acp_job_allowed,
    enforce_checkout_allowed,
    get_existing_acp_bridge,
    get_existing_checkout_by_idempotency_key,
    serialize_readiness_history,
)
from .channel_service_scheduler import ensure_egress_pools, serialize_egress_pools, serialize_scheduler_summary
from .channel_service_store import (
    AcpJobBridge,
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    PackBatch,
    init_db,
    session_scope,
)
from .ingest import create_ingest_service
from .gcs import read_json_from_gcs
from .pubsub import verify_pubsub_push
from .schemas import DiarizationReadyEvent, decode_pubsub_message
from .youtube import YouTubeClient
from src.ingest_v2.pipelines.index_youtube_captions import (
    _is_youtube_bot_check,
    _require_ytdlp,
    _ytdlp_extra_opts,
    index_youtube_channel_captions,
    index_youtube_query_captions,
    index_youtube_video_captions,
)

LOG = logging.getLogger(__name__)
app = FastAPI(title="Diarization Indexer")


@app.on_event("startup")
def _startup_channel_service() -> None:
    init_db()
    with session_scope() as session:
        ensure_egress_pools(session)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service": "diarization-indexer"}


@app.get("/diag/youtube_cookie_health")
def youtube_cookie_health(test_url: str = "https://www.youtube.com/watch?v=jNQXAC9IVRw") -> dict:
    """
    Probe yt-dlp access using the mounted cookie jar inside this container.

    This is intended for sandboxed automations that can hit localhost HTTP but
    cannot talk to the Docker daemon directly (so they can't `docker exec`).
    """
    # Determine cookie file path using the same envs as the ingestion pipelines.
    cookiefile = (os.environ.get("YTDLP_COOKIES_FILE") or os.environ.get("YTDLP_COOKIES_PATH") or "").strip()
    if not cookiefile:
        cookiefile = "/cookies/youtube.txt"
    cookie_path = Path(cookiefile)

    if not cookie_path.exists() or cookie_path.stat().st_size <= 0:
        return {
            "ok": False,
            "cookie_path": str(cookie_path),
            "error": "missing_or_empty_cookie_jar",
            "remediation": "refresh cookies: ./scripts/setup_youtube_ytdlp.sh --refresh-cookies",
        }

    st = cookie_path.stat()
    cookie_bytes = int(st.st_size)
    cookie_mtime_s = float(st.st_mtime)
    cookie_mtime = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(cookie_mtime_s))

    tmp_path: Optional[Path] = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="youtube.cookies.", suffix=".txt")
        os.close(fd)
        tmp_path = Path(tmp)
        shutil.copyfile(cookie_path, tmp_path)

        YDL = _require_ytdlp()
        ydl_opts: Dict[str, object] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            # Keep the probe bounded.
            "socket_timeout": 30,
            "retries": 1,
            "fragment_retries": 0,
        }
        ydl_opts.update(_ytdlp_extra_opts())
        # Ensure we never mutate the canonical cookie jar.
        ydl_opts["cookiefile"] = str(tmp_path)

        with YDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)

        video_id = (info or {}).get("id") if isinstance(info, dict) else None
        return {
            "ok": True,
            "cookie_path": str(cookie_path),
            "cookie_bytes": cookie_bytes,
            "cookie_mtime": cookie_mtime,
            "test_url": test_url,
            "video_id": video_id,
        }
    except Exception as exc:
        msg = str(exc) or exc.__class__.__name__
        return {
            "ok": False,
            "cookie_path": str(cookie_path),
            "cookie_bytes": cookie_bytes,
            "cookie_mtime": cookie_mtime,
            "test_url": test_url,
            "bot_check": bool(_is_youtube_bot_check(msg)),
            "error": msg[:800],
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


class IndexYoutubeReq(BaseModel):
    # Provide one or more of: `video_urls`, `channel`, `query`.
    video_urls: Optional[list[str]] = None
    channel: Optional[str] = None
    query: Optional[str] = None
    max_videos: int = Field(default=10, ge=1, le=200)
    namespace: str = Field(default="videos", min_length=1)
    language: str = Field(default="en", min_length=1)
    prefer_auto: bool = True

    # Optional metadata overrides for single-video indexing. These are especially useful
    # in the local "pubsub" pipeline where the watcher already fetched title/channel/date/duration.
    # Keeping them optional preserves backwards compatibility for existing callers.
    video_id: Optional[str] = None
    title: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    published_at: Optional[str] = None
    duration_s: Optional[float] = Field(default=None, ge=0)
    thumbnail_url: Optional[str] = None

    # Optional segmentation overrides (defaults to settings_v2 + envs).
    segment_min_s: Optional[float] = Field(default=None, ge=1)
    segment_max_s: Optional[float] = Field(default=None, ge=1)
    segment_stride_s: Optional[float] = Field(default=None, ge=1)
    min_text_chars: Optional[int] = Field(default=None, ge=40)


@app.post("/index/youtube")
def index_youtube(req: IndexYoutubeReq) -> dict:
    """
    Local-first indexing endpoint: fetch YouTube subtitles via yt-dlp and upsert into the vector store.
    Requires `OPENAI_API_KEY` for embeddings.
    """
    if not (req.video_urls or req.channel or req.query):
        raise HTTPException(status_code=400, detail="provide video_urls and/or channel and/or query")

    out: dict = {"ok": True, "indexed": [], "failed": []}
    if req.video_urls:
        for url in req.video_urls:
            try:
                metadata_override = {
                    "video_id": req.video_id,
                    "title": req.title,
                    "channel_name": req.channel_name,
                    "channel_id": req.channel_id,
                    "published_at": req.published_at,
                    "duration_s": req.duration_s,
                    "thumbnail_url": req.thumbnail_url,
                }
                res = index_youtube_video_captions(
                    video_url=url,
                    namespace=req.namespace,
                    language=req.language,
                    prefer_auto=req.prefer_auto,
                    segment_min_s=req.segment_min_s,
                    segment_max_s=req.segment_max_s,
                    segment_stride_s=req.segment_stride_s,
                    min_text_chars=req.min_text_chars,
                    metadata_override=metadata_override,
                )
                out["indexed"].append(res)
            except Exception as exc:
                out["failed"].append({"url": url, "error": str(exc)})

    if req.channel:
        try:
            res = index_youtube_channel_captions(
                channel=req.channel,
                max_videos=req.max_videos,
                namespace=req.namespace,
                language=req.language,
                prefer_auto=req.prefer_auto,
                segment_min_s=req.segment_min_s,
                segment_max_s=req.segment_max_s,
                segment_stride_s=req.segment_stride_s,
                min_text_chars=req.min_text_chars,
            )
            out["indexed"].append(res)
        except Exception as exc:
            out["failed"].append({"channel": req.channel, "error": str(exc)})

    if req.query:
        try:
            res = index_youtube_query_captions(
                query=req.query,
                max_videos=req.max_videos,
                namespace=req.namespace,
                language=req.language,
                prefer_auto=req.prefer_auto,
                segment_min_s=req.segment_min_s,
                segment_max_s=req.segment_max_s,
                segment_stride_s=req.segment_stride_s,
                min_text_chars=req.min_text_chars,
            )
            out["indexed"].append(res)
        except Exception as exc:
            out["failed"].append({"query": req.query, "error": str(exc)})

    return out


class IndexDiarizedReq(BaseModel):
    """
    Index a diarized transcript JSON payload (AssemblyAI-like) into the vector store.

    This is a local-first helper endpoint meant to be called by local workers that:
      1) download audio (yt-dlp),
      2) run diarization + entity extraction (AssemblyAI),
      3) write JSON files to a shared volume,
      4) call this endpoint to embed + upsert into Qdrant.

    `diarized_uri` and `entities_uri` accept:
      - file:// URIs
      - local paths mounted into the container
      - gs:// URIs (if GCS credentials exist)
    """

    video_id: str = Field(min_length=1)
    diarized_uri: str = Field(min_length=1)
    entities_uri: Optional[str] = None
    namespace: str = Field(default="videos", min_length=1)
    language: str = Field(default="en", min_length=1)
    source: str = Field(default="youtube", min_length=1)
    document_type: Optional[str] = None
    ingest_lane: Optional[str] = None
    transcript_provider: Optional[str] = None
    transcript_state: Optional[str] = None

    # Optional metadata overrides (avoid consuming YouTube Data API quota).
    title: Optional[str] = None
    description: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    published_at: Optional[str] = None
    duration_s: Optional[float] = Field(default=None, ge=0)
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None


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


def _youtube_indexed_channel_key(channel_name: str) -> str:
    return f"icmfyi:ops:youtube:indexed:channel:{str(channel_name or '').strip().casefold()}"


def _record_youtube_operator_truth(*, video_id: str, channel_name: str, transcript_provider: str, transcript_state: str, ingest_lane: str) -> None:
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if not redis_url or not video_id or not channel_name:
        return
    try:
        import redis

        r = redis.Redis.from_url(redis_url, decode_responses=True)
        pipe = r.pipeline()
        pipe.sadd(_youtube_indexed_channel_key(channel_name), video_id)
        if transcript_provider:
            pipe.sadd(f"icmfyi:ops:youtube:provider:{transcript_provider.strip().casefold()}", video_id)
        if transcript_state:
            pipe.sadd(f"icmfyi:ops:youtube:state:{transcript_state.strip().casefold()}", video_id)
        if ingest_lane:
            pipe.sadd(f"icmfyi:ops:youtube:lane:{ingest_lane.strip().casefold()}", video_id)
        pipe.execute()
    except Exception as exc:
        LOG.info("[index/diarized] redis operator truth write failed video=%s err=%s", video_id, exc)


@app.post("/index/diarized")
def index_diarized(req: IndexDiarizedReq) -> dict:
    """
    Index a diarized transcript (AssemblyAI JSON) into the vector store.

    Requires `OPENAI_API_KEY` for embeddings. If YouTube metadata fields aren't provided,
    `YOUTUBE_API_KEY` is used (best-effort) to enrich title/channel/publish date.
    """
    from src.ingest_v2.pipelines.build_children import build_children_from_raw
    from src.ingest_v2.pipelines.build_parents import build_parent
    from src.ingest_v2.pipelines.upsert_parents import upsert_parents
    from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
    from src.ingest_v2.pipelines.run_all_components.assemblyai import convert_assemblyai_json_to_raw

    diarized_payload = read_json_from_gcs(req.diarized_uri)
    raw_norm = convert_assemblyai_json_to_raw(diarized_payload)

    entities_payload = None
    if req.entities_uri:
        entities_payload = read_json_from_gcs(req.entities_uri)
    elif isinstance(diarized_payload, dict) and diarized_payload.get("entities"):
        entities_payload = diarized_payload.get("entities")

    entities_clean = _extract_entities(entities_payload)

    # Preserve raw entity objects for child-level entity enrichment.
    raw_entities = None
    if isinstance(entities_payload, list):
        raw_entities = entities_payload
    elif isinstance(entities_payload, dict) and isinstance(entities_payload.get("entities"), list):
        raw_entities = entities_payload.get("entities")

    source_norm = (req.source or "").strip().lower()
    is_youtube = source_norm in ("youtube", "yt")

    # Best-effort YouTube metadata lookup (avoids requiring callers to supply it).
    video_meta = None
    yt_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
    need_meta = (
        not (req.title or "").strip()
        or not (req.channel_name or "").strip()
        or not (req.channel_id or "").strip()
        or not (req.published_at or "").strip()
        or not (req.thumbnail_url or "").strip()
        or req.duration_s is None
    )
    if yt_key and is_youtube and need_meta:
        try:
            video_meta = YouTubeClient(api_key=yt_key).fetch_video_metadata(req.video_id)
        except Exception as exc:
            LOG.info("[index/diarized] youtube metadata fetch failed video=%s err=%s", req.video_id, exc)

    title = req.title or (video_meta.title if video_meta else None) or req.video_id
    description = req.description if req.description is not None else (video_meta.description if video_meta else "")
    channel_name = (
        req.channel_name
        or (video_meta.preferred_channel_name() if video_meta else None)
        or ("YouTube" if is_youtube else (req.source or "media"))
    )
    channel_id = req.channel_id or (video_meta.channel_id if video_meta else None) or ""
    published_at = req.published_at or (video_meta.published_at if video_meta else None)
    thumbnail_url = req.thumbnail_url or (video_meta.thumbnail_url if video_meta else None)

    duration_s = req.duration_s
    if duration_s is None:
        if video_meta and video_meta.duration_seconds:
            duration_s = float(video_meta.duration_seconds)
        else:
            # Derive duration from diarized segments.
            ends = [
                float(seg.get("end"))
                for seg in (raw_norm.get("segments") or [])
                if isinstance(seg, dict) and isinstance(seg.get("end"), (int, float))
            ]
            duration_s = max(ends) if ends else 0.0

    url = req.url
    if not url:
        # Avoid generating incorrect YouTube URLs for non-YouTube sources.
        url = f"https://www.youtube.com/watch?v={req.video_id}" if is_youtube else req.video_id

    document_type = (req.document_type or "").strip() or None
    if not document_type:
        if is_youtube:
            document_type = "youtube_video"
        elif source_norm in ("twitch", "twitch_vod"):
            document_type = "twitch_vod"
        elif source_norm in ("pumpfun", "pump.fun", "pump_fun"):
            document_type = "pumpfun_clip"
        else:
            document_type = "media"

    meta = {
        "video_id": req.video_id,
        "title": title,
        "description": description or "",
        "channel_name": channel_name,
        "channel_id": channel_id,
        "published_at": published_at,
        "duration_s": float(duration_s or 0.0),
        "url": url,
        "thumbnail_url": thumbnail_url,
        "language": req.language,
        "document_type": document_type,
        "source": req.source,
        "ingest_lane": (req.ingest_lane or "").strip() or None,
        "transcript_provider": (req.transcript_provider or "").strip() or None,
        "transcript_state": (req.transcript_state or "").strip() or None,
        "entities": entities_clean,
    }

    # Optional: enrich parent metadata (GPT router fields + speaker map) for catalog search.
    #
    # This is intentionally best-effort. Indexing should still succeed even if
    # enrichment fails or API keys are missing.
    router_enabled = (os.getenv("ROUTER_ENRICH", "1") or "1").strip().lower() not in ("0", "false", "no", "off")
    speakers_enabled = (os.getenv("SPEAKER_RESOLVE", "1") or "1").strip().lower() not in ("0", "false", "no", "off")

    raw_for_children = dict(raw_norm)
    raw_for_children.setdefault("caption_lines", [])
    raw_for_children.setdefault("diarization", [])
    if raw_entities is not None:
        raw_for_children["entities"] = raw_entities

    if speakers_enabled:
        try:
            from src.ingest_v2.speakers.resolve import resolve_speakers

            spk = resolve_speakers(meta, raw_for_children, audio_hint_path=None)
            if spk.get("speaker_primary"):
                meta["speaker_primary"] = spk["speaker_primary"]
            if spk.get("speaker_map"):
                meta["speaker_map"] = spk["speaker_map"]
                try:
                    meta["speaker_names"] = [
                        str(info.get("name")).strip()
                        for info in (spk.get("speaker_map") or {}).values()
                        if isinstance(info, dict) and str(info.get("name") or "").strip()
                    ]
                except Exception:
                    pass
        except Exception as exc:
            LOG.info("[index/diarized] speaker resolve failed video=%s err=%s", req.video_id, exc)

    if router_enabled:
        try:
            from src.ingest_v2.router.cache import load as router_cache_load
            from src.ingest_v2.router.cache import save as router_cache_save
            from src.ingest_v2.router.enrich_parent import enrich_parent_router_fields
            from src.ingest_v2.transcripts.normalize import normalize_to_sentences

            enrich = router_cache_load(req.video_id)
            if enrich is None:
                sentences = normalize_to_sentences(raw_for_children)
                enrich = enrich_parent_router_fields(meta, sentences)
                try:
                    router_cache_save(req.video_id, enrich)
                except Exception:
                    pass
            # Merge enriched fields into metadata for parent upsert.
            if isinstance(enrich, dict):
                meta["description"] = enrich.get("description", meta.get("description", "")) or (meta.get("description") or "")
                meta["topic_summary"] = enrich.get("topic_summary") or ""
                meta["router_tags"] = enrich.get("router_tags") or []
                meta["aliases"] = enrich.get("aliases") or []
                meta["canonical_entities"] = enrich.get("canonical_entities") or []
                meta["is_explainer"] = bool(enrich.get("is_explainer"))
                meta["router_boost"] = float(enrich.get("router_boost") or 1.0)
        except Exception as exc:
            LOG.info("[index/diarized] router enrich failed video=%s err=%s", req.video_id, exc)

    parent = build_parent(meta)
    parent_dict = parent.model_dump(mode="json")
    parent_payload = dict(parent_dict, parent_id=parent.parent_id)
    parent_payload.setdefault("video_id", parent.parent_id)
    upsert_parents([parent_payload])

    children = build_children_from_raw(parent_dict, raw_for_children)
    if not children:
        return {
            "ok": True,
            "video_id": req.video_id,
            "parent_id": parent.parent_id,
            "segments_ingested": 0,
            "details": "No children emitted (transcript empty/too short).",
        }

    stats = upsert_children(children)
    if document_type == "youtube_video":
        _record_youtube_operator_truth(
            video_id=req.video_id,
            channel_name=channel_name,
            transcript_provider=str(meta.get("transcript_provider") or ""),
            transcript_state=str(meta.get("transcript_state") or ""),
            ingest_lane=str(meta.get("ingest_lane") or ""),
        )
    return {
        "ok": True,
        "video_id": req.video_id,
        "parent_id": parent.parent_id,
        "segments_ingested": len(children),
        "upsert": stats,
    }


def _namespace_from_attributes(attributes: Optional[Dict[str, str]]) -> str:
    if attributes and "namespace" in attributes:
        return attributes["namespace"]
    return "videos"


@app.post("/pubsub/push")
async def handle_pubsub_push(request: Request) -> Response:
    audience = str(request.url)
    try:
        verify_pubsub_push(request.headers, audience=audience)
    except HTTPException:
        raise
    except RuntimeError as exc:
        LOG.error("Pub/Sub verification misconfigured: %s", exc)
        raise HTTPException(status_code=500, detail="Pub/Sub verification misconfigured") from exc
    except Exception as exc:  # pragma: no cover - defensive (should not happen)
        LOG.exception("Unexpected Pub/Sub verification error")
        raise HTTPException(status_code=500, detail="Pub/Sub verification failed") from exc

    body = await request.json()
    attributes = body.get("message", {}).get("attributes", {}) or {}
    namespace = _namespace_from_attributes(attributes)

    allowed_channels = load_namespace_channels(namespace)
    if not allowed_channels:
        LOG.info("No channels configured for namespace=%s; skipping.", namespace)
        return Response(status_code=204)

    try:
        event = decode_pubsub_message(body, model=DiarizationReadyEvent)
    except ValueError as exc:
        LOG.warning("Invalid Pub/Sub payload: %s", exc)
        return Response(status_code=204)

    try:
        service = create_ingest_service(namespace, allowed_channels)
    except Exception as exc:
        LOG.exception("Failed to initialise diarization ingest service")
        raise HTTPException(status_code=500, detail="Failed to initialise diarization ingest service") from exc

    try:
        service.handle_event(event)
    except Exception as exc:
        LOG.exception("Failed to ingest diarization-ready event")
        raise HTTPException(status_code=500, detail="Failed to ingest diarization-ready event") from exc
    return Response(status_code=204)


class ChannelPackQuoteReq(BaseModel):
    channel_handle: str = Field(min_length=1)
    max_videos: int = Field(default=10, ge=1, le=200)
    namespace: str = Field(default="videos", min_length=1)
    mode: Literal["recent_pack", "full_channel_backfill"] = "recent_pack"
    language: str = Field(default="en", min_length=1)
    prefer_auto: bool = True
    pack_id: Optional[str] = None
    published_after: Optional[str] = None
    published_before: Optional[str] = None


class CheckoutSessionReq(BaseModel):
    quote_ids: list[str] = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class ChannelPackOrderReq(BaseModel):
    quote_id: str = Field(min_length=1)
    checkout_session_id: str = Field(min_length=1)
    pack_id: Optional[str] = None
    buyer_subject_type: Optional[str] = None
    buyer_subject_id: Optional[str] = None


class AcpBuyerReq(BaseModel):
    subject_type: Optional[str] = None
    subject_id: Optional[str] = None


class AcpPaymentReq(BaseModel):
    provider: Optional[str] = None
    status: Optional[str] = None


class AcpJobReq(BaseModel):
    acp_job_id: str = Field(min_length=1)
    offering_id: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)
    buyer: Optional[AcpBuyerReq] = None
    payment: Optional[AcpPaymentReq] = None
    meta: dict = Field(default_factory=dict)


class ReadinessOverrideReq(BaseModel):
    publication_state: Optional[Literal["supported", "degraded", "paused", "internal_only"]] = None
    acceptance_scope: Optional[Literal["catalog_only", "catalog_and_arbitrary"]] = None
    reason: str = Field(min_length=1)
    created_by: Optional[str] = None
    expires_in_minutes: Optional[int] = Field(default=240, ge=1, le=7 * 24 * 60)
    clear_existing: bool = True


@app.get("/v1/channel-packs/catalog/channels")
def channel_pack_catalog(namespace: str = "videos") -> dict:
    payload = supported_channels(namespace)
    return {"ok": True, **payload}


@app.post("/v1/channel-packs/quotes")
def create_channel_pack_quote(req: ChannelPackQuoteReq) -> dict:
    with session_scope() as session:
        try:
            planning_started = time.perf_counter()
            plan = plan_quote(
                session=session,
                channel_handle=req.channel_handle,
                namespace=req.namespace,
                mode=req.mode,
                max_videos=req.max_videos,
                language=req.language,
                prefer_auto=req.prefer_auto,
                pack_id=req.pack_id,
                published_after=req.published_after,
                published_before=req.published_before,
            )
            planning_latency_ms = int((time.perf_counter() - planning_started) * 1000)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"quote planning failed: {exc}") from exc

        quote = persist_quote(
            session=session,
            request_payload=req.model_dump(mode="json"),
            plan=plan,
            planning_latency_ms=planning_latency_ms,
        )
        session.flush()
        session.refresh(quote)
        return serialize_quote(quote)


@app.get("/v1/channel-packs/quotes/{quote_id}")
def get_channel_pack_quote(quote_id: str) -> dict:
    with session_scope() as session:
        quote = session.get(ChannelQuote, quote_id)
        if quote is None:
            raise HTTPException(status_code=404, detail=f"quote {quote_id} not found")
        refresh_quote_state(session=session, quote=quote, enqueue_missing=True)
        return serialize_quote(quote)


@app.post("/v1/checkout-sessions")
def create_channel_pack_checkout(req: CheckoutSessionReq) -> dict:
    with session_scope() as session:
        existing = get_existing_checkout_by_idempotency_key(session, idempotency_key=req.idempotency_key)
        if existing is not None:
            return serialize_checkout_session(existing)
        quotes = session.execute(select(ChannelQuote).where(ChannelQuote.id.in_(req.quote_ids))).scalars().all()
        found = {quote.id for quote in quotes}
        missing = [quote_id for quote_id in req.quote_ids if quote_id not in found]
        if missing:
            raise HTTPException(status_code=400, detail=f"Unknown quote ids: {', '.join(missing)}")
        try:
            enforce_checkout_allowed(session=session, quotes=quotes)
            record = create_checkout_session(
                session=session,
                quote_ids=req.quote_ids,
                idempotency_key=req.idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return serialize_checkout_session(record)


@app.post("/v1/channel-packs/orders")
def create_channel_pack_order(req: ChannelPackOrderReq) -> dict:
    with session_scope() as session:
        quote = session.get(ChannelQuote, req.quote_id)
        if quote is None:
            raise HTTPException(status_code=404, detail=f"quote {req.quote_id} not found")
        checkout = session.get(CheckoutSessionRecord, req.checkout_session_id)
        if checkout is None:
            raise HTTPException(status_code=404, detail=f"checkout session {req.checkout_session_id} not found")
        if req.quote_id not in set(checkout.quote_ids_json or []):
            raise HTTPException(status_code=400, detail="quote is not part of the checkout session")
        if quote.status != "open":
            raise HTTPException(status_code=400, detail=f"quote {quote.id} is not open")
        if normalize_utc(quote.expires_at) < time_now_utc():
            raise HTTPException(status_code=400, detail=f"quote {quote.id} has expired")

        try:
            pack, batch, order = create_order_from_quote(
                session=session,
                quote=quote,
                checkout=checkout,
                pack_id=req.pack_id,
                buyer_subject_type=req.buyer_subject_type,
                buyer_subject_id=req.buyer_subject_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"order creation failed: {exc}") from exc

        return serialize_order(order, batch, pack)


@app.get("/v1/channel-packs/orders/{order_id}")
def get_channel_pack_order(order_id: str) -> dict:
    with session_scope() as session:
        order = session.get(ChannelOrder, order_id)
        if order is None:
            raise HTTPException(status_code=404, detail=f"order {order_id} not found")
        batch = session.get(PackBatch, order.batch_id)
        pack = session.get(ChannelPack, order.pack_id)
        if batch is None or pack is None:
            raise HTTPException(status_code=500, detail="order references missing pack or batch")
        return serialize_order(order, batch, pack)


@app.get("/v1/channel-packs/orders/{order_id}/batches")
def get_channel_pack_order_batches(order_id: str) -> dict:
    with session_scope() as session:
        order = session.get(ChannelOrder, order_id)
        if order is None:
            raise HTTPException(status_code=404, detail=f"order {order_id} not found")
        batches = (
            session.query(PackBatch)
            .filter(PackBatch.pack_id == order.pack_id)
            .order_by(PackBatch.batch_index.asc())
            .all()
        )
        return {
            "ok": True,
            "order_id": order.id,
            "pack_id": order.pack_id,
            "batches": [serialize_batch(batch) for batch in batches],
        }


@app.get("/v1/channel-packs/{pack_id}/manifest")
def get_channel_pack_manifest(pack_id: str) -> dict:
    with session_scope() as session:
        pack = session.get(ChannelPack, pack_id)
        if pack is None:
            raise HTTPException(status_code=404, detail=f"pack {pack_id} not found")
        if not pack.manifest_json:
            raise HTTPException(status_code=404, detail=f"pack {pack_id} does not have a manifest yet")
        return pack.manifest_json


@app.get("/v1/channel-packs/{pack_id}/exports/{name}")
def get_channel_pack_export(pack_id: str, name: str) -> FileResponse:
    allowed = {
        "manifest": "manifest_path",
        "videos": "videos_path",
        "links": "links_path",
        "transcripts": "transcripts_path",
        "archive": "archive_path",
    }
    key = allowed.get(name)
    if key is None:
        raise HTTPException(status_code=404, detail=f"unsupported export {name}")
    with session_scope() as session:
        pack = session.get(ChannelPack, pack_id)
        if pack is None:
            raise HTTPException(status_code=404, detail=f"pack {pack_id} not found")
        path_value = (pack.export_paths_json or {}).get(key)
        if not path_value:
            raise HTTPException(status_code=404, detail=f"export {name} is not ready for pack {pack_id}")
        export_path = Path(path_value)
        if not export_path.exists():
            raise HTTPException(status_code=404, detail=f"export file for {name} was not found")
        if name == "manifest":
            media_type = "application/json"
        elif name == "archive":
            media_type = "application/zip"
        else:
            media_type = "application/x-ndjson"
        filename = export_path.name
        return FileResponse(export_path, media_type=media_type, filename=filename)


@app.get("/v1/channel-packs/acp/offerings")
def get_channel_pack_acp_offerings(request: Request) -> dict:
    with session_scope() as session:
        readiness = compute_readiness(session, persist=True)
        visible_offering_ids = [item["offering_id"] for item in readiness["offerings"] if item["published"]]
        acp_identity_ready = bool(
            (os.getenv("SELLER_ENTITY_ID") or "").strip()
            and (os.getenv("SELLER_AGENT_WALLET_ADDRESS") or "").strip()
            and (os.getenv("WHITELISTED_WALLET_PRIVATE_KEY") or "").strip()
        )
        if not acp_identity_ready:
            visible_offering_ids = []
        return list_acp_offerings(
            base_url=channel_service_public_base_url(request),
            visible_offering_ids=visible_offering_ids,
            readiness=readiness,
        )


@app.post("/v1/channel-packs/acp/jobs")
def create_channel_pack_acp_job(req: AcpJobReq, request: Request) -> dict:
    require_acp_shared_secret(request)
    with session_scope() as session:
        try:
            bridge = get_existing_acp_bridge(session, acp_job_id=req.acp_job_id)
            if bridge is None:
                enforce_acp_job_allowed(
                    session=session,
                    offering_id=req.offering_id,
                    channel_handle=(req.input or {}).get("channel_handle"),
                    namespace=str((req.input or {}).get("namespace") or "videos"),
                )
            bridge = create_or_sync_acp_job(
                session=session,
                payload=req.model_dump(mode="json"),
                base_url=channel_service_public_base_url(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"ACP job sync failed: {exc}") from exc
        return serialize_acp_job(
            session=session,
            bridge=bridge,
            base_url=channel_service_public_base_url(request),
        )


@app.get("/v1/channel-packs/acp/jobs/{acp_job_id}")
def get_channel_pack_acp_job(acp_job_id: str, request: Request) -> dict:
    require_acp_shared_secret(request)
    with session_scope() as session:
        bridge = session.get(AcpJobBridge, acp_job_id)
        if bridge is None:
            raise HTTPException(status_code=404, detail=f"ACP job {acp_job_id} not found")
        try:
            refresh_acp_job(
                session=session,
                bridge=bridge,
                base_url=channel_service_public_base_url(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"ACP job refresh failed: {exc}") from exc
        session.refresh(bridge)
        return serialize_acp_job(
            session=session,
            bridge=bridge,
            base_url=channel_service_public_base_url(request),
        )


@app.get("/v1/channel-packs/ops/scheduler")
def get_channel_service_scheduler_summary() -> dict:
    with session_scope() as session:
        return serialize_scheduler_summary(session=session)


@app.get("/v1/channel-packs/ops/egress-pools")
def get_channel_service_egress_pools() -> dict:
    with session_scope() as session:
        return serialize_egress_pools(session=session)


@app.get("/v1/channel-packs/ops/readiness")
def get_channel_service_readiness() -> dict:
    with session_scope() as session:
        return compute_readiness(session, persist=True)


@app.get("/v1/channel-packs/ops/readiness/history")
def get_channel_service_readiness_history(limit: int = 20) -> dict:
    with session_scope() as session:
        return serialize_readiness_history(session, limit=limit)


@app.post("/v1/channel-packs/ops/readiness/overrides")
def post_channel_service_readiness_override(req: ReadinessOverrideReq, request: Request) -> dict:
    require_ops_shared_secret(request)
    with session_scope() as session:
        try:
            override = create_readiness_override(
                session=session,
                publication_state=req.publication_state,
                acceptance_scope=req.acceptance_scope,
                reason=req.reason,
                created_by=req.created_by,
                expires_in_minutes=req.expires_in_minutes,
                clear_existing=req.clear_existing,
            )
            readiness = compute_readiness(session, persist=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "override": readiness.get("override"),
            "readiness": readiness,
            "created_override_id": int(override.id),
        }


def time_now_utc():
    from .channel_service_store import utcnow

    return utcnow()


def normalize_utc(value):
    if getattr(value, "tzinfo", None) is None:
        from datetime import timezone

        return value.replace(tzinfo=timezone.utc)
    return value


def channel_service_public_base_url(request: Request) -> str:
    configured = (os.getenv("CHANNEL_SERVICE_PUBLIC_BASE_URL") or "").strip()
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


def require_acp_shared_secret(request: Request) -> None:
    expected = (os.getenv("ACP_SHARED_SECRET") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="ACP bridge is disabled: ACP_SHARED_SECRET is not configured")

    presented = (request.headers.get("x-acp-shared-secret") or "").strip()
    if not presented:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid ACP shared secret")


def require_ops_shared_secret(request: Request) -> None:
    expected = (os.getenv("CHANNEL_SERVICE_OPS_SHARED_SECRET") or os.getenv("ACP_SHARED_SECRET") or "").strip()
    if not expected:
        return
    presented = (
        (request.headers.get("x-ops-shared-secret") or "").strip()
        or (request.headers.get("x-acp-shared-secret") or "").strip()
    )
    if not presented:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid ops shared secret")
