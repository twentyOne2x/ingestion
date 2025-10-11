from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response

from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels

from .ingest import create_ingest_service
from .pubsub import verify_pubsub_push
from .schemas import DiarizationReadyEvent, decode_pubsub_message

LOG = logging.getLogger(__name__)
app = FastAPI(title="Diarization Indexer")


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
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub payload") from exc

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
