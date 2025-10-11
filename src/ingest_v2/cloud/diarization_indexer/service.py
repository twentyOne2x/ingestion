from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import FastAPI, Request, Response

from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels

from .ingest import create_ingest_service
from .schemas import DiarizationReadyEvent, decode_pubsub_message

LOG = logging.getLogger(__name__)
app = FastAPI(title="Diarization Indexer")


def _namespace_from_attributes(attributes: Optional[Dict[str, str]]) -> str:
    if attributes and "namespace" in attributes:
        return attributes["namespace"]
    return "videos"


@app.post("/pubsub/push")
async def handle_pubsub_push(request: Request) -> Response:
    body = await request.json()
    attributes = body.get("message", {}).get("attributes", {}) or {}
    namespace = _namespace_from_attributes(attributes)

    allowed_channels = load_namespace_channels(namespace)
    if not allowed_channels:
        LOG.info("No channels configured for namespace=%s; skipping.", namespace)
        return Response(status_code=204)

    event = decode_pubsub_message(body, model=DiarizationReadyEvent)
    service = create_ingest_service(namespace, allowed_channels)
    service.handle_event(event)
    return Response(status_code=204)
