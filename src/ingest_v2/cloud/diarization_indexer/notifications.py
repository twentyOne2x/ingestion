from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional at import time
    from google.cloud import pubsub_v1  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    pubsub_v1 = None  # type: ignore

LOG = logging.getLogger(__name__)
_DEFAULT_TOPIC = "ingestion-diarization-ready"


@lru_cache(maxsize=1)
def _publisher_client() -> "pubsub_v1.PublisherClient":
    if pubsub_v1 is None:
        raise RuntimeError("google-cloud-pubsub is required to publish ingestion notifications")
    return pubsub_v1.PublisherClient()


@lru_cache(maxsize=1)
def _topic_path() -> str:
    topic = os.getenv("PIPELINE_NOTIFICATIONS_TOPIC", _DEFAULT_TOPIC)
    if topic.startswith("projects/"):
        return topic
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("PROJECT")
    )
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT (or GCP_PROJECT/PROJECT) must be set to publish ingestion notifications")
    return _publisher_client().topic_path(project, topic)


def _coerce_attributes(attributes: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not attributes:
        return {}
    coerced: Dict[str, str] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        coerced[key] = str(value)
    return coerced


def publish_ingestion_event(payload: Mapping[str, Any], *, attributes: Optional[Mapping[str, Any]] = None) -> None:
    """
    Publish a JSON payload describing an ingestion completion event.

    Args:
        payload: JSON-serialisable body to send.
        attributes: Optional Pub/Sub attributes to attach to the message.
    """
    data = json.dumps(payload, default=str).encode("utf-8")
    attrs = _coerce_attributes(attributes)
    try:
        future = _publisher_client().publish(_topic_path(), data=data, **attrs)
        future.result(timeout=15.0)
    except Exception as exc:  # pragma: no cover - network failures
        LOG.warning("Failed to publish ingestion notification: %s", exc)
