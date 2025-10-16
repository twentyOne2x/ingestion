from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

T = TypeVar("T", bound="BaseEvent")


class BaseEvent(BaseModel):
    """Base schema for Pub/Sub events produced by the media pipeline."""

    event_version: str = Field(default="v1", frozen=True)

    def to_message(self) -> Dict[str, Any]:
        payload = self.model_dump()
        return payload

    @classmethod
    def from_payload(cls: Type[T], payload: Dict[str, Any]) -> T:
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid payload for {cls.__name__}: {exc}") from exc

    @classmethod
    def from_base64(cls: Type[T], data: str) -> T:
        decoded = base64.b64decode(data).decode("utf-8")
        return cls.from_payload(json.loads(decoded))


class DiarizationReadyEvent(BaseEvent):
    mp3_uri: str = Field(min_length=1)
    diarized_uri: str = Field(min_length=1)
    metadata_uri: Optional[str] = None
    video_id: Optional[str] = None
    entities_uri: Optional[str] = None

    @field_validator("mp3_uri", "diarized_uri", "metadata_uri", "entities_uri")
    @classmethod
    def _validate_uri(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value.startswith("gs://"):
            return value
        if value.startswith("file://") or value.startswith("/"):
            return value
        raise ValueError("Unsupported URI scheme")


def decode_pubsub_message(
    body: Dict[str, Any],
    *,
    model: Type[T],
) -> T:
    """Decode a Pub/Sub push payload into an event model."""
    message = body.get("message")
    if not message:
        raise ValueError("Missing 'message' in Pub/Sub request.")
    data = message.get("data")
    if not data:
        raise ValueError("Missing 'data' in Pub/Sub message.")
    return model.from_base64(data)
