import base64
import json

import pytest


def build_pubsub_payload(message: dict) -> dict:
    encoded = base64.b64encode(json.dumps(message).encode("utf-8")).decode("utf-8")
    return {
        "message": {
            "data": encoded,
            "messageId": "123",
            "attributes": {"namespace": "videos"},
        },
        "subscription": "projects/test/subscriptions/diarization-ready",
    }


def test_decode_pubsub_message_roundtrip():
    from src.ingest_v2.cloud.diarization_indexer.schemas import (
        DiarizationReadyEvent,
        decode_pubsub_message,
    )

    payload = {
        "mp3_uri": "gs://bucket/youtube_audio/2024-01-01_abc123/foo.mp3",
        "diarized_uri": "gs://bucket/youtube_diarized/abc123/abc123_diarized.json",
        "entities_uri": "gs://bucket/youtube_diarized/abc123/abc123_entities.json",
    }

    event = decode_pubsub_message(build_pubsub_payload(payload), model=DiarizationReadyEvent)
    assert event.mp3_uri == payload["mp3_uri"]
    assert event.diarized_uri == payload["diarized_uri"]
    assert event.entities_uri == payload["entities_uri"]


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("gs://bucket/youtube_diarized/ABC123/file.json", "ABC123"),
        ("gs://bucket/custom_prefix/@channel/2024-01-01_ABC123_title/file.json", "ABC123"),
    ],
)
def test_extract_video_id_from_uri(uri, expected):
    from src.ingest_v2.cloud.diarization_indexer.ingest import extract_video_id

    assert extract_video_id(uri) == expected


def test_extract_video_id_raises_for_unknown_shape():
    from src.ingest_v2.cloud.diarization_indexer.ingest import extract_video_id

    with pytest.raises(ValueError):
        extract_video_id("gs://bucket/invalid_path.json")
