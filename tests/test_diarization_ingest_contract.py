import base64
import json

import pytest

from src.ingest_v2.cloud.diarization_indexer.ingest import resolve_event_context

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
        "metadata_uri": "gs://bucket/youtube_diarized/abc123/abc123_metadata.json",
        "video_id": "abc123",
        "entities_uri": "gs://bucket/youtube_diarized/abc123/abc123_entities.json",
    }

    event = decode_pubsub_message(build_pubsub_payload(payload), model=DiarizationReadyEvent)
    assert event.mp3_uri == payload["mp3_uri"]
    assert event.diarized_uri == payload["diarized_uri"]
    assert event.metadata_uri == payload["metadata_uri"]
    assert event.video_id == payload["video_id"]
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



def test_resolve_event_context_pumpfun(monkeypatch):
    class PumpfunEvent:
        mp3_uri = 'gs://bucket/pumpfun_streams/RasMrToken/2025-10-12_clip/foo.mp3'
        diarized_uri = 'gs://bucket/pumpfun_streams/RasMrToken/2025-10-12_clip/foo_diarized.json'
        metadata_uri = 'gs://bucket/pumpfun_streams/RasMrToken/2025-10-12_clip/metadata.json'
        video_id = 'pumpfun_RasMrToken_2025-10-12_clip'

    metadata = {"clip": {"roomName": "RasMrToken", "clipId": "clip123", "startTime": "2025-10-12T00:00:00Z", "playlistUrl": "https://pump.fun/clip"}, "coin": {"name": "RasMr", "symbol": "RASMR"}}
    monkeypatch.setattr('src.ingest_v2.cloud.diarization_indexer.ingest.read_json_from_gcs', lambda uri: metadata)

    context = resolve_event_context(PumpfunEvent())
    assert context.source == 'pumpfun'
    assert context.video_id == 'pumpfun_RasMrToken_2025-10-12_clip'
    assert context.pumpfun_metadata == metadata
    assert context.pumpfun_room == 'RasMrToken'
    assert context.pumpfun_clip == '2025-10-12_clip'
