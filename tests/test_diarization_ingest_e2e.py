import json
from pathlib import Path

import pytest

from src.ingest_v2.cloud.diarization_indexer.ingest import (
    create_ingest_service,
    VideoMetadata,
)
from src.ingest_v2.cloud.diarization_indexer.schemas import DiarizationReadyEvent


@pytest.fixture()
def namespace_config(monkeypatch, tmp_path):
    config = {
        "namespaces": {
            "videos": {"channels": ["@TestChannel"]},
        }
    }
    monkeypatch.setenv("YT_NAMESPACE_CONFIG_JSON", json.dumps(config))
    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-api-key")
    return config


def test_diarization_ingest_e2e(monkeypatch, namespace_config, tmp_path):
    sample = Path("tests/data/diarization/sample_diarized.json").resolve()
    local_dir = tmp_path / "youtube_diarized" / "VIDEOABC1234"
    local_dir.mkdir(parents=True, exist_ok=True)
    diarized_path = local_dir / "VIDEOABC1234_diarized.json"
    diarized_path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    event = DiarizationReadyEvent(
        mp3_uri="gs://bucket/youtube_audio/sample.mp3",
        diarized_uri=f"file://{diarized_path}",
        entities_uri=None,
    )

    # Patch YouTube metadata fetch
    def fake_fetch(self, video_id: str) -> VideoMetadata:
        return VideoMetadata(
            video_id=video_id,
            channel_id="channel-123",
            channel_title="Test Channel",
            channel_handle="@TestChannel",
            title="Integration Test",
            description="",
            published_at="2025-01-01T00:00:00Z",
            duration_seconds=12.0,
            thumbnail_url=None,
        )

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.youtube.YouTubeClient.fetch_video_metadata",
        fake_fetch,
    )

    captured_parents = []
    captured_children = []

    def fake_upsert_parents(payload):
        captured_parents.extend(payload)

    def fake_upsert_children(children):
        captured_children.extend(children)
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    monkeypatch.setattr(
        "src.ingest_v2.pipelines.upsert_parents.upsert_parents",
        fake_upsert_parents,
    )
    monkeypatch.setattr(
        "src.ingest_v2.pipelines.upsert_pinecone.upsert_children",
        fake_upsert_children,
    )
    monkeypatch.setattr(
        "src.ingest_v2.pipelines.build_children.build_children_from_raw",
        lambda parent, raw: [{"node_type": "child", "segment_id": "seg1"}],
    )

    service = create_ingest_service("videos", ["@TestChannel"])
    service.handle_event(event)

    assert captured_parents, "Expected parent upsert payload"
    assert captured_children, "Expected child segments to be ingested"
    parent_meta = captured_parents[0]
    assert parent_meta["channel_name"] == "@TestChannel"
    assert parent_meta["video_id"] == "VIDEOABC1234"
    assert captured_children[0]["node_type"] == "child"
