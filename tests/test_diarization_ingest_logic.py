from src.ingest_v2.cloud.diarization_indexer.ingest import (
    DiarizationIngestService,
    VideoMetadata,
)


class DummyEvent:
    mp3_uri = "gs://bucket/path.mp3"
    diarized_uri = "gs://bucket/youtube_diarized/VIDEO123/out.json"
    entities_uri = None


def make_dummy_video(channel_handle: str = "@SolanaFndn") -> VideoMetadata:
    return VideoMetadata(
        video_id="VIDEO123",
        channel_id="channel-id",
        channel_title="Solana Foundation",
        channel_handle=channel_handle,
        title="Sample Video",
        description="",
        published_at="2024-01-01T00:00:00Z",
        duration_seconds=123.0,
        thumbnail_url=None,
    )


def test_service_skips_channel_outside_namespace():
    called = {}

    def fake_fetch(video_id: str) -> VideoMetadata:
        return make_dummy_video(channel_handle="@OtherChannel")

    def fake_ingest(*args, **kwargs):
        called["ingest"] = True

    service = DiarizationIngestService(
        namespace="videos",
        allowed_channels=["@SolanaFndn"],
        fetch_video=fake_fetch,
        load_artifacts=lambda event, video: (None, None),
        ingest_pipeline=fake_ingest,
    )

    service.handle_event(DummyEvent())
    assert "ingest" not in called


def test_service_processes_allowed_channel():
    captured = {}

    def fake_fetch(video_id: str) -> VideoMetadata:
        return make_dummy_video()

    def fake_ingest(meta, raw_segments, event):
        captured["meta"] = meta
        captured["event"] = event

    service = DiarizationIngestService(
        namespace="videos",
        allowed_channels=["@SolanaFndn"],
        fetch_video=fake_fetch,
        load_artifacts=lambda event, video: (
            {
                "video_id": video.video_id,
                "channel_name": video.preferred_channel_name(),
                "url": f"https://www.youtube.com/watch?v={video.video_id}",
            },
            {"segments": []},
        ),
        ingest_pipeline=fake_ingest,
    )

    service.handle_event(DummyEvent())
    assert "meta" in captured
    assert captured["meta"]["video_id"] == "VIDEO123"
