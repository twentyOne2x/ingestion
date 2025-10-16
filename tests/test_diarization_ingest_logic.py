from src.ingest_v2.cloud.diarization_indexer.ingest import (
    DiarizationIngestService,
    VideoMetadata,
    build_pumpfun_metadata,
    resolve_event_context,
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
        load_artifacts=lambda event, video, context: (None, None),
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
        load_artifacts=lambda event, video, context: (
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



def test_service_handles_pumpfun_event(monkeypatch):
    from src.ingest_v2.cloud.diarization_indexer.ingest import resolve_event_context, build_pumpfun_metadata, DiarizationIngestService, VideoMetadata

    metadata = {
        'clip': {
            'roomName': 'RasMrToken',
            'clipId': 'clip123',
            'startTime': '2025-10-12T00:00:00Z',
            'playlistUrl': 'https://pump.fun/clip',
            'duration': 900,
        },
        'coin': {'name': 'RasMr', "symbol": 'RASMR', "description": 'Pumpfun room'},
    }

    class PumpfunEvent:
        mp3_uri = 'gs://bucket/pumpfun_streams/RasMrToken/clip123/audio.mp3'
        diarized_uri = 'gs://bucket/pumpfun_streams/RasMrToken/clip123/diarized.json'
        metadata_uri = 'gs://bucket/pumpfun_streams/RasMrToken/clip123/metadata.json'
        video_id = 'pumpfun_RasMrToken_clip123'
        entities_uri = None

    monkeypatch.setattr('src.ingest_v2.cloud.diarization_indexer.ingest.read_json_from_gcs', lambda uri: metadata)

    context = resolve_event_context(PumpfunEvent())
    video_meta = build_pumpfun_metadata(PumpfunEvent(), context)

    captured = {}

    def fake_ingest(meta, raw_segments, event):
        captured['meta'] = meta
        captured['event'] = event

    service = DiarizationIngestService(
        namespace='streams',
        allowed_channels=[],
        fetch_video=lambda video_id: video_meta,
        load_artifacts=lambda event, video, ctx: ({'video_id': video.video_id, 'url': ctx.mp3_uri, 'pumpfun_room': ctx.pumpfun_room}, {'segments': []}),
        ingest_pipeline=fake_ingest,
    )

    service.handle_event(PumpfunEvent())
    assert captured['meta']['video_id'] == 'pumpfun_RasMrToken_clip123'
    assert captured['meta']['pumpfun_room'] == 'RasMrToken'
    assert captured['meta']['url'] == PumpfunEvent().mp3_uri
