from src.ingest_v2.cloud.diarization_indexer import channel_service_logic as logic


def _base_video(video_id: str) -> dict:
    return {
        "video_id": video_id,
        "title": f"title-{video_id}",
        "description": "",
        "channel_name": "OpenAI",
        "channel_handle": "@OpenAI",
        "published_at": "2024-01-01",
        "thumbnail_url": f"https://img.youtube.com/{video_id}.jpg",
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "duration_s": 300.0,
        "channel_id": "channel-123",
    }


def _stub_non_catalog(monkeypatch):
    monkeypatch.setattr(logic, "_channel_handle_is_restricted", lambda channel_handle: False)
    monkeypatch.setattr(logic, "_catalog_supports_handle", lambda namespace, channel_handle: False)
    monkeypatch.setattr(
        logic,
        "_indexed_channel_candidates",
        lambda **kwargs: {
            "channel_id": "indexed-channel",
            "channel_name": "Indexed",
            "channel_handle": kwargs["channel_handle"],
            "videos": [],
        },
    )


def test_list_channel_candidates_prefers_ytdlp_before_api(monkeypatch):
    _stub_non_catalog(monkeypatch)

    ytdlp_result = {
        "channel_id": "channel-123",
        "channel_name": "OpenAI",
        "channel_handle": "@OpenAI",
        "videos": [_base_video("vid-1")],
    }
    monkeypatch.setattr(logic, "list_channel_candidates_via_ytdlp", lambda **kwargs: ytdlp_result)

    def _unexpected(*args, **kwargs):
        raise AssertionError("YouTube API path should not run when yt-dlp already returned videos")

    monkeypatch.setattr(logic, "resolve_channel_summary", _unexpected)
    monkeypatch.setattr(logic, "_uploads_playlist_id", _unexpected)
    monkeypatch.setattr(logic, "_ytapi_get", _unexpected)

    result = logic.list_channel_candidates(
        api_key="fake-key",
        namespace="videos",
        channel_handle="@OpenAI",
        target_count=1,
    )

    assert result == ytdlp_result


def test_list_channel_candidates_uses_api_when_ytdlp_returns_no_videos(monkeypatch):
    _stub_non_catalog(monkeypatch)

    monkeypatch.setattr(
        logic,
        "list_channel_candidates_via_ytdlp",
        lambda **kwargs: {
            "channel_id": "channel-123",
            "channel_name": "OpenAI",
            "channel_handle": "@OpenAI",
            "videos": [],
        },
    )
    monkeypatch.setattr(
        logic,
        "resolve_channel_summary",
        lambda channel_handle, api_key: {
            "channel_id": "channel-123",
            "channel_name": "OpenAI",
            "channel_handle": "@OpenAI",
        },
    )
    monkeypatch.setattr(logic, "_uploads_playlist_id", lambda channel_id, api_key: "uploads-123")

    def _fake_ytapi_get(path, *, api_key, params):
        if path == "playlistItems":
            return {
                "items": [
                    {
                        "snippet": {
                            "resourceId": {"videoId": "vid-1"},
                            "title": "title-vid-1",
                            "description": "",
                            "publishedAt": "2024-01-01T00:00:00Z",
                            "thumbnails": {"high": {"url": "https://img.youtube.com/vid-1.jpg"}},
                            "channelTitle": "OpenAI",
                        }
                    }
                ]
            }
        if path == "videos":
            return {
                "items": [
                    {
                        "id": "vid-1",
                        "snippet": {
                            "publishedAt": "2024-01-01T00:00:00Z",
                            "channelTitle": "OpenAI",
                            "thumbnails": {"high": {"url": "https://img.youtube.com/vid-1.jpg"}},
                        },
                        "contentDetails": {"duration": "PT5M"},
                    }
                ]
            }
        raise AssertionError(f"unexpected YouTube API path: {path}")

    monkeypatch.setattr(logic, "_ytapi_get", _fake_ytapi_get)

    result = logic.list_channel_candidates(
        api_key="fake-key",
        namespace="videos",
        channel_handle="@OpenAI",
        target_count=1,
    )

    assert result["channel_id"] == "channel-123"
    assert result["channel_handle"] == "@OpenAI"
    assert result["videos"][0]["video_id"] == "vid-1"
    assert result["videos"][0]["duration_s"] == 300.0
