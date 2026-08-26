from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.ingest_v2.cloud.diarization_indexer import public_acquisition
from src.ingest_v2.cloud.diarization_indexer.public_acquisition import (
    PublicAcquisitionError,
    PublicItemDescriptor,
)
from src.ingest_v2.pipelines import index_youtube_captions, youtube_ytdlp_options
from src.ingest_v2.pipelines.youtube_ytdlp_options import (
    BGUTIL_PROVIDER_VERSION,
    YOUTUBE_MP4_FORMAT,
    build_youtube_ytdlp_options,
)

_YTDLP_ENV = (
    "CHANNEL_SERVICE_YTDLP_VIDEO_FORMAT",
    "YTDLP_COOKIES_FILE",
    "YTDLP_COOKIES_PATH",
    "YTDLP_DENO_BIN",
    "YTDLP_MAX_SLEEP_INTERVAL",
    "YTDLP_PLAYER_CLIENTS",
    "YTDLP_PO_TOKEN",
    "YTDLP_PO_TOKEN_PROVIDER",
    "YTDLP_PO_TOKEN_PROVIDER_DIR",
    "YTDLP_REMOTE_COMPONENTS",
    "YTDLP_SLEEP_INTERVAL",
    "YTDLP_SLEEP_REQUESTS",
    "YTDLP_SLEEP_SUBTITLES",
)


def _clear_ytdlp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _YTDLP_ENV:
        monkeypatch.delenv(name, raising=False)


def test_generic_and_index_youtube_media_share_hardened_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ytdlp_environment(monkeypatch)
    video_id = "dQw4w9WgXcQ"
    cookie_reference = tmp_path / "youtube.cookies"
    cookie_reference.write_text("fixture-cookie-reference\n", encoding="ascii")
    captured: list[dict] = []

    class FakeYDL:
        def __init__(self, options: dict) -> None:
            self.options = options
            captured.append(copy.deepcopy(options))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, *, download: bool) -> dict:
            assert download is True
            output = Path(
                self.options["outtmpl"]
                .replace("%(id)s", video_id)
                .replace("%(ext)s", "mp4")
            )
            output.write_bytes(b"bounded-youtube-fixture")
            return {
                "id": video_id,
                "channel_id": "UCfixture",
                "channel": "Fixture Channel",
                "extractor_key": "Youtube",
            }

    hot_root = tmp_path / "hot"
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(hot_root))
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookie_reference))
    monkeypatch.setenv("YTDLP_SLEEP_REQUESTS", "1")
    monkeypatch.setenv("YTDLP_SLEEP_INTERVAL", "15")
    monkeypatch.setenv("YTDLP_MAX_SLEEP_INTERVAL", "30")
    monkeypatch.setenv("YTDLP_SLEEP_SUBTITLES", "5")
    monkeypatch.setenv("YTDLP_PLAYER_CLIENTS", "mweb")
    monkeypatch.setattr(public_acquisition, "verify_hot_media", lambda spec: spec)
    monkeypatch.setattr(index_youtube_captions, "verify_hot_media", lambda spec: spec)
    monkeypatch.setattr(index_youtube_captions, "YoutubeDL", FakeYDL)

    item = PublicItemDescriptor(
        platform="youtube",
        external_id=video_id,
        channel_external_id="pending-provider-identity",
        channel_handle=None,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
    )
    public_acquisition.acquire_public_item(item, ydl_factory=FakeYDL)
    index_youtube_captions.acquire_youtube_hot_media(
        item.canonical_url,
        video_id,
    )

    assert len(captured) == 2
    generic, index_media = captured
    for key in (
        "cookiefile",
        "js_runtimes",
        "sleep_requests",
        "sleep_interval",
        "max_sleep_interval",
        "sleep_subtitles",
        "extractor_args",
        "format",
        "merge_output_format",
    ):
        assert generic[key] == index_media[key]
    assert generic["format"] == YOUTUBE_MP4_FORMAT
    assert generic["remote_components"] == set()


def test_bgutil_script_provider_is_same_host_and_secret_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_ytdlp_environment(monkeypatch)
    provider_home = tmp_path / "provider"
    (provider_home / "src").mkdir(parents=True)
    (provider_home / "src" / "generate_once.ts").write_text(
        "// fixture only\n", encoding="ascii"
    )
    (provider_home / "node_modules").mkdir()
    (provider_home / "package.json").write_text(
        '{"name":"bgutil-ytdlp-pot-provider","version":"1.3.2"}\n',
        encoding="ascii",
    )
    deno = tmp_path / "deno"
    deno.write_text("#!/bin/sh\nexit 1\n", encoding="ascii")
    deno.chmod(0o700)

    monkeypatch.setenv("YTDLP_PO_TOKEN_PROVIDER", "bgutil-script")
    monkeypatch.setenv("YTDLP_PO_TOKEN_PROVIDER_DIR", str(provider_home))
    monkeypatch.setenv("YTDLP_DENO_BIN", str(deno))
    monkeypatch.setattr(
        youtube_ytdlp_options.metadata,
        "version",
        lambda package: (
            BGUTIL_PROVIDER_VERSION
            if package == "bgutil-ytdlp-pot-provider"
            else pytest.fail(f"unexpected package lookup: {package}")
        ),
    )

    options = build_youtube_ytdlp_options()

    assert options["js_runtimes"] == {"deno": {"path": str(deno)}}
    assert options["extractor_args"]["youtube"]["player_client"] == ["mweb"]
    assert options["extractor_args"]["youtubepot-bgutilscript"] == {
        "server_home": [str(provider_home)]
    }
    assert options["extractor_args"]["youtubepot-bgutilhttp"] == {
        "base_url": ["http://127.0.0.1:0"]
    }
    assert "po_token" not in options["extractor_args"]["youtube"]
    options["logger"].debug("cookie-or-token-fixture")
    options["logger"].warning("cookie-or-token-fixture")
    options["logger"].error("cookie-or-token-fixture")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_provider_and_remote_components_fail_closed_without_secret_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ytdlp_environment(monkeypatch)
    token = "sensitive-token-fixture"
    monkeypatch.setenv("YTDLP_PO_TOKEN", token)
    with pytest.raises(RuntimeError) as token_error:
        build_youtube_ytdlp_options()
    assert token not in str(token_error.value)

    monkeypatch.delenv("YTDLP_PO_TOKEN")
    monkeypatch.setenv("YTDLP_REMOTE_COMPONENTS", "ejs:github")
    with pytest.raises(RuntimeError, match="bundled and pinned"):
        build_youtube_ytdlp_options()

    monkeypatch.delenv("YTDLP_REMOTE_COMPONENTS")
    monkeypatch.setenv("YTDLP_PO_TOKEN_PROVIDER", "bgutil-script")
    monkeypatch.setenv("YTDLP_PO_TOKEN_PROVIDER_DIR", str(tmp_path / token))
    with pytest.raises(RuntimeError) as provider_error:
        build_youtube_ytdlp_options()
    assert token not in str(provider_error.value)

    monkeypatch.setenv("YTDLP_PO_TOKEN_PROVIDER", "disabled")
    disabled = build_youtube_ytdlp_options()
    assert disabled["extractor_args"]["youtubepot-bgutilscript"] == {
        "server_home": ["/dev/null"]
    }
    assert disabled["extractor_args"]["youtubepot-bgutilhttp"] == {
        "base_url": ["http://127.0.0.1:0"]
    }


def test_extractor_failure_suppresses_secret_bearing_exception_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ytdlp_environment(monkeypatch)
    secret = "cookie-or-token-fixture"

    class FailingYDL:
        def __init__(self, _options: dict) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url: str, *, download: bool) -> dict:
            assert download is True
            raise RuntimeError(f"provider output contained {secret}")

    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(tmp_path / "hot"))
    item = PublicItemDescriptor(
        platform="youtube",
        external_id="dQw4w9WgXcQ",
        channel_external_id="UCfixture",
        channel_handle="@fixture",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )

    with pytest.raises(PublicAcquisitionError) as captured:
        public_acquisition.acquire_public_item(item, ydl_factory=FailingYDL)

    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_ytdlp_ejs_and_provider_dependencies_are_pinned_in_the_image() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    dockerfile = (root / "services" / "diarization_indexer" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "yt-dlp==2026.8.19" in requirements
    assert "yt-dlp-ejs==0.8.0" in requirements
    assert "bgutil-ytdlp-pot-provider==1.3.2" in requirements
    assert "denoland/deno:2.5.6@sha256:" in dockerfile
    assert "YTDLP_DENO_BIN=/usr/local/bin/deno" in dockerfile
    assert "YTDLP_REMOTE_COMPONENTS" not in dockerfile
