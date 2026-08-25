from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.canonical_media import (
    CanonicalPublishError,
    HotMediaSpec,
    canonical_source_video_id,
    publish_canonical_ingestion,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_config import (
    InternalRequestIdentity,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    Base,
    MediaLocation,
    MediaObject,
    SourceVideo,
    TenantChannelEntitlement,
    TranscriptRevision,
    TranscriptSegment,
    VideoMediaRef,
)
from src.ingest_v2.pipelines.index_youtube_captions import (
    CaptionCue,
    _publish_staged_hot_media,
    acquire_youtube_hot_media,
    index_youtube_video_captions,
)


IDENTITY_A = InternalRequestIdentity(
    user_id=f"usr_{'a' * 64}", tenant_id=f"ten_{'1' * 64}"
)
IDENTITY_B = InternalRequestIdentity(
    user_id=f"usr_{'b' * 64}", tenant_id=f"ten_{'2' * 64}"
)


def _publish(
    session: Session,
    *,
    identity: InternalRequestIdentity,
    hot_media: HotMediaSpec | None = None,
):
    return publish_canonical_ingestion(
        session,
        identity=identity,
        platform="youtube",
        provider_video_id="dQw4w9WgXcQ",
        channel_external_id="UC-canonical",
        channel_handle="@canonical",
        channel_name="Canonical Channel",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Canonical video",
        description="A durable canonical transcript.",
        published_at="2026-08-25T00:00:00Z",
        duration_ms=2_000,
        language="en",
        transcript_provider="yt_transcript_api",
        transcript_segments=[
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "First fact."},
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "Second fact."},
        ],
        hot_media=hot_media,
    )


def test_two_tenants_share_canonical_video_but_receive_separate_entitlements() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first = _publish(session, identity=IDENTITY_A)
        second = _publish(session, identity=IDENTITY_B)
        session.commit()

        assert first.media_id == canonical_source_video_id("youtube", "dQw4w9WgXcQ")
        assert second.media_id == first.media_id
        assert first.clip_ready is False
        assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
        assert session.scalar(select(func.count()).select_from(TranscriptRevision)) == 1
        assert session.scalar(select(func.count()).select_from(TranscriptSegment)) == 2
        assert (
            session.scalar(select(func.count()).select_from(TenantChannelEntitlement))
            == 2
        )


def test_existing_retained_video_becomes_clip_ready_only_after_sha_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the retained-media smoke")

    hot_root = tmp_path / "hot"
    hot_root.mkdir()
    media_path = hot_root / "bounded.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.25",
            "-an",
            "-c:v",
            "mpeg4",
            "-y",
            str(media_path),
        ],
        check=True,
        timeout=30,
    )
    payload = media_path.read_bytes()
    media = HotMediaSpec(
        path=media_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        mime_type="video/mp4",
    )
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(hot_root))
    monkeypatch.setenv("CHANNEL_SERVICE_FFPROBE_BIN", ffprobe)

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        transcript_only = _publish(session, identity=IDENTITY_A)
        assert transcript_only.clip_ready is False
        retained = _publish(session, identity=IDENTITY_A, hot_media=media)
        session.commit()

        assert retained.clip_ready is True
        video = session.get(SourceVideo, retained.media_id)
        assert video is not None
        assert video.archive_state == "retained_hot_verified"
        assert video.clip_candidate is True
        assert video.clip_ready is True
        media_row = session.get(MediaObject, media.sha256)
        assert media_row is not None
        assert media_row.size_bytes == len(payload)
        reference = session.execute(select(VideoMediaRef)).scalar_one()
        assert reference.video_id == retained.media_id
        assert reference.role == "source_video"
        location = session.execute(select(MediaLocation)).scalar_one()
        assert location.backend == "hot_local"
        assert location.location_key == str(media_path.resolve())
        assert location.status == "active"
        assert location.verified_at is not None


def test_hot_media_outside_declared_root_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hot"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not-a-video")
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(root))
    media = HotMediaSpec(
        path=outside,
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        size_bytes=outside.stat().st_size,
        mime_type="video/mp4",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with (
        Session(engine) as session,
        pytest.raises(CanonicalPublishError, match="outside"),
    ):
        _publish(session, identity=IDENTITY_A, hot_media=media)


def test_caption_vectors_publish_canonical_media_id_before_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    parents: list[dict] = []
    children: list[dict] = []
    canonical_payloads: list[dict] = []
    monkeypatch.setenv("ROUTER_ENRICH", "0")
    monkeypatch.setattr(
        "src.ingest_v2.pipelines.index_youtube_captions.fetch_transcript_cues",
        lambda **_kwargs: (
            [CaptionCue(start_s=0.0, end_s=2.0, text="Canonical searchable fact.")],
            "yt_transcript_api",
            {
                "title": "Canonical",
                "channel": "@canonical",
                "channel_id": "UC-canonical",
                "upload_date": "20260825",
                "duration": 2,
            },
            "dQw4w9WgXcQ",
            None,
        ),
    )

    def capture_publish(payload: dict) -> None:
        events.append("canonical")
        canonical_payloads.append(payload)

    def capture_parents(payload: list[dict]) -> None:
        events.append("parent")
        parents.extend(payload)

    def capture_children(payload: list[dict]) -> dict:
        events.append("children")
        children.extend(payload)
        return {"upserted": len(payload)}

    monkeypatch.setattr(
        "src.ingest_v2.pipelines.index_youtube_captions.upsert_parents",
        capture_parents,
    )
    monkeypatch.setattr(
        "src.ingest_v2.pipelines.index_youtube_captions.upsert_children",
        capture_children,
    )
    expected_hot = HotMediaSpec(
        path=Path("/data/hot-media/sha256/aa/example.mp4"),
        sha256="a" * 64,
        size_bytes=123,
        mime_type="video/mp4",
    )
    acquisition_calls: list[tuple[str, str]] = []

    def acquire(url: str, video_id: str) -> HotMediaSpec:
        acquisition_calls.append((url, video_id))
        return expected_hot

    result = index_youtube_video_captions(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        segment_min_s=1,
        segment_max_s=5,
        segment_stride_s=1,
        min_text_chars=1,
        canonical_publish=capture_publish,
        acquire_hot_media=True,
        media_acquire=acquire,
    )

    expected = canonical_source_video_id("youtube", "dQw4w9WgXcQ")
    assert events[0] == "canonical"
    assert result["media_id"] == expected
    assert canonical_payloads[0]["provider_video_id"] == "dQw4w9WgXcQ"
    assert canonical_payloads[0]["hot_media"] is expected_hot
    assert acquisition_calls == [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ")
    ]
    assert result["clip_ready"] is True
    assert result["hot_media_sha256"] == "a" * 64
    assert parents[0]["media_id"] == expected
    assert children
    assert {child["media_id"] for child in children} == {expected}


def test_hot_media_acquisition_is_content_addressed_and_receipt_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.pipelines.index_youtube_captions as module

    hot_root = tmp_path / "hot"
    calls: list[str] = []

    class FakeYDL:
        def __init__(self, options: dict) -> None:
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, url: str, *, download: bool) -> dict:
            assert download is True
            calls.append(url)
            output = Path(
                self.options["outtmpl"]
                .replace("%(id)s", "dQw4w9WgXcQ")
                .replace("%(ext)s", "mp4")
            )
            output.write_bytes(b"bounded-test-video")
            return {"id": "dQw4w9WgXcQ"}

    monkeypatch.setattr(module, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(module, "verify_hot_media", lambda spec: spec)
    monkeypatch.setenv("CHANNEL_SERVICE_HOT_MEDIA_ROOT", str(hot_root))

    first = acquire_youtube_hot_media(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    )
    second = acquire_youtube_hot_media(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    )

    assert calls == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    assert first == second
    assert first.path.read_bytes() == b"bounded-test-video"
    assert first.path.name == f"{first.sha256}.mp4"
    assert first.path.stat().st_mode & 0o200 == 0
    receipt = hot_root / ".staging" / "youtube" / "dQw4w9WgXcQ" / "receipt.json"
    assert receipt.is_file()


def test_identical_staging_publications_converge_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ingest_v2.pipelines.index_youtube_captions as module

    root = tmp_path / "hot"
    first_staging = root / ".staging" / "youtube" / "video-a"
    second_staging = root / ".staging" / "youtube" / "video-b"
    first_staging.mkdir(parents=True)
    second_staging.mkdir(parents=True)
    first_source = first_staging / "video-a.mp4"
    second_source = second_staging / "video-b.mp4"
    first_source.write_bytes(b"identical-video-bytes")
    second_source.write_bytes(b"identical-video-bytes")
    monkeypatch.setattr(module, "verify_hot_media", lambda spec: spec)

    first = _publish_staged_hot_media(
        root=root,
        staging_dir=first_staging,
        source=first_source,
        video_id="video-a",
        max_bytes=1024,
    )
    second = _publish_staged_hot_media(
        root=root,
        staging_dir=second_staging,
        source=second_source,
        video_id="video-b",
        max_bytes=1024,
    )

    assert first == second
    assert first.path.read_bytes() == b"identical-video-bytes"
    assert list((root / "sha256" / first.sha256[:2]).iterdir()) == [first.path]
    assert not first_source.exists()
    assert not second_source.exists()
