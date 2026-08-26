from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

import src.ingest_v2.cloud.diarization_indexer.canonical_vector_publication as publication
import src.ingest_v2.pipelines.upsert_pinecone as upsert_pipeline
from src.ingest_v2.cloud.diarization_indexer.public_acquisition import (
    PublicItemDescriptor,
)
from src.ingest_v2.cloud.diarization_indexer.transcription_runtime import (
    TranscriptResult,
)
from src.ingest_v2.utils import vector_store


@pytest.mark.parametrize(
    ("platform", "canonical_url", "expected_document_type"),
    (
        ("twitch", "https://www.twitch.tv/videos/123", "twitch_vod"),
        ("x", "https://x.com/example/status/123", "media"),
        ("pumpfun", "https://pump.fun/coin/room", "pumpfun_clip"),
        ("youtube", "https://www.youtube.com/watch?v=abc123", "youtube_video"),
    ),
)
def test_canonical_qdrant_publication_is_read_back_and_reused(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    canonical_url: str,
    expected_document_type: str,
) -> None:
    points: dict[str, dict] = {}
    upsert_batches: list[list[dict]] = []

    monkeypatch.setattr(publication, "vector_store_backend", lambda: "qdrant")
    monkeypatch.setattr(publication, "canonical_namespace", lambda: "canonical")
    monkeypatch.setattr(
        publication,
        "embedding_contract",
        lambda: {
            "provider": "sentence-transformers",
            "model": "fixture",
            "revision": "d" * 40,
            "dimension": 3,
        },
    )
    monkeypatch.setattr(publication, "settings_v2", SimpleNamespace(EMBED_DIM=3))
    monkeypatch.setattr(
        publication,
        "qdrant_collection_exists",
        lambda _collection: bool(points),
    )

    def fetch_points(*, collection_name: str, ids):
        assert collection_name == "icmfyi-v2__canonical"
        return {point_id: points[point_id] for point_id in ids if point_id in points}

    def upsert(children, *, qdrant_namespace: str, qdrant_wait: bool):
        assert qdrant_namespace == "canonical"
        assert qdrant_wait is True
        upsert_batches.append(list(children))
        for child in children:
            points[str(child["segment_id"])] = {
                "payload": dict(child),
                "vector": [0.1, 0.2, 0.3],
            }
        return {
            "t_embed": 0.0,
            "t_upsert": 0.0,
            "embed_reqs": 1,
            "pinecone_batches": 1,
        }

    monkeypatch.setattr(publication, "fetch_qdrant_points", fetch_points)
    monkeypatch.setattr(publication, "upsert_children", upsert)

    item = PublicItemDescriptor(
        platform=platform,
        external_id="123",
        channel_external_id="channel-123",
        channel_handle="example",
        canonical_url=canonical_url,
        title="Canonical vector fixture",
        duration_ms=2_000,
    )
    transcript = TranscriptResult(
        provider="local_cpu:fixture@revision",
        provider_request_id=None,
        segments=(
            {
                "ordinal": 0,
                "start_ms": 0,
                "end_ms": 2_000,
                "speaker_label": None,
                "text": (
                    "A canonical transcript point must survive exact readback, "
                    "including the deterministic summary point emitted for longer text."
                ),
            },
        ),
    )

    first = publication.publish_canonical_transcript_vectors(
        item=item,
        transcript=transcript,
        media_id=f"vid_{'a' * 40}",
        transcript_revision_id=f"trv_{'b' * 40}",
        language="en",
    )
    assert first["schema"] == publication.PUBLIC_VECTOR_PUBLICATION_SCHEMA
    assert first["collection"] == "icmfyi-v2__canonical"
    assert first["published_point_count"] == first["point_count"] == 2
    assert first["reused_point_count"] == 0
    assert len(first["readback_sha256"]) == 64
    assert len(upsert_batches) == 1
    assert {child["document_type"] for child in upsert_batches[0]} == {
        expected_document_type
    }
    assert {child["embedding_model"] for child in upsert_batches[0]} == {"fixture"}
    assert all(child["source_hash"] for child in upsert_batches[0])

    second = publication.publish_canonical_transcript_vectors(
        item=item,
        transcript=transcript,
        media_id=f"vid_{'a' * 40}",
        transcript_revision_id=f"trv_{'b' * 40}",
        language="en",
    )
    assert second["published_point_count"] == 0
    assert second["reused_point_count"] == second["point_count"] == 2
    assert second["readback_sha256"] == first["readback_sha256"]
    assert len(upsert_batches) == 1

    only_point = next(iter(points.values()))
    only_point["payload"]["embedding_model"] = "stale-model"
    repaired = publication.publish_canonical_transcript_vectors(
        item=item,
        transcript=transcript,
        media_id=f"vid_{'a' * 40}",
        transcript_revision_id=f"trv_{'b' * 40}",
        language="en",
    )
    assert repaired["published_point_count"] == 1
    assert repaired["reused_point_count"] == 1
    assert len(upsert_batches) == 2


def test_canonical_qdrant_publication_rejects_incomplete_vector_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points: dict[str, dict] = {}
    monkeypatch.setattr(publication, "vector_store_backend", lambda: "qdrant")
    monkeypatch.setattr(publication, "canonical_namespace", lambda: "canonical")
    monkeypatch.setattr(
        publication,
        "embedding_contract",
        lambda: {
            "provider": "sentence-transformers",
            "model": "fixture",
            "revision": "d" * 40,
            "dimension": 3,
        },
    )
    monkeypatch.setattr(publication, "settings_v2", SimpleNamespace(EMBED_DIM=3))
    monkeypatch.setattr(
        publication,
        "qdrant_collection_exists",
        lambda _collection: bool(points),
    )
    monkeypatch.setattr(
        publication,
        "fetch_qdrant_points",
        lambda *, collection_name, ids: {
            point_id: points[point_id] for point_id in ids if point_id in points
        },
    )

    def incomplete_upsert(children, **_kwargs):
        for child in children:
            points[str(child["segment_id"])] = {
                "payload": dict(child),
                "vector": [0.1, 0.2],
            }
        return {}

    monkeypatch.setattr(publication, "upsert_children", incomplete_upsert)
    item = PublicItemDescriptor(
        platform="youtube",
        external_id="abc123",
        channel_external_id="channel-123",
        channel_handle="example",
        canonical_url="https://www.youtube.com/watch?v=abc123",
        duration_ms=2_000,
    )
    transcript = TranscriptResult(
        provider="local_cpu:fixture@revision",
        provider_request_id=None,
        segments=(
            {
                "ordinal": 0,
                "start_ms": 0,
                "end_ms": 2_000,
                "speaker_label": None,
                "text": "This point has a deliberately invalid vector dimension.",
            },
        ),
    )

    with pytest.raises(
        publication.CanonicalVectorPublicationError,
        match="canonical Qdrant readback is incomplete",
    ):
        publication.publish_canonical_transcript_vectors(
            item=item,
            transcript=transcript,
            media_id=f"vid_{'a' * 40}",
            transcript_revision_id=f"trv_{'b' * 40}",
            language="en",
        )


def test_canonical_publication_round_trips_through_embedded_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(location=":memory:")
    embed_calls: list[tuple[str, ...]] = []
    settings = SimpleNamespace(
        EMBED_DIM=3,
        MAX_METADATA_BYTES=12_000,
        NAMESPACE_STREAMS="streams",
        NAMESPACE_VIDEOS="videos",
        PINECONE_UPSERT_BATCH=100,
    )
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "embedded-fixture")
    monkeypatch.setattr(vector_store, "qdrant_client", lambda: client)
    monkeypatch.setattr(publication, "canonical_namespace", lambda: "canonical")
    monkeypatch.setattr(publication, "settings_v2", settings)
    monkeypatch.setattr(upsert_pipeline, "settings_v2", settings)
    monkeypatch.setattr(
        publication,
        "embedding_contract",
        lambda: {
            "provider": "sentence-transformers",
            "model": "fixture",
            "revision": "d" * 40,
            "dimension": 3,
        },
    )

    def embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(tuple(texts))
        return [[0.1, 0.2, 0.3] for _text in texts]

    monkeypatch.setattr(upsert_pipeline, "_embedder", lambda: embed)
    item = PublicItemDescriptor(
        platform="x",
        external_id="123",
        channel_external_id="456",
        channel_handle="example",
        canonical_url="https://x.com/example/status/123",
        title="Embedded Qdrant fixture",
        duration_ms=2_000,
    )
    transcript = TranscriptResult(
        provider="local_cpu:fixture@revision",
        provider_request_id=None,
        segments=(
            {
                "ordinal": 0,
                "start_ms": 0,
                "end_ms": 2_000,
                "speaker_label": None,
                "text": "The embedded Qdrant point proves completed local readback.",
            },
        ),
    )
    kwargs = {
        "item": item,
        "transcript": transcript,
        "media_id": f"vid_{'a' * 40}",
        "transcript_revision_id": f"trv_{'b' * 40}",
        "language": "en",
    }

    try:
        first = publication.publish_canonical_transcript_vectors(**kwargs)
        assert first["collection"] == "embedded-fixture__canonical"
        assert first["published_point_count"] == first["point_count"] == 1
        assert len(embed_calls) == 1

        replay = publication.publish_canonical_transcript_vectors(**kwargs)
        assert replay["published_point_count"] == 0
        assert replay["reused_point_count"] == replay["point_count"] == 1
        assert replay["readback_sha256"] == first["readback_sha256"]
        assert len(embed_calls) == 1
    finally:
        client.close()
