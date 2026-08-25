from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.archive_catalog_loader import (
    ARCHIVE_CATALOG_SCHEMA,
    ArchiveCatalogError,
    load_archive_catalog,
    write_archive_catalog_receipt,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    Base,
    MediaLocation,
    MediaObject,
    SourceChannel,
    SourceVideo,
    VideoMediaRef,
)


def _records(*, clip_ready: bool = False) -> list[dict]:
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    return [
        {
            "schema": ARCHIVE_CATALOG_SCHEMA,
            "record_type": "import_contract",
            "complete": True,
            "inputs": {
                "pr6_manifest_sha256": digest_a,
                "pr6_receipt_sha256": digest_b,
                "recommended_manifest_sha256": digest_a,
                "recommended_receipt_sha256": digest_b,
                "x_registry_sha256": digest_c,
                "twitch_discovery_sha256": digest_c,
            },
            "item_identity": {
                "youtube": "provider video ID",
                "twitch": "numeric VOD/video ID without a leading v",
                "x": "post ID plus media ID",
            },
            "clip_ready_asserted": False,
            "twitch_discovery_item_IDs_available": False,
            "pumpfun_scope": "three retained thematic items only; no standalone archive",
        },
        {
            "schema": ARCHIVE_CATALOG_SCHEMA,
            "record_type": "item",
            "catalog_key": "twitch:456",
            "platform": "twitch",
            "provider_item_id": "456",
            "provider_media_id": None,
            "source_key": "twitch:id:123",
            "acquisition_state": "retained_remote_verified",
            "retained": True,
            "media_variants": [
                {
                    "dataset": "recommended-twitch-audio",
                    "source_manifest_sha256": digest_a,
                    "source_receipt_sha256": digest_b,
                    "row_id": "row-1",
                    "sha256": digest_c,
                    "bytes": 123,
                    "relative_path": "cented/456.m4a",
                    "remote_path": "archive/audio/cented/456.m4a",
                    "media_kind": "audio",
                    "container_suffix": ".m4a",
                    "complete_media": True,
                    "remote_sha256_verified": True,
                }
            ],
            "clip_candidate": False,
            "clip_ready": clip_ready,
            "clip_state": "audio_only_not_video_clip_ready",
            "topic_assertions": [],
            "evidence_ceiling": "manifest_and_remote_receipt_verified_media",
        },
        {
            "schema": ARCHIVE_CATALOG_SCHEMA,
            "record_type": "source",
            "source_key": "twitch:id:123",
            "platform": "twitch",
            "platform_entity_id": "123",
            "handle": "cented",
            "identity_state": "verified_platform_entity_id",
            "evidence_ceilings": ["remote_verified_reclaim_manifest_media"],
        },
        {
            "schema": ARCHIVE_CATALOG_SCHEMA,
            "record_type": "inventory_summary",
            "catalog_key": "twitch-source:123",
            "source_key": "twitch:id:123",
            "platform": "twitch",
            "acquisition_state": "pending_discovery",
            "auto_download_requested": True,
            "observed_feed_counts": {
                "videos": 1,
                "highlights": 2,
                "uploads": 3,
                "past_broadcasts": 4,
            },
            "observed_item_count": 10,
            "retained_item_ids_emitted": 0,
            "evidence_ceiling": "aggregate_feed_counts_without_provider_item_IDs_not_downloaded",
        },
    ]


def _packet(tmp_path: Path, records: list[dict]) -> tuple[Path, Path]:
    jsonl_path = tmp_path / "catalog.jsonl"
    payload = b"".join(
        (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    jsonl_path.write_bytes(payload)
    sidecar_path = tmp_path / "catalog.jsonl.sha256"
    sidecar_path.write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {jsonl_path.name}\n",
        encoding="ascii",
    )
    return jsonl_path, sidecar_path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_archive_packet_is_idempotent_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    jsonl_path, sidecar_path = _packet(tmp_path, _records())
    with Session(engine) as session:
        first = load_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
        )
        session.commit()
        second = load_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
        )
        session.commit()

        assert first["counts"]["items_created"] == 1
        assert first["counts"]["inventory_summaries"] == 1
        assert second["counts"]["items_created"] == 0
        assert second["counts"]["items_unchanged"] == 1
        assert session.scalar(select(func.count()).select_from(SourceChannel)) == 1
        assert session.scalar(select(func.count()).select_from(SourceVideo)) == 1
        assert session.scalar(select(func.count()).select_from(MediaObject)) == 1
        assert session.scalar(select(func.count()).select_from(VideoMediaRef)) == 1
        assert session.scalar(select(func.count()).select_from(MediaLocation)) == 1

        video = session.execute(select(SourceVideo)).scalar_one()
        assert video.archive_state == "retained_remote_verified"
        assert video.clip_candidate is False
        assert video.clip_ready is False
        assert video.metadata_json["archive_import"]["evidence_ceiling"] == (
            "manifest_and_remote_receipt_verified_media"
        )
        media = session.execute(select(MediaObject)).scalar_one()
        assert media.metadata_json["archive_evidence"][0]["row_id"] == "row-1"
        location = session.execute(select(MediaLocation)).scalar_one()
        assert location.backend == "storagebox"
        assert location.status == "active"
        assert location.verified_at is not None

        receipt_path, receipt_sha256 = write_archive_catalog_receipt(
            first,
            receipt_dir=tmp_path / "receipts",
        )
        assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == receipt_sha256
        assert receipt_path.stat().st_mode & 0o222 == 0


def test_archive_packet_rejects_bad_sidecar_and_clip_ready(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    jsonl_path, sidecar_path = _packet(tmp_path, _records())
    sidecar_path.write_text(f"{'0' * 64}  {jsonl_path.name}\n", encoding="ascii")
    with (
        Session(engine) as session,
        pytest.raises(ArchiveCatalogError, match="does not match"),
    ):
        load_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
        )

    jsonl_path, sidecar_path = _packet(tmp_path, _records(clip_ready=True))
    with (
        Session(engine) as session,
        pytest.raises(ArchiveCatalogError, match="clip_ready"),
    ):
        load_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
        )


def test_archive_packet_rejects_self_consistent_tamper_without_release_pin(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    jsonl_path, _ = _packet(tmp_path, _records())
    pinned_digest = _digest(jsonl_path)
    tampered = _records()
    tampered[1]["catalog_key"] = "twitch:tampered"
    jsonl_path, sidecar_path = _packet(tmp_path, tampered)

    with (
        Session(engine) as session,
        pytest.raises(ArchiveCatalogError, match="caller-pinned"),
    ):
        load_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=pinned_digest,
        )
