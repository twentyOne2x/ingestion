from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer import archive_admin
from src.ingest_v2.cloud.diarization_indexer.archive_admin import (
    FFPROBE_VIDEO_PROOF_SCHEMA,
    HOT_MEDIA_HYDRATION_SOURCE_SCHEMA,
    ArchiveAdminError,
    claim_archive_sources,
    register_hot_media_hydration,
)
from src.ingest_v2.cloud.diarization_indexer.archive_catalog_loader import (
    ARCHIVE_CATALOG_SCHEMA,
    ArchiveCatalogError,
    apply_archive_catalog,
    load_archive_catalog,
    write_archive_catalog_receipt,
)
from src.ingest_v2.cloud.diarization_indexer.archive_receipts import (
    ArchiveProtocolError,
    write_immutable_json_receipt,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    ArchiveCatalogImport,
    ArchiveHydrationRegistration,
    ArchiveTenantClaim,
    Base,
    MediaLocation,
    MediaObject,
    SourceChannel,
    SourceVideo,
    Tenant,
    TenantChannelEntitlement,
    TenantMembership,
    UserAccount,
    VideoMediaRef,
)

TENANT_A = f"ten_{'1' * 64}"
TENANT_B = f"ten_{'2' * 64}"
ADMIN_A = f"usr_{'3' * 64}"
ADMIN_B = f"usr_{'4' * 64}"


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


def _engine(path: Path | None = None):
    url = (
        "sqlite+pysqlite:///:memory:" if path is None else f"sqlite+pysqlite:///{path}"
    )
    engine = create_engine(
        url,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(engine)
    return engine


def _apply_packet(engine, tmp_path: Path, records: list[dict] | None = None):
    jsonl_path, sidecar_path = _packet(tmp_path, records or _records())
    with Session(engine) as session:
        result = apply_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
            receipt_dir=tmp_path / "receipts",
        )
        session.commit()
    return result, jsonl_path, sidecar_path


def _seed_admins(engine) -> None:
    with Session(engine) as session:
        session.add_all(
            [
                Tenant(id=TENANT_A, slug="tenant-a", display_name="Tenant A"),
                Tenant(id=TENANT_B, slug="tenant-b", display_name="Tenant B"),
                UserAccount(
                    id=ADMIN_A,
                    auth_provider="test",
                    auth_subject="admin-a",
                ),
                UserAccount(
                    id=ADMIN_B,
                    auth_provider="test",
                    auth_subject="admin-b",
                ),
                TenantMembership(
                    tenant_id=TENANT_A,
                    user_id=ADMIN_A,
                    role="admin",
                    status="active",
                ),
                TenantMembership(
                    tenant_id=TENANT_B,
                    user_id=ADMIN_B,
                    role="owner",
                    status="active",
                ),
            ]
        )
        session.commit()


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


def test_catalog_receipt_failure_rolls_back_every_database_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine()
    jsonl_path, sidecar_path = _packet(tmp_path, _records())

    def fail_receipt(*_args, **_kwargs):
        raise ArchiveCatalogError("injected receipt write failure")

    monkeypatch.setattr(
        "src.ingest_v2.cloud.diarization_indexer.archive_catalog_loader.write_archive_catalog_receipt",
        fail_receipt,
    )
    with (
        Session(engine) as session,
        pytest.raises(ArchiveCatalogError, match="injected receipt"),
    ):
        apply_archive_catalog(
            session,
            jsonl_path=jsonl_path,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
            receipt_dir=tmp_path / "receipts",
        )

    with Session(engine) as session:
        for model in (
            ArchiveCatalogImport,
            MediaLocation,
            MediaObject,
            SourceChannel,
            SourceVideo,
            VideoMediaRef,
        ):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_catalog_rejects_symlink_input_before_database_effect(tmp_path: Path) -> None:
    engine = _engine()
    jsonl_path, sidecar_path = _packet(tmp_path, _records())
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(jsonl_path)
    with (
        Session(engine) as session,
        pytest.raises(ArchiveCatalogError, match="symlink"),
    ):
        apply_archive_catalog(
            session,
            jsonl_path=linked,
            sidecar_path=sidecar_path,
            expected_jsonl_sha256=_digest(jsonl_path),
            receipt_dir=tmp_path / "receipts",
        )
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(SourceChannel)) == 0


def test_catalog_duplicate_and_concurrent_replay_share_one_ledger(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "archive.sqlite3"
    engine = _engine(database_path)
    jsonl_path, sidecar_path = _packet(tmp_path, _records())
    digest = _digest(jsonl_path)
    barrier = Barrier(2)

    def apply_once() -> tuple[str, bool]:
        with Session(engine) as session:
            barrier.wait(timeout=10)
            result = apply_archive_catalog(
                session,
                jsonl_path=jsonl_path,
                sidecar_path=sidecar_path,
                expected_jsonl_sha256=digest,
                receipt_dir=tmp_path / "receipts",
            )
            session.commit()
            return result.receipt_sha256, result.reconciled

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: apply_once(), range(2)))

    assert len({result[0] for result in results}) == 1
    assert sorted(result[1] for result in results) == [False, True]
    with Session(engine) as session:
        assert (
            session.scalar(select(func.count()).select_from(ArchiveCatalogImport)) == 1
        )
        assert session.scalar(select(func.count()).select_from(SourceChannel)) == 1


def test_tenant_claim_is_admin_bound_replay_safe_and_cross_tenant(
    tmp_path: Path,
) -> None:
    engine = _engine()
    applied, jsonl_path, _sidecar_path = _apply_packet(engine, tmp_path)
    _seed_admins(engine)
    digest = _digest(jsonl_path)

    with Session(engine) as session:
        first = claim_archive_sources(
            session,
            catalog_jsonl_sha256=digest,
            tenant_id=TENANT_A,
            admin_user_id=ADMIN_A,
            idempotency_key="claim-cented-v1",
            source_keys=["twitch:id:123"],
            receipt_dir=tmp_path / "claim-receipts",
        )
        session.commit()
    with Session(engine) as session:
        replay = claim_archive_sources(
            session,
            catalog_jsonl_sha256=digest,
            tenant_id=TENANT_A,
            admin_user_id=ADMIN_A,
            idempotency_key="claim-cented-v1",
            source_keys=["twitch:id:123"],
            receipt_dir=tmp_path / "claim-receipts",
        )
        session.commit()
    assert first.reconciled is False
    assert replay.reconciled is True
    assert first.receipt_sha256 == replay.receipt_sha256
    assert applied.receipt_sha256 == first.receipt["catalog"]["receipt_sha256"]

    with (
        Session(engine) as session,
        pytest.raises(ArchiveAdminError, match="active tenant admin"),
    ):
        claim_archive_sources(
            session,
            catalog_jsonl_sha256=digest,
            tenant_id=TENANT_B,
            admin_user_id=ADMIN_A,
            idempotency_key="spoofed-cross-tenant",
            source_keys=["twitch:id:123"],
            receipt_dir=tmp_path / "claim-receipts",
        )

    with Session(engine) as session:
        second_tenant = claim_archive_sources(
            session,
            catalog_jsonl_sha256=digest,
            tenant_id=TENANT_B,
            admin_user_id=ADMIN_B,
            idempotency_key="claim-cented-v1",
            source_keys=["twitch:id:123"],
            receipt_dir=tmp_path / "claim-receipts",
        )
        session.commit()
    assert second_tenant.receipt["target"]["tenant_id"] == TENANT_B
    with Session(engine) as session:
        assert (
            session.scalar(select(func.count()).select_from(TenantChannelEntitlement))
            == 2
        )
        assert session.scalar(select(func.count()).select_from(ArchiveTenantClaim)) == 2


def test_tenant_claim_concurrent_replay_and_receipt_collision(tmp_path: Path) -> None:
    database_path = tmp_path / "claims.sqlite3"
    engine = _engine(database_path)
    _applied, jsonl_path, _sidecar_path = _apply_packet(engine, tmp_path)
    _seed_admins(engine)
    digest = _digest(jsonl_path)
    barrier = Barrier(2)

    def claim_once() -> tuple[str, bool]:
        with Session(engine) as session:
            barrier.wait(timeout=10)
            result = claim_archive_sources(
                session,
                catalog_jsonl_sha256=digest,
                tenant_id=TENANT_A,
                admin_user_id=ADMIN_A,
                idempotency_key="concurrent-claim",
                source_keys=["twitch:id:123"],
                receipt_dir=tmp_path / "claim-receipts",
            )
            session.commit()
            return result.receipt_sha256, result.reconciled

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: claim_once(), range(2)))
    assert len({result[0] for result in results}) == 1
    assert sorted(result[1] for result in results) == [False, True]

    receipt_path = next((tmp_path / "claim-receipts").glob("*.json"))
    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="ascii")
    receipt_path.chmod(0o444)
    with (
        Session(engine) as session,
        pytest.raises(ArchiveProtocolError, match="caller-pinned|collision|SHA-256"),
    ):
        claim_archive_sources(
            session,
            catalog_jsonl_sha256=digest,
            tenant_id=TENANT_A,
            admin_user_id=ADMIN_A,
            idempotency_key="concurrent-claim",
            source_keys=["twitch:id:123"],
            receipt_dir=tmp_path / "claim-receipts",
        )


def test_hydration_registration_revalidates_cas_and_marks_video_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic-video-for-hash-and-registration"
    media_digest = hashlib.sha256(payload).hexdigest()
    records = _records()
    variant = records[1]["media_variants"][0]
    variant.update(
        {
            "sha256": media_digest,
            "bytes": len(payload),
            "media_kind": "video",
            "container_suffix": ".mp4",
            "relative_path": f"cented/456/{media_digest}.mp4",
            "remote_path": f"archive/video/{media_digest}.mp4",
        }
    )
    records[1]["clip_candidate"] = True
    records[1]["clip_state"] = "remote_video_requires_hot_hydration"
    engine = _engine()
    _apply_packet(engine, tmp_path, records)

    hot_root = tmp_path / "hot-media"
    cas_path = hot_root / "sha256" / media_digest[:2] / f"{media_digest}.mp4"
    cas_path.parent.mkdir(parents=True)
    cas_path.write_bytes(payload)
    cas_path.chmod(0o440)
    proof = {
        "schema": FFPROBE_VIDEO_PROOF_SCHEMA,
        "codec_names": ["h264"],
        "duration_ms": 1234,
        "format_names": ["mp4"],
        "max_height": 1080,
        "max_width": 1920,
        "video_stream_count": 1,
    }
    monkeypatch.setattr(archive_admin, "_run_ffprobe", lambda *_args, **_kwargs: proof)
    source_receipt = tmp_path / "hydration-source.json"
    source_receipt.write_text(
        json.dumps(
            {
                "schema": HOT_MEDIA_HYDRATION_SOURCE_SCHEMA,
                "cas_path": str(cas_path),
                "ffprobe": proof,
                "media_sha256": media_digest,
                "mime_type": "video/mp4",
                "size_bytes": len(payload),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    source_digest = _digest(source_receipt)

    with Session(engine) as session:
        first = register_hot_media_hydration(
            session,
            source_receipt_path=source_receipt,
            expected_source_receipt_sha256=source_digest,
            hot_media_root=hot_root,
            receipt_dir=tmp_path / "hydration-receipts",
            ffprobe_bin=Path("/usr/bin/ffprobe"),
        )
        session.commit()
    with Session(engine) as session:
        replay = register_hot_media_hydration(
            session,
            source_receipt_path=source_receipt,
            expected_source_receipt_sha256=source_digest,
            hot_media_root=hot_root,
            receipt_dir=tmp_path / "hydration-receipts",
            ffprobe_bin=Path("/usr/bin/ffprobe"),
        )
        session.commit()
        video = session.execute(select(SourceVideo)).scalar_one()
        location = session.execute(
            select(MediaLocation).where(MediaLocation.backend == "hot_local")
        ).scalar_one()
        assert video.clip_ready is True
        assert video.archive_state == "retained_hot_verified"
        assert location.location_key == str(cas_path)
        assert (
            session.scalar(
                select(func.count()).select_from(ArchiveHydrationRegistration)
            )
            == 1
        )
    assert first.reconciled is False
    assert replay.reconciled is True
    assert first.receipt_sha256 == replay.receipt_sha256


def test_immutable_receipt_detects_exact_destination_collision(tmp_path: Path) -> None:
    payload = {"schema": "test.receipt.v1", "value": 1}
    _path, digest = write_immutable_json_receipt(
        payload, receipt_dir=tmp_path, schema="test.receipt.v1"
    )
    destination = tmp_path / f"test.receipt.v1-{digest}.json"
    destination.chmod(0o644)
    destination.write_text("{}\n", encoding="ascii")
    destination.chmod(0o444)
    with pytest.raises(ArchiveProtocolError, match="caller-pinned|collision|SHA-256"):
        write_immutable_json_receipt(
            payload, receipt_dir=tmp_path, schema="test.receipt.v1"
        )


def test_archive_admin_cli_exposes_internal_subcommands() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/load_archive_catalog.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert "apply" in completed.stdout
    assert "claim" in completed.stdout
    assert "register-hydration" in completed.stdout
