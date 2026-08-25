from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

import src.ingest_v2.cloud.diarization_indexer.public_ingestion_worker as worker
from alembic import command
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    SYSTEM_COMMERCE_SCOPE,
    Base,
    ChannelOrder,
    ChannelPack,
    ChannelQuote,
    CheckoutSessionRecord,
    IngestionJob,
    IngestionRequest,
    PackBatch,
    PackVideo,
    SourceChannel,
    SourceVideo,
    Tenant,
    TenantMembership,
    TranscriptRevision,
    TranscriptSegment,
    UserAccount,
    commerce_ownership_values,
    dispose_engine,
    gateway_commerce_scope,
    get_engine,
    init_db,
    set_commerce_scope,
    utcnow,
)


def test_one_tenant_job_fans_out_to_every_requesting_principal(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'fanout.sqlite3'}")
    Base.metadata.create_all(engine)
    tenant_id = f"ten_{'a' * 64}"
    principals = (f"usr_{'a' * 64}", f"usr_{'b' * 64}")
    with Session(engine) as session:
        session.add(Tenant(id=tenant_id, slug="fanout", display_name="Fanout"))
        for principal in principals:
            session.add(
                UserAccount(
                    id=principal,
                    auth_provider="test",
                    auth_subject=principal,
                )
            )
            session.add(
                TenantMembership(
                    tenant_id=tenant_id,
                    user_id=principal,
                    role="member",
                )
            )
        job = IngestionJob(
            id="job_same_tenant",
            dedupe_key="same-tenant-public-item",
            job_kind="public_item_ingestion",
            source_kind="youtube",
            source_key="UCexample:dQw4w9WgXcQ",
            pipeline_version="test-v1",
            request_tenant_ids_json=[tenant_id],
        )
        session.add(job)
        session.flush()
        for index, principal in enumerate(principals, start=1):
            session.add(
                IngestionRequest(
                    id=f"request_{index}",
                    tenant_id=tenant_id,
                    requested_by_user_id=principal,
                    job_id=job.id,
                    idempotency_key=f"same-tenant-{index}",
                    request_fingerprint=f"{index:064x}",
                    request_json={},
                )
            )
        session.flush()

        rows = worker._tenant_requests(session, job)

    assert [row.requested_by_user_id for row in rows] == list(principals)


def _seed_ready_paid_pack(
    session: Session,
    *,
    suffix: str = "export_lock",
    scope=SYSTEM_COMMERCE_SCOPE,
) -> worker._PaidPackExportTarget:
    ownership = commerce_ownership_values(scope)
    video_external_id = f"video_{suffix}"
    quote = ChannelQuote(
        **ownership,
        id=f"quote_{suffix}",
        status="open",
        mode="recent_pack",
        namespace="videos",
        channel_handle=f"@{suffix}",
        resolved_channel_id=f"UC{suffix}",
        resolved_channel_name="Example",
        requested_max_videos=1,
        included_video_count=1,
        excluded_video_count=0,
        current_batch_index=1,
        current_batch_video_count=1,
        current_batch_amount_cents=100,
        total_included_amount_cents=100,
        per_video_cents=100,
        estimated_ready_minutes=1,
        eta_confidence="high",
        recommended_starter_batch_size=1,
        planning_latency_ms=1,
        request_json={"language": "en", "prefer_auto": True, "pack_id": None},
        batch_plan_json=[],
        price_breakdown_json={"currency": "USD", "amount_cents": 100},
        commerce_json={},
        expires_at=utcnow() + timedelta(minutes=30),
    )
    checkout = CheckoutSessionRecord(
        **ownership,
        id=f"checkout_{suffix}",
        status="completed",
        idempotency_key=f"export-lock-{suffix}",
        currency="USD",
        total_amount_cents=100,
        quote_ids_json=[quote.id],
        line_items_json=[{"quote_id": quote.id, "amount_cents": 100}],
        payment_provider="x402",
        payment_status="settled_x402",
    )
    pack = ChannelPack(
        **ownership,
        id=f"pack_{suffix}",
        status="partial",
        mode="recent_pack",
        namespace="videos",
        channel_handle=f"@{suffix}",
        resolved_channel_id=f"UC{suffix}",
        resolved_channel_name="Example",
        total_purchased_video_count=1,
        ready_video_count=1,
        batch_count=1,
    )
    batch = PackBatch(
        **ownership,
        id=f"batch_{suffix}",
        pack_id=pack.id,
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        batch_index=1,
        status="partial",
        billable_video_count=1,
        ready_video_count=1,
        amount_cents=100,
        estimated_ready_minutes=1,
        build_notes_json={},
    )
    order = ChannelOrder(
        **ownership,
        id=f"order_{suffix}",
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        pack_id=pack.id,
        batch_id=batch.id,
        status="partial",
        payment_status="settled_x402",
        payment_provider="x402",
        amount_cents=100,
        currency="USD",
        notes_json={},
    )
    channel = SourceChannel(
        id=f"channel_{suffix}",
        platform="youtube",
        external_id=f"UC{suffix}",
        handle=f"@{suffix}",
    )
    source = SourceVideo(
        id=f"source_{suffix}",
        channel_id=channel.id,
        platform="youtube",
        external_id=video_external_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_external_id}",
        title="Example",
        duration_ms=2_000,
        archive_state="retained_hot_verified",
        clip_candidate=True,
        clip_ready=True,
    )
    revision = TranscriptRevision(
        id=f"revision_{suffix}",
        video_id=source.id,
        provider="local_cpu:test@fixture",
        provider_revision_id=f"fixture-revision-{suffix}",
        language="en",
        content_sha256=hashlib.sha256(
            f"one transcript {suffix}".encode()
        ).hexdigest(),
        is_current=True,
    )
    segment = TranscriptSegment(
        id=f"segment_{suffix}",
        revision_id=revision.id,
        ordinal=0,
        start_ms=0,
        end_ms=2_000,
        text=f"One canonical transcript for {suffix}.",
    )
    pack_video = PackVideo(
        **ownership,
        pack_id=pack.id,
        batch_id=batch.id,
        quote_id=quote.id,
        position=1,
        video_id=source.external_id,
        title=source.title,
        channel_name="Example",
        channel_handle=f"@{suffix}",
        duration_s=2.0,
        video_url=source.canonical_url,
        transcript_source=revision.provider,
        indexed_parent_id=source.id,
        status="ready",
    )
    # Explicit stages exercise the same database lineage triggers as production
    # instead of relying on ORM unit-of-work ordering between unrelated objects.
    session.add_all([quote, checkout, pack, channel])
    session.flush()
    session.add_all([batch, source])
    session.flush()
    session.add_all([order, revision])
    session.flush()
    session.add_all([segment, pack_video])
    session.flush()
    return worker._PaidPackExportTarget(
        scope=scope,
        order_id=order.id,
        pack_id=pack.id,
        batch_id=batch.id,
        quote_id=quote.id,
    )


def _add_ready_second_batch(
    session: Session, *, target: worker._PaidPackExportTarget
) -> tuple[PackBatch, ChannelOrder]:
    ownership = commerce_ownership_values(SYSTEM_COMMERCE_SCOPE)
    pack = session.get(ChannelPack, target.pack_id)
    first_batch = session.get(PackBatch, target.batch_id)
    first_order = session.get(ChannelOrder, target.order_id)
    assert pack is not None and first_batch is not None and first_order is not None
    quote = ChannelQuote(
        **ownership,
        id=f"quote_{target.pack_id}_second",
        status="open",
        mode=pack.mode,
        namespace=pack.namespace,
        channel_handle=pack.channel_handle,
        resolved_channel_id=pack.resolved_channel_id,
        resolved_channel_name=pack.resolved_channel_name,
        requested_max_videos=1,
        included_video_count=1,
        excluded_video_count=0,
        current_batch_index=2,
        current_batch_video_count=1,
        current_batch_amount_cents=100,
        total_included_amount_cents=100,
        per_video_cents=100,
        estimated_ready_minutes=1,
        eta_confidence="high",
        recommended_starter_batch_size=1,
        planning_latency_ms=1,
        request_json={
            "language": "en",
            "prefer_auto": True,
            "pack_id": pack.id,
        },
        batch_plan_json=[],
        price_breakdown_json={"currency": "USD", "amount_cents": 100},
        commerce_json={},
        expires_at=utcnow() + timedelta(minutes=30),
    )
    checkout = CheckoutSessionRecord(
        **ownership,
        id=f"checkout_{target.pack_id}_second",
        status="completed",
        idempotency_key=f"export-lock-{target.pack_id}-second",
        currency="USD",
        total_amount_cents=100,
        quote_ids_json=[quote.id],
        line_items_json=[{"quote_id": quote.id, "amount_cents": 100}],
        payment_provider="x402",
        payment_status="settled_x402",
    )
    batch = PackBatch(
        **ownership,
        id=f"batch_{target.pack_id}_second",
        pack_id=pack.id,
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        batch_index=2,
        status="partial",
        billable_video_count=1,
        ready_video_count=1,
        amount_cents=100,
        estimated_ready_minutes=1,
        build_notes_json={},
    )
    order = ChannelOrder(
        **ownership,
        id=f"order_{target.pack_id}_second",
        quote_id=quote.id,
        checkout_session_id=checkout.id,
        pack_id=pack.id,
        batch_id=batch.id,
        status="partial",
        payment_status="settled_x402",
        payment_provider="x402",
        amount_cents=100,
        currency="USD",
        notes_json={},
    )
    channel = SourceChannel(
        id=f"channel_{target.pack_id}_second",
        platform="youtube",
        external_id=f"UC{target.pack_id}second",
        handle=f"@{target.pack_id}second",
    )
    source = SourceVideo(
        id=f"source_{target.pack_id}_second",
        channel_id=channel.id,
        platform="youtube",
        external_id=f"video_{target.pack_id}_second",
        canonical_url=(
            "https://www.youtube.com/watch?v="
            f"video_{target.pack_id}_second"
        ),
        title="Second batch video",
        duration_ms=3_000,
        archive_state="retained_hot_verified",
        clip_candidate=True,
        clip_ready=True,
    )
    revision = TranscriptRevision(
        id=f"revision_{target.pack_id}_second",
        video_id=source.id,
        provider="local_cpu:test@fixture",
        provider_revision_id=f"fixture-{target.pack_id}-second",
        language="en",
        content_sha256=hashlib.sha256(b"second transcript").hexdigest(),
        is_current=True,
    )
    segment = TranscriptSegment(
        id=f"segment_{target.pack_id}_second",
        revision_id=revision.id,
        ordinal=0,
        start_ms=0,
        end_ms=3_000,
        text="Second canonical transcript.",
    )
    pack_video = PackVideo(
        **ownership,
        pack_id=pack.id,
        batch_id=batch.id,
        quote_id=quote.id,
        position=1,
        video_id=source.external_id,
        title=source.title,
        channel_name="Example",
        channel_handle=channel.handle,
        duration_s=3.0,
        video_url=source.canonical_url,
        transcript_source=revision.provider,
        indexed_parent_id=source.id,
        status="ready",
    )
    session.add_all([quote, checkout, channel])
    session.flush()
    session.add_all([batch, source])
    session.flush()
    session.add_all([order, revision])
    session.flush()
    session.add_all([segment, pack_video])
    pack.batch_count = 2
    pack.total_purchased_video_count = 2
    pack.ready_video_count = 2
    pack.status = "partial"
    first_batch.status = "partial"
    first_order.status = "partial"
    session.flush()
    return batch, order


def test_paid_pack_files_are_built_before_commerce_rows_are_locked(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "export-lock.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()

        row_lock_phase = False
        original_rows = worker._paid_export_rows
        original_builder = worker._build_pack_artifacts
        advertised_files: dict[Path, bytes] = {}
        advertised_generation: str | None = None

        def observed_rows(session, *, target, lock):
            nonlocal row_lock_phase
            if lock:
                row_lock_phase = True
            return original_rows(session, target=target, lock=lock)

        def observed_builder(**kwargs):
            assert row_lock_phase is False
            if advertised_generation is not None:
                assert kwargs["artifact_generation"] != advertised_generation
                assert all(
                    path.read_bytes() == data
                    for path, data in advertised_files.items()
                )
            result = original_builder(**kwargs)
            if advertised_generation is not None:
                assert all(
                    path.read_bytes() == data
                    for path, data in advertised_files.items()
                )
            return result

        monkeypatch.setattr(worker, "_paid_export_rows", observed_rows)
        monkeypatch.setattr(worker, "_build_pack_artifacts", observed_builder)

        worker._finalize_paid_pack_export(target)

        assert row_lock_phase is True
        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            order = session.get(ChannelOrder, target.order_id)
            assert pack is not None and order is not None
            assert (pack.status, order.status) == ("ready", "ready")
            receipt = dict(pack.export_paths_json or {})
            for key in ("manifest", "videos", "links", "transcripts", "archive"):
                path = Path(receipt[f"{key}_path"])
                assert path.is_file()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt[
                    f"{key}_sha256"
                ]
            assert session.scalars(select(PackVideo)).one().indexed_parent_id

        advertised_generation = Path(receipt["manifest_path"]).parent.name
        advertised_files.update(
            {
                Path(receipt[f"{key}_path"]): Path(
                    receipt[f"{key}_path"]
                ).read_bytes()
                for key in ("manifest", "videos", "links", "transcripts")
            }
        )

        # A replay never trusts path/hash strings under the row lock, but it
        # also does not pre-demote an already-ready pack before unlocked
        # verification. A missing file is repaired and reattached.
        with Session(engine) as session:
            state = worker._paid_export_rows(
                session, target=target, lock=True
            )
            assert worker._aggregate_paid_pack(
                session,
                scope=target.scope,
                order=next(
                    order for order in state.orders if order.id == target.order_id
                ),
                pack=state.pack,
                batch=next(
                    batch for batch in state.batches if batch.id == target.batch_id
                ),
                quote=state.quotes_by_id[target.quote_id],
            )
            session.commit()
            assert state.pack.status == "ready"
        archive_path = Path(receipt["archive_path"])
        archive_path.unlink()
        row_lock_phase = False
        worker._finalize_paid_pack_export(target)
        assert not archive_path.exists()
        with Session(engine) as session:
            repaired = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
        repaired_archive = Path(repaired["archive_path"])
        assert repaired_archive.parent.name != advertised_generation
        assert repaired_archive.is_file()
        assert hashlib.sha256(repaired_archive.read_bytes()).hexdigest() == repaired[
            "archive_sha256"
        ]
        assert all(
            path.read_bytes() == data
            for path, data in advertised_files.items()
        )
    finally:
        dispose_engine()


def test_fresh_crash_orphan_is_collected_on_successful_retry(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "crash-orphan.sqlite3"
    export_root = tmp_path / "exports"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()

        orphan = (
            export_root
            / target.pack_id
            / f"paid-{'a' * 64}-{'b' * 32}"
        )
        orphan.mkdir(parents=True)
        (orphan / "manifest.json").write_text("crash-partial", encoding="utf-8")
        assert (
            time.time() - orphan.stat().st_mtime
            < worker._PAID_PACK_GENERATION_GC_MIN_AGE_SECONDS
        )

        worker._finalize_paid_pack_export(target)

        assert not orphan.exists()
        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "ready"
            advertised = Path((pack.export_paths_json or {})["manifest_path"]).parent
            assert advertised.is_dir()
            assert advertised != orphan
            assert (advertised / worker._PAID_PACK_PUBLISHED_MARKER).is_file()
            jobs = list(
                session.scalars(
                    select(IngestionJob).where(
                        IngestionJob.job_kind == "paid_pack_export_gc"
                    )
                )
            )
            assert jobs == []
    finally:
        dispose_engine()


def test_durable_gc_job_reaps_aged_retired_root_and_preserves_advertised_root(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "advertised-generation.sqlite3"
    export_root = tmp_path / "exports"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()
        worker._finalize_paid_pack_export(target)
        with Session(engine) as session:
            receipt = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
        retired = Path(receipt["manifest_path"]).parent
        retained_retired_bytes = {
            path.name: path.read_bytes()
            for path in retired.iterdir()
            if path.is_file() and path.name != Path(receipt["archive_path"]).name
        }
        Path(receipt["archive_path"]).unlink()

        # Repair atomically advertises a fresh generation. The former root is
        # marked as once-published and remains available during its reader
        # grace period; the same transaction queues the durable delayed sweep.
        real_utcnow = worker.utcnow
        retirement_clock = real_utcnow() - timedelta(hours=2)
        monkeypatch.setattr(worker, "utcnow", lambda: retirement_clock)
        worker._finalize_paid_pack_export(target)
        monkeypatch.setattr(worker, "utcnow", real_utcnow)

        with Session(engine) as session:
            replay = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
            jobs = list(
                session.scalars(
                    select(IngestionJob).where(
                        IngestionJob.job_kind == "paid_pack_export_gc"
                    )
                )
            )
        replacement = Path(replay["manifest_path"]).parent
        assert replacement != retired
        assert retired.is_dir()
        assert {
            path.name: path.read_bytes()
            for path in retired.iterdir()
            if path.is_file() and path.name != Path(receipt["archive_path"]).name
        } == retained_retired_bytes
        assert len(jobs) == 1
        delayed = jobs[0]
        assert delayed.next_run_at is not None
        assert delayed.payload_json["retiredGeneration"] == retired.name
        assert delayed.payload_json["replacementGeneration"] == replacement.name

        # A second immediate repair must not opportunistically delete the
        # just-retired first root, even if that root itself is old. Only its
        # exact grace-expiry job may reclaim it.
        Path(replay["archive_path"]).unlink()
        worker._finalize_paid_pack_export(target)
        assert retired.is_dir()
        assert replacement.is_dir()
        with Session(engine) as session:
            current = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
            jobs = list(
                session.scalars(
                    select(IngestionJob).where(
                        IngestionJob.job_kind == "paid_pack_export_gc"
                    )
                )
            )
        advertised = Path(current["manifest_path"]).parent
        assert advertised not in {retired, replacement}
        assert len(jobs) == 2

        assert worker.process_next_public_ingestion_job(
            worker_id="gc-test-worker"
        )

        assert not retired.exists()
        assert replacement.is_dir()
        assert advertised.is_dir()
        advertised_bytes = {
            path.name: path.read_bytes()
            for path in advertised.iterdir()
            if path.is_file()
        }
        with Session(engine) as session:
            after_gc = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
            completed = session.get(IngestionJob, delayed.id)
            assert completed is not None and completed.status == "succeeded"
            assert completed.result_json["deleted_generations"] == [retired.name]
        assert Path(after_gc["manifest_path"]).parent == advertised
        assert {
            path.name: path.read_bytes()
            for path in advertised.iterdir()
            if path.is_file()
        } == advertised_bytes
    finally:
        dispose_engine()


def test_generation_gc_preserves_everything_when_db_paths_are_ambiguous(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "ambiguous-generation.sqlite3"
    export_root = tmp_path / "exports"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()
            ambiguous = (
                export_root
                / target.pack_id
                / f"paid-{'e' * 64}-{'f' * 32}"
            )
            ambiguous.mkdir(parents=True)
            manifest_path = ambiguous / "manifest.json"
            manifest_path.write_text("ambiguous", encoding="utf-8")
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None
            pack.export_paths_json = {"manifest_path": str(manifest_path)}
            session.commit()
        old = time.time() - worker._PAID_PACK_GENERATION_GC_MIN_AGE_SECONDS - 60
        os.utime(ambiguous, (old, old))

        worker._finalize_paid_pack_export(target)

        assert ambiguous.is_dir()
        assert manifest_path.read_text(encoding="utf-8") == "ambiguous"
        with Session(engine) as session:
            repaired = session.get(ChannelPack, target.pack_id)
            assert repaired is not None and repaired.status == "ready"
            assert Path((repaired.export_paths_json or {})["manifest_path"]).parent != ambiguous
    finally:
        dispose_engine()


def test_failed_reconciliation_queues_grace_for_cleared_advertised_root(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "failed-retirement.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()
        worker._finalize_paid_pack_export(target)
        with Session(engine) as session:
            receipt = dict(
                session.get(ChannelPack, target.pack_id).export_paths_json or {}
            )
        retired = Path(receipt["manifest_path"]).parent

        real_utcnow = worker.utcnow
        retirement_clock = real_utcnow() - timedelta(hours=2)
        monkeypatch.setattr(worker, "utcnow", lambda: retirement_clock)

        def fail_snapshot(*_args, **_kwargs):
            raise RuntimeError("simulated post-publication reconciliation failure")

        monkeypatch.setattr(
            worker,
            "_paid_export_snapshot",
            fail_snapshot,
        )
        with pytest.raises(RuntimeError, match="post-publication"):
            worker._finalize_paid_pack_export(target)
        monkeypatch.setattr(worker, "utcnow", real_utcnow)

        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "partial"
            assert pack.export_paths_json is None
            gc_job = session.scalar(
                select(IngestionJob).where(
                    IngestionJob.job_kind == "paid_pack_export_gc"
                )
            )
            assert gc_job is not None
            gc_job_id = gc_job.id
            assert gc_job.payload_json["retiredGeneration"] == retired.name
            assert gc_job.payload_json["replacementGeneration"] is None

        assert retired.is_dir()
        assert worker.process_next_public_ingestion_job(
            worker_id="failed-retirement-gc-worker"
        )
        assert not retired.exists()
        with Session(engine) as session:
            completed = session.get(IngestionJob, gc_job_id)
            assert completed is not None and completed.status == "succeeded"
    finally:
        dispose_engine()


def test_multi_batch_export_is_deterministic_and_reconciles_every_order(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "multi-batch.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            second_batch, second_order = _add_ready_second_batch(
                session, target=target
            )
            second_ids = (second_batch.id, second_order.id)
            session.commit()

        worker._finalize_paid_pack_export(target)

        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "ready"
            batches = list(
                session.scalars(
                    select(PackBatch)
                    .where(PackBatch.pack_id == pack.id)
                    .order_by(PackBatch.batch_index.asc())
                )
            )
            orders = list(
                session.scalars(
                    select(ChannelOrder)
                    .where(ChannelOrder.pack_id == pack.id)
                    .order_by(ChannelOrder.id.asc())
                )
            )
            assert [(row.batch_index, row.status) for row in batches] == [
                (1, "ready"),
                (2, "ready"),
            ]
            assert {row.status for row in orders} == {"ready"}
            assert session.get(PackBatch, second_ids[0]).status == "ready"
            assert session.get(ChannelOrder, second_ids[1]).status == "ready"
            receipt = dict(pack.export_paths_json or {})
            manifest = json.loads(Path(receipt["manifest_path"]).read_text())
            assert manifest["batch_count"] == 2
            assert manifest["latest_batch_index"] == 2
            video_rows = [
                json.loads(line)
                for line in Path(receipt["videos_path"])
                .read_text()
                .splitlines()
            ]
            # Both batches restart position at one. Batch index is therefore
            # part of the total, deterministic export order.
            assert [row["video_id"] for row in video_rows] == [
                "video_export_lock",
                "video_pack_export_lock_second",
            ]
    finally:
        dispose_engine()


def test_paid_export_rejects_same_owner_checkout_misbinding(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "checkout-misbinding.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            order = session.get(ChannelOrder, target.order_id)
            assert order is not None
            alternate = CheckoutSessionRecord(
                **commerce_ownership_values(SYSTEM_COMMERCE_SCOPE),
                id="checkout_export_lock_alternate",
                status="completed",
                idempotency_key="export-lock-alternate",
                currency="USD",
                total_amount_cents=100,
                quote_ids_json=[target.quote_id],
                line_items_json=[
                    {"quote_id": target.quote_id, "amount_cents": 100}
                ],
                payment_provider="x402",
                payment_status="settled_x402",
            )
            session.add(alternate)
            session.flush()
            order.checkout_session_id = alternate.id
            session.commit()

        with Session(engine) as session, pytest.raises(
            RuntimeError, match="export lineage is inconsistent"
        ):
            worker._paid_export_rows(session, target=target, lock=False)
    finally:
        dispose_engine()


def test_paid_export_excludes_retired_transcript_content(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "retired-transcript.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            revision = session.get(TranscriptRevision, "revision_export_lock")
            assert revision is not None
            session.add(
                TranscriptSegment(
                    id="segment_export_lock_retired",
                    revision_id=revision.id,
                    ordinal=1,
                    start_ms=2_000,
                    end_ms=3_000,
                    text="This retired text must never be exported.",
                    status="retired",
                )
            )
            session.commit()

        worker._finalize_paid_pack_export(target)
        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            transcript_path = Path((pack.export_paths_json or {})["transcripts_path"])
            exported = transcript_path.read_text()
            assert "One canonical transcript" in exported
            assert "retired text" not in exported
            revision = session.get(TranscriptRevision, "revision_export_lock")
            revision.status = "retired"
            session.commit()

        with pytest.raises(
            RuntimeError,
            match="cannot export before every canonical transcript exists",
        ):
            worker._finalize_paid_pack_export(target)
    finally:
        dispose_engine()


def test_paid_builder_uses_only_frozen_rows_and_never_qdrant_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "frozen-builder.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()

        original_builder = worker._build_pack_artifacts
        ownership = commerce_ownership_values(SYSTEM_COMMERCE_SCOPE)

        def forbidden_qdrant(*_args, **_kwargs):
            raise AssertionError("frozen paid export must not read Qdrant")

        def mutate_after_snapshot(**kwargs):
            frozen = list(kwargs.get("authoritative_pack_rows") or [])
            assert [row.video_id for row in frozen] == ["video_export_lock"]
            kwargs["session"].add(
                PackVideo(
                    **ownership,
                    pack_id=target.pack_id,
                    batch_id=target.batch_id,
                    quote_id=target.quote_id,
                    position=2,
                    video_id="concurrent_video",
                    title="Concurrent row",
                    video_url=(
                        "https://www.youtube.com/watch?v=concurrent_video"
                    ),
                    transcript_source="concurrent",
                    indexed_parent_id="concurrent_parent",
                    status="ready",
                )
            )
            kwargs["session"].flush()
            return original_builder(**kwargs)

        monkeypatch.setattr(
            "src.ingest_v2.cloud.diarization_indexer.channel_service_logic."
            "child_segments_by_parent",
            forbidden_qdrant,
        )
        monkeypatch.setattr(worker, "_build_pack_artifacts", mutate_after_snapshot)
        with pytest.raises(RuntimeError, match="canonical transcript exists"):
            worker._finalize_paid_pack_export(target)

        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "partial"
            assert pack.export_paths_json is None
            assert session.scalar(
                select(IngestionJob).where(
                    IngestionJob.job_kind == "paid_pack_export_gc"
                )
            ) is None
            assert session.scalar(select(PackVideo).where(
                PackVideo.video_id == "concurrent_video"
            )) is None
        pack_root = tmp_path / "exports" / target.pack_id
        assert list(pack_root.glob("paid-*")) == []
    finally:
        dispose_engine()


def test_partial_unpublished_generation_is_removed_after_builder_failure(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "partial-generation.sqlite3"
    export_root = tmp_path / "exports"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(export_root))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()

        def partial_builder(**kwargs):
            root = (
                export_root
                / kwargs["pack"].id
                / kwargs["artifact_generation"]
            )
            root.mkdir(parents=True)
            (root / "manifest.json").write_text("partial", encoding="utf-8")
            raise OSError("simulated interrupted artifact build")

        monkeypatch.setattr(worker, "_build_pack_artifacts", partial_builder)
        with pytest.raises(OSError, match="interrupted artifact build"):
            worker._finalize_paid_pack_export(target)
        assert list((export_root / target.pack_id).glob("paid-*")) == []
        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "partial"
            assert pack.export_paths_json is None
    finally:
        dispose_engine()


def test_failed_finalizer_marks_before_next_finalizer_can_attach(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "atomic-failure-marker.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    marker_entered = threading.Event()
    release_marker = threading.Event()
    second_verified = threading.Event()
    call_guard = threading.Lock()
    calls = 0
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []
    try:
        with Session(engine) as session:
            target = _seed_ready_paid_pack(session)
            session.commit()

        original_verifier = worker._verified_pack_artifacts
        original_marker = worker._mark_paid_pack_export_state_incomplete

        def fail_once(*args, **kwargs):
            nonlocal calls
            with call_guard:
                calls += 1
                call_number = calls
            if call_number == 1:
                raise OSError("first generation failed")
            second_verified.set()
            return original_verifier(*args, **kwargs)

        def blocked_marker(state, *, error):
            marker_entered.set()
            assert release_marker.wait(timeout=5)
            return original_marker(state, error=error)

        monkeypatch.setattr(worker, "_verified_pack_artifacts", fail_once)
        monkeypatch.setattr(
            worker, "_mark_paid_pack_export_state_incomplete", blocked_marker
        )

        def run(errors):
            try:
                worker._finalize_paid_pack_export(target)
            except BaseException as exc:  # test thread must retain the failure
                errors.append(exc)

        first_thread = threading.Thread(target=run, args=(first_errors,))
        first_thread.start()
        assert marker_entered.wait(timeout=5)
        second_thread = threading.Thread(target=run, args=(second_errors,))
        second_thread.start()
        assert not second_verified.wait(timeout=0.25)
        release_marker.set()
        first_thread.join(timeout=10)
        second_thread.join(timeout=10)
        assert not first_thread.is_alive() and not second_thread.is_alive()
        assert len(first_errors) == 1
        assert second_errors == []
        assert second_verified.is_set()

        with Session(engine) as session:
            pack = session.get(ChannelPack, target.pack_id)
            assert pack is not None and pack.status == "ready"
            assert pack.export_paths_json
    finally:
        release_marker.set()
        dispose_engine()


def test_one_pack_export_failure_does_not_strand_later_buyer(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "export-isolation.sqlite3"
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", f"sqlite+pysqlite:///{database}"
    )
    monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
    dispose_engine()
    init_db()
    engine = get_engine()
    try:
        with Session(engine) as session:
            first = _seed_ready_paid_pack(session, suffix="first_buyer")
            second = _seed_ready_paid_pack(session, suffix="second_buyer")
            session.commit()

        original_verifier = worker._verified_pack_artifacts

        def fail_first(*args, **kwargs):
            if kwargs["pack"].id == first.pack_id:
                raise OSError("simulated first-buyer export failure")
            return original_verifier(*args, **kwargs)

        monkeypatch.setattr(worker, "_verified_pack_artifacts", fail_first)
        heartbeat = worker._LeaseHeartbeat(job_id="test-job", worker_id="test-worker")
        with pytest.raises(
            RuntimeError, match="paid pack export finalization failed"
        ):
            worker._finalize_paid_pack_exports(
                (first, second), heartbeat=heartbeat
            )

        with Session(engine) as session:
            first_pack = session.get(ChannelPack, first.pack_id)
            second_pack = session.get(ChannelPack, second.pack_id)
            assert first_pack is not None and second_pack is not None
            assert first_pack.status == "partial"
            assert first_pack.export_paths_json is None
            assert second_pack.status == "ready"
            assert second_pack.export_paths_json
            assert session.get(ChannelOrder, first.order_id).status == "partial"
            assert session.get(ChannelOrder, second.order_id).status == "ready"
    finally:
        dispose_engine()


@pytest.mark.skipif(
    not (os.getenv("ICMFYI_TEST_POSTGRES_ADMIN_URL") or "").strip(),
    reason="ICMFYI_TEST_POSTGRES_ADMIN_URL is required for the real row-lock gate",
)
def test_pg16_pack_builder_holds_no_commerce_row_lock(
    tmp_path: Path, monkeypatch
) -> None:
    admin_url = make_url(os.environ["ICMFYI_TEST_POSTGRES_ADMIN_URL"])
    database_name = f"icmfyi_export_lock_{uuid.uuid4().hex[:12]}"
    target_url = admin_url.set(database=database_name)
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    verifier = None
    try:
        with admin.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
        monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
        monkeypatch.setenv(
            "CHANNEL_SERVICE_DATABASE_URL", target_url.render_as_string(False)
        )
        monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
        monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
        monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
        command.upgrade(Config("alembic.ini"), "head")
        monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
        dispose_engine()
        init_db()
        engine = get_engine()
        with Session(engine) as session:
            set_commerce_scope(session, SYSTEM_COMMERCE_SCOPE)
            target = _seed_ready_paid_pack(session)
            session.commit()

        verifier = create_engine(target_url, future=True)
        original_builder = worker._build_pack_artifacts
        observed = {"builder_reached": False}

        def lock_probe_builder(**kwargs):
            # A second transaction can NOWAIT-lock the exact commerce rows at
            # the instant filesystem generation begins. This would raise if
            # the worker still held its earlier FOR UPDATE locks.
            with verifier.begin() as connection:
                connection.execute(
                    text(
                        "SELECT set_config('app.commerce_authority', "
                        "'system_internal', true)"
                    )
                )
                connection.execute(
                    text(
                        "SELECT id FROM channel_orders WHERE id=:value "
                        "FOR UPDATE NOWAIT"
                    ),
                    {"value": target.order_id},
                ).one()
                connection.execute(
                    text(
                        "SELECT id FROM channel_packs WHERE id=:value "
                        "FOR UPDATE NOWAIT"
                    ),
                    {"value": target.pack_id},
                ).one()
                connection.execute(
                    text(
                        "SELECT id FROM pack_batches WHERE id=:value "
                        "FOR UPDATE NOWAIT"
                    ),
                    {"value": target.batch_id},
                ).one()
                connection.execute(
                    text(
                        "SELECT id FROM pack_videos WHERE pack_id=:pack_id "
                        "FOR UPDATE NOWAIT"
                    ),
                    {"pack_id": target.pack_id},
                ).one()
            observed["builder_reached"] = True
            return original_builder(**kwargs)

        monkeypatch.setattr(worker, "_build_pack_artifacts", lock_probe_builder)
        worker._finalize_paid_pack_export(target)
        assert observed == {"builder_reached": True}
        with Session(engine) as session:
            set_commerce_scope(session, SYSTEM_COMMERCE_SCOPE)
            assert session.get(ChannelPack, target.pack_id).status == "ready"
    finally:
        dispose_engine()
        if verifier is not None:
            verifier.dispose()
        with admin.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname=:database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database_name}"')
        admin.dispose()


@pytest.mark.skipif(
    not (os.getenv("ICMFYI_TEST_POSTGRES_ADMIN_URL") or "").strip(),
    reason="ICMFYI_TEST_POSTGRES_ADMIN_URL is required for the real lock-order gate",
)
def test_pg16_paid_reconciler_waits_on_pack_before_locking_order(
    tmp_path: Path, monkeypatch
) -> None:
    admin_url = make_url(os.environ["ICMFYI_TEST_POSTGRES_ADMIN_URL"])
    database_name = f"icmfyi_lock_order_{uuid.uuid4().hex[:12]}"
    target_url = admin_url.set(database=database_name)
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    engine = None
    reconciler_engine = None
    blocker = None
    try:
        with admin.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
        monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
        monkeypatch.setenv(
            "CHANNEL_SERVICE_DATABASE_URL", target_url.render_as_string(False)
        )
        monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
        monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
        monkeypatch.setenv("CHANNEL_SERVICE_EXPORT_ROOT", str(tmp_path / "exports"))
        command.upgrade(Config("alembic.ini"), "head")
        monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
        dispose_engine()
        init_db()
        engine = get_engine()
        tenant_id = f"ten_{'a' * 64}"
        principal_id = f"usr_{'b' * 64}"
        scope = gateway_commerce_scope(
            tenant_id=tenant_id, principal_user_id=principal_id
        )
        with Session(engine) as session:
            session.add_all(
                [
                    Tenant(
                        id=tenant_id,
                        slug="lock-order",
                        display_name="Lock order",
                    ),
                    UserAccount(
                        id=principal_id,
                        auth_provider="test",
                        auth_subject="lock-order",
                    ),
                ]
            )
            session.flush()
            session.add(
                TenantMembership(
                    tenant_id=tenant_id,
                    user_id=principal_id,
                    role="member",
                )
            )
            session.flush()
            target = _seed_ready_paid_pack(
                session, suffix="lock_order", scope=scope
            )
            job = IngestionJob(
                id="job_lock_order",
                dedupe_key="paid-lock-order",
                job_kind="public_item_ingestion",
                source_kind="youtube",
                source_key="UC:video_lock_order",
                pipeline_version="test-v1",
                request_tenant_ids_json=[tenant_id],
            )
            binding = {
                "schema": worker.PAID_PUBLIC_INGESTION_SCHEMA,
                "intentId": "intent_lock_order",
                "outboxId": "outbox_lock_order",
                "tenantId": tenant_id,
                "principalId": principal_id,
                "orderId": target.order_id,
                "packId": target.pack_id,
                "batchId": target.batch_id,
                "quoteId": target.quote_id,
                "quoteHash": "a" * 64,
                "videoId": "video_lock_order",
                "position": 1,
            }
            request = IngestionRequest(
                id="request_lock_order",
                tenant_id=tenant_id,
                requested_by_user_id=principal_id,
                job_id=job.id,
                idempotency_key="lock-order",
                request_fingerprint="b" * 64,
                request_json={"paidWork": binding},
            )
            session.add_all([job, request])
            session.commit()

        blocker = Session(engine)
        set_commerce_scope(blocker, scope)
        blocker.execute(
            select(ChannelPack)
            .where(ChannelPack.id == target.pack_id)
            .with_for_update()
        ).scalar_one()

        reconciler_engine = create_engine(target_url, future=True)
        pack_lock_attempted = threading.Event()
        errors: list[BaseException] = []

        def observe_pack_lock(_conn, _cursor, statement, *_args):
            normalized = " ".join(statement.lower().split())
            if "from channel_packs" in normalized and "for update" in normalized:
                pack_lock_attempted.set()

        event.listen(
            reconciler_engine, "before_cursor_execute", observe_pack_lock
        )

        def reconcile() -> None:
            try:
                with Session(reconciler_engine) as session:
                    request = session.get(IngestionRequest, "request_lock_order")
                    assert request is not None
                    worker._locked_paid_rows(
                        session, request=request, binding=binding
                    )
                    session.rollback()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=reconcile)
        thread.start()
        assert pack_lock_attempted.wait(timeout=5)

        # The reconciler is blocked on the pack. It must not already own the
        # narrower order row, so an independent NOWAIT probe can lock it.
        with engine.begin() as connection:
            connection.execute(
                text(
                    "SELECT id FROM channel_orders WHERE id=:order_id "
                    "FOR UPDATE NOWAIT"
                ),
                {"order_id": target.order_id},
            ).one()

        blocker.rollback()
        blocker.close()
        blocker = None
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        dispose_engine()
        if reconciler_engine is not None:
            reconciler_engine.dispose()
        if engine is not None:
            engine.dispose()
        with admin.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname=:database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database_name}"')
        admin.dispose()
