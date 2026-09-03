from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from alembic import command
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import Base

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_TABLES = {
    "hot_media_custody_manifests",
    "hot_media_custody_items",
    "hot_media_rehydration_attempts",
}
POSTGRES_ADMIN_URL = (os.getenv("ICMFYI_TEST_POSTGRES_ADMIN_URL") or "").strip()


def _alembic(monkeypatch: pytest.MonkeyPatch, database_url: str) -> Config:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv("CHANNEL_SERVICE_DATABASE_URL", database_url)
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    return Config(str(REPOSITORY_ROOT / "alembic.ini"))


def _seed_retained_media(engine) -> tuple[str, str]:
    digest = "a" * 64
    hot_path = f"/data/hot-media/sha256/aa/{digest}.mp4"
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO source_channels"
                "(id,platform,external_id,status,metadata_json,created_at,updated_at) "
                "VALUES ('channel','youtube','channel','active',"
                "CAST('{}' AS json),CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO source_videos"
                "(id,channel_id,platform,external_id,archive_state,clip_candidate,"
                "clip_ready,status,metadata_json,created_at,updated_at) VALUES "
                "('video','channel','youtube','video','retained_hot_verified',"
                "TRUE,TRUE,'active',CAST('{}' AS json),CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO media_objects"
                "(sha256,size_bytes,mime_type,status,metadata_json,created_at) VALUES "
                "(:digest,1234,'video/mp4','active',CAST('{}' AS json),CURRENT_TIMESTAMP)"
            ),
            {"digest": digest},
        )
        connection.execute(
            text(
                "INSERT INTO media_locations"
                "(id,media_sha256,backend,location_key,status,bytes,verified_at,"
                "created_at,updated_at) VALUES "
                "('hot_1',:digest,'hot_local',:hot_path,'active',1234,"
                "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            ),
            {"digest": digest, "hot_path": hot_path},
        )
        connection.execute(
            text(
                "INSERT INTO video_media_refs"
                "(id,video_id,media_sha256,role,status,created_at) VALUES "
                "('ref_1','video',:digest,'source_video','active',CURRENT_TIMESTAMP)"
            ),
            {"digest": digest},
        )
    return digest, hot_path


def _assert_retained_media(engine, digest: str, hot_path: str) -> None:
    with engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT size_bytes,mime_type,status FROM media_objects "
                "WHERE sha256=:digest"
            ),
            {"digest": digest},
        ).one() == (1234, "video/mp4", "active")
        assert connection.execute(
            text(
                "SELECT backend,location_key,status,bytes FROM media_locations "
                "WHERE id='hot_1'"
            )
        ).one() == ("hot_local", hot_path, "active", 1234)
        assert connection.execute(
            text(
                "SELECT archive_state,clip_ready,status FROM source_videos "
                "WHERE id='video'"
            )
        ).one() == ("retained_hot_verified", True, "active")


def test_metadata_owns_global_lifecycle_extension() -> None:
    assert LIFECYCLE_TABLES <= set(Base.metadata.tables)
    for table_name in LIFECYCLE_TABLES:
        assert "tenant_id" not in Base.metadata.tables[table_name].columns
    custody = Base.metadata.tables["hot_media_custody_items"]
    assert "appliance_hot_path" in custody.columns
    assert {foreign_key.target_fullname for foreign_key in custody.foreign_keys} == {
        "hot_media_custody_manifests.manifest_sha256",
        "media_locations.id",
        "media_objects.sha256",
    }
    attempts = Base.metadata.tables["hot_media_rehydration_attempts"]
    assert "final_appliance_path" in attempts.columns
    assert {foreign_key.target_fullname for foreign_key in attempts.foreign_keys} == {
        "hot_media_custody_manifests.manifest_sha256",
        "media_locations.id",
        "media_objects.sha256",
    }


def test_sqlite_empty_and_retained_upgrade_is_idempotent_and_reversible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "hot-media-retained.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    alembic = _alembic(monkeypatch, database_url)
    command.upgrade(alembic, "20260825_0005")
    engine = create_engine(database_url, future=True)
    digest, hot_path = _seed_retained_media(engine)
    engine.dispose()

    command.upgrade(alembic, "head")
    command.upgrade(alembic, "head")
    engine = create_engine(database_url, future=True)
    assert LIFECYCLE_TABLES <= set(inspect(engine).get_table_names())
    _assert_retained_media(engine, digest, hot_path)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260903_0007"
        )
        assert all(
            connection.execute(text(f"SELECT count(*) FROM {table_name}")).scalar_one()
            == 0
            for table_name in LIFECYCLE_TABLES
        )
    engine.dispose()

    command.downgrade(alembic, "20260825_0005")
    engine = create_engine(database_url, future=True)
    assert not (LIFECYCLE_TABLES & set(inspect(engine).get_table_names()))
    _assert_retained_media(engine, digest, hot_path)
    engine.dispose()

    command.upgrade(alembic, "head")
    engine = create_engine(database_url, future=True)
    assert LIFECYCLE_TABLES <= set(inspect(engine).get_table_names())
    _assert_retained_media(engine, digest, hot_path)
    engine.dispose()


def test_sqlite_downgrade_refuses_durable_lifecycle_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "hot-media-downgrade.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    alembic = _alembic(monkeypatch, database_url)
    command.upgrade(alembic, "head")
    engine = create_engine(database_url, future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO hot_media_custody_manifests"
                "(manifest_sha256,manifest_json,manifest_bytes,items_count,media_bytes,"
                "remote_root,status,custody_receipt_sha256,custody_receipt_json) VALUES "
                "(:manifest_sha,CAST('{}' AS json),1,1,1,'storagebox:archive',"
                "'custodied',:receipt_sha,CAST('{}' AS json))"
            ),
            {"manifest_sha": "b" * 64, "receipt_sha": "c" * 64},
        )
    engine.dispose()

    with pytest.raises(
        RuntimeError, match="downgrade would destroy durable hot-media lifecycle state"
    ):
        command.downgrade(alembic, "20260825_0005")

    engine = create_engine(database_url, future=True)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260826_0006"
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM hot_media_custody_manifests")
            ).scalar_one()
            == 1
        )
    engine.dispose()


@pytest.mark.skipif(
    not POSTGRES_ADMIN_URL,
    reason="ICMFYI_TEST_POSTGRES_ADMIN_URL is required for PostgreSQL lifecycle migration proof",
)
def test_postgres_empty_retained_acl_and_rollback_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_hot_media_{suffix}"
    probe_role = f"icmfyi_hot_media_probe_{suffix}"
    admin_url = make_url(POSTGRES_ADMIN_URL)
    target_url = admin_url.set(database=database_name)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    owner_engine = None
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(
            f'CREATE ROLE "{probe_role}" NOSUPERUSER NOCREATEDB NOCREATEROLE '
            "NOINHERIT NOLOGIN NOREPLICATION NOBYPASSRLS"
        )
        connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
    try:
        alembic = _alembic(monkeypatch, target_url.render_as_string(False))
        command.upgrade(alembic, "20260825_0005")
        owner_engine = create_engine(target_url, future=True)
        digest, hot_path = _seed_retained_media(owner_engine)
        owner_engine.dispose()
        owner_engine = None

        command.upgrade(alembic, "head")
        command.upgrade(alembic, "head")
        owner_engine = create_engine(target_url, future=True)
        _assert_retained_media(owner_engine, digest, hot_path)
        with owner_engine.begin() as connection:
            connection.execute(text(f'GRANT USAGE ON SCHEMA public TO "{probe_role}"'))
            acl_rows = connection.execute(
                text(
                    "SELECT table_name FROM information_schema.table_privileges "
                    "WHERE table_schema='public' AND table_name = ANY(:tables) "
                    "AND grantee IN ('PUBLIC',:probe_role)"
                ),
                {"tables": sorted(LIFECYCLE_TABLES), "probe_role": probe_role},
            ).all()
            assert acl_rows == []
            policy_rows = connection.execute(
                text(
                    "SELECT relname,relrowsecurity,relforcerowsecurity "
                    "FROM pg_class WHERE relname = ANY(:tables) ORDER BY relname"
                ),
                {"tables": sorted(LIFECYCLE_TABLES)},
            ).all()
            assert policy_rows == [
                (table_name, False, False) for table_name in sorted(LIFECYCLE_TABLES)
            ]
        owner_engine.dispose()
        owner_engine = None

        command.downgrade(alembic, "20260825_0005")
        owner_engine = create_engine(target_url, future=True)
        _assert_retained_media(owner_engine, digest, hot_path)
        assert not (LIFECYCLE_TABLES & set(inspect(owner_engine).get_table_names()))
        owner_engine.dispose()
        owner_engine = None
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname=:database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database_name}"')
            connection.exec_driver_sql(f'DROP ROLE IF EXISTS "{probe_role}"')
        admin_engine.dispose()
