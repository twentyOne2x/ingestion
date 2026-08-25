from __future__ import annotations

import os
import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    claim_ingestion_jobs,
    reserve_ingestion_effect,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    IngestionEffect,
    utcnow,
)


ADMIN_URL = (os.getenv("ICMFYI_TEST_POSTGRES_ADMIN_URL") or "").strip()
pytestmark = pytest.mark.skipif(
    not ADMIN_URL,
    reason="ICMFYI_TEST_POSTGRES_ADMIN_URL is required for destructive isolated PostgreSQL smoke",
)


def test_alembic_roundtrip_and_non_bypass_rls(monkeypatch: pytest.MonkeyPatch) -> None:
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_ingestion_{suffix}"
    role_name = f"icmfyi_runtime_{suffix}"
    runtime_password = f"runtime-{suffix}-only"
    admin_url = make_url(ADMIN_URL)
    target_url = admin_url.set(database=database_name)
    runtime_url = target_url.set(username=role_name, password=runtime_password)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    owner_engine = None
    runtime_engine = None

    with admin_engine.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
        connection.exec_driver_sql(
            f"CREATE ROLE \"{role_name}\" LOGIN PASSWORD '{runtime_password}' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
        )

    try:
        monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
        monkeypatch.setenv(
            "CHANNEL_SERVICE_DATABASE_URL", target_url.render_as_string(False)
        )
        monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
        monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
        alembic = Config("alembic.ini")
        command.upgrade(alembic, "head")
        command.downgrade(alembic, "20260825_0001")
        command.upgrade(alembic, "head")

        owner_engine = create_engine(target_url, future=True)
        tenant_a = f"ten_{'a' * 64}"
        tenant_b = f"ten_{'b' * 64}"
        user_a = f"usr_{'a' * 64}"
        user_b = f"usr_{'b' * 64}"
        with owner_engine.begin() as connection:
            identity_widths = dict(
                connection.execute(
                    text(
                        "SELECT table_name || '.' || column_name, "
                        "character_maximum_length FROM information_schema.columns "
                        "WHERE (table_name, column_name) IN "
                        "(('user_accounts','id'),('tenants','id'),"
                        "('ingestion_requests','tenant_id'))"
                    )
                ).all()
            )
            assert set(identity_widths.values()) == {68}
            policies = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT tablename FROM pg_policies WHERE schemaname='public' "
                        "AND tablename IN ('tenant_channel_entitlements',"
                        "'ingestion_requests','tenant_exports')"
                    )
                )
            }
            assert policies == {
                "tenant_channel_entitlements",
                "ingestion_requests",
                "tenant_exports",
            }
            connection.exec_driver_sql(
                f'GRANT CONNECT ON DATABASE "{database_name}" TO "{role_name}"'
            )
            connection.exec_driver_sql(f'GRANT USAGE ON SCHEMA public TO "{role_name}"')
            connection.exec_driver_sql(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{role_name}"'
            )
            connection.exec_driver_sql(
                f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{role_name}"'
            )
            seed_parameters = {
                "tenant_a": tenant_a,
                "tenant_b": tenant_b,
                "user_a": user_a,
                "user_b": user_b,
            }
            seed_statements = (
                (
                    "INSERT INTO user_accounts"
                    "(id,auth_provider,auth_subject,status,created_at,updated_at) VALUES "
                    "(:user_a,'test','a','active',now(),now()),"
                    "(:user_b,'test','b','active',now(),now())"
                ),
                (
                    "INSERT INTO tenants(id,slug,display_name,status,created_at,updated_at) VALUES "
                    "(:tenant_a,'tenant-a','Tenant A','active',now(),now()),"
                    "(:tenant_b,'tenant-b','Tenant B','active',now(),now())"
                ),
                (
                    "INSERT INTO source_channels"
                    "(id,platform,external_id,metadata_json,created_at,updated_at,status) VALUES "
                    "('chn_a','youtube','a','{}'::json,now(),now(),'active'),"
                    "('chn_b','youtube','b','{}'::json,now(),now(),'active')"
                ),
                (
                    "INSERT INTO tenant_channel_entitlements"
                    "(id,tenant_id,channel_id,granted_by_user_id,access_level,status,created_at,updated_at) VALUES "
                    "('ent_a',:tenant_a,'chn_a',:user_a,'query','active',now(),now()),"
                    "('ent_b',:tenant_b,'chn_b',:user_b,'query','active',now(),now())"
                ),
                (
                    "INSERT INTO ingestion_jobs"
                    "(id,dedupe_key,job_kind,source_kind,source_key,pipeline_version,status,"
                    "priority,attempt_count,max_attempts,payload_json,result_json,created_at,updated_at) "
                    "VALUES ('job_a','dedupe-a','video','youtube','a','v1','queued',0,0,5,"
                    "'{}'::json,'{}'::json,now(),now())"
                ),
                (
                    "INSERT INTO ingestion_jobs"
                    "(id,dedupe_key,job_kind,source_kind,source_key,pipeline_version,status,"
                    "priority,attempt_count,max_attempts,payload_json,result_json,created_at,updated_at) "
                    "VALUES ('job_hot','dedupe-hot','youtube_hot_media','youtube','dQw4w9WgXcQ',"
                    "'youtube-hot-media-v1','queued',0,0,5,"
                    '\'{"video_id":"dQw4w9WgXcQ","canonical_url":'
                    '"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}\'::json,'
                    "'{}'::json,now(),now())"
                ),
                (
                    "INSERT INTO ingestion_requests"
                    "(id,tenant_id,requested_by_user_id,job_id,idempotency_key,request_fingerprint,"
                    "status,request_json,created_at,updated_at) VALUES "
                    "('req_a',:tenant_a,:user_a,'job_a','a',repeat('1',64),'accepted','{}'::json,now(),now()),"
                    "('req_b',:tenant_b,:user_b,'job_a','b',repeat('2',64),'accepted','{}'::json,now(),now())"
                ),
                (
                    "INSERT INTO tenant_exports"
                    "(id,tenant_id,requested_by_user_id,idempotency_key,request_fingerprint,"
                    "schema_version,status,counts_json,manifest_json,created_at,updated_at) VALUES "
                    "('exp_a',:tenant_a,:user_a,'a',repeat('3',64),'tenant-sqlite-v1','completed',"
                    "'{}'::json,'{}'::json,now(),now()),"
                    "('exp_b',:tenant_b,:user_b,'b',repeat('4',64),'tenant-sqlite-v1','completed',"
                    "'{}'::json,'{}'::json,now(),now())"
                ),
            )
            for statement in seed_statements:
                connection.execute(text(statement), seed_parameters)

        runtime_engine = create_engine(runtime_url, future=True)
        with runtime_engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM tenant_channel_entitlements")
                ).scalar_one()
                == 0
            )
        with runtime_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": tenant_a},
            )
            counts = connection.execute(
                text(
                    "SELECT "
                    "(SELECT count(*) FROM tenant_channel_entitlements),"
                    "(SELECT count(*) FROM ingestion_requests),"
                    "(SELECT count(*) FROM tenant_exports)"
                )
            ).one()
            assert tuple(counts) == (1, 1, 1)
        with runtime_engine.begin() as connection, pytest.raises(DBAPIError):
            connection.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": tenant_a},
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_channel_entitlements"
                    "(id,tenant_id,channel_id,granted_by_user_id,access_level,status,created_at,updated_at) "
                    "VALUES ('ent_cross',:tenant_b,'chn_b',:user_b,'query','active',now(),now())"
                ),
                {"tenant_b": tenant_b, "user_b": user_b},
            )

        first_worker = Session(runtime_engine)
        second_worker = Session(runtime_engine)
        try:
            claimed = claim_ingestion_jobs(
                first_worker,
                worker_id="postgres-acquirer-a",
                limit=1,
                lease_seconds=60,
                now=utcnow(),
                job_kinds=["youtube_hot_media"],
            )
            assert [row.id for row in claimed] == ["job_hot"]
            assert (
                claim_ingestion_jobs(
                    second_worker,
                    worker_id="postgres-acquirer-b",
                    limit=1,
                    lease_seconds=60,
                    now=utcnow(),
                    job_kinds=["youtube_hot_media"],
                )
                == []
            )
            effect, created = reserve_ingestion_effect(
                first_worker,
                job_id="job_hot",
                provider="youtube_ytdlp",
                effect_kind="public_video_download",
                idempotency_key="youtube-hot-media-v1:dQw4w9WgXcQ",
                request_payload={
                    "video_id": "dQw4w9WgXcQ",
                    "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                },
            )
            assert created is True
            assert effect.status == "reserved"
            first_worker.commit()
            second_worker.rollback()
            with Session(runtime_engine) as readback:
                assert readback.query(IngestionEffect).count() == 1
        finally:
            first_worker.close()
            second_worker.close()
    finally:
        if runtime_engine is not None:
            runtime_engine.dispose()
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
            connection.exec_driver_sql(f'DROP ROLE IF EXISTS "{role_name}"')
        admin_engine.dispose()
