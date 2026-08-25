from __future__ import annotations

import os
import uuid

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from alembic import command
from src.ingest_v2.cloud.diarization_indexer.channel_service_jobs import (
    claim_ingestion_jobs,
    reserve_ingestion_effect,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_scheduler import (
    _pending_quote_summary,
    _resolve_canary_target,
    sync_scheduler_jobs,
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


def _create_isolated_database(prefix: str):
    database_name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    admin_url = make_url(ADMIN_URL)
    target_url = admin_url.set(database=database_name)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
    return admin_engine, target_url, database_name


def _drop_isolated_database(
    admin_engine, database_name: str, *, role_name: str | None = None
) -> None:
    with admin_engine.connect() as connection:
        connection.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname=:database_name AND pid <> pg_backend_pid()"
            ),
            {"database_name": database_name},
        )
        connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{database_name}"')
        if role_name is not None:
            connection.exec_driver_sql(f'DROP ROLE IF EXISTS "{role_name}"')
    admin_engine.dispose()


def _configure_alembic_database(monkeypatch: pytest.MonkeyPatch, target_url) -> Config:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "production")
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL", target_url.render_as_string(False)
    )
    monkeypatch.setenv("CHANNEL_SERVICE_INTERNAL_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CHANNEL_SERVICE_CANONICAL_NAMESPACE", "videos")
    return Config("alembic.ini")


def _seed_pre_principal_commerce(connection) -> None:
    connection.execute(
        text(
            "INSERT INTO channel_packs"
            "(id,status,mode,namespace,channel_handle,total_purchased_video_count,"
            "ready_video_count,batch_count,created_at,updated_at) VALUES "
            "('p_acp','draft','recent_pack','videos','@acp',0,0,0,now(),now()),"
            "('p_system','draft','recent_pack','videos','@system',0,0,0,now(),now())"
        )
    )
    connection.execute(
        text(
            "INSERT INTO channel_quotes"
            "(id,status,mode,namespace,channel_handle,requested_max_videos,"
            "included_video_count,excluded_video_count,current_batch_index,"
            "current_batch_video_count,current_batch_amount_cents,"
            "total_included_amount_cents,per_video_cents,estimated_ready_minutes,"
            "eta_confidence,recommended_starter_batch_size,planning_latency_ms,"
            "request_json,batch_plan_json,price_breakdown_json,expires_at,created_at,updated_at) "
            "VALUES "
            "('q_acp','open','recent_pack','videos','@acp',1,1,0,1,1,1,1,1,1,"
            "'high',1,0,CAST(:acp_request AS json),CAST('[]' AS json),CAST('{}' AS json),"
            "now() + interval '1 hour',now(),now()),"
            "('q_system','open','recent_pack','videos','@system',1,1,0,1,1,1,1,1,1,"
            "'high',1,0,CAST('{}' AS json),CAST('[]' AS json),CAST('{}' AS json),"
            "now() + interval '1 hour',now(),now())"
        ),
        {"acp_request": '{"pack_id":"p_acp"}'},
    )
    connection.execute(
        text(
            "INSERT INTO quote_videos"
            "(quote_id,position,batch_index,included,video_id,status) VALUES "
            "('q_acp',1,1,true,'video_acp','included'),"
            "('q_system',1,1,true,'video_system','included')"
        )
    )
    connection.execute(
        text(
            "INSERT INTO acp_job_bridges"
            "(acp_job_id,offering_id,status,quote_id,fixed_price_cents,currency,"
            "payment_provider,payment_status,request_json,delivery_json,created_at,updated_at) "
            "VALUES ('job_acp','transcript_pack_starter_10','received','q_acp',1,'USD',"
            "'acp','settled_acp',CAST('{}' AS json),CAST('{}' AS json),now(),now())"
        )
    )
    connection.execute(
        text(
            "INSERT INTO entitlements"
            "(id,pack_id,subject_type,subject_id,status,created_at) VALUES "
            "('e_acp','p_acp','acp_buyer','buyer','active',now()),"
            "('e_system','p_system','legacy','operator','active',now())"
        )
    )


def test_commerce_upgrade_closes_lineage_and_quarantines_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_engine, target_url, database_name = _create_isolated_database(
        "icmfyi_lineage"
    )
    owner_engine = None
    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "20260825_0004")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            _seed_pre_principal_commerce(connection)
        owner_engine.dispose()
        owner_engine = None

        command.upgrade(alembic, "head")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.connect() as connection:
            authorities = dict(
                connection.execute(
                    text("SELECT id,authority_kind FROM channel_quotes ORDER BY id")
                ).all()
            )
            assert authorities == {
                "q_acp": "acp_internal",
                "q_system": "system_internal",
            }
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM quote_videos "
                        "WHERE authority_kind='acp_internal'"
                    )
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text("SELECT authority_kind FROM channel_packs WHERE id='p_acp'")
                ).scalar_one()
                == "acp_internal"
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM ("
                        "SELECT authority_kind FROM channel_quotes UNION ALL "
                        "SELECT authority_kind FROM quote_videos UNION ALL "
                        "SELECT authority_kind FROM channel_packs UNION ALL "
                        "SELECT authority_kind FROM acp_job_bridges UNION ALL "
                        "SELECT authority_kind FROM entitlements"
                        ") AS commerce WHERE authority_kind='legacy_internal'"
                    )
                ).scalar_one()
                == 0
            )

        with owner_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO source_channels"
                    "(id,platform,external_id,status,metadata_json,created_at,updated_at) "
                    "VALUES ('age_channel','youtube','age_channel','active',"
                    "CAST('{}' AS json),now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO source_videos"
                    "(id,channel_id,platform,external_id,archive_state,clip_candidate,"
                    "clip_ready,status,metadata_json,created_at,updated_at) VALUES "
                    "('age_video','age_channel','youtube','age_video',"
                    "'blocked_public_age_gate',false,false,'active',"
                    "CAST('{}' AS json),now(),now())"
                )
            )

        with (
            owner_engine.connect() as connection,
            pytest.raises(DBAPIError),
            connection.begin(),
        ):
            connection.execute(
                text(
                    "INSERT INTO checkout_sessions"
                    "(id,status,idempotency_key,currency,total_amount_cents,"
                    "quote_ids_json,line_items_json,payment_provider,payment_status,"
                    "authority_kind,created_at,updated_at) VALUES "
                    "('cross','open','cross','USD',1,CAST(:quotes AS json),"
                    "CAST(:items AS json),'acp','settled_acp','acp_internal',"
                    "now(),now())"
                ),
                {
                    "quotes": '["q_acp"]',
                    "items": '[{"quote_id":"q_system"}]',
                },
            )

        with (
            owner_engine.connect() as connection,
            pytest.raises(DBAPIError),
            connection.begin(),
        ):
            connection.execute(
                text(
                    "UPDATE acp_job_bridges SET delivery_json=CAST(:delivery AS json) "
                    "WHERE acp_job_id='job_acp'"
                ),
                {"delivery": '{"quote_id":"q_system"}'},
            )
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name)


def test_commerce_rls_realms_are_exclusive_and_owner_tuple_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_realm_guard_{suffix}"
    role_name = f"icmfyi_realm_probe_{suffix}"
    role_password = f"realm-{suffix}-only"
    admin_url = make_url(ADMIN_URL)
    target_url = admin_url.set(database=database_name)
    probe_url = target_url.set(username=role_name, password=role_password)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    owner_engine = None
    probe_engine = None
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(
            f'CREATE ROLE "{role_name}" LOGIN PASSWORD \'{role_password}\' '
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
        )
        connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')

    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "head")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO user_accounts"
                    "(id,auth_provider,auth_subject,status,created_at,updated_at) VALUES "
                    "('principal','test','principal','active',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO tenants"
                    "(id,slug,display_name,status,created_at,updated_at) VALUES "
                    "('tenant','tenant','Tenant','active',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_memberships"
                    "(tenant_id,user_id,role,status,created_at,updated_at) VALUES "
                    "('tenant','principal','member','active',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO channel_packs"
                    "(id,status,mode,namespace,channel_handle,"
                    "total_purchased_video_count,ready_video_count,batch_count,"
                    "authority_kind,created_at,updated_at) VALUES "
                    "('p_system','draft','recent_pack','videos','@system',0,0,0,"
                    "'system_internal',now(),now())"
                )
            )
            connection.exec_driver_sql(
                f'GRANT CONNECT ON DATABASE "{database_name}" TO "{role_name}"'
            )
            connection.exec_driver_sql(
                f'GRANT USAGE ON SCHEMA public TO "{role_name}"'
            )
            connection.exec_driver_sql(
                f'GRANT SELECT, INSERT, UPDATE ON channel_packs TO "{role_name}"'
            )

        probe_engine = create_engine(probe_url, future=True)
        with probe_engine.begin() as connection:
            connection.execute(
                text(
                    "SELECT set_config('app.commerce_authority',"
                    "'system_internal',true)"
                )
            )
            connection.execute(
                text("SELECT set_config('app.tenant_id','tenant',true)")
            )
            connection.execute(
                text(
                    "SELECT set_config('app.principal_user_id','principal',true)"
                )
            )
            assert connection.execute(
                text("SELECT count(*) FROM channel_packs WHERE id='p_system'")
            ).scalar_one() == 0
            assert (
                connection.execute(
                    text(
                        "UPDATE channel_packs SET authority_kind='gateway',"
                        "tenant_id='tenant',principal_user_id='principal' "
                        "WHERE id='p_system'"
                    )
                ).rowcount
                == 0
            )

        # A policy-only fix is insufficient: volatile target expressions can
        # switch GUC realms after USING selected the old row but before WITH
        # CHECK validates the new row.  The immutable ownership trigger must
        # reject that same-statement transition.
        with probe_engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(
                    text(
                        "SELECT set_config('app.commerce_authority',"
                        "'system_internal',true)"
                    )
                )
                connection.execute(
                    text("SELECT set_config('app.tenant_id','',true)")
                )
                connection.execute(
                    text("SELECT set_config('app.principal_user_id','',true)")
                )
                with pytest.raises(
                    DBAPIError, match="commerce ownership is immutable"
                ):
                    connection.execute(
                        text(
                            "UPDATE channel_packs SET "
                            "authority_kind=CASE WHEN set_config("
                            "'app.commerce_authority','',true)='' "
                            "THEN 'gateway' ELSE 'gateway' END,"
                            "tenant_id=set_config('app.tenant_id','tenant',true),"
                            "principal_user_id=set_config("
                            "'app.principal_user_id','principal',true) "
                            "WHERE id='p_system'"
                        )
                    )
            finally:
                transaction.rollback()

        with owner_engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT authority_kind,tenant_id,principal_user_id,status "
                    "FROM channel_packs WHERE id='p_system'"
                )
            ).one() == ("system_internal", None, None, "draft")

        # Exact gateway-only settings preserve ordinary insert and non-owner
        # updates for the restricted worker's legitimate path.
        with probe_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.commerce_authority','',true)")
            )
            connection.execute(
                text("SELECT set_config('app.tenant_id','tenant',true)")
            )
            connection.execute(
                text(
                    "SELECT set_config('app.principal_user_id','principal',true)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO channel_packs"
                    "(id,status,mode,namespace,channel_handle,"
                    "total_purchased_video_count,ready_video_count,batch_count,"
                    "authority_kind,tenant_id,principal_user_id,created_at,updated_at) "
                    "VALUES ('p_gateway','draft','recent_pack','videos','@gateway',"
                    "0,0,0,'gateway','tenant','principal',now(),now())"
                )
            )
            assert (
                connection.execute(
                    text(
                        "UPDATE channel_packs SET status='ready' "
                        "WHERE id='p_gateway'"
                    )
                ).rowcount
                == 1
            )
            assert connection.execute(
                text(
                    "SELECT status FROM channel_packs WHERE id='p_gateway'"
                )
            ).scalar_one() == "ready"
    finally:
        if probe_engine is not None:
            probe_engine.dispose()
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name, role_name=role_name)


def test_commerce_upgrade_rejects_json_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_engine, target_url, database_name = _create_isolated_database(
        "icmfyi_lineage_orphan"
    )
    owner_engine = None
    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "20260825_0004")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO checkout_sessions"
                    "(id,status,idempotency_key,currency,total_amount_cents,"
                    "quote_ids_json,line_items_json,payment_provider,payment_status,"
                    "created_at,updated_at) VALUES "
                    "('orphan','open','orphan','USD',1,CAST(:quotes AS json),"
                    "CAST('[]' AS json),'legacy','pending',now(),now())"
                ),
                {"quotes": '["missing_quote"]'},
            )
        owner_engine.dispose()
        owner_engine = None

        with pytest.raises(
            RuntimeError,
            match="references missing channel_quotes:missing_quote",
        ):
            command.upgrade(alembic, "head")

        owner_engine = create_engine(target_url, future=True)
        with owner_engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "20260825_0004"
            )
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name)


def test_commerce_upgrade_rejects_checkout_projection_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_engine, target_url, database_name = _create_isolated_database(
        "icmfyi_lineage_projection"
    )
    owner_engine = None
    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "20260825_0004")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            _seed_pre_principal_commerce(connection)
            connection.execute(
                text(
                    "INSERT INTO checkout_sessions"
                    "(id,status,idempotency_key,currency,total_amount_cents,"
                    "quote_ids_json,line_items_json,payment_provider,payment_status,"
                    "created_at,updated_at) VALUES "
                    "('cross','open','cross','USD',1,CAST(:quotes AS json),"
                    "CAST(:items AS json),'legacy','pending',now(),now())"
                ),
                {
                    "quotes": '["q_acp"]',
                    "items": '[{"quote_id":"q_system"}]',
                },
            )
        owner_engine.dispose()
        owner_engine = None

        with pytest.raises(
            RuntimeError,
            match="line_items_json quote ids must exactly match quote_ids_json",
        ):
            command.upgrade(alembic, "head")

        owner_engine = create_engine(target_url, future=True)
        with owner_engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "20260825_0004"
            )
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name)


def test_commerce_downgrade_refuses_to_erase_gateway_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commerce_tables = (
        "channel_quotes",
        "quote_videos",
        "checkout_sessions",
        "channel_packs",
        "pack_batches",
        "pack_videos",
        "channel_orders",
        "payment_receipts",
        "acp_job_bridges",
        "entitlements",
    )
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_lineage_downgrade_{suffix}"
    role_name = f"icmfyi_migration_owner_{suffix}"
    role_password = f"migration-{suffix}-only"
    admin_url = make_url(ADMIN_URL)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(
            f'CREATE ROLE "{role_name}" LOGIN PASSWORD \'{role_password}\' '
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
        )
        connection.exec_driver_sql(
            f'CREATE DATABASE "{database_name}" OWNER "{role_name}"'
        )
    target_url = admin_url.set(
        database=database_name,
        username=role_name,
        password=role_password,
    )
    owner_engine = None
    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "20260825_0004")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            _seed_pre_principal_commerce(connection)
        owner_engine.dispose()
        owner_engine = None
        command.upgrade(alembic, "head")

        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            assert (
                connection.execute(text("SELECT current_user")).scalar_one()
                == role_name
            )
            assert connection.execute(
                text(
                    "SELECT rolsuper,rolbypassrls FROM pg_roles "
                    "WHERE rolname=current_user"
                )
            ).one() == (False, False)
            assert connection.execute(
                text(
                    "SELECT pg_get_userbyid(relowner),relrowsecurity,"
                    "relforcerowsecurity FROM pg_class "
                    "WHERE oid='public.channel_quotes'::regclass"
                )
            ).one() == (role_name, True, True)
            assert connection.execute(
                text(
                    "SELECT count(*) FROM channel_quotes "
                    "WHERE id IN ('q_acp','q_system')"
                )
            ).scalar_one() == 0
            connection.execute(
                text(
                    "INSERT INTO user_accounts"
                    "(id,auth_provider,auth_subject,status,created_at,updated_at) VALUES "
                    "('principal','test','principal','active',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO tenants"
                    "(id,slug,display_name,status,created_at,updated_at) VALUES "
                    "('tenant','tenant','Tenant','active',now(),now())"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_memberships"
                    "(tenant_id,user_id,role,status,created_at,updated_at) VALUES "
                    "('tenant','principal','member','active',now(),now())"
                )
            )
            # Manufacture a non-reconstructible downgrade fixture as the table
            # owner. Runtime ownership transitions are intentionally blocked;
            # lifting FORCE RLS and this user trigger is test-only scaffolding.
            for table_name in ("channel_quotes", "quote_videos"):
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} DISABLE TRIGGER "
                    f"{table_name}_commerce_lineage"
                )
            connection.execute(
                text(
                    "UPDATE channel_quotes SET authority_kind='gateway',"
                    "tenant_id='tenant',principal_user_id='principal' "
                    "WHERE id='q_system'"
                )
            )
            connection.execute(
                text(
                    "UPDATE quote_videos SET authority_kind='gateway',"
                    "tenant_id='tenant',principal_user_id='principal' "
                    "WHERE quote_id='q_system'"
                )
            )
            for table_name in ("quote_videos", "channel_quotes"):
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} ENABLE TRIGGER "
                    f"{table_name}_commerce_lineage"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY"
                )
        owner_engine.dispose()
        owner_engine = None

        with pytest.raises(
            RuntimeError,
            match="downgrade would erase non-reconstructible commerce ownership",
        ):
            command.downgrade(alembic, "20260825_0004")

        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "20260825_0005"
            )
            policy_state = connection.execute(
                text(
                    "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity "
                    "FROM pg_class AS c "
                    "JOIN pg_namespace AS n ON n.oid=c.relnamespace "
                    "WHERE n.nspname='public' AND c.relname = ANY(:table_names) "
                    "ORDER BY c.relname"
                ),
                {"table_names": list(commerce_tables)},
            ).all()
            assert len(policy_state) == len(commerce_tables)
            assert all(
                row.relrowsecurity and row.relforcerowsecurity for row in policy_state
            )
            assert connection.execute(
                text(
                    "SELECT count(*) FROM pg_policies WHERE schemaname='public' "
                    "AND tablename = ANY(:table_names)"
                ),
                {"table_names": list(commerce_tables)},
            ).scalar_one() == len(commerce_tables)
            assert connection.execute(
                text("SELECT count(*) FROM channel_quotes WHERE id='q_system'")
            ).scalar_one() == 0
            connection.execute(
                text("SELECT set_config('app.tenant_id','tenant',true)")
            )
            connection.execute(
                text("SELECT set_config('app.principal_user_id','principal',true)")
            )
            connection.execute(
                text("SELECT set_config('app.commerce_authority','',true)")
            )
            assert connection.execute(
                text(
                    "SELECT authority_kind,tenant_id,principal_user_id "
                    "FROM channel_quotes WHERE id='q_system'"
                )
            ).one() == ("gateway", "tenant", "principal")
            for table_name in ("channel_quotes", "quote_videos"):
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} DISABLE TRIGGER "
                    f"{table_name}_commerce_lineage"
                )
            connection.execute(
                text(
                    "UPDATE channel_quotes SET authority_kind='system_internal',"
                    "tenant_id=NULL,principal_user_id=NULL WHERE id='q_system'"
                )
            )
            connection.execute(
                text(
                    "UPDATE quote_videos SET authority_kind='system_internal',"
                    "tenant_id=NULL,principal_user_id=NULL WHERE quote_id='q_system'"
                )
            )
            for table_name in ("quote_videos", "channel_quotes"):
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} ENABLE TRIGGER "
                    f"{table_name}_commerce_lineage"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY"
                )
        owner_engine.dispose()
        owner_engine = None

        command.downgrade(alembic, "20260825_0004")

        owner_engine = create_engine(target_url, future=True)
        with owner_engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == "20260825_0004"
            )
            assert connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_schema='public' "
                    "AND table_name = ANY(:table_names) "
                    "AND column_name IN "
                    "('authority_kind','tenant_id','principal_user_id')"
                ),
                {"table_names": list(commerce_tables)},
            ).scalar_one() == 0
            assert connection.execute(
                text(
                    "SELECT count(*) FROM pg_policies WHERE schemaname='public' "
                    "AND tablename = ANY(:table_names)"
                ),
                {"table_names": list(commerce_tables)},
            ).scalar_one() == 0
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name, role_name=role_name)


def test_scheduler_projection_preserves_cross_realm_demand_under_forced_rls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_scheduler_projection_{suffix}"
    role_name = f"icmfyi_scheduler_owner_{suffix}"
    role_password = f"scheduler-{suffix}-only"
    admin_url = make_url(ADMIN_URL)
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    with admin_engine.connect() as connection:
        connection.exec_driver_sql(
            f'CREATE ROLE "{role_name}" LOGIN PASSWORD \'{role_password}\' '
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
        )
        connection.exec_driver_sql(
            f'CREATE DATABASE "{database_name}" OWNER "{role_name}"'
        )
    target_url = admin_url.set(
        database=database_name,
        username=role_name,
        password=role_password,
    )
    owner_engine = None
    gateway_tenant = f"ten_{'a' * 64}"
    gateway_principal = f"usr_{'b' * 64}"
    try:
        alembic = _configure_alembic_database(monkeypatch, target_url)
        command.upgrade(alembic, "20260825_0004")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            _seed_pre_principal_commerce(connection)
        owner_engine.dispose()
        owner_engine = None
        command.upgrade(alembic, "head")
        owner_engine = create_engine(target_url, future=True)

        with owner_engine.begin() as connection:
            assert connection.execute(
                text("SELECT count(*) FROM quote_videos")
            ).scalar_one() == 0

            connection.execute(
                text(
                    "SELECT set_config('app.commerce_authority',"
                    "'acp_internal',true)"
                )
            )
            connection.execute(
                text(
                    "UPDATE quote_videos SET video_id='AcpCanary01',"
                    "video_url=:video_url "
                    "WHERE quote_id='q_acp'"
                ),
                {"video_url": "https://www.youtube.com/watch?v=AcpCanary01"},
            )
            connection.execute(
                text(
                    "INSERT INTO quote_videos"
                    "(quote_id,position,batch_index,included,video_id,video_url,"
                    "status,authority_kind,tenant_id,principal_user_id) VALUES "
                    "('q_acp',5,0,false,'SharedVid01',"
                    "'https://www.youtube.com/watch?v=SharedVid01',"
                    "'pending_acquisition','acp_internal',NULL,NULL)"
                )
            )

            connection.execute(
                text(
                    "SELECT set_config('app.commerce_authority',"
                    "'system_internal',true)"
                )
            )
            connection.execute(
                text(
                    "UPDATE quote_videos SET position=20,included=false,"
                    "video_id='SysDemand01',"
                    "video_url='https://www.youtube.com/watch?v=SysDemand01',"
                    "status='pending_acquisition' WHERE quote_id='q_system'"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO quote_videos"
                    "(quote_id,position,batch_index,included,video_id,video_url,"
                    "status,authority_kind,tenant_id,principal_user_id) VALUES "
                    "('q_system',30,0,false,'SharedVid01',"
                    "'https://www.youtube.com/watch?v=SharedVid01',"
                    "'pending_acquisition','system_internal',NULL,NULL)"
                )
            )

            connection.execute(
                text(
                    "INSERT INTO user_accounts"
                    "(id,auth_provider,auth_subject,status,created_at,updated_at) VALUES "
                    "(:principal,'test','gateway','active',now(),now())"
                ),
                {"principal": gateway_principal},
            )
            connection.execute(
                text(
                    "INSERT INTO tenants"
                    "(id,slug,display_name,status,created_at,updated_at) VALUES "
                    "(:tenant,'gateway','Gateway','active',now(),now())"
                ),
                {"tenant": gateway_tenant},
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_memberships"
                    "(tenant_id,user_id,role,status,created_at,updated_at) VALUES "
                    "(:tenant,:principal,'member','active',now(),now())"
                ),
                {"tenant": gateway_tenant, "principal": gateway_principal},
            )
            connection.execute(
                text("SELECT set_config('app.commerce_authority','',true)")
            )
            connection.execute(
                text(
                    "SELECT set_config('app.tenant_id',:tenant,true)"
                ),
                {"tenant": gateway_tenant},
            )
            connection.execute(
                text(
                    "SELECT set_config('app.principal_user_id',"
                    ":principal,true)"
                ),
                {"principal": gateway_principal},
            )
            connection.execute(
                text(
                    "INSERT INTO channel_quotes"
                    "(id,status,mode,namespace,channel_handle,requested_max_videos,"
                    "included_video_count,excluded_video_count,current_batch_index,"
                    "current_batch_video_count,current_batch_amount_cents,"
                    "total_included_amount_cents,per_video_cents,estimated_ready_minutes,"
                    "eta_confidence,recommended_starter_batch_size,planning_latency_ms,"
                    "request_json,batch_plan_json,price_breakdown_json,commerce_json,"
                    "expires_at,created_at,updated_at,authority_kind,tenant_id,"
                    "principal_user_id) VALUES "
                    "('q_gateway','open','recent_pack','videos','@gateway',1,0,0,"
                    "1,0,0,0,0,1,'low',1,0,CAST('{}' AS json),CAST('[]' AS json),"
                    "CAST('{}' AS json),CAST('{}' AS json),now()+interval '1 hour',"
                    "now(),now(),'gateway',:tenant,:principal)"
                ),
                {"tenant": gateway_tenant, "principal": gateway_principal},
            )
            connection.execute(
                text(
                    "INSERT INTO quote_videos"
                    "(quote_id,position,batch_index,included,video_id,video_url,"
                    "status,authority_kind,tenant_id,principal_user_id) VALUES "
                    "('q_gateway',1,0,false,'SharedVid01',"
                    "'https://www.youtube.com/watch?v=SharedVid01',"
                    "'pending_acquisition','gateway',:tenant,:principal)"
                ),
                {"tenant": gateway_tenant, "principal": gateway_principal},
            )
            connection.execute(text("SELECT set_config('app.tenant_id','',true)"))
            connection.execute(
                text("SELECT set_config('app.principal_user_id','',true)")
            )

            connection.execute(
                text(
                    "INSERT INTO transcript_probes"
                    "(key,video_id,video_url,language,prefer_auto,status,attempt_count,"
                    "created_at,updated_at) VALUES "
                    "('probe_shared','SharedVid01',"
                    "'https://www.youtube.com/watch?v=SharedVid01','en',true,"
                    "'queued',0,now(),now()),"
                    "('probe_system','SysDemand01',"
                    "'https://www.youtube.com/watch?v=SysDemand01','en',true,"
                    "'queued',0,now(),now())"
                )
            )

        with Session(owner_engine) as session:
            assert _pending_quote_summary(session) == {
                "SharedVid01": {"subscriber_count": 3, "min_position": 1},
                "SysDemand01": {"subscriber_count": 1, "min_position": 20},
            }
            sync_scheduler_jobs(session)
            session.commit()
            lanes = dict(
                session.execute(
                    text("SELECT video_id,lane FROM scheduler_jobs")
                ).all()
            )
            assert lanes == {
                "SharedVid01": "quote_starter_probe",
                "SysDemand01": "quote_deferred_probe",
            }
            assert _resolve_canary_target(
                session=session,
                pool=None,
                profile={"canary_language": "en"},
            ) == {
                "video_url": "https://www.youtube.com/watch?v=AcpCanary01",
                "video_id": "AcpCanary01",
                "language": "en",
            }

        for statement in (
            "INSERT INTO scheduler_quote_video_projection"
            "(quote_video_id,video_id,position,included,status) "
            "VALUES (-1,'HostileVid1',1,false,'pending_acquisition')",
            "UPDATE scheduler_quote_video_projection "
            "SET position=1 WHERE video_id='SysDemand01'",
            "DELETE FROM scheduler_quote_video_projection "
            "WHERE video_id='SysDemand01'",
        ):
            with (
                pytest.raises(DBAPIError, match="trigger-maintained"),
                owner_engine.begin() as connection,
            ):
                connection.execute(text(statement))

        with owner_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.commerce_authority','',true)")
            )
            connection.execute(
                text("SELECT set_config('app.tenant_id',:tenant,true)"),
                {"tenant": gateway_tenant},
            )
            connection.execute(
                text(
                    "SELECT set_config('app.principal_user_id',:principal,true)"
                ),
                {"principal": gateway_principal},
            )
            connection.execute(
                text(
                    "DELETE FROM quote_videos WHERE quote_id='q_gateway' "
                    "AND video_id='SharedVid01'"
                )
            )
            connection.execute(text("SELECT set_config('app.tenant_id','',true)"))
            connection.execute(
                text("SELECT set_config('app.principal_user_id','',true)")
            )
            connection.execute(
                text(
                    "SELECT set_config('app.commerce_authority',"
                    "'acp_internal',true)"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM quote_videos WHERE quote_id='q_acp' "
                    "AND video_id='SharedVid01'"
                )
            )

        with Session(owner_engine) as session:
            assert _pending_quote_summary(session)["SharedVid01"] == {
                "subscriber_count": 1,
                "min_position": 30,
            }
            sync_scheduler_jobs(session)
            assert session.execute(
                text(
                    "SELECT lane FROM scheduler_jobs "
                    "WHERE video_id='SharedVid01'"
                )
            ).scalar_one() == "quote_deferred_probe"
    finally:
        if owner_engine is not None:
            owner_engine.dispose()
        _drop_isolated_database(admin_engine, database_name, role_name=role_name)


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
