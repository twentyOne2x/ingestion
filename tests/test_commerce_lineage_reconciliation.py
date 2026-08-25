from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text

from alembic import command
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import (
    _reconcile_sqlite_commerce_lineage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMERCE_TABLES = (
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


def _load_migration_reconciler() -> Callable:
    migration_path = (
        REPOSITORY_ROOT
        / "alembic"
        / "versions"
        / "20260825_0005_commerce_principal_isolation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_commerce_principal_isolation_0005", migration_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._reconcile_commerce_lineage


@pytest.fixture(params=("sqlite_startup", "alembic_0005"))
def reconciler(request: pytest.FixtureRequest) -> Callable:
    if request.param == "sqlite_startup":
        return _reconcile_sqlite_commerce_lineage
    return _load_migration_reconciler()


def _create_graph_schema(connection) -> None:
    ownership = (
        "authority_kind TEXT NOT NULL DEFAULT 'legacy_internal', "
        "tenant_id TEXT, principal_user_id TEXT"
    )
    statements = (
        f"CREATE TABLE channel_quotes (id TEXT PRIMARY KEY, request_json JSON NOT NULL, {ownership})",
        f"CREATE TABLE quote_videos (id INTEGER PRIMARY KEY, quote_id TEXT NOT NULL, {ownership})",
        f"CREATE TABLE checkout_sessions (id TEXT PRIMARY KEY, quote_ids_json JSON NOT NULL, line_items_json JSON NOT NULL, {ownership})",
        f"CREATE TABLE channel_packs (id TEXT PRIMARY KEY, {ownership})",
        f"CREATE TABLE pack_batches (id TEXT PRIMARY KEY, pack_id TEXT NOT NULL, quote_id TEXT NOT NULL, checkout_session_id TEXT NOT NULL, {ownership})",
        f"CREATE TABLE pack_videos (id INTEGER PRIMARY KEY, pack_id TEXT NOT NULL, batch_id TEXT NOT NULL, quote_id TEXT NOT NULL, {ownership})",
        f"CREATE TABLE channel_orders (id TEXT PRIMARY KEY, quote_id TEXT NOT NULL, checkout_session_id TEXT NOT NULL, pack_id TEXT NOT NULL, batch_id TEXT NOT NULL, {ownership})",
        f"CREATE TABLE payment_receipts (id TEXT PRIMARY KEY, checkout_session_id TEXT NOT NULL, order_id TEXT, {ownership})",
        f"CREATE TABLE acp_job_bridges (acp_job_id TEXT PRIMARY KEY, quote_id TEXT, checkout_session_id TEXT, order_id TEXT, pack_id TEXT, request_json JSON NOT NULL, delivery_json JSON NOT NULL, {ownership})",
        f"CREATE TABLE entitlements (id TEXT PRIMARY KEY, pack_id TEXT NOT NULL, {ownership})",
    )
    for statement in statements:
        connection.exec_driver_sql(statement)


def _seed_complete_graph(connection) -> None:
    statements = (
        (
            "INSERT INTO channel_quotes(id,request_json) VALUES "
            "('q_acp','{\"pack_id\":null}'),"
            "('q_expand','{\"pack_id\":\"p_acp\"}'),"
            "('q_system','{}')"
        ),
        (
            "INSERT INTO quote_videos(id,quote_id) VALUES "
            "(1,'q_acp'),(2,'q_expand'),(3,'q_system')"
        ),
        (
            "INSERT INTO checkout_sessions(id,quote_ids_json,line_items_json) VALUES "
            "('c_acp','[\"q_acp\",\"q_expand\"]',"
            ' \'[{"quote_id":"q_acp"},{"quote_id":"q_expand"}]\'),'
            "('c_system','[\"q_system\"]','[{\"quote_id\":\"q_system\"}]')"
        ),
        "INSERT INTO channel_packs(id) VALUES ('p_acp'),('p_system')",
        (
            "INSERT INTO pack_batches"
            "(id,pack_id,quote_id,checkout_session_id) VALUES "
            "('b_acp','p_acp','q_acp','c_acp'),"
            "('b_system','p_system','q_system','c_system')"
        ),
        (
            "INSERT INTO pack_videos(id,pack_id,batch_id,quote_id) VALUES "
            "(1,'p_acp','b_acp','q_acp'),"
            "(2,'p_system','b_system','q_system')"
        ),
        (
            "INSERT INTO channel_orders"
            "(id,quote_id,checkout_session_id,pack_id,batch_id) VALUES "
            "('o_acp','q_acp','c_acp','p_acp','b_acp'),"
            "('o_system','q_system','c_system','p_system','b_system')"
        ),
        (
            "INSERT INTO payment_receipts"
            "(id,checkout_session_id,order_id) VALUES "
            "('r_acp','c_acp','o_acp'),('r_system','c_system','o_system')"
        ),
        # The bridge intentionally reaches the lifecycle through structured JSON
        # only. Closure must walk upward from the leaf-side order, then downward
        # to siblings and the expansion quote's request_json pack edge.
        (
            "INSERT INTO acp_job_bridges"
            "(acp_job_id,request_json,delivery_json) VALUES "
            '(\'job_acp\',\'{"acp_job_id":"job_acp","pack_id":"p_acp"}\','
            '\'{"acp_job_id":"job_acp","quote_id":"q_acp",'
            '"order_id":"o_acp","pack_id":"p_acp",'
            '"batch_id":"b_acp"}\')'
        ),
        (
            "INSERT INTO entitlements(id,pack_id) VALUES "
            "('e_acp','p_acp'),('e_system','p_system')"
        ),
    )
    for statement in statements:
        connection.exec_driver_sql(statement)


def _authority_rows(connection) -> dict[tuple[str, str], tuple]:
    primary_keys = {
        "acp_job_bridges": "acp_job_id",
        **{
            table_name: "id"
            for table_name in COMMERCE_TABLES
            if table_name != "acp_job_bridges"
        },
    }
    rows: dict[tuple[str, str], tuple] = {}
    for table_name in COMMERCE_TABLES:
        key = primary_keys[table_name]
        for row in connection.execute(
            text(
                f"SELECT {key},authority_kind,tenant_id,principal_user_id "
                f"FROM {table_name}"
            )
        ):
            rows[(table_name, str(row[0]))] = tuple(row[1:])
    return rows


def test_full_undirected_closure_and_residual_system_quarantine(
    reconciler: Callable,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        _seed_complete_graph(connection)
        reconciler(connection, COMMERCE_TABLES)
        rows = _authority_rows(connection)

    acp_nodes = {
        ("channel_quotes", "q_acp"),
        ("channel_quotes", "q_expand"),
        ("quote_videos", "1"),
        ("quote_videos", "2"),
        ("checkout_sessions", "c_acp"),
        ("channel_packs", "p_acp"),
        ("pack_batches", "b_acp"),
        ("pack_videos", "1"),
        ("channel_orders", "o_acp"),
        ("payment_receipts", "r_acp"),
        ("acp_job_bridges", "job_acp"),
        ("entitlements", "e_acp"),
    }
    assert {
        key for key, value in rows.items() if value[0] == "acp_internal"
    } == acp_nodes
    assert all(value[1:] == (None, None) for value in rows.values())
    assert all(
        value[0] == ("acp_internal" if key in acp_nodes else "system_internal")
        for key, value in rows.items()
    )
    assert not any(value[0] == "legacy_internal" for value in rows.values())
    engine.dispose()


def test_orphan_fails_before_any_legacy_row_is_rewritten(reconciler: Callable) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        connection.execute(
            text("INSERT INTO quote_videos(id,quote_id) VALUES (1,'missing_quote')")
        )
        with pytest.raises(
            RuntimeError, match="references missing channel_quotes:missing_quote"
        ):
            reconciler(connection, COMMERCE_TABLES)
        assert (
            connection.execute(
                text("SELECT authority_kind FROM quote_videos WHERE id=1")
            ).scalar_one()
            == "legacy_internal"
        )
    engine.dispose()


def test_mixed_explicit_ownership_fails_closed(reconciler: Callable) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        connection.execute(
            text(
                "INSERT INTO channel_quotes"
                "(id,request_json,authority_kind,tenant_id,principal_user_id) VALUES "
                "('q_gateway','{}','gateway','ten_a','usr_a')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO quote_videos"
                "(id,quote_id,authority_kind) VALUES "
                "(1,'q_gateway','system_internal')"
            )
        )
        with pytest.raises(RuntimeError, match="crosses ownership"):
            reconciler(connection, COMMERCE_TABLES)
    engine.dispose()


@pytest.mark.parametrize(
    ("statement", "error_type", "message"),
    (
        (
            (
                "INSERT INTO checkout_sessions(id,quote_ids_json,line_items_json) "
                "VALUES ('empty','[]','[]')"
            ),
            RuntimeError,
            "must contain at least one quote id",
        ),
        (
            (
                "INSERT INTO acp_job_bridges"
                "(acp_job_id,request_json,delivery_json) "
                "VALUES ('detached','{}','{}')"
            ),
            RuntimeError,
            "detached from every commerce lifecycle row",
        ),
        (
            (
                "INSERT INTO checkout_sessions(id,quote_ids_json,line_items_json) "
                "VALUES ('typed','[7]','[]')"
            ),
            TypeError,
            "quote_ids_json must be a string id",
        ),
    ),
)
def test_invalid_detached_lifecycle_fails_closed(
    reconciler: Callable,
    statement: str,
    error_type: type[Exception],
    message: str,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        connection.exec_driver_sql(statement)
        with pytest.raises(error_type, match=message):
            reconciler(connection, COMMERCE_TABLES)
    engine.dispose()


def test_checkout_line_items_cannot_smuggle_a_different_quote(
    reconciler: Callable,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        connection.exec_driver_sql(
            "INSERT INTO channel_quotes(id,request_json) VALUES "
            "('q_acp','{}'),('q_system','{}')"
        )
        connection.exec_driver_sql(
            "INSERT INTO checkout_sessions"
            "(id,quote_ids_json,line_items_json) VALUES "
            "('cross','[\"q_acp\"]','[{\"quote_id\":\"q_system\"}]')"
        )
        with pytest.raises(
            RuntimeError,
            match="line_items_json quote ids must exactly match quote_ids_json",
        ):
            reconciler(connection, COMMERCE_TABLES)
        assert set(
            connection.execute(
                text("SELECT authority_kind FROM channel_quotes")
            ).scalars()
        ) == {"legacy_internal"}
    engine.dispose()


@pytest.mark.parametrize(
    ("bridge_json", "message"),
    (
        (
            "'{\"pack_id\":7}', '{}'",
            "request_json.pack_id must be a string id",
        ),
        (
            "'{\"acp_job_id\":\"other\"}', '{}'",
            "request_json.acp_job_id disagrees with the bridge primary key",
        ),
        (
            "'{}', '{\"order_id\":\"missing\"}'",
            "delivery_json.order_id references missing channel_orders:missing",
        ),
    ),
)
def test_acp_bridge_structured_lineage_fails_closed(
    reconciler: Callable,
    bridge_json: str,
    message: str,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as connection:
        _create_graph_schema(connection)
        connection.exec_driver_sql(
            "INSERT INTO acp_job_bridges"
            "(acp_job_id,request_json,delivery_json) VALUES "
            f"('job',{bridge_json})"
        )
        with pytest.raises((RuntimeError, TypeError), match=message):
            reconciler(connection, COMMERCE_TABLES)
    engine.dispose()


def _seed_alembic_sqlite_commerce(connection) -> None:
    connection.exec_driver_sql(
        "INSERT INTO channel_packs"
        "(id,status,mode,namespace,channel_handle,total_purchased_video_count,"
        "ready_video_count,batch_count,created_at,updated_at) VALUES "
        "('p_acp','draft','recent_pack','videos','@acp',0,0,0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),"
        "('p_system','draft','recent_pack','videos','@system',0,0,0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
    )
    connection.exec_driver_sql(
        "INSERT INTO channel_quotes"
        "(id,status,mode,namespace,channel_handle,requested_max_videos,"
        "included_video_count,excluded_video_count,current_batch_index,"
        "current_batch_video_count,current_batch_amount_cents,"
        "total_included_amount_cents,per_video_cents,estimated_ready_minutes,"
        "eta_confidence,recommended_starter_batch_size,planning_latency_ms,"
        "request_json,batch_plan_json,price_breakdown_json,expires_at,created_at,updated_at) "
        "VALUES "
        "('q_acp','open','recent_pack','videos','@acp',1,1,0,1,1,1,1,1,1,"
        "'high',1,0,'{\"pack_id\":\"p_acp\"}','[]','{}',CURRENT_TIMESTAMP,"
        "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),"
        "('q_system','open','recent_pack','videos','@system',1,1,0,1,1,1,1,1,1,"
        "'high',1,0,'{}','[]','{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
    )
    connection.exec_driver_sql(
        "INSERT INTO quote_videos"
        "(quote_id,position,batch_index,included,video_id,status) VALUES "
        "('q_acp',1,1,1,'video_acp','included'),"
        "('q_system',1,1,1,'video_system','included')"
    )
    connection.exec_driver_sql(
        "INSERT INTO acp_job_bridges"
        "(acp_job_id,offering_id,status,quote_id,fixed_price_cents,currency,"
        "payment_provider,payment_status,request_json,delivery_json,created_at,updated_at) "
        "VALUES ('job_acp','transcript_pack_starter_10','received','q_acp',1,'USD',"
        "'acp','settled_acp','{}','{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
    )
    connection.exec_driver_sql(
        "INSERT INTO entitlements"
        "(id,pack_id,subject_type,subject_id,status,created_at) VALUES "
        "('e_acp','p_acp','acp_buyer','buyer','active',CURRENT_TIMESTAMP),"
        "('e_system','p_system','legacy','operator','active',CURRENT_TIMESTAMP)"
    )


def _sqlite_alembic_config(
    monkeypatch: pytest.MonkeyPatch, database_path: Path
) -> Config:
    monkeypatch.setenv(
        "CHANNEL_SERVICE_DATABASE_URL",
        f"sqlite+pysqlite:///{database_path}",
    )
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    return Config(str(REPOSITORY_ROOT / "alembic.ini"))


def test_real_sqlite_alembic_upgrade_reconciles_before_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "commerce-upgrade.sqlite3"
    alembic = _sqlite_alembic_config(monkeypatch, database_path)
    command.upgrade(alembic, "20260825_0004")
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
        _seed_alembic_sqlite_commerce(connection)
    engine.dispose()

    command.upgrade(alembic, "head")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.connect() as connection:
        assert dict(
            connection.execute(
                text("SELECT id,authority_kind FROM channel_quotes ORDER BY id")
            ).all()
        ) == {"q_acp": "acp_internal", "q_system": "system_internal"}
        assert (
            connection.execute(
                text("SELECT authority_kind FROM channel_packs WHERE id='p_acp'")
            ).scalar_one()
            == "acp_internal"
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM channel_packs "
                    "WHERE authority_kind='legacy_internal'"
                )
            ).scalar_one()
            == 0
        )
    engine.dispose()


def test_real_sqlite_alembic_upgrade_rejects_json_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "commerce-orphan.sqlite3"
    alembic = _sqlite_alembic_config(monkeypatch, database_path)
    command.upgrade(alembic, "20260825_0004")
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO checkout_sessions"
            "(id,status,idempotency_key,currency,total_amount_cents,quote_ids_json,"
            "line_items_json,payment_provider,payment_status,created_at,updated_at) VALUES "
            "('orphan','open','orphan','USD',1,'[\"missing_quote\"]','[]','legacy',"
            "'pending',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
        )
    engine.dispose()

    with pytest.raises(
        RuntimeError,
        match="references missing channel_quotes:missing_quote",
    ):
        command.upgrade(alembic, "head")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260825_0004"
        )
        checkout_columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(checkout_sessions)"
            )
        }
        assert "authority_kind" not in checkout_columns
        connection.execute(text("DELETE FROM checkout_sessions WHERE id='orphan'"))
    engine.dispose()

    command.upgrade(alembic, "head")
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260825_0005"
        )
        assert "authority_kind" in {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(checkout_sessions)"
            )
        }
    engine.dispose()


def test_real_sqlite_downgrade_refuses_to_erase_gateway_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "commerce-gateway-downgrade.sqlite3"
    alembic = _sqlite_alembic_config(monkeypatch, database_path)
    command.upgrade(alembic, "20260825_0004")
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
        _seed_alembic_sqlite_commerce(connection)
    engine.dispose()
    command.upgrade(alembic, "head")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
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
    engine.dispose()

    with pytest.raises(
        RuntimeError,
        match="downgrade would erase non-reconstructible commerce ownership",
    ):
        command.downgrade(alembic, "20260825_0004")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260825_0005"
        )
        assert connection.execute(
            text(
                "SELECT authority_kind,tenant_id,principal_user_id "
                "FROM channel_quotes WHERE id='q_system'"
            )
        ).one() == ("gateway", "tenant", "principal")
    engine.dispose()


def test_source_video_age_gate_state_survives_refused_sqlite_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "source-video-age-gate.sqlite3"
    alembic = _sqlite_alembic_config(monkeypatch, database_path)
    command.upgrade(alembic, "head")
    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO source_channels"
                "(id,platform,external_id,status,metadata_json,created_at,updated_at) "
                "VALUES ('channel','youtube','channel','active','{}',"
                "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO source_videos"
                "(id,channel_id,platform,external_id,archive_state,clip_candidate,"
                "clip_ready,status,metadata_json,created_at,updated_at) VALUES "
                "('video','channel','youtube','video','blocked_public_age_gate',"
                "0,0,'active','{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
            )
        )
    engine.dispose()

    with pytest.raises(RuntimeError, match="blocked_public_age_gate source video"):
        command.downgrade(alembic, "20260825_0004")

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT archive_state FROM source_videos WHERE id='video'")
            ).scalar_one()
            == "blocked_public_age_gate"
        )
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == "20260825_0005"
        )
    engine.dispose()
