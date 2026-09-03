from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from alembic import command

POSTGRES_ADMIN_URL = (os.getenv("ICMFYI_TEST_POSTGRES_ADMIN_URL") or "").strip()


def _alembic(monkeypatch: pytest.MonkeyPatch, database_url: str) -> Config:
    monkeypatch.setenv("CHANNEL_SERVICE_ENV", "development")
    monkeypatch.setenv("CHANNEL_SERVICE_DATABASE_URL", database_url)
    return Config("alembic.ini")


def test_app_chat_migration_is_additive_and_refuses_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'app-chat.sqlite3'}"
    alembic = _alembic(monkeypatch, database_url)
    command.upgrade(alembic, "head")
    command.upgrade(alembic, "head")

    engine = create_engine(database_url, future=True)
    assert "app_chats" in inspect(engine).get_table_names()
    columns = {column["name"] for column in inspect(engine).get_columns("app_chats")}
    assert columns == {
        "id",
        "tenant_id",
        "principal_user_id",
        "created_at_ms",
        "is_shared",
        "original_chat_id",
        "original_chat_is_shared",
        "payload_json",
        "created_at",
        "updated_at",
    }
    with engine.begin() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == "20260903_0007"
        )
        payload = {
            "id": "chat-a",
            "title": "Retained",
            "userId": f"usr_{'a' * 64}",
            "createdAt": 1788422400000,
            "path": "/chat/chat-a",
            "messages": [],
            "structured_metadata": [],
        }
        connection.execute(
            text(
                "INSERT INTO app_chats"
                "(id,tenant_id,principal_user_id,created_at_ms,is_shared,payload_json) "
                "VALUES (:id,:tenant,:principal,:created,false,:payload)"
            ),
            {
                "id": payload["id"],
                "tenant": f"ten_{'b' * 64}",
                "principal": payload["userId"],
                "created": payload["createdAt"],
                "payload": json.dumps(payload, sort_keys=True),
            },
        )
    engine.dispose()

    with pytest.raises(RuntimeError, match="refusing to drop authoritative app chat rows"):
        command.downgrade(alembic, "20260826_0006")

    engine = create_engine(database_url, future=True)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM app_chats")).scalar_one() == 1
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == "20260903_0007"
        )
    engine.dispose()


def test_postgres_contract_is_forced_rls_and_exact_share_scoped() -> None:
    source = (
        Path("alembic/versions/20260903_0007_app_chat_persistence.py")
        .read_text(encoding="utf-8")
    )
    assert "ALTER TABLE public.app_chats FORCE ROW LEVEL SECURITY" in source
    assert "current_setting('app.tenant_id', true)" in source
    assert "current_setting('app.principal_user_id', true)" in source
    assert "id = current_setting('app.share_id', true)" in source
    assert "FOR UPDATE\n        USING (\n            NOT is_shared" in source
    assert "FOR DELETE\n        USING (\n            NOT is_shared" in source
    assert 'name="fk_app_chats_private_original"' in source
    assert 'name="uq_app_chats_owner_id_kind"' in source
    assert '"AND original_chat_is_shared IS FALSE "' in source
    assert '"AND id ~ \'^shr_[A-Za-z0-9_-]{32}$\' "' in source
    assert '"AND payload_json->>\'createdAt\' = created_at_ms::text) IS TRUE"' in source
    assert '"AND payload_json->\'readOnly\' = \'true\'::jsonb "' in source
    assert '"AND payload_json->>\'sharePath\' = \'/share/\' || id) IS TRUE"' in source
    assert "REVOKE ALL PRIVILEGES ON TABLE public.app_chats FROM PUBLIC" in source
    assert "ALTER TABLE public.app_chats NO FORCE ROW LEVEL SECURITY" in source
    assert source.count("ALTER TABLE public.app_chats FORCE ROW LEVEL SECURITY") == 2


@pytest.mark.skipif(
    not POSTGRES_ADMIN_URL,
    reason="ICMFYI_TEST_POSTGRES_ADMIN_URL is required for isolated PostgreSQL chat proof",
)
def test_postgres_chat_share_constraints_rls_cascade_and_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    database_name = f"icmfyi_chat_{suffix}"
    role_name = f"icmfyi_chat_probe_{suffix}"
    role_password = f"chat-{suffix}-only"
    admin_url = make_url(POSTGRES_ADMIN_URL)
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

    tenant = f"ten_{'a' * 64}"
    principal = f"usr_{'b' * 64}"
    other_tenant = f"ten_{'c' * 64}"
    valid_share = f"shr_{'d' * 32}"

    def payload(chat_id: str, *, shared: bool = False, original: str | None = None):
        value = {
            "id": chat_id,
            "title": "Retained",
            "userId": principal,
            "createdAt": 1788422400000,
            "path": f"/chat/{chat_id}",
            "messages": [],
            "structured_metadata": [],
        }
        if shared:
            value.update(
                {
                    "readOnly": True,
                    "sharePath": f"/share/{chat_id}",
                    "originalChatId": original,
                }
            )
        return json.dumps(value, sort_keys=True)

    def insert_share(
        connection,
        chat_id: str,
        original: str,
        *,
        row_tenant: str = tenant,
        discriminator: bool | None = False,
        raw_payload: str | None = None,
    ) -> None:
        connection.execute(
            text(
                "INSERT INTO app_chats"
                "(id,tenant_id,principal_user_id,created_at_ms,is_shared,"
                "original_chat_id,original_chat_is_shared,payload_json) "
                "VALUES (:id,:tenant,:principal,1788422400000,true,:original,"
                ":discriminator,CAST(:payload AS jsonb))"
            ),
            {
                "id": chat_id,
                "tenant": row_tenant,
                "principal": principal,
                "original": original,
                "discriminator": discriminator,
                "payload": raw_payload
                if raw_payload is not None
                else payload(chat_id, shared=True, original=original),
            },
        )

    def insert_private(connection, chat_id: str, raw_payload: str) -> None:
        connection.execute(
            text(
                "INSERT INTO app_chats"
                "(id,tenant_id,principal_user_id,created_at_ms,is_shared,payload_json) "
                "VALUES (:id,:tenant,:principal,1788422400000,false,CAST(:payload AS jsonb))"
            ),
            {
                "id": chat_id,
                "tenant": tenant,
                "principal": principal,
                "payload": raw_payload,
            },
        )

    try:
        alembic = _alembic(monkeypatch, target_url.render_as_string(False))
        command.upgrade(alembic, "head")
        owner_engine = create_engine(target_url, future=True)
        with owner_engine.begin() as connection:
            connection.exec_driver_sql(
                f'GRANT CONNECT ON DATABASE "{database_name}" TO "{role_name}"'
            )
            connection.exec_driver_sql(
                f'GRANT USAGE ON SCHEMA public TO "{role_name}"'
            )
            connection.exec_driver_sql(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON app_chats TO "{role_name}"'
            )
            connection.execute(
                text(
                    "INSERT INTO app_chats"
                    "(id,tenant_id,principal_user_id,created_at_ms,is_shared,payload_json) "
                    "VALUES ('private-a',:tenant,:principal,1788422400000,false,"
                    "CAST(:payload AS jsonb))"
                ),
                {
                    "tenant": tenant,
                    "principal": principal,
                    "payload": payload("private-a"),
                },
            )
            insert_share(connection, valid_share, "private-a")

        invalid_rows = [
            ("short-share", "private-a", tenant, False),
            (f"shr_{'e' * 31}", "private-a", tenant, False),
            (f"shr_{'e' * 33}", "private-a", tenant, False),
            (f"shr_{'f' * 32}", "private-a", tenant, None),
            (f"shr_{'1' * 32}", "private-a", other_tenant, False),
            (f"shr_{'2' * 32}", f"shr_{'2' * 32}", tenant, False),
            (f"shr_{'3' * 32}", valid_share, tenant, False),
        ]
        for chat_id, original, row_tenant, discriminator in invalid_rows:
            with pytest.raises(DBAPIError), owner_engine.begin() as connection:
                insert_share(
                    connection,
                    chat_id,
                    original,
                    row_tenant=row_tenant,
                    discriminator=discriminator,
                )

        private_payload_mutations = [
            ("id", "missing"),
            ("id", None),
            ("id", 42),
            ("userId", "missing"),
            ("userId", None),
            ("userId", [principal]),
            ("createdAt", "missing"),
            ("createdAt", None),
            ("createdAt", "1788422400000"),
        ]
        for index, (key, invalid_value) in enumerate(private_payload_mutations):
            chat_id = f"invalid-private-{index}"
            invalid_payload = json.loads(payload(chat_id))
            if invalid_value == "missing":
                invalid_payload.pop(key)
            else:
                invalid_payload[key] = invalid_value
            with pytest.raises(DBAPIError), owner_engine.begin() as connection:
                insert_private(connection, chat_id, json.dumps(invalid_payload))

        share_payload_mutations = []
        for key in ("readOnly", "originalChatId", "sharePath"):
            missing = json.loads(payload(valid_share, shared=True, original="private-a"))
            missing.pop(key)
            share_payload_mutations.append(missing)
            null_value = json.loads(payload(valid_share, shared=True, original="private-a"))
            null_value[key] = None
            share_payload_mutations.append(null_value)
        string_true = json.loads(payload(valid_share, shared=True, original="private-a"))
        string_true["readOnly"] = "true"
        share_payload_mutations.append(string_true)
        wrong_original = json.loads(payload(valid_share, shared=True, original="private-a"))
        wrong_original["originalChatId"] = 42
        share_payload_mutations.append(wrong_original)
        wrong_path = json.loads(payload(valid_share, shared=True, original="private-a"))
        wrong_path["sharePath"] = [f"/share/{valid_share}"]
        share_payload_mutations.append(wrong_path)
        for index, invalid_payload in enumerate(share_payload_mutations, start=10):
            chat_id = f"shr_{index:032d}"
            invalid_payload["id"] = chat_id
            if isinstance(invalid_payload.get("sharePath"), str):
                invalid_payload["sharePath"] = f"/share/{chat_id}"
            with pytest.raises(DBAPIError), owner_engine.begin() as connection:
                insert_share(
                    connection,
                    chat_id,
                    "private-a",
                    raw_payload=json.dumps(invalid_payload),
                )

        probe_engine = create_engine(probe_url, future=True)
        with probe_engine.begin() as connection:
            assert connection.execute(text("SELECT count(*) FROM app_chats")).scalar_one() == 0

        with probe_engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.tenant_id',:v,true)"), {"v": tenant})
            connection.execute(
                text("SELECT set_config('app.principal_user_id',:v,true)"),
                {"v": principal},
            )
            assert connection.execute(text("SELECT count(*) FROM app_chats")).scalar_one() == 2
            assert (
                connection.execute(
                    text("UPDATE app_chats SET updated_at=now() WHERE id=:id"),
                    {"id": valid_share},
                ).rowcount
                == 0
            )
            assert (
                connection.execute(
                    text("DELETE FROM app_chats WHERE id=:id"), {"id": valid_share}
                ).rowcount
                == 0
            )

        with probe_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.share_id',:v,true)"), {"v": valid_share}
            )
            assert connection.execute(text("SELECT id FROM app_chats")).scalar_one() == valid_share

        with pytest.raises(RuntimeError, match="refusing to drop authoritative app chat rows"):
            command.downgrade(alembic, "20260826_0006")
        with owner_engine.connect() as connection:
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == "20260903_0007"
            )
            assert connection.execute(
                text(
                    "SELECT relrowsecurity AND relforcerowsecurity FROM pg_class "
                    "WHERE oid='public.app_chats'::regclass"
                )
            ).scalar_one()

        with probe_engine.begin() as connection:
            connection.execute(text("SELECT set_config('app.tenant_id',:v,true)"), {"v": tenant})
            connection.execute(
                text("SELECT set_config('app.principal_user_id',:v,true)"),
                {"v": principal},
            )
            assert (
                connection.execute(
                    text("DELETE FROM app_chats WHERE id='private-a'")
                ).rowcount
                == 1
            )
        with owner_engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM app_chats")).scalar_one() == 0
        command.downgrade(alembic, "20260826_0006")
        with owner_engine.connect() as connection:
            assert connection.execute(text("SELECT to_regclass('public.app_chats')")).scalar_one() is None
    finally:
        if probe_engine is not None:
            probe_engine.dispose()
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
