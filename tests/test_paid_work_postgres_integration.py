from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from src.ingest_v2.cloud.diarization_indexer.paid_work_worker import (
    PAID_WORK_SCHEMA,
    PAID_WORK_TOPIC,
    PaidWorkError,
    _assert_worker_connection,
    claim_settled_paid_work,
    fail_settled_paid_work,
)

ADMIN_URL_ENV = "ICMFYI_TEST_POSTGRES_ADMIN_URL"
WORKER_URL_ENV = "ICMFYI_TEST_PAYMENT_WORKER_URL"


def _clear_prior_fixture_rows(admin) -> None:
    """Keep reruns deterministic without touching non-test payment traffic."""
    predicate = "left(tenant_id, 7) = 'ten_pg_'"
    with admin.begin() as connection:
        connection.execute(text(f"DELETE FROM payment_work_outbox WHERE {predicate}"))
        connection.execute(text(f"DELETE FROM payment_settlements WHERE {predicate}"))
        connection.execute(text(f"DELETE FROM payment_work_intents WHERE {predicate}"))


@pytest.fixture()
def pg_engines():
    admin_url = (os.getenv(ADMIN_URL_ENV) or "").strip()
    worker_url = (os.getenv(WORKER_URL_ENV) or "").strip()
    if not admin_url or not worker_url:
        pytest.skip("real PostgreSQL paid-work integration URLs were not supplied")
    admin = create_engine(admin_url, future=True)
    worker = create_engine(worker_url, future=True)
    try:
        with admin.connect() as connection:
            assert int(connection.execute(text("SHOW server_version_num")).scalar_one()) >= 160000
        _clear_prior_fixture_rows(admin)
        yield admin, worker
    finally:
        _clear_prior_fixture_rows(admin)
        worker.dispose()
        admin.dispose()


def _insert_settled(
    admin,
    *,
    ordinal: int,
    extra_payload_key: bool = False,
    commerce_quote_id: str | None = None,
) -> tuple[str, str]:
    intent_id = str(uuid.uuid4())
    outbox_id = str(uuid.uuid4())
    tenant_id = f"ten_pg_{uuid.uuid4().hex}"
    principal_id = f"usr_pg_{uuid.uuid4().hex}"
    request_hash = f"{ordinal:064x}"[-64:]
    quote_hash = f"{ordinal + 100:064x}"[-64:]
    quote_id = commerce_quote_id or f"quote_pg_{uuid.uuid4().hex}"
    payload = {
        "schema": PAID_WORK_SCHEMA,
        "tenantId": tenant_id,
        "principalId": principal_id,
        "toolName": "icmfyi.ingest.youtube",
        "idempotencyKey": f"pg-work-{uuid.uuid4().hex}",
        "requestHash": request_hash,
        "commerce": {
            "provider": "icmfyi-acp",
            "quoteId": quote_id,
            "offeringId": "youtube-channel-pack-v1",
            "quoteHash": quote_hash,
        },
        "work": {
            "schema": "icmfyi.channel-pack-work.v1",
            "operation": "create_settled_channel_pack_order",
            "quoteId": quote_id,
            "packId": None,
        },
    }
    if extra_payload_key:
        payload["poison"] = True
    idempotency_key = payload["idempotencyKey"]
    with admin.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO payment_work_intents "
                "(id,tenant_id,principal_id,tool_name,idempotency_key,request_hash,"
                "commerce_quote_id,commerce_quote_hash,scheme,network,pay_to,asset,"
                "amount_atomic,status) VALUES "
                "(CAST(:id AS uuid),:tenant,:principal,'icmfyi.ingest.youtube',:key,"
                ":request_hash,:quote_id,:quote_hash,'exact','eip155:8453',"
                "'0x0000000000000000000000000000000000000001',"
                "'0x0000000000000000000000000000000000000002',1,'settled')"
            ),
            {
                "id": intent_id,
                "tenant": tenant_id,
                "principal": principal_id,
                "key": idempotency_key,
                "request_hash": request_hash,
                "quote_id": quote_id,
                "quote_hash": quote_hash,
            },
        )
        connection.execute(
            text(
                "INSERT INTO payment_settlements "
                "(intent_id,tenant_id,request_hash,network,transaction,settlement) "
                "VALUES (CAST(:intent AS uuid),:tenant,:request_hash,'eip155:8453',"
                ":transaction,CAST('{}' AS jsonb))"
            ),
            {
                "intent": intent_id,
                "tenant": tenant_id,
                "request_hash": request_hash,
                "transaction": f"0x{uuid.uuid4().hex}{uuid.uuid4().hex}",
            },
        )
        connection.execute(
            text(
                "INSERT INTO payment_work_outbox "
                "(id,intent_id,tenant_id,topic,idempotency_key,request_hash,payload) "
                "VALUES (CAST(:id AS uuid),CAST(:intent AS uuid),:tenant,:topic,:key,"
                ":request_hash,CAST(:payload AS jsonb))"
            ),
            {
                "id": outbox_id,
                "intent": intent_id,
                "tenant": tenant_id,
                "topic": PAID_WORK_TOPIC,
                "key": idempotency_key,
                "request_hash": request_hash,
                "payload": json.dumps(payload),
            },
        )
    return outbox_id, intent_id

def test_pg16_claim_crash_poison_fairness_rls_and_two_worker_skip_locked(pg_engines) -> None:
    admin, worker = pg_engines
    poison = _insert_settled(admin, ordinal=1, extra_payload_key=True)
    first = _insert_settled(admin, ordinal=2)
    second = _insert_settled(admin, ordinal=3)

    with Session(worker) as session:
        _assert_worker_connection(session)
        assert session.execute(
            text(
                "SELECT has_table_privilege(current_user,"
                "'public.scheduler_quote_video_projection','SELECT')"
            )
        ).scalar_one() is False
        with pytest.raises(ProgrammingError):
            session.execute(text("SELECT * FROM payment_work_outbox")).all()
        session.rollback()

        # A process crash/rollback releases the row without manufacturing an ACK.
        claimed = claim_settled_paid_work(session)
        assert claimed is not None and claimed.outbox_id == poison[0]
        session.rollback()
    with Session(worker) as session:
        replay = claim_settled_paid_work(session)
        assert replay is not None and replay.outbox_id == poison[0]
        fail_settled_paid_work(
            session,
            replay,
            error_code="paid_work_invalid",
            error_detail="hostile exact-shape poison",
            retryable=False,
            retry_delay_seconds=0,
        )
        session.commit()

    # Two simultaneously open transactions claim different rows via SKIP LOCKED.
    left = Session(worker)
    right = Session(worker)
    try:
        left_claim = claim_settled_paid_work(left)
        right_claim = claim_settled_paid_work(right)
        assert left_claim is not None and right_claim is not None
        assert {left_claim.outbox_id, right_claim.outbox_id} == {first[0], second[0]}
        fail_settled_paid_work(
            left,
            left_claim,
            error_code="paid_work_database_transient",
            error_detail="bounded retry",
            retryable=True,
            retry_delay_seconds=600,
        )
        left.commit()
        fail_settled_paid_work(
            right,
            right_claim,
            error_code="paid_work_database_transient",
            error_detail="bounded retry",
            retryable=True,
            retry_delay_seconds=600,
        )
        right.commit()
    finally:
        left.close()
        right.close()

    later = _insert_settled(admin, ordinal=4)
    with Session(worker) as session:
        fair = claim_settled_paid_work(session)
        assert fair is not None and fair.outbox_id == later[0]
        fail_settled_paid_work(
            session,
            fair,
            error_code="paid_work_invalid",
            error_detail="terminal fixture",
            retryable=False,
            retry_delay_seconds=0,
        )
        session.commit()

    with admin.connect() as connection:
        row = connection.execute(
            text(
                "SELECT published_at IS NULL, dead_lettered_at IS NOT NULL, "
                "delivery_attempt_count, last_delivery_error_code "
                "FROM payment_work_outbox WHERE id=CAST(:id AS uuid)"
            ),
            {"id": poison[0]},
        ).one()
        assert tuple(row) == (True, True, 1, "paid_work_invalid")


def test_pg16_same_quote_claims_are_serialized_before_fulfillment(pg_engines) -> None:
    admin, worker = pg_engines
    quote_id = f"quote_pg_shared_{uuid.uuid4().hex}"
    first = _insert_settled(admin, ordinal=10, commerce_quote_id=quote_id)
    second = _insert_settled(admin, ordinal=11, commerce_quote_id=quote_id)

    left = Session(worker)
    right = Session(worker)
    try:
        left_claim = claim_settled_paid_work(left)
        assert left_claim is not None and left_claim.outbox_id == first[0]
        assert claim_settled_paid_work(right) is None
        right.commit()
        fail_settled_paid_work(
            left,
            left_claim,
            error_code="paid_work_invalid",
            error_detail="release quote-scoped advisory lock",
            retryable=False,
            retry_delay_seconds=0,
        )
        left.commit()
    finally:
        left.close()
        right.close()

    with Session(worker) as session:
        next_claim = claim_settled_paid_work(session)
        assert next_claim is not None and next_claim.outbox_id == second[0]
        fail_settled_paid_work(
            session,
            next_claim,
            error_code="paid_work_invalid",
            error_detail="terminal same-quote fixture",
            retryable=False,
            retry_delay_seconds=0,
        )
        session.commit()


def test_pg16_payment_worker_cannot_copy_internal_pack_into_gateway(
    pg_engines,
) -> None:
    admin, worker = pg_engines
    suffix = uuid.uuid4().hex
    tenant_id = f"ten_pg_{suffix}"
    principal_id = f"usr_pg_{suffix}"
    source_pack_id = f"pack_pg_internal_{suffix}"
    copied_pack_id = f"pack_pg_gateway_copy_{suffix}"
    try:
        with admin.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO user_accounts"
                    "(id,auth_provider,auth_subject,status,created_at,updated_at) VALUES "
                    "(:principal,'test',:principal,'active',now(),now())"
                ),
                {"principal": principal_id},
            )
            connection.execute(
                text(
                    "INSERT INTO tenants"
                    "(id,slug,display_name,status,created_at,updated_at) VALUES "
                    "(:tenant,:tenant,:tenant,'active',now(),now())"
                ),
                {"tenant": tenant_id},
            )
            connection.execute(
                text(
                    "INSERT INTO tenant_memberships"
                    "(tenant_id,user_id,role,status,created_at,updated_at) VALUES "
                    "(:tenant,:principal,'member','active',now(),now())"
                ),
                {"tenant": tenant_id, "principal": principal_id},
            )
            connection.execute(
                text(
                    "INSERT INTO channel_packs"
                    "(id,status,mode,namespace,channel_handle,"
                    "total_purchased_video_count,ready_video_count,batch_count,"
                    "manifest_json,export_paths_json,authority_kind,created_at,updated_at) "
                    "VALUES (:pack,'ready','recent_pack','videos','@internal',1,1,1,"
                    "CAST(:manifest AS json),CAST(:exports AS json),"
                    "'system_internal',now(),now())"
                ),
                {
                    "pack": source_pack_id,
                    "manifest": json.dumps({"secret": "internal-manifest"}),
                    "exports": json.dumps(
                        {"manifest_path": "/internal/manifest.json"}
                    ),
                },
            )

        with Session(worker) as session:
            _assert_worker_connection(session)
            session.execute(
                text(
                    "SELECT set_config('app.commerce_authority',"
                    "'system_internal',true)"
                )
            )
            session.execute(text("SELECT set_config('app.tenant_id','',true)"))
            session.execute(
                text("SELECT set_config('app.principal_user_id','',true)")
            )
            assert session.execute(
                text("SELECT count(*) FROM channel_packs WHERE id=:pack"),
                {"pack": source_pack_id},
            ).scalar_one() == 0
            assert (
                session.execute(
                    text(
                        "INSERT INTO channel_packs"
                        "(id,status,mode,namespace,channel_handle,"
                        "total_purchased_video_count,ready_video_count,batch_count,"
                        "manifest_json,export_paths_json,authority_kind,tenant_id,"
                        "principal_user_id,created_at,updated_at) "
                        "SELECT :copy,status,mode,namespace,channel_handle,"
                        "total_purchased_video_count,ready_video_count,batch_count,"
                        "manifest_json,export_paths_json,"
                        "CASE WHEN set_config('app.commerce_authority','',true)='' "
                        "THEN 'gateway' ELSE 'gateway' END,"
                        "set_config('app.tenant_id',:tenant,true),"
                        "set_config('app.principal_user_id',:principal,true),"
                        "now(),now() FROM channel_packs WHERE id=:source"
                    ),
                    {
                        "copy": copied_pack_id,
                        "tenant": tenant_id,
                        "principal": principal_id,
                        "source": source_pack_id,
                    },
                ).rowcount
                == 0
            )
            session.commit()

        with admin.connect() as connection:
            assert connection.execute(
                text("SELECT count(*) FROM channel_packs WHERE id=:copy"),
                {"copy": copied_pack_id},
            ).scalar_one() == 0
            assert connection.execute(
                text(
                    "SELECT authority_kind,tenant_id,principal_user_id,"
                    "manifest_json,export_paths_json FROM channel_packs WHERE id=:pack"
                ),
                {"pack": source_pack_id},
            ).one() == (
                "system_internal",
                None,
                None,
                {"secret": "internal-manifest"},
                {"manifest_path": "/internal/manifest.json"},
            )
    finally:
        with admin.begin() as connection:
            connection.execute(
                text("DELETE FROM channel_packs WHERE id IN (:source,:copy)"),
                {"source": source_pack_id, "copy": copied_pack_id},
            )
            connection.execute(
                text(
                    "DELETE FROM tenant_memberships "
                    "WHERE tenant_id=:tenant AND user_id=:principal"
                ),
                {"tenant": tenant_id, "principal": principal_id},
            )
            connection.execute(
                text("DELETE FROM tenants WHERE id=:tenant"),
                {"tenant": tenant_id},
            )
            connection.execute(
                text("DELETE FROM user_accounts WHERE id=:principal"),
                {"principal": principal_id},
            )


@pytest.mark.parametrize(
    ("grant", "revoke"),
    (
        (
            "GRANT CREATE ON SCHEMA public TO icmfyi_payment_worker",
            "REVOKE CREATE ON SCHEMA public FROM icmfyi_payment_worker",
        ),
        (
            "GRANT CREATE ON DATABASE icmfyi TO icmfyi_payment_worker",
            "REVOKE CREATE ON DATABASE icmfyi FROM icmfyi_payment_worker",
        ),
    ),
)
def test_pg16_readiness_rejects_database_and_schema_ddl_drift(
    pg_engines, grant: str, revoke: str
) -> None:
    admin, worker = pg_engines
    try:
        with admin.begin() as connection:
            connection.execute(text(grant))
        with Session(worker) as session, pytest.raises(
            PaidWorkError, match="database or schema privileges"
        ):
            _assert_worker_connection(session)
    finally:
        with admin.begin() as connection:
            connection.execute(text(revoke))
    with Session(worker) as session:
        _assert_worker_connection(session)


@pytest.mark.parametrize("membership_direction", ("member_of_role", "granted_to_member"))
def test_pg16_readiness_rejects_role_membership_in_both_directions(
    pg_engines, membership_direction: str
) -> None:
    admin, worker = pg_engines
    probe_role = f"icmfyi_membership_probe_{uuid.uuid4().hex}"
    quoted_probe = f'"{probe_role}"'
    if membership_direction == "member_of_role":
        grant = f"GRANT {quoted_probe} TO icmfyi_payment_worker"
        revoke = f"REVOKE {quoted_probe} FROM icmfyi_payment_worker"
    else:
        grant = f"GRANT icmfyi_payment_worker TO {quoted_probe}"
        revoke = f"REVOKE icmfyi_payment_worker FROM {quoted_probe}"
    try:
        with admin.begin() as connection:
            connection.execute(text(f"CREATE ROLE {quoted_probe} NOLOGIN"))
            connection.execute(text(grant))
        with Session(worker) as session, pytest.raises(
            PaidWorkError, match="role capabilities are unsafe"
        ):
            _assert_worker_connection(session)
    finally:
        with admin.begin() as connection:
            connection.execute(text(revoke))
            connection.execute(text(f"DROP ROLE {quoted_probe}"))
    with Session(worker) as session:
        _assert_worker_connection(session)
