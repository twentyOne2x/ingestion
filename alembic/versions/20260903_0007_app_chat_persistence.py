"""Add authoritative tenant-scoped application chat persistence.

Chat payloads previously lived in a remote Redis-compatible KV store or a
process-local fallback.  This additive table makes PostgreSQL authoritative
for private conversations and immutable public share copies.  The application
role is forced through transaction-local tenant/principal scope; public share
reads are limited to one exact unguessable share id per transaction.

Revision ID: 20260903_0007
Revises: 20260826_0006
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260903_0007"
down_revision: str | None = "20260826_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


KNOWN_RUNTIME_ROLES = (
    "icmfyi_runtime",
    "icmfyi_clip_api",
    "icmfyi_clip_worker",
    "icmfyi_payment_seller",
    "icmfyi_payment_worker",
    "icmfyi_app",
)


def _install_postgres_contract() -> None:
    op.create_check_constraint(
        "ck_app_chats_id_charset",
        "app_chats",
        "id ~ '^[A-Za-z0-9_-]{1,64}$'",
    )
    op.create_check_constraint(
        "ck_app_chats_tenant_id",
        "app_chats",
        "tenant_id ~ '^ten_[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_app_chats_principal_user_id",
        "app_chats",
        "principal_user_id ~ '^usr_[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_app_chats_payload_object",
        "app_chats",
        "jsonb_typeof(payload_json) = 'object'",
    )
    op.create_check_constraint(
        "ck_app_chats_payload_identity",
        "app_chats",
        "(jsonb_typeof(payload_json->'id') = 'string' "
        "AND jsonb_typeof(payload_json->'userId') = 'string' "
        "AND jsonb_typeof(payload_json->'createdAt') = 'number' "
        "AND payload_json->>'id' = id "
        "AND payload_json->>'userId' = principal_user_id "
        "AND payload_json->>'createdAt' = created_at_ms::text) IS TRUE",
    )
    op.create_check_constraint(
        "ck_app_chats_share_shape",
        "app_chats",
        "((NOT is_shared AND original_chat_id IS NULL "
        "AND original_chat_is_shared IS NULL "
        "AND NOT (payload_json ? 'readOnly') "
        "AND NOT (payload_json ? 'sharePath') "
        "AND NOT (payload_json ? 'originalChatId')) IS TRUE) OR "
        "(is_shared AND original_chat_id IS NOT NULL "
        "AND original_chat_id <> id "
        "AND original_chat_is_shared IS FALSE "
        "AND id ~ '^shr_[A-Za-z0-9_-]{32}$' "
        "AND jsonb_typeof(payload_json->'readOnly') = 'boolean' "
        "AND payload_json->'readOnly' = 'true'::jsonb "
        "AND jsonb_typeof(payload_json->'originalChatId') = 'string' "
        "AND payload_json->>'originalChatId' = original_chat_id "
        "AND jsonb_typeof(payload_json->'sharePath') = 'string' "
        "AND payload_json->>'sharePath' = '/share/' || id) IS TRUE",
    )
    op.execute("ALTER TABLE public.app_chats ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.app_chats FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY app_chats_select_scope ON public.app_chats
        FOR SELECT
        USING (
            (
                tenant_id = current_setting('app.tenant_id', true)
                AND principal_user_id =
                    current_setting('app.principal_user_id', true)
            )
            OR (
                is_shared
                AND id = current_setting('app.share_id', true)
                AND payload_json->>'sharePath' = '/share/' || id
            )
        )
        """
    )
    op.execute(
        """
        CREATE POLICY app_chats_insert_scope ON public.app_chats
        FOR INSERT
        WITH CHECK (
            tenant_id = current_setting('app.tenant_id', true)
            AND principal_user_id =
                current_setting('app.principal_user_id', true)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY app_chats_update_scope ON public.app_chats
        FOR UPDATE
        USING (
            NOT is_shared
            AND
            tenant_id = current_setting('app.tenant_id', true)
            AND principal_user_id =
                current_setting('app.principal_user_id', true)
        )
        WITH CHECK (
            NOT is_shared
            AND
            tenant_id = current_setting('app.tenant_id', true)
            AND principal_user_id =
                current_setting('app.principal_user_id', true)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY app_chats_delete_scope ON public.app_chats
        FOR DELETE
        USING (
            NOT is_shared
            AND
            tenant_id = current_setting('app.tenant_id', true)
            AND principal_user_id =
                current_setting('app.principal_user_id', true)
        )
        """
    )
    op.execute("REVOKE ALL PRIVILEGES ON TABLE public.app_chats FROM PUBLIC")
    roles = ",".join(f"'{role}'" for role in KNOWN_RUNTIME_ROLES)
    op.execute(
        f"""
        DO $app_chat_privileges$
        DECLARE
            runtime_role text;
        BEGIN
            FOREACH runtime_role IN ARRAY ARRAY[{roles}]
            LOOP
                IF EXISTS (
                    SELECT 1 FROM pg_catalog.pg_roles
                    WHERE rolname = runtime_role
                ) THEN
                    EXECUTE format(
                        'REVOKE ALL PRIVILEGES ON TABLE public.app_chats FROM %I',
                        runtime_role
                    );
                END IF;
            END LOOP;
        END
        $app_chat_privileges$
        """
    )
    op.execute(
        "COMMENT ON TABLE public.app_chats IS "
        "'Authoritative forced-RLS chat and immutable share payloads'"
    )


def upgrade() -> None:
    op.create_table(
        "app_chats",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=68), nullable=False),
        sa.Column("principal_user_id", sa.String(length=68), nullable=False),
        sa.Column("created_at_ms", sa.BigInteger(), nullable=False),
        sa.Column(
            "is_shared",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "original_chat_id",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column("original_chat_is_shared", sa.Boolean(), nullable=True),
        sa.Column(
            "payload_json",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "created_at_ms > 0",
            name="ck_app_chats_created_at_ms",
        ),
        sa.ForeignKeyConstraint(
            [
                "tenant_id",
                "principal_user_id",
                "original_chat_id",
                "original_chat_is_shared",
            ],
            [
                "app_chats.tenant_id",
                "app_chats.principal_user_id",
                "app_chats.id",
                "app_chats.is_shared",
            ],
            name="fk_app_chats_private_original",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "principal_user_id",
            "id",
            "is_shared",
            name="uq_app_chats_owner_id_kind",
        ),
    )
    op.create_index(
        "ix_app_chats_owner_created",
        "app_chats",
        ["tenant_id", "principal_user_id", "is_shared", "created_at_ms"],
    )
    op.create_index(
        "ix_app_chats_original_chat_id",
        "app_chats",
        ["original_chat_id"],
    )
    if op.get_bind().dialect.name == "postgresql":
        _install_postgres_contract()


def downgrade() -> None:
    connection = op.get_bind()
    is_postgresql = connection.dialect.name == "postgresql"
    if is_postgresql:
        # FORCE RLS also applies to a NOBYPASSRLS table owner. Temporarily
        # restore ordinary owner visibility so the destructive downgrade gate
        # cannot mistake policy-hidden data for an empty table. PostgreSQL DDL
        # is transactional, and FORCE is restored explicitly before refusing.
        op.execute("ALTER TABLE public.app_chats NO FORCE ROW LEVEL SECURITY")
    has_rows = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM app_chats)")
    ).scalar_one()
    if has_rows:
        if is_postgresql:
            op.execute("ALTER TABLE public.app_chats FORCE ROW LEVEL SECURITY")
        raise RuntimeError(
            "refusing to drop authoritative app chat rows; export and remove them first"
        )
    op.drop_index("ix_app_chats_original_chat_id", table_name="app_chats")
    op.drop_index("ix_app_chats_owner_created", table_name="app_chats")
    op.drop_table("app_chats")
