"""Add crash-safe archive import, tenant claim, and hydration ledgers.

Revision ID: 20260825_0003
Revises: 20260825_0002
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260825_0003"
down_revision: str | None = "20260825_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "archive_catalog_imports",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("jsonl_sha256", sa.String(length=64), nullable=False),
        sa.Column("sidecar_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_filename", sa.String(length=255), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("receipt_json", sa.JSON(), nullable=False),
        sa.Column("source_keys_json", sa.JSON(), nullable=False),
        sa.Column("source_ids_json", sa.JSON(), nullable=False),
        sa.Column("video_ids_json", sa.JSON(), nullable=False),
        sa.Column("media_sha256s_json", sa.JSON(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="applied"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status = 'applied'", name="ck_archive_catalog_imports_status"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("jsonl_sha256"),
        sa.UniqueConstraint("receipt_sha256"),
    )
    op.create_index(
        "ix_archive_catalog_imports_jsonl_sha256",
        "archive_catalog_imports",
        ["jsonl_sha256"],
    )
    op.create_index(
        "ix_archive_catalog_imports_receipt_sha256",
        "archive_catalog_imports",
        ["receipt_sha256"],
    )
    op.create_index(
        "ix_archive_catalog_imports_status", "archive_catalog_imports", ["status"]
    )

    op.create_table(
        "archive_tenant_claims",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=68), nullable=False),
        sa.Column("admin_user_id", sa.String(length=68), nullable=False),
        sa.Column("catalog_import_id", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("source_ids_json", sa.JSON(), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("receipt_json", sa.JSON(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="applied"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status = 'applied'", name="ck_archive_tenant_claims_status"
        ),
        sa.ForeignKeyConstraint(["admin_user_id"], ["user_accounts.id"]),
        sa.ForeignKeyConstraint(["catalog_import_id"], ["archive_catalog_imports.id"]),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_archive_tenant_claims_key"
        ),
        sa.UniqueConstraint("receipt_sha256"),
    )
    for column in (
        "admin_user_id",
        "catalog_import_id",
        "receipt_sha256",
        "status",
        "tenant_id",
    ):
        op.create_index(
            f"ix_archive_tenant_claims_{column}", "archive_tenant_claims", [column]
        )

    op.create_table(
        "archive_hydration_registrations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("input_receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("media_sha256", sa.String(length=64), nullable=False),
        sa.Column("location_id", sa.String(length=64), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("receipt_json", sa.JSON(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="applied"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status = 'applied'", name="ck_archive_hydration_registrations_status"
        ),
        sa.ForeignKeyConstraint(["location_id"], ["media_locations.id"]),
        sa.ForeignKeyConstraint(["media_sha256"], ["media_objects.sha256"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("input_receipt_sha256"),
        sa.UniqueConstraint("receipt_sha256"),
    )
    for column in (
        "input_receipt_sha256",
        "location_id",
        "media_sha256",
        "receipt_sha256",
        "status",
    ):
        op.create_index(
            f"ix_archive_hydration_registrations_{column}",
            "archive_hydration_registrations",
            [column],
        )


def downgrade() -> None:
    for column in (
        "status",
        "receipt_sha256",
        "media_sha256",
        "location_id",
        "input_receipt_sha256",
    ):
        op.drop_index(
            f"ix_archive_hydration_registrations_{column}",
            table_name="archive_hydration_registrations",
        )
    op.drop_table("archive_hydration_registrations")

    for column in (
        "tenant_id",
        "status",
        "receipt_sha256",
        "catalog_import_id",
        "admin_user_id",
    ):
        op.drop_index(
            f"ix_archive_tenant_claims_{column}", table_name="archive_tenant_claims"
        )
    op.drop_table("archive_tenant_claims")

    op.drop_index(
        "ix_archive_catalog_imports_status", table_name="archive_catalog_imports"
    )
    op.drop_index(
        "ix_archive_catalog_imports_receipt_sha256",
        table_name="archive_catalog_imports",
    )
    op.drop_index(
        "ix_archive_catalog_imports_jsonl_sha256",
        table_name="archive_catalog_imports",
    )
    op.drop_table("archive_catalog_imports")
