"""Add durable hot-media custody and rehydration state.

The lifecycle rows extend the ingestion-owned global content-addressed media
catalog. They are operational facts about shared media objects, not tenant
records, so tenant RLS does not apply. They remain owner-only: PostgreSQL
PUBLIC and every known application runtime role receive no table privilege.

The upgrade is additive and leaves retained canonical media rows unchanged.
The downgrade is intentionally refused once lifecycle state exists because
dropping those rows would destroy custody evidence.

Revision ID: 20260826_0006
Revises: 20260825_0005
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260826_0006"
down_revision: str | None = "20260825_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


LIFECYCLE_TABLES = (
    "hot_media_custody_manifests",
    "hot_media_custody_items",
    "hot_media_rehydration_attempts",
)
CANONICAL_MEDIA_TABLES = (
    "media_objects",
    "media_locations",
    "source_videos",
    "video_media_refs",
)
KNOWN_RUNTIME_ROLES = (
    "icmfyi_runtime",
    "icmfyi_clip_api",
    "icmfyi_clip_worker",
    "icmfyi_payment_seller",
    "icmfyi_payment_worker",
)


def _assert_canonical_media_schema(connection) -> None:
    existing = set(sa.inspect(connection).get_table_names())
    missing = sorted(set(CANONICAL_MEDIA_TABLES) - existing)
    if missing:
        raise RuntimeError(
            "hot-media lifecycle requires ingestion canonical media tables: "
            + ",".join(missing)
        )


def _postgres_constraints_and_privileges() -> None:
    op.create_check_constraint(
        "ck_hot_media_custody_manifests_manifest_sha256_hex",
        "hot_media_custody_manifests",
        "manifest_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_custody_manifests_custody_receipt_sha256_hex",
        "hot_media_custody_manifests",
        "custody_receipt_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_custody_manifests_eviction_receipt_sha256_hex",
        "hot_media_custody_manifests",
        "eviction_receipt_sha256 IS NULL OR eviction_receipt_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_custody_items_media_sha256_hex",
        "hot_media_custody_items",
        "media_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_rehydration_attempts_request_key_sha256_hex",
        "hot_media_rehydration_attempts",
        "request_key_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_rehydrate_custody_manifest_sha_hex",
        "hot_media_rehydration_attempts",
        "custody_manifest_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_rehydration_attempts_media_sha256_hex",
        "hot_media_rehydration_attempts",
        "media_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_hot_media_rehydration_attempts_receipt_sha256_hex",
        "hot_media_rehydration_attempts",
        "receipt_sha256 IS NULL OR receipt_sha256 ~ '^[0-9a-f]{64}$'",
    )
    table_list = ", ".join(f"public.{name}" for name in LIFECYCLE_TABLES)
    op.execute(f"REVOKE ALL PRIVILEGES ON TABLE {table_list} FROM PUBLIC")
    roles = ",".join(f"'{role}'" for role in KNOWN_RUNTIME_ROLES)
    op.execute(
        f"""
        DO $lifecycle_privileges$
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
                        'REVOKE ALL PRIVILEGES ON TABLE {table_list} FROM %I',
                        runtime_role
                    );
                END IF;
            END LOOP;
        END
        $lifecycle_privileges$
        """
    )
    for table_name in LIFECYCLE_TABLES:
        op.execute(
            "COMMENT ON TABLE public."
            f"{table_name} IS "
            "'Global owner-only hot-media lifecycle state; not tenant-scoped'"
        )


def upgrade() -> None:
    connection = op.get_bind()
    _assert_canonical_media_schema(connection)

    op.create_table(
        "hot_media_custody_manifests",
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("manifest_bytes", sa.BigInteger(), nullable=False),
        sa.Column("items_count", sa.Integer(), nullable=False),
        sa.Column("media_bytes", sa.BigInteger(), nullable=False),
        sa.Column("remote_root", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("custody_receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("custody_receipt_json", sa.JSON(), nullable=False),
        sa.Column("eviction_receipt_sha256", sa.String(length=64), nullable=True),
        sa.Column("eviction_receipt_json", sa.JSON(), nullable=True),
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
        sa.Column(
            "custodied_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("eviction_prepared_at", sa.DateTime(timezone=True)),
        sa.Column("evicted_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "length(manifest_sha256) = 64",
            name="ck_hot_media_custody_manifests_manifest_sha256_length",
        ),
        sa.CheckConstraint(
            "manifest_bytes > 0",
            name="ck_hot_media_custody_manifests_manifest_bytes",
        ),
        sa.CheckConstraint(
            "items_count > 0",
            name="ck_hot_media_custody_manifests_items_count",
        ),
        sa.CheckConstraint(
            "media_bytes > 0",
            name="ck_hot_media_custody_manifests_media_bytes",
        ),
        sa.CheckConstraint(
            "status IN ('custodied', 'eviction_prepared', 'evicted')",
            name="ck_hot_media_custody_manifests_status",
        ),
        sa.CheckConstraint(
            "length(custody_receipt_sha256) = 64",
            name="ck_hot_media_custody_manifests_custody_receipt_sha256_length",
        ),
        sa.CheckConstraint(
            "(eviction_receipt_sha256 IS NULL AND eviction_receipt_json IS NULL) "
            "OR (length(eviction_receipt_sha256) = 64 "
            "AND eviction_receipt_json IS NOT NULL)",
            name="ck_hot_media_custody_manifests_eviction_receipt_pair",
        ),
        sa.PrimaryKeyConstraint("manifest_sha256"),
        sa.UniqueConstraint(
            "custody_receipt_sha256",
            name="uq_hot_media_custody_manifests_custody_receipt",
        ),
        sa.UniqueConstraint(
            "eviction_receipt_sha256",
            name="uq_hot_media_custody_manifests_eviction_receipt",
        ),
    )

    op.create_table(
        "hot_media_custody_items",
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("media_sha256", sa.String(length=64), nullable=False),
        sa.Column("hot_location_id", sa.String(length=64), nullable=False),
        sa.Column("hot_path", sa.Text(), nullable=False),
        sa.Column("appliance_hot_path", sa.Text(), nullable=False),
        sa.Column("remote_location_id", sa.String(length=64), nullable=False),
        sa.Column("remote_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
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
        sa.CheckConstraint("ordinal >= 0", name="ck_hot_media_custody_items_ordinal"),
        sa.CheckConstraint(
            "length(media_sha256) = 64",
            name="ck_hot_media_custody_items_media_sha256_length",
        ),
        sa.CheckConstraint(
            "size_bytes > 0", name="ck_hot_media_custody_items_size_bytes"
        ),
        sa.CheckConstraint(
            "mime_type LIKE 'video/%'",
            name="ck_hot_media_custody_items_video_mime",
        ),
        sa.CheckConstraint(
            "state IN ('custodied', 'eviction_prepared', 'evicted')",
            name="ck_hot_media_custody_items_state",
        ),
        sa.ForeignKeyConstraint(
            ["manifest_sha256"],
            ["hot_media_custody_manifests.manifest_sha256"],
            name="fk_hot_media_custody_items_manifest",
        ),
        sa.ForeignKeyConstraint(
            ["media_sha256"],
            ["media_objects.sha256"],
            name="fk_hot_media_custody_items_media",
        ),
        sa.ForeignKeyConstraint(
            ["hot_location_id"],
            ["media_locations.id"],
            name="fk_hot_media_custody_items_hot_location",
        ),
        sa.ForeignKeyConstraint(
            ["remote_location_id"],
            ["media_locations.id"],
            name="fk_hot_media_custody_items_remote_location",
        ),
        sa.PrimaryKeyConstraint("manifest_sha256", "media_sha256"),
        sa.UniqueConstraint(
            "manifest_sha256",
            "ordinal",
            name="uq_hot_media_custody_items_manifest_ordinal",
        ),
        sa.UniqueConstraint(
            "manifest_sha256",
            "hot_path",
            name="uq_hot_media_custody_items_manifest_hot_path",
        ),
        sa.UniqueConstraint(
            "manifest_sha256",
            "appliance_hot_path",
            name="uq_hot_media_custody_items_manifest_appliance_hot_path",
        ),
        sa.UniqueConstraint(
            "manifest_sha256",
            "remote_path",
            name="uq_hot_media_custody_items_manifest_remote_path",
        ),
    )
    op.create_index(
        "ix_hot_media_custody_items_media_sha256",
        "hot_media_custody_items",
        ["media_sha256"],
    )
    op.create_index(
        "ix_hot_media_custody_items_state",
        "hot_media_custody_items",
        ["state"],
    )

    op.create_table(
        "hot_media_rehydration_attempts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("request_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("custody_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("media_sha256", sa.String(length=64), nullable=False),
        sa.Column("storagebox_location_id", sa.String(length=64), nullable=False),
        sa.Column("hot_location_id", sa.String(length=64), nullable=False),
        sa.Column("attempt_path", sa.Text(), nullable=False),
        sa.Column("final_hot_path", sa.Text(), nullable=False),
        sa.Column("final_appliance_path", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column("receipt_sha256", sa.String(length=64)),
        sa.Column("receipt_json", sa.JSON()),
        sa.Column("last_error_code", sa.String(length=128)),
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
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "length(request_key_sha256) = 64",
            name="ck_hot_media_rehydration_attempts_request_key_sha256_length",
        ),
        sa.CheckConstraint(
            "length(custody_manifest_sha256) = 64",
            name="ck_hot_media_rehydrate_custody_manifest_sha_length",
        ),
        sa.CheckConstraint(
            "length(media_sha256) = 64",
            name="ck_hot_media_rehydration_attempts_media_sha256_length",
        ),
        sa.CheckConstraint(
            "state IN ('downloading', 'failed', 'ready')",
            name="ck_hot_media_rehydration_attempts_state",
        ),
        sa.CheckConstraint(
            "attempt_count > 0",
            name="ck_hot_media_rehydration_attempts_attempt_count",
        ),
        sa.CheckConstraint(
            "(state = 'ready' AND length(receipt_sha256) = 64 "
            "AND receipt_json IS NOT NULL) OR "
            "(state <> 'ready' AND receipt_sha256 IS NULL "
            "AND receipt_json IS NULL)",
            name="ck_hot_media_rehydration_attempts_receipt_state",
        ),
        sa.ForeignKeyConstraint(
            ["custody_manifest_sha256"],
            ["hot_media_custody_manifests.manifest_sha256"],
            name="fk_hot_media_rehydration_attempts_custody_manifest",
        ),
        sa.ForeignKeyConstraint(
            ["media_sha256"],
            ["media_objects.sha256"],
            name="fk_hot_media_rehydration_attempts_media",
        ),
        sa.ForeignKeyConstraint(
            ["storagebox_location_id"],
            ["media_locations.id"],
            name="fk_hot_media_rehydration_attempts_storagebox_location",
        ),
        sa.ForeignKeyConstraint(
            ["hot_location_id"],
            ["media_locations.id"],
            name="fk_hot_media_rehydration_attempts_hot_location",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_key_sha256",
            "media_sha256",
            "custody_manifest_sha256",
            name="uq_hot_media_rehydration_attempts_request_media_manifest",
        ),
        sa.UniqueConstraint(
            "receipt_sha256",
            name="uq_hot_media_rehydration_attempts_receipt",
        ),
    )
    op.create_index(
        "ix_hot_media_rehydration_attempts_custody_manifest_sha256",
        "hot_media_rehydration_attempts",
        ["custody_manifest_sha256"],
    )
    op.create_index(
        "ix_hot_media_rehydration_attempts_media_sha256",
        "hot_media_rehydration_attempts",
        ["media_sha256"],
    )
    op.create_index(
        "ix_hot_media_rehydration_attempts_state",
        "hot_media_rehydration_attempts",
        ["state"],
    )

    if connection.dialect.name == "postgresql":
        _postgres_constraints_and_privileges()


def _assert_lifecycle_empty(connection) -> None:
    counts = {
        table_name: int(
            connection.execute(
                sa.text(f"SELECT count(*) FROM {table_name}")
            ).scalar_one()
        )
        for table_name in LIFECYCLE_TABLES
    }
    retained = {name: count for name, count in counts.items() if count}
    if retained:
        summary = ",".join(f"{name}={retained[name]}" for name in sorted(retained))
        raise RuntimeError(
            "downgrade would destroy durable hot-media lifecycle state: " + summary
        )


def downgrade() -> None:
    connection = op.get_bind()
    _assert_lifecycle_empty(connection)
    op.drop_table("hot_media_rehydration_attempts")
    op.drop_table("hot_media_custody_items")
    op.drop_table("hot_media_custody_manifests")
