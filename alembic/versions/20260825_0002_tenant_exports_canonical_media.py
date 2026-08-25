"""Add canonical media, transcript, and tenant export schema.

Revision ID: 20260825_0002
Revises: 20260825_0001
Create Date: 2026-08-25
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260825_0002"
down_revision: str | None = "20260825_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES = (
    "tenant_channel_entitlements",
    "ingestion_requests",
    "tenant_exports",
)


def upgrade() -> None:
    _alter_identity_columns(from_length=64, to_length=68)
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "request_tenant_ids_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    _backfill_request_tenants()
    op.add_column(
        "source_channels",
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
    )
    op.create_index("ix_source_channels_status", "source_channels", ["status"])

    op.create_table(
        "source_videos",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("channel_id", sa.String(length=64), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column(
            "archive_state",
            sa.String(length=32),
            nullable=False,
            server_default="pending_discovery",
        ),
        sa.Column(
            "clip_candidate", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "clip_ready", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["channel_id"], ["source_channels.id"]),
        sa.CheckConstraint(
            "archive_state IN ('pending_discovery', 'retained_remote_verified', "
            "'retained_hot_verified', 'partial_only')",
            name="ck_source_videos_archive_state",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "platform", "external_id", name="uq_source_videos_provider_item"
        ),
    )
    op.create_index("ix_source_videos_channel_id", "source_videos", ["channel_id"])
    op.create_index(
        "ix_source_videos_archive_state", "source_videos", ["archive_state"]
    )
    op.create_index(
        "ix_source_videos_clip_candidate", "source_videos", ["clip_candidate"]
    )
    op.create_index("ix_source_videos_clip_ready", "source_videos", ["clip_ready"])
    op.create_index("ix_source_videos_platform", "source_videos", ["platform"])
    op.create_index("ix_source_videos_published_at", "source_videos", ["published_at"])
    op.create_index("ix_source_videos_status", "source_videos", ["status"])

    op.create_table(
        "media_objects",
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("sha256"),
    )
    op.create_index("ix_media_objects_status", "media_objects", ["status"])

    op.create_table(
        "transcript_revisions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("video_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("provider_revision_id", sa.String(length=255), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["source_videos.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "provider_revision_id",
            name="uq_transcript_revisions_provider_item",
        ),
    )
    op.create_index(
        "ix_transcript_revisions_content_sha256",
        "transcript_revisions",
        ["content_sha256"],
    )
    op.create_index(
        "ix_transcript_revisions_is_current",
        "transcript_revisions",
        ["is_current"],
    )
    op.create_index(
        "ix_transcript_revisions_provider", "transcript_revisions", ["provider"]
    )
    op.create_index(
        "ix_transcript_revisions_status", "transcript_revisions", ["status"]
    )
    op.create_index(
        "ix_transcript_revisions_video_id", "transcript_revisions", ["video_id"]
    )
    op.create_index(
        "uq_transcript_revisions_current_video",
        "transcript_revisions",
        ["video_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current = 1"),
    )

    op.create_table(
        "media_locations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("media_sha256", sa.String(length=64), nullable=False),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("location_key", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "backend IN ('hot_local', 'storagebox')", name="ck_media_locations_backend"
        ),
        sa.CheckConstraint(
            "status IN ('active', 'pending', 'missing', 'corrupt')",
            name="ck_media_locations_status",
        ),
        sa.ForeignKeyConstraint(["media_sha256"], ["media_objects.sha256"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "backend", "location_key", name="uq_media_locations_backend_key"
        ),
    )
    op.create_index("ix_media_locations_backend", "media_locations", ["backend"])
    op.create_index(
        "ix_media_locations_media_sha256", "media_locations", ["media_sha256"]
    )
    op.create_index("ix_media_locations_status", "media_locations", ["status"])
    op.create_index(
        "ix_media_locations_verified_at", "media_locations", ["verified_at"]
    )
    op.create_index(
        "uq_media_locations_active_hot_local",
        "media_locations",
        ["media_sha256"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND backend = 'hot_local'"),
        sqlite_where=sa.text("status = 'active' AND backend = 'hot_local'"),
    )

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("revision_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("speaker_label", sa.String(length=255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["revision_id"], ["transcript_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "revision_id", "ordinal", name="uq_transcript_segments_ordinal"
        ),
    )
    op.create_index(
        "ix_transcript_segments_revision_id", "transcript_segments", ["revision_id"]
    )
    op.create_index("ix_transcript_segments_status", "transcript_segments", ["status"])

    op.create_table(
        "tenant_exports",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=68), nullable=False),
        sa.Column("requested_by_user_id", sa.String(length=68), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="running"
        ),
        sa.Column("snapshot_sha256", sa.String(length=64), nullable=True),
        sa.Column("database_sha256", sa.String(length=64), nullable=True),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("database_path", sa.Text(), nullable=True),
        sa.Column("manifest_path", sa.Text(), nullable=True),
        sa.Column("counts_json", sa.JSON(), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["user_accounts.id"]),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_tenant_exports_tenant_idempotency",
        ),
    )
    op.create_index(
        "ix_tenant_exports_database_sha256", "tenant_exports", ["database_sha256"]
    )
    op.create_index(
        "ix_tenant_exports_requested_by_user_id",
        "tenant_exports",
        ["requested_by_user_id"],
    )
    op.create_index("ix_tenant_exports_status", "tenant_exports", ["status"])
    op.create_index("ix_tenant_exports_tenant_id", "tenant_exports", ["tenant_id"])

    op.create_table(
        "video_media_refs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("video_id", sa.String(length=64), nullable=False),
        sa.Column("media_sha256", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="active"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role IN ('source_video', 'proxy', 'audio', 'thumbnail')",
            name="ck_video_media_refs_role",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'inactive')", name="ck_video_media_refs_status"
        ),
        sa.ForeignKeyConstraint(["media_sha256"], ["media_objects.sha256"]),
        sa.ForeignKeyConstraint(["video_id"], ["source_videos.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "video_id",
            "media_sha256",
            "role",
            name="uq_video_media_refs_identity",
        ),
    )
    op.create_index(
        "ix_video_media_refs_media_sha256", "video_media_refs", ["media_sha256"]
    )
    op.create_index("ix_video_media_refs_status", "video_media_refs", ["status"])
    op.create_index("ix_video_media_refs_video_id", "video_media_refs", ["video_id"])

    if op.get_bind().dialect.name == "postgresql":
        for table_name in _TENANT_TABLES:
            op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"""
                CREATE POLICY {table_name}_tenant_isolation ON {table_name}
                USING (tenant_id = current_setting('app.tenant_id', true))
                WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
                """
            )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for table_name in reversed(_TENANT_TABLES):
            op.execute(
                f"DROP POLICY IF EXISTS {table_name}_tenant_isolation ON {table_name}"
            )
            op.execute(f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")

    op.drop_index("ix_video_media_refs_video_id", table_name="video_media_refs")
    op.drop_index("ix_video_media_refs_status", table_name="video_media_refs")
    op.drop_index("ix_video_media_refs_media_sha256", table_name="video_media_refs")
    op.drop_table("video_media_refs")

    op.drop_index("ix_tenant_exports_tenant_id", table_name="tenant_exports")
    op.drop_index("ix_tenant_exports_status", table_name="tenant_exports")
    op.drop_index("ix_tenant_exports_requested_by_user_id", table_name="tenant_exports")
    op.drop_index("ix_tenant_exports_database_sha256", table_name="tenant_exports")
    op.drop_table("tenant_exports")

    op.drop_index("ix_transcript_segments_status", table_name="transcript_segments")
    op.drop_index(
        "ix_transcript_segments_revision_id", table_name="transcript_segments"
    )
    op.drop_table("transcript_segments")

    op.drop_index("uq_media_locations_active_hot_local", table_name="media_locations")
    op.drop_index("ix_media_locations_verified_at", table_name="media_locations")
    op.drop_index("ix_media_locations_status", table_name="media_locations")
    op.drop_index("ix_media_locations_media_sha256", table_name="media_locations")
    op.drop_index("ix_media_locations_backend", table_name="media_locations")
    op.drop_table("media_locations")

    op.drop_index(
        "uq_transcript_revisions_current_video", table_name="transcript_revisions"
    )
    op.drop_index("ix_transcript_revisions_video_id", table_name="transcript_revisions")
    op.drop_index("ix_transcript_revisions_status", table_name="transcript_revisions")
    op.drop_index("ix_transcript_revisions_provider", table_name="transcript_revisions")
    op.drop_index(
        "ix_transcript_revisions_is_current", table_name="transcript_revisions"
    )
    op.drop_index(
        "ix_transcript_revisions_content_sha256", table_name="transcript_revisions"
    )
    op.drop_table("transcript_revisions")

    op.drop_index("ix_media_objects_status", table_name="media_objects")
    op.drop_table("media_objects")

    op.drop_index("ix_source_videos_status", table_name="source_videos")
    op.drop_index("ix_source_videos_published_at", table_name="source_videos")
    op.drop_index("ix_source_videos_platform", table_name="source_videos")
    op.drop_index("ix_source_videos_clip_ready", table_name="source_videos")
    op.drop_index("ix_source_videos_clip_candidate", table_name="source_videos")
    op.drop_index("ix_source_videos_archive_state", table_name="source_videos")
    op.drop_index("ix_source_videos_channel_id", table_name="source_videos")
    op.drop_table("source_videos")

    op.drop_index("ix_source_channels_status", table_name="source_channels")
    op.drop_column("source_channels", "status")
    op.drop_column("ingestion_jobs", "request_tenant_ids_json")
    _alter_identity_columns(from_length=68, to_length=64)


def _alter_identity_columns(*, from_length: int, to_length: int) -> None:
    columns = (
        ("user_accounts", "id", False),
        ("tenants", "id", False),
        ("tenant_memberships", "tenant_id", False),
        ("tenant_memberships", "user_id", False),
        ("tenant_channel_entitlements", "tenant_id", False),
        ("tenant_channel_entitlements", "granted_by_user_id", True),
        ("ingestion_requests", "tenant_id", False),
        ("ingestion_requests", "requested_by_user_id", True),
    )
    ordered_columns = columns if to_length > from_length else tuple(reversed(columns))
    for table_name, column_name, nullable in ordered_columns:
        if op.get_bind().dialect.name == "sqlite":
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.alter_column(
                    column_name,
                    existing_type=sa.String(length=from_length),
                    type_=sa.String(length=to_length),
                    existing_nullable=nullable,
                )
        else:
            op.alter_column(
                table_name,
                column_name,
                existing_type=sa.String(length=from_length),
                type_=sa.String(length=to_length),
                existing_nullable=nullable,
            )


def _backfill_request_tenants() -> None:
    jobs = sa.table(
        "ingestion_jobs",
        sa.column("id", sa.String(length=64)),
        sa.column("request_tenant_ids_json", sa.JSON()),
    )
    requests = sa.table(
        "ingestion_requests",
        sa.column("job_id", sa.String(length=64)),
        sa.column("tenant_id", sa.String(length=68)),
    )
    bind = op.get_bind()
    tenants_by_job: dict[str, set[str]] = {}
    for job_id, tenant_id in bind.execute(
        sa.select(requests.c.job_id, requests.c.tenant_id)
    ):
        tenants_by_job.setdefault(str(job_id), set()).add(str(tenant_id))
    for job_id, tenant_ids in tenants_by_job.items():
        bind.execute(
            sa.update(jobs)
            .where(jobs.c.id == job_id)
            .values(request_tenant_ids_json=sorted(tenant_ids))
        )
