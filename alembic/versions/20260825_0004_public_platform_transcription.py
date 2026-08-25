"""Add durable public-platform transcription attempt ledger.

Revision ID: 20260825_0004
Revises: 20260825_0003
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260825_0004"
down_revision: str | None = "20260825_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "transcription_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=255), nullable=True),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="prepared"
        ),
        sa.Column("temp_audio_path", sa.Text(), nullable=False),
        sa.Column(
            "cleanup_status",
            sa.String(length=32),
            nullable=False,
            server_default="not_created",
        ),
        sa.Column("audio_sha256", sa.String(length=64), nullable=True),
        sa.Column("transcript_sha256", sa.String(length=64), nullable=True),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cleaned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "mode IN ('openai', 'local_cpu')", name="ck_transcription_runs_mode"
        ),
        sa.CheckConstraint(
            "status IN ('prepared', 'running', 'succeeded', 'failed', 'unknown')",
            name="ck_transcription_runs_status",
        ),
        sa.CheckConstraint(
            "cleanup_status IN ('pending', 'deleted', 'not_created', 'cleanup_failed')",
            name="ck_transcription_runs_cleanup_status",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["ingestion_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "attempt_number", name="uq_transcription_runs_attempt"
        ),
    )
    for column in (
        "audio_sha256",
        "cleanup_status",
        "job_id",
        "mode",
        "provider_request_id",
        "status",
        "transcript_sha256",
    ):
        op.create_index(
            f"ix_transcription_runs_{column}", "transcription_runs", [column]
        )


def downgrade() -> None:
    for column in (
        "transcript_sha256",
        "status",
        "provider_request_id",
        "mode",
        "job_id",
        "cleanup_status",
        "audio_sha256",
    ):
        op.drop_index(
            f"ix_transcription_runs_{column}", table_name="transcription_runs"
        )
    op.drop_table("transcription_runs")
