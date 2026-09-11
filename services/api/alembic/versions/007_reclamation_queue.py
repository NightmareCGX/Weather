"""Add reclamation_queue table for Data Lifecycle V3 Phase 4 granular physical reclamation.

Revision ID: 007_reclamation_queue
Revises: 006_city_elevation
Create Date: 2026-09-11 00:00:00.000000

Data Lifecycle V3 Phase 4 introduces dependency-aware granular reclamation of
individual physical variable shards in sharded_v1 Zarr containers.

This table records the physical lifecycle of individual variable shards:
    queued   -> shard is queued for reclamation; still physically available
    deleting -> shard is leased and physically fenced from new serving
    deleted  -> shard has been physically deleted from object storage
    failed   -> shard deletion failed max retries and is quarantined

The run-scoped uniqueness constraint:
    UNIQUE (run_id, lead_time_hours, variable_code, target_kind, member_index)
guarantees idempotency and prevents concurrent duplicate work.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "007_reclamation_queue"
down_revision = "006_city_elevation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reclamation_queue",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("model_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            sa.String(),
            sa.ForeignKey("models.model_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cycle_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lead_time_hours", sa.Integer(), nullable=False),
        sa.Column("variable_code", sa.String(), nullable=False),
        sa.Column("target_kind", sa.String(length=16), nullable=False),
        sa.Column(
            "member_index",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("store_path", sa.String(), nullable=False),
        sa.Column("physical_key", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="queued",
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(), nullable=True),
        sa.Column("reclaimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id",
            "lead_time_hours",
            "variable_code",
            "target_kind",
            "member_index",
            name="uq_reclamation_queue_target",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'deleting', 'deleted', 'failed')",
            name="ck_reclamation_queue_status",
        ),
        sa.CheckConstraint(
            "target_kind IN ('det', 'mean', 'mem')",
            name="ck_reclamation_queue_target_kind",
        ),
        sa.CheckConstraint(
            "(target_kind = 'det' AND member_index = 0) OR "
            "(target_kind = 'mean' AND member_index = -1) OR "
            "(target_kind = 'mem' AND member_index >= 1 AND member_index <= 30)",
            name="ck_reclamation_queue_member_index",
        ),
    )

    op.create_index(
        "idx_reclamation_claim",
        "reclamation_queue",
        ["status", "next_retry_at", "lease_expires_at"],
        unique=False,
    )
    op.create_index(
        "idx_reclamation_run_status",
        "reclamation_queue",
        ["run_id", "status"],
        unique=False,
    )
    op.create_index(
        "idx_reclamation_physical_fence",
        "reclamation_queue",
        ["run_id", "physical_key", "status"],
        unique=False,
    )
    op.create_index(
        "idx_reclamation_region_audit",
        "reclamation_queue",
        ["run_id", "lead_time_hours", "target_kind", "member_index", "status"],
        unique=False,
    )
    op.create_index(
        "idx_reclamation_model_cycle",
        "reclamation_queue",
        ["model_id", "cycle_time", "lead_time_hours"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_reclamation_model_cycle", table_name="reclamation_queue")
    op.drop_index("idx_reclamation_region_audit", table_name="reclamation_queue")
    op.drop_index("idx_reclamation_physical_fence", table_name="reclamation_queue")
    op.drop_index("idx_reclamation_run_status", table_name="reclamation_queue")
    op.drop_index("idx_reclamation_claim", table_name="reclamation_queue")
    op.drop_table("reclamation_queue")
