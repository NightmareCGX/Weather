"""Drop retired_at, retired_by_cycle_time, and idx_cycle_lifecycle_retired (Deployment 2).

Revision ID: 008_drop_retired_fields
Revises: 007_reclamation_queue
Create Date: 2026-09-12 00:00:00.000000

Data Lifecycle V3 Milestone 4 decoupled all serving and recovery behavior from
legacy V2 retirement fields. In Deployment 2, the schema is physically contracted
to remove:
    forecast_cycle_lifecycle.retired_at
    forecast_cycle_lifecycle.retired_by_cycle_time
    idx_cycle_lifecycle_retired

Authoritative lifecycle state is exclusively governed by:
    deletion_started_at (durable physical deletion claim / serving + mutation fence)
    deleted_at (permanent anti-resurrection tombstone)
    reclamation_queue (granular variable shard reclamation)

Downgrade Policy:
Downgrade recreates the columns as nullable TIMESTAMPTZ and restores the legacy
index for schema rollback compatibility, leaving existing rows as NULL. Downgrade
does NOT restore legacy V2 runtime semantics or undo physical data deletion.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "008_drop_retired_fields"
down_revision = "007_reclamation_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("forecast_cycle_lifecycle") as batch_op:
            batch_op.drop_index("idx_cycle_lifecycle_retired")
            batch_op.drop_column("retired_by_cycle_time")
            batch_op.drop_column("retired_at")
    else:
        op.drop_index("idx_cycle_lifecycle_retired", table_name="forecast_cycle_lifecycle")
        op.drop_column("forecast_cycle_lifecycle", "retired_by_cycle_time")
        op.drop_column("forecast_cycle_lifecycle", "retired_at")


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("forecast_cycle_lifecycle") as batch_op:
            batch_op.add_column(
                sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True)
            )
            batch_op.add_column(
                sa.Column("retired_by_cycle_time", sa.DateTime(timezone=True), nullable=True)
            )
            batch_op.create_index(
                "idx_cycle_lifecycle_retired",
                ["model_id", "retired_at"],
                unique=False,
            )
    else:
        op.add_column(
            "forecast_cycle_lifecycle",
            sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.add_column(
            "forecast_cycle_lifecycle",
            sa.Column("retired_by_cycle_time", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(
            "idx_cycle_lifecycle_retired",
            "forecast_cycle_lifecycle",
            ["model_id", "retired_at"],
            unique=False,
        )
