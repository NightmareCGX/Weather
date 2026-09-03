"""Add deletion_started_at column to forecast_cycle_lifecycle for crash-safe deletion fencing.

Revision ID: 004_deletion_fence
Revises: 003_cycle_lifecycle
Create Date: 2026-09-03 00:00:00.000000

When physical GC begins deleting a retired, GC-eligible cycle, it sets
``deletion_started_at = NOW()`` to establish a durable cycle-level deletion
fence before acquiring per-model store gates. This prevents new/stale ingestion
writers from entering the cycle after GFS store deletion and before final
tombstoning (deleted_at).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "004_deletion_fence"
down_revision = "003_cycle_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "forecast_cycle_lifecycle",
        sa.Column("deletion_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_cycle_lifecycle_claimed",
        "forecast_cycle_lifecycle",
        ["deletion_started_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_cycle_lifecycle_claimed", table_name="forecast_cycle_lifecycle")
    op.drop_column("forecast_cycle_lifecycle", "deletion_started_at")
