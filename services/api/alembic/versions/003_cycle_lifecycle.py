"""Add forecast_cycle_lifecycle table for durable cycle retirement and retention metadata.

Revision ID: 003_cycle_lifecycle
Revises: 002_ensemble_member_products
Create Date: 2026-09-02 00:00:00.000000

The forecast lifecycle unit is a logical forecast cycle (paired GFS + GEFS at
cycle_time C). This table tracks cycle-level visibility retirement and physical
GC eligibility timestamps independently of individual model_runs rows.
Tombstones (deleted_at) survive physical forecast store deletion for auditability
and to prevent accidental resurrection by stale manual jobs.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "003_cycle_lifecycle"
down_revision = "002_ensemble_member_products"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_cycle_lifecycle",
        sa.Column("cycle_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_by_cycle_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("cycle_time"),
    )
    op.create_index(
        "idx_cycle_lifecycle_retired",
        "forecast_cycle_lifecycle",
        ["retired_at"],
        unique=False,
    )
    op.create_index(
        "idx_cycle_lifecycle_deleted",
        "forecast_cycle_lifecycle",
        ["deleted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_cycle_lifecycle_deleted", table_name="forecast_cycle_lifecycle")
    op.drop_index("idx_cycle_lifecycle_retired", table_name="forecast_cycle_lifecycle")
    op.drop_table("forecast_cycle_lifecycle")
