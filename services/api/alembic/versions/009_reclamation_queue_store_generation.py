"""Add reclamation_queue.store_generation for the I14 replacement-evidence gate.

Revision ID: 009_reclamation_store_gen
Revises: 008_drop_retired_fields
Create Date: 2026-09-13 00:00:00.000000

Data Lifecycle V3 (architecture doc I14 / §11.2-2): the planner snapshots the
cycle store's committed-manifest generation when it enqueues a shard target,
and the worker refuses physical deletion while that baseline disagrees with
the store's current generation. A generation bump between observation and
deletion is physical evidence that the store was replaced (every EXCLUSIVE
finalizer commit bumps the generation, including same-set same-cycle
replacements), which invalidates the enqueue-time necessity judgement.

The column is nullable: rows enqueued before this migration carry no baseline
and are backfilled by the worker at first claim.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "009_reclamation_store_gen"
down_revision = "008_drop_retired_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reclamation_queue",
        sa.Column("store_generation", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reclamation_queue", "store_generation")
