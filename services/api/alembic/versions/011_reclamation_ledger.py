"""Add reclamation_ledger — the durable per-target-group reclamation record.

Revision ID: 011_reclamation_ledger
Revises: 009_reclamation_store_gen
Create Date: 2026-10-02 00:00:00.000000

Two changes land together (docs/investigations/early-cycle-retirement/DESIGN.md):

*Part A — early cycle retirement.* A cycle whose every run is promoted and
whose every committed reclamation unit is physically gone may be claimed and
tombstoned without waiting for the 240h horizon gate. This is behavior-only
(the finalizer eligibility, flag-gated); no schema change.

*Part B — the reclamation ledger.* One row per
``(run_id, lead_time_hours, variable_code, target_kind)`` with a member bitmask
for the ensemble-member kind (bit *m* ⇔ member *m*, members 1..30). The worker
writes it in the same transaction that removes the terminal
``reclamation_queue`` row, so the queue becomes a pure in-flight work list
(queued/deleting/failed). The ledger is the "this unit is gone" record consumed
by planner idempotency, finalizer terminality, the serving physical fence, and
the counterfactual revalidation; it cascades away with the cycle's catalog at
the tombstone+retention sweep.

No backfill: pre-existing ``deleted`` queue rows keep fencing and satisfying
terminality through the union reads, and they drain within ~a day of Part A's
first tombstones.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "011_reclamation_ledger"
down_revision = "009_reclamation_store_gen"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reclamation_ledger",
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("model_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("lead_time_hours", sa.Integer(), nullable=False),
        sa.Column("variable_code", sa.String(), nullable=False),
        sa.Column("target_kind", sa.String(length=16), nullable=False),
        sa.Column(
            "deleted_members_mask", sa.BigInteger(), nullable=False,
            server_default="0",
        ),
        sa.Column("store_path", sa.String(), nullable=False),
        sa.Column("reclaimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "run_id", "lead_time_hours", "variable_code", "target_kind"
        ),
        sa.CheckConstraint(
            "target_kind IN ('det', 'mean', 'mem')",
            name="ck_reclamation_ledger_target_kind",
        ),
    )
    op.create_index(
        "idx_reclamation_ledger_store", "reclamation_ledger", ["store_path"]
    )


def downgrade() -> None:
    op.drop_index(
        "idx_reclamation_ledger_store", table_name="reclamation_ledger"
    )
    op.drop_table("reclamation_ledger")
