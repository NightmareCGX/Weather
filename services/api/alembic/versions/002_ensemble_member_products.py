"""Add ensemble_member_products to track committed (member, lead) pairs.

Revision ID: 002_ensemble_member_products
Revises: 001_initial_schema
Create Date: 2026-08-14 00:00:00.000000

The ingestion catalog's ``forecast_products`` rows record lead completion
without member identity, and ``ensemble_members`` rows record member presence
without lead identity. Neither can answer "has member 3 committed lead 6".
This table records exactly the committed ``(member_index, lead_time_hours)``
pairs (one row per per-file ensemble ingest) so run-level readiness can
enforce the Cartesian product of expected members × expected leads.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "002_ensemble_member_products"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ensemble_member_products",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("member_index", sa.Integer(), nullable=False),
        sa.Column("lead_time_hours", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["model_runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "member_index",
            "lead_time_hours",
            name="uq_ensemble_member_product",
        ),
    )


def downgrade() -> None:
    op.drop_table("ensemble_member_products")
