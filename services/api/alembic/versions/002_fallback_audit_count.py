"""Add fallback_count to the point_query_fallback_audit audit ledger.

Revision ID: 002_fallback_audit_count
Revises: 001_initial_schema
Create Date: 2026-08-04 00:00:00.000000

Adds a cumulative ``fallback_count`` column to ``point_query_fallback_audit``
so concurrent fallback events for the same ``cache_key`` are accumulated by a
PostgreSQL ``ON CONFLICT`` upsert rather than dropped on a primary-key
collision (see DATABASE.md section 2).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "002_fallback_audit_count"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing rows are treated as having recorded a single fallback.
    op.add_column(
        "point_query_fallback_audit",
        sa.Column(
            "fallback_count",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )


def downgrade() -> None:
    op.drop_column("point_query_fallback_audit", "fallback_count")
