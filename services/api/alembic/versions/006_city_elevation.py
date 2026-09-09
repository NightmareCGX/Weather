"""Add nullable elevation_m column to cities table.

Revision ID: 006_city_elevation
Revises: 005_model_scoped_lifecycle
Create Date: 2026-09-08 00:00:00.000000

Tier 1 Known-Location Elevation:
Adds a nullable elevation_m column (meters above sea level) to the cities table
so persistent cities can store authoritative elevation metadata and resolve
without external API calls at query time. Schema-only; no external network calls.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "006_city_elevation"
down_revision = "005_model_scoped_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("cities") as batch_op:
            batch_op.add_column(sa.Column("elevation_m", sa.Float(), nullable=True))
    else:
        op.add_column("cities", sa.Column("elevation_m", sa.Float(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("cities") as batch_op:
            batch_op.drop_column("elevation_m")
    else:
        op.drop_column("cities", "elevation_m")
