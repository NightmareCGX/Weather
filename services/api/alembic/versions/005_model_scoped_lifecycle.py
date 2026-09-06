"""Update forecast_cycle_lifecycle to model-scoped primary key (model_id, cycle_time).

Revision ID: 005_model_scoped_lifecycle
Revises: 004_deletion_fence
Create Date: 2026-09-04 00:00:00.000000

Data Lifecycle V2 decouples model lifecycles so GFS and GEFS advance retention
independently. This migration updates the forecast_cycle_lifecycle table to be
keyed by (model_id, cycle_time). Existing rows (representing paired cycles) are
mapped safely to both 'gfs' and 'gefs' rows so no pre-existing lifecycle audit
metadata is lost.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic (max 32 chars).
revision = "005_model_scoped_lifecycle"
down_revision = "004_deletion_fence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("forecast_cycle_lifecycle", recreate="always") as batch_op:
            batch_op.add_column(
                sa.Column("model_id", sa.String(), nullable=False, server_default="gfs")
            )
            # Recreate with composite primary key and foreign key
            batch_op.create_primary_key("pk_forecast_cycle_lifecycle", ["model_id", "cycle_time"])
            batch_op.create_foreign_key(
                "fk_cycle_lifecycle_model_id",
                "models",
                ["model_id"],
                ["model_id"],
                ondelete="CASCADE",
            )
            batch_op.create_index(
                "idx_cycle_lifecycle_retired",
                ["model_id", "retired_at"],
                unique=False,
            )
            batch_op.create_index(
                "idx_cycle_lifecycle_claimed",
                ["model_id", "deletion_started_at"],
                unique=False,
            )
            batch_op.create_index(
                "idx_cycle_lifecycle_deleted",
                ["model_id", "deleted_at"],
                unique=False,
            )
    else:
        # PostgreSQL / production path
        # 1. Drop old indexes
        op.drop_index("idx_cycle_lifecycle_claimed", table_name="forecast_cycle_lifecycle")
        op.drop_index("idx_cycle_lifecycle_deleted", table_name="forecast_cycle_lifecycle")
        op.drop_index("idx_cycle_lifecycle_retired", table_name="forecast_cycle_lifecycle")

        # 2. Add nullable model_id column
        op.add_column(
            "forecast_cycle_lifecycle",
            sa.Column("model_id", sa.String(), nullable=True),
        )

        # 3. Migrate existing paired rows: assign 'gfs' to existing rows, duplicate for 'gefs'
        conn = op.get_bind()
        conn.execute(
            sa.text(
                "UPDATE forecast_cycle_lifecycle SET model_id = 'gfs' WHERE model_id IS NULL"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO forecast_cycle_lifecycle "
                "(model_id, cycle_time, retired_at, retired_by_cycle_time, deletion_started_at, deleted_at, created_at, updated_at) "
                "SELECT 'gefs', cycle_time, retired_at, retired_by_cycle_time, deletion_started_at, deleted_at, created_at, updated_at "
                "FROM forecast_cycle_lifecycle WHERE model_id = 'gfs'"
            )
        )

        # 4. Alter model_id to NOT NULL
        op.alter_column("forecast_cycle_lifecycle", "model_id", nullable=False)

        # 5. Drop old primary key constraint
        # In PostgreSQL the default PK constraint is named <tablename>_pkey
        op.drop_constraint("forecast_cycle_lifecycle_pkey", "forecast_cycle_lifecycle", type_="primary")

        # 6. Create composite primary key
        op.create_primary_key(
            "pk_forecast_cycle_lifecycle",
            "forecast_cycle_lifecycle",
            ["model_id", "cycle_time"],
        )

        # 7. Add foreign key to models
        op.create_foreign_key(
            "fk_cycle_lifecycle_model_id",
            "forecast_cycle_lifecycle",
            "models",
            ["model_id"],
            ["model_id"],
            ondelete="CASCADE",
        )

        # 8. Recreate indexes with model_id
        op.create_index(
            "idx_cycle_lifecycle_retired",
            "forecast_cycle_lifecycle",
            ["model_id", "retired_at"],
            unique=False,
        )
        op.create_index(
            "idx_cycle_lifecycle_claimed",
            "forecast_cycle_lifecycle",
            ["model_id", "deletion_started_at"],
            unique=False,
        )
        op.create_index(
            "idx_cycle_lifecycle_deleted",
            "forecast_cycle_lifecycle",
            ["model_id", "deleted_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if is_sqlite:
        with op.batch_alter_table("forecast_cycle_lifecycle", recreate="always") as batch_op:
            batch_op.drop_constraint("fk_cycle_lifecycle_model_id", type_="foreignkey")
            batch_op.create_primary_key("pk_forecast_cycle_lifecycle", ["cycle_time"])
            batch_op.drop_column("model_id")
            batch_op.create_index("idx_cycle_lifecycle_retired", ["retired_at"], unique=False)
            batch_op.create_index("idx_cycle_lifecycle_claimed", ["deletion_started_at"], unique=False)
            batch_op.create_index("idx_cycle_lifecycle_deleted", ["deleted_at"], unique=False)
    else:
        op.drop_index("idx_cycle_lifecycle_deleted", table_name="forecast_cycle_lifecycle")
        op.drop_index("idx_cycle_lifecycle_claimed", table_name="forecast_cycle_lifecycle")
        op.drop_index("idx_cycle_lifecycle_retired", table_name="forecast_cycle_lifecycle")
        op.drop_constraint("fk_cycle_lifecycle_model_id", "forecast_cycle_lifecycle", type_="foreignkey")
        op.drop_constraint("pk_forecast_cycle_lifecycle", "forecast_cycle_lifecycle", type_="primary")

        conn = op.get_bind()
        conn.execute(sa.text("DELETE FROM forecast_cycle_lifecycle WHERE model_id != 'gfs'"))

        op.create_primary_key("forecast_cycle_lifecycle_pkey", "forecast_cycle_lifecycle", ["cycle_time"])
        op.drop_column("forecast_cycle_lifecycle", "model_id")
        op.create_index("idx_cycle_lifecycle_retired", "forecast_cycle_lifecycle", ["retired_at"], unique=False)
        op.create_index("idx_cycle_lifecycle_claimed", "forecast_cycle_lifecycle", ["deletion_started_at"], unique=False)
        op.create_index("idx_cycle_lifecycle_deleted", "forecast_cycle_lifecycle", ["deleted_at"], unique=False)
