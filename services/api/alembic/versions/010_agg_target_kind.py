"""Add 'agg' to the reclamation_queue target_kind domain.

Revision ID: 010_agg_target_kind
Revises: 009_reclamation_store_gen
Create Date: 2026-09-20 00:00:00.000000

The ensemble aggregate encodings replace a variable's 30 member shards with one
statistic container per ``(variable, lead)``. That container is a deletion unit of
its own -- it is what gets reclaimed when a *newer* aggregate replaces it -- so
``reclamation_queue`` has to be able to name it.

Two check constraints are relaxed, and only these two:

* ``ck_reclamation_queue_target_kind`` gains ``'agg'``;
* ``ck_reclamation_queue_member_index`` gains ``(target_kind = 'agg' AND
  member_index = 0)``. A container is not a member, so 0 is its normalized index --
  the same convention ``det`` uses -- while ``mem`` keeps its 1..30 range. The
  bound stays a check constraint rather than a looser ``>= 0`` so a member row with
  member_index 0 (a missing identity, not a member) is still refused.

The uniqueness constraint is unchanged and already admits the new kind: it is keyed
on ``(run_id, lead_time_hours, variable_code, target_kind, member_index)``, and
``agg`` has exactly one row per ``(run, lead, variable)``.
"""

from alembic import op

# revision identifiers, used by Alembic (max 32 chars).
revision = "010_agg_target_kind"
down_revision = "009_reclamation_store_gen"
branch_labels = None
depends_on = None


_TARGET_KIND_UPGRADE = (
    "(target_kind = 'det' AND member_index = 0) OR "
    "(target_kind = 'mean' AND member_index = -1) OR "
    "(target_kind = 'agg' AND member_index = 0) OR "
    "(target_kind = 'mem' AND member_index >= 1 AND member_index <= 30)"
)

_TARGET_KIND_DOWNGRADE = (
    "(target_kind = 'det' AND member_index = 0) OR "
    "(target_kind = 'mean' AND member_index = -1) OR "
    "(target_kind = 'mem' AND member_index >= 1 AND member_index <= 30)"
)


def _rebuild(name: str, condition: str) -> None:
    """Replace one check constraint, batching for SQLite.

    SQLite cannot alter a constraint in place, so the table is rebuilt; PostgreSQL can drop and
    re-add. Both paths are here because the test suite runs on SQLite and the deployment on
    PostgreSQL, and a migration that only works on one of them would fail silently in the other.
    """
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("reclamation_queue") as batch_op:
            batch_op.drop_constraint(name, type_="check")
            batch_op.create_check_constraint(name, condition)
        return
    op.drop_constraint(name, "reclamation_queue", type_="check")
    op.create_check_constraint(name, "reclamation_queue", condition)


def upgrade() -> None:
    _rebuild("ck_reclamation_queue_target_kind", "target_kind IN ('det', 'mean', 'mem', 'agg')")
    _rebuild("ck_reclamation_queue_member_index", _TARGET_KIND_UPGRADE)


def downgrade() -> None:
    # Rows of the new kind cannot satisfy the narrowed constraint, so they are removed before it
    # is re-added. Downgrade is a schema rollback, not a data recovery: an aggregate row names an
    # object this schema version cannot describe.
    op.execute("DELETE FROM reclamation_queue WHERE target_kind = 'agg'")
    _rebuild("ck_reclamation_queue_member_index", _TARGET_KIND_DOWNGRADE)
    _rebuild("ck_reclamation_queue_target_kind", "target_kind IN ('det', 'mean', 'mem')")
