"""Add the covering index for the serving physical-fence filter.

Revision ID: 010_reclamation_fence_idx
Revises: 009_reclamation_store_gen
Create Date: 2026-09-14 00:00:00.000000

Every serving read path (``api.services.resolver``, ``api.services.availability``)
filters physical shards through a correlated ``NOT EXISTS`` over
``reclamation_queue`` keyed on:

    (run_id, lead_time_hours, variable_code, target_kind, status)

No existing index covers that combination: ``uq_reclamation_queue_target``
carries the first four columns but not ``status``, and
``idx_reclamation_region_audit`` carries ``run_id``/``lead_time_hours``/
``target_kind``/``member_index``/``status`` but not ``variable_code``. The
predicate therefore fell back to a sequential scan of ``reclamation_queue`` on
every serving query:

    Hash Right Anti Join  (actual time=62.869..62.899 rows=100)
      ->  Seq Scan on reclamation_queue rq  (Rows Removed by Filter: 236235)
          Buffers: shared hit=11858
    Execution Time: 63.073 ms

with a cost that grows linearly with the (append-mostly, unbounded) queue.

``status`` is placed last because the fence predicate is an ``IN`` over a small
set of statuses while the identity columns are equality predicates on the
correlated outer row, so the identity prefix does the bulk of the selectivity.

``member_index`` is deliberately absent: the ensemble-member fence joins on
``(run_id, lead_time_hours, member_index)`` with a constant ``target_kind``,
which this index still serves through the ``run_id``/``lead_time_hours`` prefix
(one extra heap fetch per candidate member row, versus a second multi-column
index that would have to be written on every queue state transition).
"""

from alembic import op

# revision identifiers, used by Alembic (max 32 chars).
revision = "010_reclamation_fence_idx"
down_revision = "009_reclamation_store_gen"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_reclamation_serving_fence",
        "reclamation_queue",
        [
            "run_id",
            "lead_time_hours",
            "variable_code",
            "target_kind",
            "status",
        ],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_reclamation_serving_fence", table_name="reclamation_queue"
    )
