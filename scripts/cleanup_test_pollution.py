"""One-time cleanup of pytest pollution rows in the live catalog (§11 item 10).

Historical local pytest runs (before test isolation was fixed) connected to the
live compose database by default and seeded ``model_runs`` rows whose
``zarr_store_path`` points at pytest temp directories (``pytest-of-*``). The
physical stores behind those paths no longer exist, so the rows are dead
catalog pollution that confuses GC/resolver bookkeeping.

The predicate is deliberately conservative: only rows whose
``zarr_store_path`` matches the pytest temp-path pattern are touched. Related
rows (forecast products, ensemble members, reclamation queue entries) are
removed through the catalog's own foreign-key CASCADE chain.

Usage:
    python scripts/cleanup_test_pollution.py            # dry-run: report only
    python scripts/cleanup_test_pollution.py --apply    # actually delete
    DATABASE_URL=postgresql://... python scripts/cleanup_test_pollution.py
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, text

LIVE_DEFAULT = "postgresql://weather_user:weather_password@localhost:5432/weather_db"
POLLUTION_PREDICATE = "zarr_store_path LIKE '%pytest-of-%'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the matching rows (default: dry-run report only).",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or LIVE_DEFAULT,
        help="Catalog database URL (defaults to DATABASE_URL env, then the compose URL).",
    )
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    with engine.connect() as conn:
        count = conn.execute(
            text(f"SELECT count(*) FROM model_runs WHERE {POLLUTION_PREDICATE}")
        ).scalar_one()
        print(f"Polluted model_runs rows (zarr_store_path LIKE '%pytest-of-%'): {count}")
        if count:
            samples = conn.execute(
                text(
                    "SELECT id, model_version_id, cycle_time, status, zarr_store_path "
                    f"FROM model_runs WHERE {POLLUTION_PREDICATE} ORDER BY cycle_time LIMIT 10"
                )
            ).all()
            print("Sample rows:")
            for row in samples:
                print(f"  {row}")
            # Tables holding FKs into model_runs must be cleaned first (the
            # catalog FKs are not all ON DELETE CASCADE).
            referencing = conn.execute(
                text(
                    "SELECT DISTINCT tc.table_name, kcu.column_name "
                    "FROM information_schema.table_constraints tc "
                    "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
                    "JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name "
                    "WHERE tc.constraint_type = 'FOREIGN KEY' AND ccu.table_name = 'model_runs' "
                    "AND tc.table_name != 'model_runs'"
                )
            ).all()
            dependent_total = 0
            for table, column in referencing:
                n = conn.execute(
                    text(
                        f"SELECT count(*) FROM {table} WHERE {column} IN "
                        f"(SELECT id FROM model_runs WHERE {POLLUTION_PREDICATE})"
                    )
                ).scalar_one()
                dependent_total += n
                print(f"  dependent rows: {table}.{column} = {n}")
            if dependent_total == 0 and not referencing:
                print("No FK dependents found.")

        if not args.apply:
            print("Dry-run only: pass --apply to delete these rows.")
            return 0

        # FK-safe deletion: children first (two rounds to cover chained FKs),
        # then the polluted model_runs rows.
        for _round in range(2):
            for table, column in referencing:
                conn.execute(
                    text(
                        f"DELETE FROM {table} WHERE {column} IN "
                        f"(SELECT id FROM model_runs WHERE {POLLUTION_PREDICATE})"
                    )
                )
        result = conn.execute(
            text(f"DELETE FROM model_runs WHERE {POLLUTION_PREDICATE}")
        )
        conn.commit()
        print(f"Deleted {result.rowcount} polluted model_runs rows (dependents included).")
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
