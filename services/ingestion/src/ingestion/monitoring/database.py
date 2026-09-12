"""PostgreSQL health and capacity monitoring with bounded query overhead.

Collects connection pool statistics, active/max connections, long-running queries,
lock waits, deadlocks, database size, table sizes, dead tuples, and retention
invariants without running unbounded full-table scans.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ingestion.monitoring.metrics import REGISTRY

logger = logging.getLogger(__name__)

# Register Prometheus metrics
DB_CONNECTED = REGISTRY.gauge(
    "weather_postgres_connected",
    "PostgreSQL connectivity status (1 if connected, 0 otherwise)",
)
DB_ACTIVE_CONNECTIONS = REGISTRY.gauge(
    "weather_postgres_active_connections",
    "Number of active server connections from pg_stat_activity",
)
DB_MAX_CONNECTIONS = REGISTRY.gauge(
    "weather_postgres_max_connections",
    "Configured max_connections on the PostgreSQL server",
)
DB_CONNECTION_UTILIZATION_PERCENT = REGISTRY.gauge(
    "weather_postgres_connection_utilization_percent",
    "PostgreSQL connection utilization percentage (active / max * 100)",
)
DB_POOL_CHECKED_OUT = REGISTRY.gauge(
    "weather_postgres_pool_checked_out",
    "SQLAlchemy connection pool checked-out connections",
)
DB_POOL_SIZE = REGISTRY.gauge(
    "weather_postgres_pool_size",
    "SQLAlchemy connection pool configured base size",
)
DB_POOL_OVERFLOW = REGISTRY.gauge(
    "weather_postgres_pool_overflow",
    "SQLAlchemy connection pool current overflow connections",
)
DB_TOTAL_SIZE_BYTES = REGISTRY.gauge(
    "weather_postgres_total_size_bytes",
    "Total PostgreSQL database size in bytes",
)
DB_TABLE_SIZE_BYTES = REGISTRY.gauge(
    "weather_postgres_table_size_bytes",
    "Total relation size (table + indexes) in bytes",
    labelnames=("table_name",),
)
DB_TABLE_DEAD_TUPLES = REGISTRY.gauge(
    "weather_postgres_table_dead_tuples",
    "Estimated dead tuples from pg_stat_user_tables",
    labelnames=("table_name",),
)
DB_LONG_RUNNING_TRANSACTIONS = REGISTRY.gauge(
    "weather_postgres_long_running_transactions",
    "Number of active transactions running longer than threshold",
)
DB_LOCK_WAITS = REGISTRY.gauge(
    "weather_postgres_lock_waits",
    "Number of sessions waiting on locks",
)
DB_DEADLOCKS_TOTAL = REGISTRY.gauge(
    "weather_postgres_deadlocks_total",
    "Cumulative deadlocks count from pg_stat_database",
)


@dataclass(frozen=True)
class TableStat:
    """Table size and tuple statistics from PostgreSQL catalogs."""

    table_name: str
    total_bytes: int
    live_tuples: int
    dead_tuples: int
    last_autovacuum: datetime | None


@dataclass(frozen=True)
class LongRunningQuery:
    """Details of a long-running transaction or query."""

    pid: int
    duration_seconds: float
    state: str
    query_snippet: str
    wait_event: str | None


@dataclass
class PostgresHealthReport:
    """Comprehensive health and performance report for PostgreSQL."""

    connected: bool
    active_connections: int = 0
    max_connections: int = 100
    connection_utilization_pct: float = 0.0
    pool_checked_out: int = 0
    pool_size: int = 0
    pool_overflow: int = 0
    total_size_bytes: int = 0
    tables: dict[str, TableStat] = field(default_factory=dict)
    long_running_queries: list[LongRunningQuery] = field(default_factory=list)
    lock_waits: int = 0
    deadlocks: int = 0
    deadlocks_delta: int = 0
    error: str | None = None

    @property
    def is_connection_warning(self) -> bool:
        return self.connection_utilization_pct >= 80.0

    @property
    def is_connection_critical(self) -> bool:
        return self.connection_utilization_pct >= 95.0


class PostgresHealthCollector:
    """Collects bounded PostgreSQL health and capacity metrics."""

    TRACKED_TABLES: tuple[str, ...] = (
        "model_runs",
        "forecast_products",
        "ensemble_members",
        "ensemble_member_products",
        "forecast_cycle_lifecycle",
        "reclamation_queue",
    )

    def __init__(self, engine: Engine | Connection) -> None:
        self.engine = engine
        self._last_deadlocks: int | None = None

    def collect(
        self,
        long_query_threshold_seconds: float = 30.0,
    ) -> PostgresHealthReport:
        """Query PostgreSQL catalog views using strictly bounded, indexed queries."""
        try:
            pool = getattr(self.engine, "pool", None)
            pool_checked_out = pool.checkedout() if pool and hasattr(pool, "checkedout") else 0
            pool_size = pool.size() if pool and hasattr(pool, "size") else 0
            pool_overflow = pool.overflow() if pool and hasattr(pool, "overflow") else 0
        except Exception:
            pool_checked_out = 0
            pool_size = 0
            pool_overflow = 0

        DB_POOL_CHECKED_OUT.set(float(pool_checked_out))
        DB_POOL_SIZE.set(float(pool_size))
        DB_POOL_OVERFLOW.set(float(pool_overflow))

        try:
            conn_ctx = self.engine.connect() if hasattr(self.engine, "connect") else nullcontext(self.engine)
            with conn_ctx as conn:
                # 1. Connectivity & connection limits
                res_max = conn.execute(text("SHOW max_connections")).scalar()
                max_conn = int(res_max) if res_max is not None else 100

                res_act = conn.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE state IS NOT NULL")
                ).scalar()
                active_conn = int(res_act) if res_act is not None else 0

                util_pct = (active_conn / max_conn * 100.0) if max_conn > 0 else 0.0

                # 2. Database total size
                res_db_size = conn.execute(
                    text("SELECT pg_database_size(current_database())")
                ).scalar()
                total_db_size = int(res_db_size) if res_db_size is not None else 0

                # 3. Deadlocks count
                res_deadlocks = conn.execute(
                    text(
                        "SELECT deadlocks FROM pg_stat_database WHERE datname = current_database()"
                    )
                ).scalar()
                deadlocks = int(res_deadlocks) if res_deadlocks is not None else 0
                if self._last_deadlocks is None:
                    deadlocks_delta = 0
                else:
                    deadlocks_delta = max(0, deadlocks - self._last_deadlocks)
                self._last_deadlocks = deadlocks

                # 4. Long running queries / transactions (bounded to LIMIT 10)
                q_long = text(
                    """
                    SELECT pid,
                           EXTRACT(EPOCH FROM (clock_timestamp() - xact_start)) AS duration_s,
                           state,
                           substr(query, 1, 100) AS q_snippet,
                           wait_event
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                      AND xact_start IS NOT NULL
                      AND clock_timestamp() - xact_start > make_interval(secs => :thresh)
                    ORDER BY duration_s DESC
                    LIMIT 10
                    """
                )
                long_queries: list[LongRunningQuery] = []
                for row in conn.execute(q_long, {"thresh": long_query_threshold_seconds}):
                    long_queries.append(
                        LongRunningQuery(
                            pid=int(row.pid),
                            duration_seconds=round(float(row.duration_s), 2),
                            state=str(row.state),
                            query_snippet=str(row.q_snippet),
                            wait_event=str(row.wait_event) if row.wait_event else None,
                        )
                    )

                # 5. Lock waits
                res_locks = conn.execute(
                    text("SELECT count(*) FROM pg_locks WHERE NOT granted")
                ).scalar()
                lock_waits = int(res_locks) if res_locks is not None else 0

                # 6. Table sizes and tuple counts via pg_stat_user_tables & pg_total_relation_size
                q_tables = text(
                    """
                    SELECT s.relname,
                           pg_total_relation_size(s.relid) AS total_bytes,
                           s.n_live_tup,
                           s.n_dead_tup,
                           s.last_autovacuum
                    FROM pg_stat_user_tables s
                    WHERE s.relname = ANY(:tables)
                    """
                )
                table_stats: dict[str, TableStat] = {}
                for row in conn.execute(q_tables, {"tables": list(self.TRACKED_TABLES)}):
                    tname = str(row.relname)
                    t_bytes = int(row.total_bytes or 0)
                    t_dead = int(row.n_dead_tup or 0)
                    table_stats[tname] = TableStat(
                        table_name=tname,
                        total_bytes=t_bytes,
                        live_tuples=int(row.n_live_tup or 0),
                        dead_tuples=t_dead,
                        last_autovacuum=row.last_autovacuum,
                    )
                    DB_TABLE_SIZE_BYTES.labels(table_name=tname).set(float(t_bytes))
                    DB_TABLE_DEAD_TUPLES.labels(table_name=tname).set(float(t_dead))

            # Update Prometheus gauges
            DB_CONNECTED.set(1.0)
            DB_ACTIVE_CONNECTIONS.set(float(active_conn))
            DB_MAX_CONNECTIONS.set(float(max_conn))
            DB_CONNECTION_UTILIZATION_PERCENT.set(round(util_pct, 2))
            DB_TOTAL_SIZE_BYTES.set(float(total_db_size))
            DB_DEADLOCKS_TOTAL.set(float(deadlocks))
            DB_LONG_RUNNING_TRANSACTIONS.set(float(len(long_queries)))
            DB_LOCK_WAITS.set(float(lock_waits))

            return PostgresHealthReport(
                connected=True,
                active_connections=active_conn,
                max_connections=max_conn,
                connection_utilization_pct=round(util_pct, 2),
                pool_checked_out=pool_checked_out,
                pool_size=pool_size,
                pool_overflow=pool_overflow,
                total_size_bytes=total_db_size,
                tables=table_stats,
                long_running_queries=long_queries,
                lock_waits=lock_waits,
                deadlocks=deadlocks,
                deadlocks_delta=deadlocks_delta,
            )
        except Exception as exc:
            logger.warning("PostgreSQL health probe failed: %s", exc)
            DB_CONNECTED.set(0.0)
            return PostgresHealthReport(
                connected=False,
                pool_checked_out=pool_checked_out,
                pool_size=pool_size,
                pool_overflow=pool_overflow,
                error=str(exc),
            )
