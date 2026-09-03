"""Integration tests for the API ForecastCycleLifecycle ORM entity."""

from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.models.entities import ForecastCycleLifecycle


def test_forecast_cycle_lifecycle_entity_crud() -> None:
    engine = create_engine("sqlite:///:memory:")
    ForecastCycleLifecycle.__table__.create(engine)

    cycle_time = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
    retired_at = datetime(2026, 9, 2, 6, 0, tzinfo=timezone.utc)
    retired_by = datetime(2026, 9, 2, 6, 0, tzinfo=timezone.utc)

    with Session(engine) as session:
        record = ForecastCycleLifecycle(
            cycle_time=cycle_time,
            retired_at=retired_at,
            retired_by_cycle_time=retired_by,
        )
        session.add(record)
        session.commit()

        # Retrieve and verify
        row = session.get(ForecastCycleLifecycle, cycle_time)
        assert row is not None

        def _utc(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        assert _utc(row.cycle_time) == cycle_time
        assert _utc(row.retired_at) == retired_at
        assert _utc(row.retired_by_cycle_time) == retired_by
        assert row.deletion_started_at is None
        assert row.deleted_at is None
        assert row.created_at is not None
        assert row.updated_at is not None
