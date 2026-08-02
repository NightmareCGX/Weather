import os
import sys

# Ensure services/api/src is on sys.path for test execution
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

"""Unit tests for models and metadata configuration without requiring a real database."""

from api.core.database import Base
from api.models import (
    ForecastCenter,
    Model,
    Station,
    City,
    SkiResort,
)


def test_orm_metadata_tables_registered():
    """Verify all 14 expected tables are correctly registered in SQLAlchemy metadata."""
    expected_tables = [
        "forecast_centers",
        "models",
        "model_versions",
        "model_runs",
        "ensemble_members",
        "forecast_variables",
        "forecast_grids",
        "forecast_products",
        "stations",
        "cities",
        "ski_resorts",
        "verification_observations",
        "point_query_fallback_audit",
    ]

    registered_tables = list(Base.metadata.tables.keys())
    for table in expected_tables:
        assert table in registered_tables, f"Table {table} is missing from SQLAlchemy metadata registry."


def test_model_attributes_and_relationships():
    """Verify core ORM models and relationships are correctly defined."""
    assert hasattr(ForecastCenter, "models")
    assert hasattr(Model, "center")
    assert hasattr(Model, "versions")
    assert hasattr(Station, "observations")
    assert hasattr(Station, "geom")
    assert hasattr(City, "geom")
    assert hasattr(SkiResort, "geom")
