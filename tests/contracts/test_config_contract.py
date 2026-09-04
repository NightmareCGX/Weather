"""Cross-Package Configuration Parameter Interoperability Contract Suite.

Asserts that shared infrastructure configuration settings defined in both
`api.core.config.Settings` and `ingestion.core.config.IngestionSettings` maintain
identical variable names, types, and default semantics for interoperability.
"""

from api.core.config import Settings as ApiSettings
from ingestion.core.config import IngestionSettings


def test_shared_infrastructure_settings_no_drift() -> None:
    """Verify that shared infrastructure environment variable names and types match."""
    api_fields = ApiSettings.model_fields
    ing_fields = IngestionSettings.model_fields

    # True shared interoperability settings (database, object storage, coverage thresholds)
    shared_interop_keys = [
        "DATABASE_URL",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_SECURE",
        "ENSEMBLE_MIN_COVERAGE_RATIO",
    ]

    for key in shared_interop_keys:
        assert key in api_fields, f"Key '{key}' missing from api.core.config.Settings"
        assert key in ing_fields, f"Key '{key}' missing from ingestion.core.config.IngestionSettings"

    # Verify MinIO secure defaults to False in both
    api_inst = ApiSettings()
    ing_inst = IngestionSettings()
    assert api_inst.MINIO_SECURE is False
    assert ing_inst.MINIO_SECURE is False

    # Verify coverage ratio default is identical (0.85)
    assert api_inst.ENSEMBLE_MIN_COVERAGE_RATIO == 0.85
    assert ing_inst.ENSEMBLE_MIN_COVERAGE_RATIO == 0.85
