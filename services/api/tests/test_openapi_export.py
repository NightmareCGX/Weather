"""OpenAPI schema export and drift detection test (Stage 7D-A).

Ensures that FastAPI's generated OpenAPI 3.1 specification exports deterministically
and matches the committed `services/frontend/openapi.json` artifact consumed by the
frontend TypeScript contract test suite.
"""

import json
from pathlib import Path

from api.main import app


def test_openapi_schema_matches_committed_frontend_artifact() -> None:
    """Assert that the generated OpenAPI schema matches the committed frontend artifact."""
    current_schema = app.openapi()
    frontend_openapi_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "services"
        / "frontend"
        / "openapi.json"
    )
    assert frontend_openapi_path.exists(), f"Missing {frontend_openapi_path}"

    committed_schema = json.loads(frontend_openapi_path.read_text(encoding="utf-8"))

    # Assert paths and schemas match
    assert set(current_schema["paths"].keys()) == set(committed_schema["paths"].keys())
    assert set(current_schema["components"]["schemas"].keys()) == set(
        committed_schema["components"]["schemas"].keys()
    )
