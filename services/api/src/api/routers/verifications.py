"""Verification metrics endpoint (API.md section 7.1).

Returns historical error metrics (RMSE, bias, MAE) for a specific model over a
date window. The router is thin (ENGINEERING_CONTRACT section 2): it validates
parameters, calls the verification service, and serializes the documented
``verification_report`` envelope. The metric math lives in ``domain.verification``
and forecast/observation retrieval and pairing live in the service layer.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.schemas import VerificationReportEnvelope
from api.services.verification import build_verification_report

router = APIRouter()

#: Database session dependency (module-level to satisfy ruff B008).
DB = Depends(get_db)

#: Cache policy for verification metrics (API.md 7.1: 24 hours).
CACHE_CONTROL_VERIFICATION = "public, max-age=86400"


@router.get(
    "/verifications",
    response_model=VerificationReportEnvelope,
    summary="Get verification metrics",
)
def get_verification_metrics(
    response: Response,
    model: Annotated[str, Query(description="A model identifier.")],
    start_date: Annotated[
        date, Query(description="Inclusive start date (YYYY-MM-DD, UTC).")
    ],
    end_date: Annotated[
        date, Query(description="Inclusive end date (YYYY-MM-DD, UTC).")
    ],
    db: Session = DB,
) -> VerificationReportEnvelope:
    """Return RMSE/bias/MAE verification metrics for a model and date window.

    Metrics are computed over the pooled sample of every forecast/observation
    pair whose valid time falls in the window (see the verification service).
    """
    data = build_verification_report(
        db,
        model=model,
        start_date=start_date,
        end_date=end_date,
    )
    response.headers["Cache-Control"] = CACHE_CONTROL_VERIFICATION
    return VerificationReportEnvelope(data=data)
