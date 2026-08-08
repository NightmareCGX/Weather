"""Forecast verification metrics (RMSE, bias, MAE).

Public API (all functions return primitive ``float`` values):

- ``root_mean_squared_error``
- ``mean_absolute_error``
- ``bias``

Shared input validation and the ``VerificationError`` exception are also
exported for callers and tests.
"""

from domain.verification.metrics import (
    VerificationError,
    bias,
    mean_absolute_error,
    root_mean_squared_error,
)

__all__ = [
    "root_mean_squared_error",
    "mean_absolute_error",
    "bias",
    "VerificationError",
]
