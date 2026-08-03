"""Domain-layer exceptions for the weather forecasting platform.

Exceptions raised by domain models and calculation modules are defined here so
the API layer can map them to RFC 7807 problem details without coupling API
code to domain internals (see ``ENGINEERING_CONTRACT.md`` section 6).
"""


class DomainError(Exception):
    """Base exception for all domain-layer failures."""


class InvalidCoordinatesError(DomainError, ValueError):
    """Raised when coordinates fall outside valid WGS 84 bounds."""


class InvalidGridError(DomainError, ValueError):
    """Raised when a grid definition or field shape is invalid."""


class PointOutsideGridError(DomainError, ValueError):
    """Raised when a geographic point lies outside a grid's bounds."""


class EmptyEnsembleError(DomainError, ValueError):
    """Raised when an ensemble member sequence is empty."""


class InvalidEnsembleError(DomainError, ValueError):
    """Raised when ensemble members are not a 1-D numeric sequence of finite values."""


class InvalidPercentileError(DomainError, ValueError):
    """Raised when a requested percentile is outside the valid [0, 100] range."""


class InvalidThresholdError(DomainError, ValueError):
    """Raised when a probability threshold is invalid (e.g. upper below lower)."""
