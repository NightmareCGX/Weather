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
