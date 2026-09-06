"""Authoritative cycle cadence contract per platform model.

This module centralizes the platform's forecast cycle cadence: the nominal
time interval between consecutive forecast generations for a model (e.g. GFS=6h,
GEFS=6h). It is a model/product specification in the pure domain layer —
deliberately NOT a runtime or scheduler configuration value — because physical
storage lifecycle reclamation (Lifecycle V2) and forecast time serving must
agree on one authoritative cadence per model.

Contract rules (Data Lifecycle V2):
* Default operational cadence: 6 hours for GFS and GEFS (00Z, 06Z, 12Z, 18Z).
* Extensible: future models (e.g. high-resolution 1h or 3h products) register
  their cadence here without altering lifecycle algorithms or requiring model-specific
  branching.
* Pure & side-effect-free: no database connections, environment dependencies, or
  network I/O.
"""

from __future__ import annotations

from datetime import timedelta

#: Default fallback cycle cadence in hours.
DEFAULT_CYCLE_CADENCE_HOURS: int = 6

#: Authoritative fixed cycle cadence in hours per platform model.
MODEL_CYCLE_CADENCE_HOURS: dict[str, int] = {
    "gfs": 6,
    "gefs": 6,
}


def canonical_cycle_cadence_hours(
    model_id: str, default_if_unknown: int | None = None
) -> int:
    """Return the authoritative cycle cadence in hours for a model.

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        default_if_unknown: Optional fallback cadence in hours if model is unregistered.
            When ``None``, an unknown model raises ``ValueError``.

    Returns:
        Cadence in hours (positive integer).

    Raises:
        ValueError: If model is unknown and ``default_if_unknown`` is None.
    """
    key = model_id.lower().strip()
    if key in MODEL_CYCLE_CADENCE_HOURS:
        return MODEL_CYCLE_CADENCE_HOURS[key]
    if default_if_unknown is not None:
        if default_if_unknown <= 0:
            raise ValueError(f"Cadence must be positive, got {default_if_unknown}")
        return default_if_unknown
    raise ValueError(
        f"Unknown model identifier {model_id!r}; registered models: "
        f"{sorted(MODEL_CYCLE_CADENCE_HOURS)}"
    )


def canonical_cycle_cadence(
    model_id: str, default_if_unknown: timedelta | None = None
) -> timedelta:
    """Return the authoritative cycle cadence as a ``timedelta`` for a model.

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        default_if_unknown: Optional fallback cadence as ``timedelta`` if unregistered.
            When ``None``, an unknown model raises ``ValueError``.

    Returns:
        Cadence as a positive ``timedelta``.

    Raises:
        ValueError: If model is unknown and ``default_if_unknown`` is None.
    """
    default_hours = (
        int(default_if_unknown.total_seconds() // 3600)
        if default_if_unknown is not None
        else None
    )
    hours = canonical_cycle_cadence_hours(model_id, default_if_unknown=default_hours)
    return timedelta(hours=hours)


def register_canonical_cycle_cadence(model_id: str, cadence_hours: int) -> None:
    """Register or override the cycle cadence for a model (test/startup).

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``, ``future_3h``).
        cadence_hours: Strictly positive integer representing cycle interval in hours.

    Raises:
        ValueError: If cadence_hours is not a strictly positive integer.
    """
    if cadence_hours <= 0:
        raise ValueError(f"Cycle cadence must be strictly positive: {cadence_hours}")
    MODEL_CYCLE_CADENCE_HOURS[model_id.lower().strip()] = int(cadence_hours)
