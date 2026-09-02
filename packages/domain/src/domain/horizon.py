"""Authoritative canonical forecast-lead horizon contract per platform model.

This module centralizes the canonical cycle-horizon lead sequence: the
complete, ordered set of forecast leads a model's cycle is expected to serve
when fully ingested (the "canonical cycle horizon"). It is a model/product
contract — deliberately NOT a runtime or scheduler configuration value —
because both big-batch ingestion (store pre-allocation, run-level readiness)
and future realtime lead-wave ingestion (wave planning) must agree on one
horizon, and the distinction between an invocation's *targets* and the
cycle's *expected horizon* must be impossible to lose.

Contract facts (``docs/investigations/gfs-gefs-variable-inventory/README.md``):

* Horizon: 0–240 hours at 3-hour cadence → 81 leads (f000, f003, …, f240)
  for both GFS and GEFS 0.25° surface products.
* The horizon is independent of upstream publication cadence: upstream GFS
  also publishes hourly files through f120, which are not platform targets;
  upstream GEFS ``pgrb2sp25`` publishes exactly this sequence per member.
* This horizon is NOT the connector's transport ceiling (0–384 h), which only
  bounds upstream URL validation.

The registry mirrors :mod:`domain.coverage` (``MODEL_EXPECTED_MEMBERS``):
fixed per-model specification, not dynamically inferred. Tests may inject a
reduced horizon via :func:`register_canonical_lead_horizon` (the same pattern
as ``coverage.register_expected_members``); production always uses the
canonical 81-lead sequence registered below.
"""

from __future__ import annotations

#: Lead cadence of the canonical platform horizon, in hours.
CANONICAL_LEAD_CADENCE_HOURS: int = 3

#: Last (highest) lead of the canonical platform horizon, in hours.
CANONICAL_MAX_LEAD_HOURS: int = 240


def _build_lead_sequence(
    start_hour: int, cadence_hours: int, max_hour: int
) -> tuple[int, ...]:
    """Build the ordered lead sequence ``start, start+cadence, …, max``.

    The arguments are module-level contract constants (never runtime input),
    so the sequence is built without defensive validation: a literal change
    to the constants is a contract change reviewed in code.

    Args:
        start_hour: First lead of the horizon (the analysis lead, 0).
        cadence_hours: Spacing between consecutive leads.
        max_hour: Last lead of the horizon (inclusive).

    Returns:
        The ordered lead sequence, ending at ``max_hour``.
    """
    return tuple(range(start_hour, max_hour + 1, cadence_hours))


#: Authoritative fixed canonical horizon per model (the complete lead set a
#: cycle store is expected to serve when fully ingested).
MODEL_CANONICAL_HORIZONS: dict[str, tuple[int, ...]] = {
    "gfs": _build_lead_sequence(0, CANONICAL_LEAD_CADENCE_HOURS, CANONICAL_MAX_LEAD_HOURS),
    "gefs": _build_lead_sequence(0, CANONICAL_LEAD_CADENCE_HOURS, CANONICAL_MAX_LEAD_HOURS),
}


def register_canonical_lead_horizon(model_id: str, leads: tuple[int, ...]) -> None:
    """Register or override the canonical horizon for a model (test/startup).

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        leads: The complete ordered lead sequence. Must be non-empty and
            strictly increasing (the horizon is walked in order by frontier
            logic), with non-negative lead values.

    Raises:
        ValueError: If the sequence is empty, not strictly increasing, or
            contains a negative lead.
    """
    if not leads:
        raise ValueError("canonical lead horizon must not be empty")
    if any(lead < 0 for lead in leads):
        raise ValueError(f"canonical lead horizon must be non-negative: {leads!r}")
    if any(b <= a for a, b in zip(leads, leads[1:])):
        raise ValueError(
            f"canonical lead horizon must be strictly increasing: {leads!r}"
        )
    MODEL_CANONICAL_HORIZONS[model_id.lower().strip()] = tuple(leads)


def canonical_lead_time_hours(
    model_id: str, default_if_unknown: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    """Return the authoritative canonical horizon for a model.

    The horizon is a fixed specification of the platform's product contract.
    It is NOT dynamically inferred from the current run, upstream publication,
    or store occupancy, and it is NOT the invocation's requested subset.

    Args:
        model_id: Platform model identifier (e.g. ``gfs``, ``gefs``).
        default_if_unknown: Optional fallback horizon if the model is not in
            the authoritative registry. When ``None`` (the default), an
            unrecognized model raises ``ValueError``.

    Returns:
        The canonical, strictly increasing lead sequence for the model.

    Raises:
        ValueError: If ``model_id`` is not registered and ``default_if_unknown``
            is ``None``.
    """
    normalized = model_id.lower().strip()
    if normalized in MODEL_CANONICAL_HORIZONS:
        return MODEL_CANONICAL_HORIZONS[normalized]
    if default_if_unknown is not None:
        return default_if_unknown
    raise ValueError(
        f"Unknown model identifier {model_id!r}; registered models: "
        f"{sorted(MODEL_CANONICAL_HORIZONS)}"
    )
