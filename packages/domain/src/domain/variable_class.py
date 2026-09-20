"""Variable → aggregate-encoding mapping (the approved storage classification).

The platform stores two aggregate encodings, and which one a variable gets is a property of
the variable, not a local choice. Both were chosen from measurement on real GEFS data
(``docs/investigations/numeric-encoding/REPORT.md`` §15/§16):

``A`` -- near-Gaussian (temperature, wind components): ``MEAN + STD + 32`` normalised bins.
    Zero-valued cells are absent, ``max / mean(sigma)`` stays under ~30 and the CDF is
    smooth, so per-cell normalisation is representative. 32 bins is the approved count: 16
    suffice through the body but leave a rare-threshold error of 2.0-2.3x the ensemble's own
    sampling noise, while 64 buys nothing further on the tail for another 1.5x the bytes.

``B`` and ``C`` -- zero-inflated / heavy-tailed (precipitation, gust, snow depth) and
    bounded / censored / bimodal (humidity, cloud cover, cloud ceiling, visibility): the
    quantile function at 19 body-dense levels. These classes are treated identically on
    purpose. The bin encoding fails for both, for the same reason: its bounded +-4 sigma
    support discards the tail, and where the CDF is steep -- the far tail for B, and the
    *median* of a zero-inflated or capped field for C -- the resolution collapses. Building a
    separate censored-point-mass model for C was tried and measured: it was bit-identical to
    the bins on two variables and markedly worse on the other two, so the endpoint mass is
    not the binding error source and the extra machinery was dropped.

``D`` -- 0/1 flags (crain, csnow, cfrzr, cicep): a per-cell exceedance fraction rather than
    an :class:`~domain.aggregate.AggregateSpec`, because a shape over two values carries
    nothing. Cost is negligible either way.

This module is the only place that mapping is written down. A variable with no entry is an
error, not a default: silently aggregating a heavy-tailed variable with the near-Gaussian
encoding would produce a plausible-looking field that cannot answer its product's thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from domain.aggregate import (
    DEFAULT_N_BINS,
    KIND_MEAN_STD_BINS,
    KIND_QUANTILE_FUNCTION,
    AggregateSpec,
)

#: Class identifiers, as they appear in a store's metadata.
CLASS_NEAR_GAUSSIAN: Final[str] = "A"
CLASS_ZERO_INFLATED: Final[str] = "B"
CLASS_BOUNDED: Final[str] = "C"
CLASS_FLAG: Final[str] = "D"
#: A variable whose representation is **entirely** product-defined supplementary fields: it has
#: no distribution spec because no scalar or percentile statistic describes it.
#:
#: ``wind_10m`` is the case. It is synthesised by the API from ``wind_u_10m`` and
#: ``wind_v_10m`` rather than stored itself, and everything its products show -- consensus
#: vector, wind rose, directional probability -- is a function of each member's ``(u, v)`` pair.
#: Its speed histogram is the rose summed over its sectors, so even the distribution path is
#: served by the rose. Class D is the other group-only case, but there the reason is the
#: opposite: a 0/1 flag's shape carries too little to store, where wind's carries too much for a
#: single ordering.
CLASS_PRODUCT_FIELDS: Final[str] = "S"

VALID_CLASSES: Final[frozenset[str]] = frozenset(
    {CLASS_NEAR_GAUSSIAN, CLASS_ZERO_INFLATED, CLASS_BOUNDED, CLASS_FLAG, CLASS_PRODUCT_FIELDS}
)

#: Classes that carry no :class:`~domain.aggregate.AggregateSpec`.
SPECLESS_CLASSES: Final[frozenset[str]] = frozenset(
    {CLASS_FLAG, CLASS_PRODUCT_FIELDS}
)

#: Approved bin count for the near-Gaussian class.
APPROVED_BIN_COUNT: Final[int] = DEFAULT_N_BINS

#: Approved level count for the quantile class. 19 body-dense levels measured best: 17 left a
#: 0.200 error at the finest precipitation threshold, 26 were worse (a 30-member sample cannot
#: resolve a denser grid).
APPROVED_LEVEL_COUNT: Final[int] = 19


class VariableClassError(ValueError):
    """Raised when a variable's storage encoding cannot be resolved."""


@dataclass(frozen=True)
class VariableEncoding:
    """The storage encoding for one variable.

    Attributes:
        variable: Variable code, as it appears in the store.
        variable_class: One of :data:`VALID_CLASSES`.
        spec: The aggregate spec for classes A/B/C, or ``None`` for a flag (too little shape to
            store) and for a product-fields variable (no single distribution to store).
    """

    variable: str
    variable_class: str
    spec: AggregateSpec | None

    def __post_init__(self) -> None:
        if self.variable_class not in VALID_CLASSES:
            raise VariableClassError(
                f"unknown variable class {self.variable_class!r} for {self.variable!r}"
            )
        specless = self.variable_class in SPECLESS_CLASSES
        if specless and self.spec is not None:
            raise VariableClassError(
                f"{self.variable!r} is class {self.variable_class} and must not carry an "
                "aggregate spec"
            )
        if not specless and self.spec is None:
            raise VariableClassError(
                f"{self.variable!r} is class {self.variable_class} and requires an aggregate spec"
            )

    @property
    def is_flag(self) -> bool:
        """Whether the variable is a 0/1 flag encoded as a per-cell fraction."""
        return self.variable_class == CLASS_FLAG

    @property
    def has_distribution(self) -> bool:
        """Whether the variable has distribution fields, or only product-defined ones."""
        return self.spec is not None

    @property
    def field_count(self) -> int:
        """Distribution fields: one per ``AggregateSpec`` field, none without a spec.

        The product-defined groups a variable carries are not counted here -- they are
        ``domain.field_layout``'s business, since they belong to the container rather than to the
        distribution.
        """
        return 0 if self.spec is None else self.spec.n_fields


def _near_gaussian(variable: str) -> VariableEncoding:
    return VariableEncoding(
        variable=variable,
        variable_class=CLASS_NEAR_GAUSSIAN,
        spec=AggregateSpec(kind=KIND_MEAN_STD_BINS, n_bins=APPROVED_BIN_COUNT),
    )


#: The quantile spec's level count is part of the approved design, so the two constants are
#: asserted to agree at import rather than checked on every registration.
assert len(AggregateSpec(kind=KIND_QUANTILE_FUNCTION).levels) == APPROVED_LEVEL_COUNT, (
    "APPROVED_LEVEL_COUNT disagrees with the quantile spec's default level set"
)


def _quantile(variable: str, variable_class: str) -> VariableEncoding:
    return VariableEncoding(
        variable=variable,
        variable_class=variable_class,
        spec=AggregateSpec(kind=KIND_QUANTILE_FUNCTION),
    )


def _flag(variable: str) -> VariableEncoding:
    return VariableEncoding(variable=variable, variable_class=CLASS_FLAG, spec=None)


def _product_fields(variable: str) -> VariableEncoding:
    return VariableEncoding(
        variable=variable, variable_class=CLASS_PRODUCT_FIELDS, spec=None
    )


#: The approved classification. Keys are the canonical variable codes the platform stores
#: (``domain.reclamation.get_expected_region_variables`` is the authority on which exist), plus
#: ``wind_10m``, which the API synthesises from its two components and which therefore has a
#: container of its own without appearing in the stored-variable list.
_VARIABLE_ENCODINGS: Final[dict[str, VariableEncoding]] = {
    # A -- near-Gaussian
    "temperature_2m": _near_gaussian("temperature_2m"),
    "wind_u_10m": _near_gaussian("wind_u_10m"),
    "wind_v_10m": _near_gaussian("wind_v_10m"),
    # B -- zero-inflated / heavy-tailed
    "precipitation_rate": _quantile("precipitation_rate", CLASS_ZERO_INFLATED),
    "precipitation_amount_3h": _quantile(
        "precipitation_amount_3h", CLASS_ZERO_INFLATED
    ),
    "snow_depth": _quantile("snow_depth", CLASS_ZERO_INFLATED),
    "wind_gust": _quantile("wind_gust", CLASS_ZERO_INFLATED),
    # C -- bounded / censored / bimodal
    "relative_humidity_2m": _quantile("relative_humidity_2m", CLASS_BOUNDED),
    "cloud_cover_3h": _quantile("cloud_cover_3h", CLASS_BOUNDED),
    "cloud_ceiling": _quantile("cloud_ceiling", CLASS_BOUNDED),
    "visibility": _quantile("visibility", CLASS_BOUNDED),
    # D -- 0/1 flags
    "crain": _flag("crain"),
    "csnow": _flag("csnow"),
    "cfrzr": _flag("cfrzr"),
    "cicep": _flag("cicep"),
    # S -- product-defined supplementary fields only
    "wind_10m": _product_fields("wind_10m"),
}

#: The set of variables this module classifies.
REGISTERED_VARIABLES: Final[frozenset[str]] = frozenset(_VARIABLE_ENCODINGS)

#: Classified variables the platform does not store as such: they are derived at serve time from
#: other stored variables. Kept separate because ``domain.reclamation``'s stored-variable list is
#: the authority on which variables have member shards, and these do not.
DERIVED_VARIABLES: Final[frozenset[str]] = frozenset({"wind_10m"})


def encoding_for(variable: str) -> VariableEncoding:
    """Return the approved encoding for a variable.

    Raises:
        VariableClassError: if the variable has no approved encoding. There is deliberately
            no default: falling back to the near-Gaussian encoding for a heavy-tailed
            variable would store a field that looks right and cannot answer its thresholds.
    """
    key = variable.strip()
    try:
        return _VARIABLE_ENCODINGS[key]
    except KeyError as exc:
        raise VariableClassError(
            f"no approved aggregate encoding for {variable!r}; "
            f"registered: {sorted(REGISTERED_VARIABLES)}"
        ) from exc


def is_registered(variable: str) -> bool:
    """Whether a variable has an approved encoding."""
    return variable.strip() in _VARIABLE_ENCODINGS


def spec_for(variable: str) -> AggregateSpec:
    """Return the aggregate spec for a variable.

    Raises:
        VariableClassError: for an unknown variable or for a flag, which carries no spec.
    """
    encoding = encoding_for(variable)
    if encoding.spec is None:
        raise VariableClassError(
            f"{variable!r} is a flag; it stores a per-cell fraction, not an aggregate spec"
        )
    return encoding.spec


def register_encoding(encoding: VariableEncoding) -> None:
    """Add or replace a variable's encoding.

    Used when a new variable is added to the platform: the classification is an engineering
    decision recorded here, not something the pipeline may infer.
    """
    _VARIABLE_ENCODINGS[encoding.variable.strip()] = encoding


def unclassified(variables: frozenset[str] | set[str]) -> frozenset[str]:
    """Return the subset of ``variables`` with no approved encoding.

    Callers use this to assert that the platform's authoritative variable list and this
    registry agree, so adding a variable without classifying it fails loudly instead of
    falling through to some default at write time.
    """
    return frozenset(v for v in variables if not is_registered(v))
