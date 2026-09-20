"""Pure domain types and rules for Data Lifecycle V3 Phase 4 granular reclamation.

This module contains no database connections, ORM models, or cloud storage I/O.
It defines:
- PhysicalShardTarget: The granular physical deletion unit.
- IngestionRegionIdentity: The atomic region commit identity.
- Shard naming and key resolution rules for sharded_v1 containers.
- Ingestion predecessor dependency rules (L % 6 == 0 requires L - 3).
- Region marker cleanup validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

#: Target kind identifiers.
TARGET_KIND_DET = "det"
TARGET_KIND_MEAN = "mean"
TARGET_KIND_MEM = "mem"
#: The aggregate (statistic) container a variable's members are replaced by.
#:
#: It is a **deletion unit** -- the object whose existence authorizes a member's reclamation -- and
#: deliberately not a member shard. ``parse_shard_filename``/``is_shard_filename`` must keep
#: rejecting ``shard.agg_L0006.shard``: store-layout detection, the ingestion reader's reassembly
#: and the API reader all identify member shards by that filename grammar, so a key that parsed as
#: one would have a reader try to reassemble statistic planes as a member field. The aggregate has
#: its own key builder (:func:`make_aggregate_relative_key`) for exactly that reason.
TARGET_KIND_AGG = "agg"
VALID_TARGET_KINDS: frozenset[str] = frozenset(
    {TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM, TARGET_KIND_AGG}
)

#: Reclamation queue statuses.
RECLAMATION_STATUS_QUEUED = "queued"
RECLAMATION_STATUS_DELETING = "deleting"
RECLAMATION_STATUS_DELETED = "deleted"
RECLAMATION_STATUS_FAILED = "failed"
VALID_RECLAMATION_STATUSES: frozenset[str] = frozenset(
    {
        RECLAMATION_STATUS_QUEUED,
        RECLAMATION_STATUS_DELETING,
        RECLAMATION_STATUS_DELETED,
        RECLAMATION_STATUS_FAILED,
    }
)

#: Variables that require predecessor retention at 6-hour reset leads.
PREDECESSOR_VARIABLES: frozenset[str] = frozenset(
    {
        "precipitation_amount_3h",
        "cloud_cover_3h",
    }
)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def normalize_member_index(target_kind: str, member_index: int | None = None) -> int:
    """Return the normalized member_index for a target kind.

    - det  -> 0
    - mean -> -1
    - mem  -> 1..30 (defaults to 1 if unspecified)
    - agg  -> 0 (one container per ``(variable, lead)``: a container is not a member)
    """
    kind = target_kind.lower().strip()
    if kind == TARGET_KIND_DET:
        return 0
    if kind == TARGET_KIND_MEAN:
        return -1
    if kind == TARGET_KIND_AGG:
        return 0
    if kind == TARGET_KIND_MEM:
        if member_index is None:
            return 1
        return int(member_index)
    raise ValueError(f"Unknown target_kind: {target_kind!r}")


@dataclass(frozen=True)
class PhysicalShardTarget:
    """The granular physical deletion unit: one variable shard in a Zarr container.

    Attributes:
        run_id: Model run database ID.
        model_id: Model identifier ('gfs', 'gefs').
        cycle_time: Cycle UTC datetime.
        lead_time_hours: Lead time hours offset.
        variable_code: The data variable identifier (e.g. 'temperature_2m').
        target_kind: 'det', 'mean', or 'mem'.
        member_index: 0 for det, -1 for mean, 1..30 for mem.
        valid_time: Valid UTC datetime.
        store_path: Canonical Zarr store path/URL.
        physical_key: Relative or full physical key of the shard file.
    """

    run_id: str
    model_id: str
    cycle_time: datetime
    lead_time_hours: int
    variable_code: str
    target_kind: str
    member_index: int
    valid_time: datetime
    store_path: str
    physical_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", self.model_id.lower().strip())
        object.__setattr__(self, "target_kind", self.target_kind.lower().strip())
        object.__setattr__(self, "cycle_time", _ensure_utc(self.cycle_time))
        object.__setattr__(self, "valid_time", _ensure_utc(self.valid_time))
        if self.target_kind not in VALID_TARGET_KINDS:
            raise ValueError(f"Invalid target_kind: {self.target_kind!r}")
        normalized_mem = normalize_member_index(self.target_kind, self.member_index)
        if self.member_index != normalized_mem:
            object.__setattr__(self, "member_index", normalized_mem)

    @property
    def target_tuple(self) -> tuple[str, int, str, str, int]:
        """Unique target tuple:
        (run_id, lead_time_hours, variable_code, target_kind, member_index).
        """
        return (
            self.run_id,
            self.lead_time_hours,
            self.variable_code,
            self.target_kind,
            self.member_index,
        )


@dataclass(frozen=True)
class IngestionRegionIdentity:
    """The atomic unit of NOAA acquisition and region commit markers.

    One region contains multiple variable shards:
    - GFS: det_L{lead:04d}
    - GEFS mean: mean_L{lead:04d}
    - GEFS member: mem{member:03d}_L{lead:04d}
    """

    run_id: str
    lead_time_hours: int
    target_kind: str
    member_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_kind", self.target_kind.lower().strip())
        normalized_mem = normalize_member_index(self.target_kind, self.member_index)
        object.__setattr__(self, "member_index", normalized_mem)


def make_aggregate_relative_key(variable_code: str, lead_time_hours: int) -> str:
    """Generate the relative object key for a variable's aggregate container.

    Shaped like a member shard's key because that is what it is: one object per
    ``(variable, lead)`` holding that variable's statistics. The difference is that
    :func:`parse_shard_filename` **rejects** it, deliberately -- see
    :data:`TARGET_KIND_AGG`.

    Examples:
        make_aggregate_relative_key('temperature_2m', 6)
        -> 'temperature_2m/shard.agg_L0006.shard'
    """
    return f"{variable_code}/shard.{TARGET_KIND_AGG}_L{lead_time_hours:04d}.shard"


def make_shard_relative_key(
    variable_code: str,
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the relative object key for a stored object of a target kind.

    Examples:
        make_shard_relative_key('temperature_2m', 'det', 0)
        -> 'temperature_2m/shard.det_L0000.shard'
        make_shard_relative_key('temperature_2m', 'mean', 6)
        -> 'temperature_2m/shard.mean_L0006.shard'
        make_shard_relative_key('temperature_2m', 'mem', 6, member_index=3)
        -> 'temperature_2m/shard.mem003_L0006.shard'
        make_shard_relative_key('temperature_2m', 'agg', 6)
        -> 'temperature_2m/shard.agg_L0006.shard'

    ``agg`` is the one kind this builds whose key :func:`parse_shard_filename` refuses to parse
    back, and that asymmetry is the point: the aggregate is a deletion unit, not a member shard,
    so a reader that walks the member grammar must not pick it up. The key template lives here
    rather than in the ingestion writer for the same reason every other one does.
    """
    kind = target_kind.lower().strip()
    if kind == TARGET_KIND_AGG:
        return make_aggregate_relative_key(variable_code, lead_time_hours)
    if kind == TARGET_KIND_DET:
        return f"{variable_code}/shard.det_L{lead_time_hours:04d}.shard"
    if kind == TARGET_KIND_MEAN:
        return f"{variable_code}/shard.mean_L{lead_time_hours:04d}.shard"
    if kind == TARGET_KIND_MEM:
        mem_idx = normalize_member_index(kind, member_index)
        return f"{variable_code}/shard.mem{mem_idx:03d}_L{lead_time_hours:04d}.shard"
    raise ValueError(f"Unknown target_kind: {target_kind!r}")


def make_shard_physical_key(
    store_path: str,
    variable_code: str,
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the full storage path/key for a physical shard."""
    rel = make_shard_relative_key(
        variable_code, target_kind, lead_time_hours, member_index
    )
    clean_store = store_path.rstrip("/")
    return f"{clean_store}/{rel}"


def target_kind_for(member: int | None, *, is_mean: bool = False) -> tuple[str, int]:
    """Map the acquisition-side ``(member, is_mean)`` convention onto a target kind.

    This is the single translation point between the two ways the platform names a
    region: the download/decode side speaks ``member: int | None`` plus an ``is_mean``
    flag, while reclamation and the shard keys speak ``target_kind`` plus a normalized
    ``member_index``. Duplicating the mapping is how the two drift apart.

    Returns:
        ``(target_kind, member_index)`` ready for :func:`make_shard_relative_key`.
    """
    if is_mean:
        return TARGET_KIND_MEAN, -1
    if member is None:
        return TARGET_KIND_DET, 0
    return TARGET_KIND_MEM, int(member)


def make_shard_filename(
    variable_code: str,
    *,
    member: int | None,
    lead_time_hours: int,
    is_mean: bool = False,
) -> str:
    """Generate a shard's store-relative key from the acquisition convention.

    Identical to :func:`make_shard_relative_key` after :func:`target_kind_for`, and the
    exact inverse of :func:`parse_shard_filename`. Prefer this over hand-built f-strings:
    the key template is duplicated nowhere else.

    Examples:
        make_shard_filename('temperature_2m', member=None, lead_time_hours=0)
        -> 'temperature_2m/shard.det_L0000.shard'
        make_shard_filename('temperature_2m', member=None, lead_time_hours=6, is_mean=True)
        -> 'temperature_2m/shard.mean_L0006.shard'
        make_shard_filename('temperature_2m', member=3, lead_time_hours=6)
        -> 'temperature_2m/shard.mem003_L0006.shard'
    """
    kind, member_index = target_kind_for(member, is_mean=is_mean)
    return make_shard_relative_key(variable_code, kind, lead_time_hours, member_index)


#: Every physical shard object ends with this suffix. Store layout detection keys on it.
SHARD_SUFFIX = ".shard"

#: Prefixes of the shard filename body, longest first so no prefix shadows another.
_KIND_FILENAME_PREFIX: tuple[tuple[str, str], ...] = (
    ("shard.mean_L", TARGET_KIND_MEAN),
    ("shard.mem", TARGET_KIND_MEM),
    ("shard.det_L", TARGET_KIND_DET),
)


def parse_shard_filename(filename: str) -> tuple[int | None, int, bool]:
    """Parse ``(member, lead_time_hours, is_mean)`` from a shard filename.

    The inverse of :func:`make_shard_filename`, and the only place the filename grammar
    is decoded. Accepts a bare filename or a store-relative key with a variable prefix.

    Raises:
        ValueError: if the name is not a recognized shard filename. Callers that enumerate
            a store may encounter unrelated objects; they should test
            :func:`is_shard_filename` first rather than catching this.
    """
    base = filename.rsplit("/", 1)[-1]
    if not base.endswith(SHARD_SUFFIX):
        raise ValueError(f"Not a shard filename: {filename!r}")
    stem = base[: -len(SHARD_SUFFIX)]
    for prefix, kind in _KIND_FILENAME_PREFIX:
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :]
        if kind == TARGET_KIND_MEM:
            member_str, _, lead_str = rest.partition("_L")
            if not member_str or not lead_str:
                break
            return int(member_str), int(lead_str), False
        return None, int(rest), kind == TARGET_KIND_MEAN
    raise ValueError(f"Unrecognized shard filename: {filename!r}")


def is_shard_filename(filename: str) -> bool:
    """Return True if ``filename`` is a well-formed shard object name."""
    try:
        parse_shard_filename(filename)
    except (ValueError, IndexError):
        return False
    return True


def make_region_marker_relative_key(
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the relative object key for an ingestion region commit marker.

    ``agg`` has no marker: a region marker records that a *member region was acquired*, and the
    aggregate is computed from members rather than acquired, so it has nothing to mark. The
    refusal is explicit rather than inherited from the final ``raise``, so the message says which
    kind is missing and why rather than "unknown target_kind".

    Examples:
        make_region_marker_relative_key('det', 0)
        -> '__commit__/v1/regions/det_L0000.json'
        make_region_marker_relative_key('mean', 6)
        -> '__commit__/v1/regions/mean_L0006.json'
        make_region_marker_relative_key('mem', 6, member_index=3)
        -> '__commit__/v1/regions/mem003_L0006.json'
    """
    kind = target_kind.lower().strip()
    if kind == TARGET_KIND_DET:
        return f"__commit__/v1/regions/det_L{lead_time_hours:04d}.json"
    if kind == TARGET_KIND_MEAN:
        return f"__commit__/v1/regions/mean_L{lead_time_hours:04d}.json"
    if kind == TARGET_KIND_MEM:
        mem_idx = normalize_member_index(kind, member_index)
        return f"__commit__/v1/regions/mem{mem_idx:03d}_L{lead_time_hours:04d}.json"
    if kind == TARGET_KIND_AGG:
        raise ValueError(
            "the agg kind has no region marker: a marker records an acquired member region, "
            "and an aggregate is computed from members rather than acquired"
        )
    raise ValueError(f"Unknown target_kind: {target_kind!r}")


def make_region_marker_physical_key(
    store_path: str,
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the full storage path/key for an ingestion region commit marker."""
    rel = make_region_marker_relative_key(target_kind, lead_time_hours, member_index)
    clean_store = store_path.rstrip("/")
    return f"{clean_store}/{rel}"


def is_predecessor_dependent_lead(
    lead_time_hours: int,
    *,
    reset_period_hours: int = 6,
) -> bool:
    """Return True if lead L is a variable-reset lead requiring a predecessor hold.

    Reset leads are ``L > 0 and L % R == 0`` with ``R = variable.reset_period_hours``.
    The reset period is per-variable metadata (architecture doc I19) — it is NOT
    the model cycle cadence and must never be hardcoded to 6 in new call sites.
    """
    return lead_time_hours > 0 and lead_time_hours % reset_period_hours == 0


def get_predecessor_lead(
    lead_time_hours: int,
    *,
    interval_width_hours: int = 3,
    reset_period_hours: int = 6,
) -> int:
    """Return the required predecessor lead ``L - W`` for a reset lead.

    ``W = variable.interval_width_hours``: at a reset lead L the upstream
    quantity is running-since-reset over R hours, so the trailing W-width
    interval is reconstructed from the samples at L and L - W. The historically
    hardcoded ``L - 3`` was the W=3 special case; assuming ``W == R / 2`` in
    generic logic is forbidden (architecture doc I19).

    Raises:
        ValueError: If lead_time_hours is not a reset lead for the given R, or
            the metadata combination is invalid (W <= 0, R <= 0, or W > R).
    """
    if reset_period_hours <= 0 or interval_width_hours <= 0:
        raise ValueError(
            "reset_period_hours and interval_width_hours must be positive, got "
            f"W={interval_width_hours}, R={reset_period_hours}"
        )
    if interval_width_hours > reset_period_hours:
        raise ValueError(
            f"interval width W={interval_width_hours} must not exceed reset "
            f"period R={reset_period_hours}"
        )
    if not is_predecessor_dependent_lead(
        lead_time_hours, reset_period_hours=reset_period_hours
    ):
        raise ValueError(
            f"Lead {lead_time_hours} is not a reset lead for R={reset_period_hours}."
        )
    return lead_time_hours - interval_width_hours


def is_predecessor_variable(variable: str | None) -> bool:
    """Return True if variable requires predecessor retention."""
    if variable is None:
        return False
    return variable.strip().lower() in PREDECESSOR_VARIABLES


def can_delete_region_marker(
    expected_variables: frozenset[str] | set[str],
    deleted_variables: frozenset[str] | set[str],
) -> bool:
    """Return True if every expected variable shard for a region has been deleted."""
    if not expected_variables:
        return False
    return set(expected_variables).issubset(set(deleted_variables))


#: Authoritative schema-expected variables per (model_id, version_string, target_kind).
#: GFS deterministic has 15 variables.
#: GEFS mean and GEFS member have 14 variables (no instant precipitation_rate in pgrb2sp25).
DEFAULT_EXPECTED_REGION_VARIABLES: dict[tuple[str, str, str] | tuple[str, str], frozenset[str]] = {
    ("gfs", "v1.0", TARGET_KIND_DET): frozenset(
        {
            "temperature_2m",
            "precipitation_rate",
            "precipitation_amount_3h",
            "crain",
            "csnow",
            "cfrzr",
            "cicep",
            "relative_humidity_2m",
            "wind_gust",
            "visibility",
            "snow_depth",
            "wind_u_10m",
            "wind_v_10m",
            "cloud_cover_3h",
            "cloud_ceiling",
        }
    ),
    ("gefs", "v1.0", TARGET_KIND_MEAN): frozenset(
        {
            "temperature_2m",
            "precipitation_amount_3h",
            "crain",
            "csnow",
            "cfrzr",
            "cicep",
            "relative_humidity_2m",
            "wind_gust",
            "visibility",
            "snow_depth",
            "wind_u_10m",
            "wind_v_10m",
            "cloud_cover_3h",
            "cloud_ceiling",
        }
    ),
    ("gefs", "v1.0", TARGET_KIND_MEM): frozenset(
        {
            "temperature_2m",
            "precipitation_amount_3h",
            "crain",
            "csnow",
            "cfrzr",
            "cicep",
            "relative_humidity_2m",
            "wind_gust",
            "visibility",
            "snow_depth",
            "wind_u_10m",
            "wind_v_10m",
            "cloud_cover_3h",
            "cloud_ceiling",
        }
    ),
}

# Add two-tuple aliases for backward compatibility
DEFAULT_EXPECTED_REGION_VARIABLES[("gfs", TARGET_KIND_DET)] = (
    DEFAULT_EXPECTED_REGION_VARIABLES[("gfs", "v1.0", TARGET_KIND_DET)]
)
DEFAULT_EXPECTED_REGION_VARIABLES[("gefs", TARGET_KIND_MEAN)] = (
    DEFAULT_EXPECTED_REGION_VARIABLES[("gefs", "v1.0", TARGET_KIND_MEAN)]
)
DEFAULT_EXPECTED_REGION_VARIABLES[("gefs", TARGET_KIND_MEM)] = (
    DEFAULT_EXPECTED_REGION_VARIABLES[("gefs", "v1.0", TARGET_KIND_MEM)]
)

_EXPECTED_REGION_VARIABLES: dict[tuple[str, str, str] | tuple[str, str], frozenset[str]] = dict(
    DEFAULT_EXPECTED_REGION_VARIABLES
)


def register_expected_region_variables(
    model_id: str,
    target_kind: str,
    variables: Iterable[str],
    version_string: str | None = None,
) -> None:
    """Register or override the authoritative expected variables for a model/version/kind."""
    m_clean = model_id.lower().strip()
    k_clean = target_kind.lower().strip()
    var_set = frozenset(variables)
    if version_string is not None:
        v_clean = version_string.lower().strip()
        _EXPECTED_REGION_VARIABLES[(m_clean, v_clean, k_clean)] = var_set
    else:
        _EXPECTED_REGION_VARIABLES[(m_clean, "v1.0", k_clean)] = var_set
    _EXPECTED_REGION_VARIABLES[(m_clean, k_clean)] = var_set


def get_expected_region_variables(
    model_id: str,
    target_kind: str,
    version_string: str | None = None,
    fallback: Iterable[str] | None = None,
) -> frozenset[str]:
    """Return the authoritative schema-expected variables for a model, version, and region kind.

    Never relies on whatever incomplete rows happen to exist in catalog.
    """
    m_clean = model_id.lower().strip()
    k_clean = target_kind.lower().strip()
    if version_string is not None:
        v_clean = version_string.lower().strip()
        if (m_clean, v_clean, k_clean) in _EXPECTED_REGION_VARIABLES:
            return _EXPECTED_REGION_VARIABLES[(m_clean, v_clean, k_clean)]
    if (m_clean, k_clean) in _EXPECTED_REGION_VARIABLES:
        return _EXPECTED_REGION_VARIABLES[(m_clean, k_clean)]
    if fallback is not None:
        return frozenset(fallback)
    raise ValueError(f"No authoritative variable schema registered for {(m_clean, k_clean)!r}")

