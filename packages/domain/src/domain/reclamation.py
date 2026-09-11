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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

#: Target kind identifiers.
TARGET_KIND_DET = "det"
TARGET_KIND_MEAN = "mean"
TARGET_KIND_MEM = "mem"
VALID_TARGET_KINDS: frozenset[str] = frozenset(
    {TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM}
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
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_member_index(target_kind: str, member_index: int | None = None) -> int:
    """Return the normalized member_index for a target kind.

    - det  -> 0
    - mean -> -1
    - mem  -> 1..30 (defaults to 1 if unspecified)
    """
    kind = target_kind.lower().strip()
    if kind == TARGET_KIND_DET:
        return 0
    if kind == TARGET_KIND_MEAN:
        return -1
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


def make_shard_relative_key(
    variable_code: str,
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the relative object key for a sharded_v1 variable shard.

    Examples:
        make_shard_relative_key('temperature_2m', 'det', 0)
        -> 'temperature_2m/shard.det_L0000.shard'
        make_shard_relative_key('temperature_2m', 'mean', 6)
        -> 'temperature_2m/shard.mean_L0006.shard'
        make_shard_relative_key('temperature_2m', 'mem', 6, member_index=3)
        -> 'temperature_2m/shard.mem003_L0006.shard'
    """
    kind = target_kind.lower().strip()
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


def make_region_marker_relative_key(
    target_kind: str,
    lead_time_hours: int,
    member_index: int = 0,
) -> str:
    """Generate the relative object key for an ingestion region commit marker.

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


def is_predecessor_dependent_lead(lead_time_hours: int) -> bool:
    """Return True if lead_time_hours is a 6-hour reset lead requiring predecessor L - 3."""
    return lead_time_hours > 0 and lead_time_hours % 6 == 0


def get_predecessor_lead(lead_time_hours: int) -> int:
    """Return the required predecessor lead time for a 6-hour reset lead.

    Raises:
        ValueError: If lead_time_hours is not a 6-hour reset lead.
    """
    if not is_predecessor_dependent_lead(lead_time_hours):
        raise ValueError(f"Lead {lead_time_hours} is not a 6-hour reset lead.")
    return lead_time_hours - 3


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

