"""Domain data structures and functions for forecast points and wind fields."""

from domain.models.point import ForecastPoint
from domain.models.wind import (
    CALM_WIND_THRESHOLD_MPS,
    CARDINAL_DIRECTIONS_8,
    CARDINAL_DIRECTIONS_16,
    WIND_ROSE_SPEED_BINS_MPS,
    ConsensusVectorData,
    WindRoseData,
    WindRoseSector,
    compute_consensus_vector,
    compute_directional_probability,
    compute_wind_rose,
    derive_ensemble_mean_scalar_speed,
    derive_ensemble_mean_vector,
    derive_ensemble_member_speeds,
    derive_meteorological_direction,
    derive_meteorological_direction_array,
    derive_wind_speed,
    get_cardinal_direction,
)

__all__ = [
    "CALM_WIND_THRESHOLD_MPS",
    "CARDINAL_DIRECTIONS_8",
    "CARDINAL_DIRECTIONS_16",
    "ConsensusVectorData",
    "ForecastPoint",
    "WIND_ROSE_SPEED_BINS_MPS",
    "WindRoseData",
    "WindRoseSector",
    "compute_consensus_vector",
    "compute_directional_probability",
    "compute_wind_rose",
    "derive_ensemble_mean_scalar_speed",
    "derive_ensemble_mean_vector",
    "derive_ensemble_member_speeds",
    "derive_meteorological_direction",
    "derive_meteorological_direction_array",
    "derive_wind_speed",
    "get_cardinal_direction",
]
