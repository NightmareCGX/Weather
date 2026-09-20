"""The deletion admission gate: does the stored field vector reproduce the member path?

Members may be deleted only where the container answers everything the members did, so before any
deletion the two have to be compared on real data, per variable and per **displayed** quantity,
against a yardstick that says whether a difference matters. The yardstick is the investigation's
own (``REPORT.md`` §9): the **ensemble's sampling noise**, measured by splitting the 30 members
into two disjoint 15-member halves and taking their disagreement. An error below that is below the
difference a different 30 members would have produced, so it is not something a reader could act
on.

How the two sides are produced
------------------------------
**The stored side is the production path.** The members are written to a staging area with the
real v1 encoder, the container is built by ``aggregate_staged_lead`` -- the same call a publication
makes -- and the stored fields are read back by the API's own ``AggregateShardReader``. So the
numbers compared are the bytes a reader gets, including the fixed-point round trip they went
through.

**The member side is the arithmetic the response applies.** Each compared quantity is computed
from the member values with the same function the API uses (``domain.ensemble``,
``domain.models.wind``, ``domain.models.precipitation``, ``domain.models.cloud``,
``domain.product_fields``). That is the definition: it is what the platform serves today, and what
a reader would stop getting if the members went away.

The comparison is per cell over a sample of cells inside one chunk. One chunk is a real 1x1 degree
region and using it removes every index-conversion question from the comparison, which is not what
is being tested here.

The rose needs one word: its buckets are **quantile edges of this cycle's member set**, stored in
the container. So a bucket index is compared against the member path *binned with the same edges*
-- exactly what a reader does, since the edges travel with the fields.

Result on real GEFS 20260920 00Z, f006 with f003 as its predecessor, 30 members, 200 sampled
cells: **128 judged comparisons, none at or above the sampling noise**; the worst is 0.564
(``temperature_2m``'s p90) and the median is 0.003. Two findings came out of building it, both
fixed: the dry/``trace`` comparison disagreed with the classifier by one ulp (3.5x the noise at the
cell it bit), and the censoring counts had been stored as fractions at a one-member step.

Run:
    .venv/Scripts/python.exe gefs_fetch_all.py 20260920 00 f006 f003
    .venv/Scripts/python.exe compare_paths.py
    .venv/Scripts/python.exe compare_paths.py --quick   # fewer cells
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in (
    _REPO_ROOT,
    _REPO_ROOT / "packages" / "domain" / "src",
    _REPO_ROOT / "services" / "api" / "src",
    _REPO_ROOT / "services" / "ingestion" / "src",
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE",
    os.path.join(tempfile.gettempdir(), "weather_realdata"),
)

DATE, CYCLE = "20260920", "00"
LEAD, PREDECESSOR_LEAD = 6, 3
N_MEMBERS = 30
GRID_LAT, GRID_LON = 721, 1440
CHUNK = 100

#: Cells sampled inside the chunk both sides are read from.
SAMPLES = 200
SEED = 20260920

#: Variables whose container is a distribution and nothing else, so its fields can be built from
#: its own members alone. ``precipitation_amount_3h`` is not here: its container also carries the
#: phase groups, which read the four flags, and that is compared separately.
PLAIN_VARIABLES = (
    "temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_gust",
    "relative_humidity_2m",
    "visibility",
    "snow_depth",
)
FLAG_VARIABLES = ("crain", "csnow", "cfrzr", "cicep")
CLOUD_VARIABLES = ("cloud_ceiling", "cloud_cover_3h")
WIND_COMPONENTS = ("wind_u_10m", "wind_v_10m")
PRECIPITATION_INPUTS = ("precipitation_amount_3h", *FLAG_VARIABLES)

#: The chunk both sides are read from. A central one, so no cell is on the grid edge.
CHUNK_ROW, CHUNK_COL = GRID_LAT // (2 * CHUNK), GRID_LON // (2 * CHUNK)


#: GRIB's own units to the platform's canonical ones, per variable. The platform stores Celsius,
#: km/h, kilometres and millimetres; GRIB reports Kelvin, m/s, metres and kg m-2. This mirrors
#: ``ingestion.core.pipeline``'s table for the variables this bench compares, and the flags are
#: the fourth case: GRIB reports them as an average over the accumulation window and the platform
#: stores the share of that window.
_TO_CANONICAL = {
    "temperature_2m": lambda plane: plane - np.float32(273.15),
    "wind_gust": lambda plane: plane * np.float32(3.6),
    "visibility": lambda plane: plane / np.float32(1000.0),
    # Ceiling comes as geopotential metres, which the platform stores in kilometres.
    "cloud_ceiling": lambda plane: plane / np.float32(1000.0),
    "crain": lambda plane: plane / np.float32(6.0),
    "csnow": lambda plane: plane / np.float32(6.0),
    "cfrzr": lambda plane: plane / np.float32(6.0),
    "cicep": lambda plane: plane / np.float32(6.0),
}


def message_path(lead: str, member: int, variable: str) -> str:
    """The cached GRIB2 message for one ``(member, lead, variable)``."""
    return os.path.join(CACHE, f"{DATE}{CYCLE}_gep{member:02d}_{lead}_{variable}.grib2")


def load_plane(path: str) -> np.ndarray:
    """One GRIB2 message as a float32 ``(lat, lon)`` plane."""
    import xarray as xr

    dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        return np.asarray(
            dataset[list(dataset.data_vars)[0]].values, dtype=np.float32
        ).squeeze()
    finally:
        dataset.close()


def load_stack(lead: str, variable: str) -> np.ndarray:
    """The real GEFS members of one variable at one lead, ``(n_members, lat, lon)``.

    Converted to the platform's canonical unit, because that is the unit the container stores:
    comparing against Kelvin while the store holds Celsius would report a 273-degree encoding
    error that is really a unit mismatch.
    """
    planes = []
    for member in range(1, N_MEMBERS + 1):
        path = message_path(lead, member, variable)
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}; run:  python gefs_fetch_all.py {DATE} {CYCLE} "
                f"f{LEAD:03d} f{PREDECESSOR_LEAD:03d}"
            )
        planes.append(load_plane(path))
    stack = np.stack(planes)
    if stack.shape[1:] != (GRID_LAT, GRID_LON):
        raise SystemExit(
            f"{variable} came out {stack.shape[1:]}, expected {(GRID_LAT, GRID_LON)}"
        )
    if variable in _TO_CANONICAL:
        stack = _TO_CANONICAL[variable](stack).astype(np.float32)
    return stack


def read_container(
    variable: str,
    staged: list[tuple[str, np.ndarray, int]],
    *,
    lead: int,
):
    """Build a variable's container the way a publication does, and read it back.

    ``staged`` is a list rather than a mapping keyed by variable name, because the same variable
    appears once per *lead*: the phase and transition groups read the predecessor interval's
    ``precipitation_amount_3h`` and its flags, which live at a different lead under the same name.
    A mapping would silently keep only one of the two.

    One container is written **per lead**, which is again what a publication does -- each lead has
    its own object and its own member set -- so the lead under comparison is the one whose
    container is read.

    Args:
        variable: The variable to build a container for.
        staged: ``(name, members, lead)`` for every input to stage.
        lead: The lead whose container to read.

    Returns:
        ``(store, stack)`` where ``stack`` is the reader's ``(n_fields, chunk, chunk)`` view at
        the sampled chunk. The caller removes ``store``.
    """
    import xarray as xr

    from api.core.aggregate_reader import AggregateShardReader
    from ingestion.core import aggregate_staging as staging
    from ingestion.core.zarr_writer import encode_region_sharded_v1

    store = tempfile.mkdtemp(prefix="compare_paths_")
    for name, members, members_lead in staged:
        for index in range(members.shape[0]):
            dataset = xr.Dataset(
                {
                    name: (
                        ("member", "lead_time_hours", "latitude", "longitude"),
                        members[index][None, None],
                    )
                }
            )
            for _key, blob in encode_region_sharded_v1(
                dataset, member=index + 1, lead_time_hours=members_lead
            ):
                relative = staging.staging_relative_key(name, index + 1, members_lead)
                full = os.path.join(store, *relative.split("/"))
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as handle:
                    handle.write(blob)
    leads = sorted({members_lead for _name, _members, members_lead in staged})
    reader = AggregateShardReader(store)
    try:
        # Every lead is built with its staging kept, and the release happens once at the end --
        # the order the production pass uses, and not cosmetic here either: the lead under
        # comparison reads the *earlier* lead's amount and flags, so releasing lead 3's staging
        # while writing lead 3's own container would leave lead 6 with no predecessor, and the
        # pass would (correctly) refuse to build it.
        for staged_lead in leads:
            staging.aggregate_staged_lead(
                store,
                variable,
                staged_lead,
                grid_lat=GRID_LAT,
                grid_lon=GRID_LON,
                expected_members=N_MEMBERS,
                wave_leads=tuple(leads),
                drop_staging=False,
            )
        stack = reader.read_location(
            variable,
            lead_time_hours=lead,
            chunk_row=CHUNK_ROW,
            chunk_col=CHUNK_COL,
        )
    except Exception:
        shutil.rmtree(store, ignore_errors=True)
        raise
    if stack is None:
        shutil.rmtree(store, ignore_errors=True)
        raise SystemExit(f"the container for {variable!r} could not be read back")
    return store, stack


class Report:
    """One comparison per (variable, quantity), printed as it is measured."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, float, float]] = []

    def add(self, variable: str, field: str, error: float, noise: float) -> None:
        self.rows.append((variable, field, error, noise))
        if noise > 0:
            value = error / noise
            verdict = "ok" if value < 1.0 else "OVER"
            ratio_text = f"{value:6.3f}"
        else:
            # No yardstick: the two halves agreed exactly at every sampled cell, so there is no
            # sampling noise to divide by. Reported as a pass or a fail on an absolute bound
            # instead -- a quantisation step -- because a ratio would be an artefact.
            verdict = "ok" if error < 0.01 else "OVER"
            ratio_text = "  n/a "
        print(
            f"  {variable:<24} {field:<30} err {error:10.5f}  noise {noise:10.5f}  "
            f"{ratio_text}  {verdict}",
            flush=True,
        )

    def failures(self) -> list[tuple[str, str, float]]:
        return [
            (variable, field, error / noise)
            for variable, field, error, noise in self.rows
            if noise > 0 and error / noise >= 1.0
        ] + [
            (variable, field, error)
            for variable, field, error, noise in self.rows
            if noise == 0 and error >= 0.01
        ]


class Sample:
    """The sampled cells, the member stacks, and the two halves used for the noise estimate."""

    def __init__(self, rng: np.random.Generator, samples: int) -> None:
        self.rows = rng.integers(0, CHUNK, samples)
        self.cols = rng.integers(0, CHUNK, samples)
        half = N_MEMBERS // 2
        # Cell coordinates in the full grid, which is how the member stacks are indexed.
        self.grid_rows = CHUNK_ROW * CHUNK + self.rows
        self.grid_cols = CHUNK_COL * CHUNK + self.cols
        self.first = slice(0, half)
        self.second = slice(half, 2 * half)

    def member(self, stack: np.ndarray, part: slice | None = None) -> np.ndarray:
        """The member values at the sampled cells, ``(n_members, n_samples)``."""

        def take(row: int, col: int) -> np.ndarray:
            column = stack[:, row, col]
            return column if part is None else column[part]

        return np.stack(
            [
                take(int(row), int(col))
                for row, col in zip(self.grid_rows, self.grid_cols, strict=True)
            ],
            axis=1,
        )

    def stored(self, field: np.ndarray) -> np.ndarray:
        """One stored field's values at the sampled cells."""
        return np.array(
            [
                float(field[int(row), int(col)])
                for row, col in zip(self.rows, self.cols, strict=True)
            ]
        )


def compare(
    aggregated: np.ndarray,
    member: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    """The worst disagreement, and the yardstick to judge it by.

    NaN-aware on both sides, and only where the two agree on whether there *is* a value: a cell
    the container refuses and the member path also refuses is agreement, and letting NaN count as
    a difference would report the coverage rule as an encoding error.

    The yardstick is the disagreement between the two halves where they both have a value. Where
    they do not -- ``cloud_ceiling``'s summary refuses fewer than 21 valid members, so a 15-member
    half can never be summarised, and its halves-pair is empty at *every* cell -- the fallback is
    the larger of each half's distance from the full set, which asks the same question ("how far
    does a different 15 members move this number?") through a comparison that is defined. A zero
    yardstick with a non-zero error is left as zero rather than floored: an exact halves agreement
    is a real outcome, and the caller reports it as "no yardstick" instead of a ratio.
    """
    both = np.isfinite(aggregated) & np.isfinite(member)
    error = float(np.max(np.abs(aggregated[both] - member[both]))) if both.any() else 0.0
    halves = np.isfinite(first) & np.isfinite(second)
    if halves.any():
        return error, float(np.max(np.abs(first[halves] - second[halves])))
    fallback = np.isfinite(first) & np.isfinite(member), np.isfinite(second) & np.isfinite(member)
    distances = [
        float(np.max(np.abs(first[mask] - member[mask]))) if mask.any() else 0.0
        for mask in fallback
    ]
    return error, max(distances)


def statistic(vector: np.ndarray, name: str, expected_members: int) -> np.ndarray:
    """One response statistic per cell, from that cell's member values."""
    from domain.coverage import is_cell_statistically_valid
    from domain.ensemble import (
        ensemble_mean,
        ensemble_median,
        ensemble_percentile,
        ensemble_spread,
    )
    from domain.models.precipitation import PrecipitationPhaseState  # noqa: F401

    out = np.full(vector.shape[1], np.nan)
    for index in range(vector.shape[1]):
        column = vector[:, index]
        values = [float(value) for value in column if np.isfinite(value)]
        if not values or not is_cell_statistically_valid(len(values), expected_members):
            continue
        out[index] = {
            "mean": lambda: ensemble_mean(values),
            "median": lambda: ensemble_median(values),
            "spread": lambda: ensemble_spread(values),
            "p10": lambda: ensemble_percentile(values, 10),
            "p25": lambda: ensemble_percentile(values, 25),
            "p50": lambda: ensemble_percentile(values, 50),
            "p75": lambda: ensemble_percentile(values, 75),
            "p90": lambda: ensemble_percentile(values, 90),
        }[name]()
    return out


def run_plain(report: Report, sample: Sample, stacks: dict[str, np.ndarray]) -> None:
    """The distribution's statistics, for every variable whose container is a distribution."""
    from domain.aggregate import KIND_MEAN_STD_BINS
    from domain.field_layout import aggregate_fields_for
    from domain.variable_class import spec_for

    from api.services.aggregate_serving import (
        _statistics_from_bins,
        _statistics_from_quantiles,
    )

    names = ("mean", "median", "spread", "p10", "p25", "p50", "p75", "p90")
    for variable in PLAIN_VARIABLES:
        stack = stacks[variable]
        store, stored = read_container(variable, [(variable, stack, LEAD)], lead=LEAD)
        try:
            layout = aggregate_fields_for(variable)
            spec = spec_for(variable)
            # The stored side: read the statistics from the stored fields the way a reader does.
            aggregated = {name: np.full(sample.rows.size, np.nan) for name in names}
            for index, (row, col) in enumerate(
                zip(sample.rows, sample.cols, strict=True)
            ):
                point = np.asarray(stored[:, int(row), int(col)], dtype=np.float32)
                fields = point[layout.distribution_slice]
                try:
                    if spec.kind == KIND_MEAN_STD_BINS:
                        values, _exact = _statistics_from_bins(fields, spec)
                    else:
                        values, _exact = _statistics_from_quantiles(fields, spec)
                except Exception:  # noqa: BLE001 - an uninterpretable cell is a gap
                    continue
                for name in names:
                    if name in values and np.isfinite(values[name]):
                        aggregated[name][index] = values[name]
            member_all = sample.member(stack)
            member_a = sample.member(stack, sample.first)
            member_b = sample.member(stack, sample.second)
            for name in names:
                report.add(
                    variable,
                    name,
                    *compare(
                        aggregated[name],
                        statistic(member_all, name, N_MEMBERS),
                        statistic(member_a, name, N_MEMBERS // 2),
                        statistic(member_b, name, N_MEMBERS // 2),
                    ),
                )
        finally:
            shutil.rmtree(store, ignore_errors=True)


def run_flags(report: Report, sample: Sample, stacks: dict[str, np.ndarray]) -> None:
    """A flag's per-cell fraction, against the share of members that set it."""
    from domain.field_layout import aggregate_fields_for

    def share(vector: np.ndarray) -> np.ndarray:
        finite = np.isfinite(vector)
        counts = finite.sum(axis=0)
        set_here = np.where(finite, vector >= 0.5, False).sum(axis=0)
        return np.where(counts > 0, set_here / np.maximum(counts, 1), np.nan)

    for variable in FLAG_VARIABLES:
        stack = stacks[variable]
        store, stored = read_container(variable, [(variable, stack, LEAD)], lead=LEAD)
        try:
            layout = aggregate_fields_for(variable)
            fraction = stored[layout.group_slice("fraction").start]
            report.add(
                variable,
                "fraction",
                *compare(
                    sample.stored(fraction),
                    share(sample.member(stack)),
                    share(sample.member(stack, sample.first)),
                    share(sample.member(stack, sample.second)),
                ),
            )
        finally:
            shutil.rmtree(store, ignore_errors=True)


def run_wind(report: Report, sample: Sample, stacks: dict[str, np.ndarray]) -> None:
    """The consensus vector and the rose, against the member path's own two functions.

    The rose is compared at **bucket index**, with both sides binned by the edges the container
    stores: those edges are quantile quantiles of this member set, so a bucket has no meaning
    without them, and a reader takes them from the same container it takes the counts from.
    """
    from domain.field_layout import aggregate_fields_for, group_field_names
    from domain.models.wind import (
        CALM_WIND_THRESHOLD_MPS,
        CARDINAL_DIRECTIONS_8,
        compute_consensus_vector,
    )

    u_stack, v_stack = stacks["wind_u_10m"], stacks["wind_v_10m"]
    store, stored = read_container(
        "wind_10m",
        [("wind_u_10m", u_stack, LEAD), ("wind_v_10m", v_stack, LEAD)],
        lead=LEAD,
    )
    try:
        layout = aggregate_fields_for("wind_10m")
        rose_slice = layout.group_slice("rose")
        names = group_field_names("rose")
        cells = len([name for name in names if name.startswith("ROSE_") and "EDGE" not in name])
        scalar_names = names[cells : cells + 4]
        del scalar_names

        # The stored side.
        consensus_speed = np.full(sample.rows.size, np.nan)
        direction = np.full(sample.rows.size, np.nan)
        coherence = np.full(sample.rows.size, np.nan)
        calm_share = np.full(sample.rows.size, np.nan)
        sector_bucket = np.full((8, 8, sample.rows.size), np.nan)
        edges_per_point: list[np.ndarray] = []
        for index, (row, col) in enumerate(zip(sample.rows, sample.cols, strict=True)):
            point = np.asarray(
                stored[rose_slice.start : rose_slice.stop, int(row), int(col)],
                dtype=np.float32,
            )
            edges = point[-9:]
            edges_per_point.append(edges)
            sector_bucket[:, :, index] = point[:64].reshape(8, 8)
            scalars = point[cells : cells + 4]
            consensus_speed[index] = scalars[0]
            coherence[index] = scalars[1]
            if np.isfinite(scalars[2]) and np.isfinite(scalars[3]):
                direction[index] = np.degrees(np.arctan2(scalars[2], scalars[3])) % 360.0
            accounted = float(np.nansum(point[:64]))
            calm_share[index] = 1.0 - accounted

        # The member side, binned by the *same* edges the container carries.
        def member_point(
            u_vector: np.ndarray, v_vector: np.ndarray, edges: np.ndarray
        ) -> tuple[float, float, float, float, np.ndarray]:
            pairs = [
                (float(u), float(v))
                for u, v in zip(u_vector, v_vector, strict=True)
                if np.isfinite(u) and np.isfinite(v)
            ]
            counts = np.zeros((8, 8))
            if not pairs:
                return np.nan, np.nan, np.nan, np.nan, counts
            u_values = np.array([pair[0] for pair in pairs])
            v_values = np.array([pair[1] for pair in pairs])
            consensus = compute_consensus_vector(u_values, v_values)
            speeds = np.hypot(u_values, v_values)
            live = speeds >= CALM_WIND_THRESHOLD_MPS
            degrees = (np.degrees(np.arctan2(-u_values, -v_values)) % 360.0)[live]
            sectors = (
                np.floor((degrees + 22.5) / 45.0).astype(int) % 8
            )
            buckets = np.clip(
                np.searchsorted(edges, speeds[live], side="right") - 1, 0, 7
            )
            for sector, bucket in zip(sectors, buckets, strict=True):
                counts[sector, bucket] += 1.0 / len(pairs)
            return (
                consensus.speed_mps,
                consensus.direction_deg if consensus.direction_deg is not None else np.nan,
                consensus.coherence,
                float((~live).sum()) / len(pairs),
                counts,
            )

        def member_all(part: slice | None) -> tuple[np.ndarray, ...]:
            speeds = np.full(sample.rows.size, np.nan)
            directions = np.full(sample.rows.size, np.nan)
            coherences = np.full(sample.rows.size, np.nan)
            calms = np.full(sample.rows.size, np.nan)
            counts = np.full((8, 8, sample.rows.size), np.nan)
            for index in range(sample.rows.size):
                u_vector = sample.member(u_stack, part)[:, index]
                v_vector = sample.member(v_stack, part)[:, index]
                speed, deg, coh, calm, per_cell = member_point(
                    u_vector, v_vector, edges_per_point[index]
                )
                speeds[index] = speed
                directions[index] = deg
                coherences[index] = coh
                calms[index] = calm
                counts[:, :, index] = per_cell
            return speeds, directions, coherences, calms, counts

        member_speed, member_dir, member_coh, member_calm, member_counts = member_all(None)
        first_speed, first_dir, first_coh, first_calm, first_counts = member_all(sample.first)
        second_speed, second_dir, second_coh, second_calm, second_counts = member_all(
            sample.second
        )
        # A direction near 0/360 must be compared as an angle: the raw difference across the seam
        # is 360, which is not an error of 360 degrees.
        def angular_error(left: np.ndarray, right: np.ndarray) -> float:
            both = np.isfinite(left) & np.isfinite(right)
            if not both.any():
                return 0.0
            delta = np.abs(left[both] - right[both]) % 360.0
            return float(np.max(np.minimum(delta, 360.0 - delta)))

        def angular_noise(left: np.ndarray, right: np.ndarray) -> float:
            both = np.isfinite(left) & np.isfinite(right)
            if not both.any():
                return 1e-12
            delta = np.abs(left[both] - right[both]) % 360.0
            return max(float(np.max(np.minimum(delta, 360.0 - delta))), 1e-12)

        for name, aggregated_values, member_values, first_values, second_values in (
            ("consensus.speed_mps", consensus_speed, member_speed, first_speed, second_speed),
            ("consensus.coherence", coherence, member_coh, first_coh, second_coh),
            ("rose.calm_probability", calm_share, member_calm, first_calm, second_calm),
        ):
            report.add(
                "wind_10m",
                name,
                *compare(aggregated_values, member_values, first_values, second_values),
            )
        report.add(
            "wind_10m",
            "consensus.direction_deg",
            angular_error(direction, member_dir),
            angular_noise(first_dir, second_dir),
        )
        for sector_index in range(8):
            for bucket_index in range(8):
                report.add(
                    "wind_10m",
                    f"rose.{CARDINAL_DIRECTIONS_8[sector_index]}.b{bucket_index}",
                    *compare(
                        sector_bucket[sector_index, bucket_index],
                        member_counts[sector_index, bucket_index],
                        first_counts[sector_index, bucket_index],
                        second_counts[sector_index, bucket_index],
                    ),
                )
    finally:
        shutil.rmtree(store, ignore_errors=True)


def run_precipitation(
    report: Report,
    sample: Sample,
    stacks: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
) -> None:
    """The phase support and the transition frequencies, against the member path's aggregates."""
    from domain.field_layout import aggregate_fields_for, group_field_names
    from domain.models.precipitation import (
        aggregate_ensemble_phase_support,
        classify_precipitation_phase,
        compute_transition_frequencies,
    )

    # Both intervals of every input are staged: the container under comparison is lead 6's, and
    # its phase and transition groups read lead 3's amount *and flags* as the predecessor. The
    # earlier lead's own container is written as well, because a publication writes one per lead.
    inputs = [
        (name, stacks[name], LEAD) for name in PRECIPITATION_INPUTS
    ] + [
        (name, previous[name], PREDECESSOR_LEAD) for name in PRECIPITATION_INPUTS
    ]
    store, stored = read_container("precipitation_amount_3h", inputs, lead=LEAD)
    try:
        layout = aggregate_fields_for("precipitation_amount_3h")
        phase_slice = layout.group_slice("phase")
        transition_slice = layout.group_slice("transition")
        # The names are positional (``PHASE_0``..``PHASE_5``, ``TRANSITION_00``..``_19``), so the
        # comparison addresses the planes by index plus offset and needs no name list; the
        # response's own keys come from the enum orders.
        phase_names = group_field_names("phase")

        amounts = stacks["precipitation_amount_3h"]
        flags = {name: stacks[name] for name in FLAG_VARIABLES}
        previous_amounts = previous["precipitation_amount_3h"]
        previous_flags = {name: previous[name] for name in FLAG_VARIABLES}

        phase_order = _physical_phase_order()
        transition_order = _transition_order()

        def member_support(part: slice | None) -> dict[str, np.ndarray]:
            """Phase support per sampled cell, keyed by the class's storage names."""
            out = {
                f"PHASE_{index}": np.full(sample.rows.size, np.nan)
                for index in range(len(phase_order))
            }
            amount_matrix = sample.member(amounts, part)
            flag_matrices = {
                name: sample.member(plane, part) for name, plane in flags.items()
            }
            for index in range(sample.rows.size):
                amounts_vector = amount_matrix[:, index]
                flags_vector = {
                    name: matrix[:, index] for name, matrix in flag_matrices.items()
                }
                states = [
                    classify_precipitation_phase(
                        float(amount),
                        {key: int(plane[member] >= 0.5) for key, plane in flags_vector.items()},
                    )
                    for member, amount in enumerate(amounts_vector)
                    if np.isfinite(amount)
                ]
                if not states:
                    continue
                support = aggregate_ensemble_phase_support(states)
                for position, phase in enumerate(phase_order):
                    out[f"PHASE_{position}"][index] = support[phase]
            return out

        member_support_values = member_support(None)
        first_support = member_support(sample.first)
        second_support = member_support(sample.second)
        for index, name in enumerate(phase_names[:6]):
            report.add(
                "precipitation_amount_3h",
                f"phase.{phase_order[index].value}",
                *compare(
                    sample.stored(stored[phase_slice.start + index]),
                    member_support_values[name],
                    first_support[name],
                    second_support[name],
                ),
            )
        # The predecessor planes are the earlier interval's own current planes, whose own
        # container is the thing that answers them; comparing them here would compare this
        # container against a container, which is not the question. They are covered by that
        # lead's own row in this table.

        def member_transitions(part: slice | None) -> dict[str, np.ndarray]:
            out = {
                f"TRANSITION_{index:02d}": np.full(sample.rows.size, np.nan)
                for index in range(len(transition_order))
            }
            amount_matrix = sample.member(amounts, part)
            flag_matrices = {
                name: sample.member(plane, part) for name, plane in flags.items()
            }
            previous_matrix = sample.member(previous_amounts, part)
            previous_flag_matrices = {
                name: sample.member(plane, part) for name, plane in previous_flags.items()
            }
            for index in range(sample.rows.size):
                amounts_vector = amount_matrix[:, index]
                flags_vector = {
                    name: matrix[:, index] for name, matrix in flag_matrices.items()
                }
                previous_vector = previous_matrix[:, index]
                previous_flags_vector = {
                    name: matrix[:, index]
                    for name, matrix in previous_flag_matrices.items()
                }
                states = []
                for member, amount in enumerate(amounts_vector):
                    if not np.isfinite(amount):
                        continue
                    has_previous = np.isfinite(previous_vector[member]) and all(
                        np.isfinite(plane[member]) for plane in previous_flags_vector.values()
                    )
                    states.append(
                        classify_precipitation_phase(
                            float(amount),
                            {key: int(plane[member] >= 0.5) for key, plane in flags_vector.items()},
                            amount_prev=float(previous_vector[member]) if has_previous else None,
                            flags_prev=(
                                {
                                    key: int(plane[member] >= 0.5)
                                    for key, plane in previous_flags_vector.items()
                                }
                                if has_previous
                                else None
                            ),
                        )
                    )
                if not states:
                    continue
                frequency = compute_transition_frequencies(states)
                for position, transition in enumerate(transition_order):
                    out[f"TRANSITION_{position:02d}"][index] = frequency[transition]
            return out

        member_transition_values = member_transitions(None)
        first_transition = member_transitions(sample.first)
        second_transition = member_transitions(sample.second)
        for index, transition in enumerate(transition_order):
            name = f"TRANSITION_{index:02d}"
            report.add(
                "precipitation_amount_3h",
                f"transition.{transition.value}",
                *compare(
                    sample.stored(stored[transition_slice.start + index]),
                    member_transition_values[name],
                    first_transition[name],
                    second_transition[name],
                ),
            )
    finally:
        shutil.rmtree(store, ignore_errors=True)


def run_clouds(report: Report, sample: Sample, stacks: dict[str, np.ndarray]) -> None:
    """The cloud variables' censoring counts and conditional statistics."""
    from domain.field_layout import aggregate_fields_for, group_field_names
    from domain.models.cloud import (
        cloud_ceiling_ensemble_summary,
        cloud_cover_ensemble_summary,
    )

    for variable in ("cloud_ceiling", "cloud_cover_3h"):
        stack = stacks[variable]
        store, stored = read_container(variable, [(variable, stack, LEAD)], lead=LEAD)
        try:
            layout = aggregate_fields_for(variable)
            censoring = layout.group_slice("censoring")
            conditional = layout.group_slice("conditional")
            count_names = group_field_names("censoring")
            statistic_names = group_field_names("conditional")

            def summary(part: slice | None, index: int):
                vector = sample.member(stack, part)[:, index]
                if variable == "cloud_ceiling":
                    return cloud_ceiling_ensemble_summary(vector)
                return cloud_cover_ensemble_summary(vector, min_valid=1)

            def summary_finite(part: slice, index: int):
                """The conditional statistics over half of one cell's finite members.

                A half of the *member set* can carry fewer finite members than the summariser's
                own minimum -- ``cloud_ceiling`` requires 21 valid members and a half has 15 --
                which makes its conditional summary ``None`` and the row unjudgeable from halves.
                Halving the finite members keeps the yardstick defined and asks the same question:
                how far does a different sample of that size move the number?
                """
                full = sample.member(stack, None)[:, index]
                if variable == "cloud_ceiling":
                    finite = full[np.isfinite(full) & (full >= 0.0) & (full < 19.99)]
                else:
                    finite = full[np.isfinite(full) & (full >= 0.0) & (full <= 100.0)]
                half = finite.size // 2
                selected = finite[0:half] if part.start == 0 else finite[half : 2 * half]
                if selected.size < 2:
                    return None
                if variable == "cloud_ceiling":
                    return cloud_ceiling_ensemble_summary(
                        selected, min_valid=1, min_finite=1
                    )
                return cloud_cover_ensemble_summary(selected, min_valid=1)

            def count_of(value, name: str) -> float | None:
                if value is None:
                    return None
                if name == "VALID_COUNT":
                    return float(value.valid_member_count)
                if name == "UNLIMITED_COUNT":
                    return float(value.unlimited_member_count) if variable == "cloud_ceiling" else 0.0
                return (
                    float(value.finite_member_count)
                    if variable == "cloud_ceiling"
                    else float(value.valid_member_count)
                )

            def statistic_of(value, name: str) -> float | None:
                if value is None:
                    return None
                key = name.removeprefix("COND_").lower()
                if variable == "cloud_ceiling":
                    if value.conditional_percentiles is None:
                        return None
                    if key == "mean":
                        return value.conditional_mean
                    if key == "spread":
                        return value.conditional_spread
                    return value.conditional_percentiles.get(key)
                if key == "mean":
                    return value.mean
                if key == "spread":
                    return value.spread
                return value.percentiles.get(key)

            # The counts are exact integers over the members at the cell, so a half's count is a
            # count of the half. The conditional statistics cannot be summarised from a half at
            # all where the rule demands more members than a half has -- ``cloud_ceiling``
            # requires 21 valid members and a half has 15 -- so those rows fall back to the
            # halves-versus-full comparison ``compare`` documents.
            for offset, name in enumerate(count_names):
                aggregated = sample.stored(stored[censoring.start + offset])
                member = np.array(
                    [
                        count_of(summary(None, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                first = np.array(
                    [
                        count_of(summary(sample.first, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                second = np.array(
                    [
                        count_of(summary(sample.second, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                report.add(variable, name.lower(), *compare(aggregated, member, first, second))
            for offset, name in enumerate(statistic_names):
                aggregated = sample.stored(stored[conditional.start + offset])
                member = np.array(
                    [
                        statistic_of(summary(None, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                # The conditional statistics are summarised over the *finite* members, so the
                # halves that can be compared are the finite members' two halves -- not the
                # member set's. Splitting the member set instead would leave a half with fewer
                # than the rule's minimum and no yardstick at all, which is what the first
                # version of this row did before it was measured.
                first = np.array(
                    [
                        statistic_of(summary_finite(sample.first, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                second = np.array(
                    [
                        statistic_of(summary_finite(sample.second, index), name) or np.nan
                        for index in range(sample.rows.size)
                    ]
                )
                report.add(variable, name.lower(), *compare(aggregated, member, first, second))
        finally:
            shutil.rmtree(store, ignore_errors=True)


def _physical_phase_order():
    from domain.models.precipitation import PhysicalPhase

    return tuple(PhysicalPhase)


def _transition_order():
    from domain.models.precipitation import PrecipitationTransition

    return tuple(PrecipitationTransition)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="fewer sampled cells")
    args = parser.parse_args()
    samples = 24 if args.quick else SAMPLES

    print()
    print("=" * 110)
    print(
        f"STORED FIELDS vs MEMBER PATH   (real GEFS {DATE} {CYCLE} f{LEAD:03d}, "
        f"{N_MEMBERS} members, {GRID_LAT}x{GRID_LON}, {samples} cells in chunk "
        f"({CHUNK_ROW},{CHUNK_COL}))"
    )
    print("=" * 110)
    print()
    print("Yardstick: the ensemble's own sampling noise -- the disagreement between two disjoint")
    print("15-member halves. A ratio below 1 means the stored answer is closer to the member")
    print("answer than a different 30 members would have been, so it is below what a reader")
    print("could act on.")
    print()
    print(f"{'variable':<24} {'quantity':<30} {'error':>10}  {'noise':>10}  {'ratio':>6}")
    print("-" * 110)

    print("\n--- loading real members ---", flush=True)
    # The lead's own stacks: every variable with a container of its own, plus the amount the
    # precipitation groups read (which is a variable with a container as well, and is compared
    # here under its own distribution as well as under its phase group).
    stacks = {
        variable: load_stack(f"f{LEAD:03d}", variable)
        for variable in {*PLAIN_VARIABLES, *PRECIPITATION_INPUTS, *CLOUD_VARIABLES}
    }
    previous = {
        variable: load_stack(f"f{PREDECESSOR_LEAD:03d}", variable)
        for variable in PRECIPITATION_INPUTS
    }
    for variable, stack in stacks.items():
        print(f"  {variable:<24} {stack.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    sample = Sample(rng, samples)
    report = Report()
    run_plain(report, sample, stacks)
    run_flags(report, sample, stacks)
    run_wind(report, sample, stacks)
    run_precipitation(report, sample, stacks, previous)
    run_clouds(report, sample, stacks)

    print()
    failures = report.failures()
    if failures:
        print(f"{len(failures)} comparison(s) ABOVE the sampling noise:")
        for variable, field, value in failures:
            print(f"  {variable}.{field}: {value:.3f}x")
        print()
        print("Per the plan, a variable above the noise does not have its members deleted and does")
        print("not move to the front end.")
    else:
        print("Every compared quantity is within the ensemble's own sampling noise.")
    print()
    print("Reading the table:")
    print("  * a ratio near zero is agreement at the level of the stored fixed-point step -- the")
    print("    fields are quantised, so exact equality is neither expected nor needed;")
    print("  * the compared quantities are the ones the response reports, not intermediate ones,")
    print("    because those are what a reader sees after the members are gone;")
    print("  * a cell both sides refuse counts as agreement, not as an error;")
    print("  * the rose is compared at bucket index with both sides binned by the edges the")
    print("    container stores, which is what a reader does with them.")


if __name__ == "__main__":
    main()
