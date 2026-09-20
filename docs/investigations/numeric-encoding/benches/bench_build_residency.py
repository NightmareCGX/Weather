"""Experiment 33: what the container builder costs in memory at the production grid.

The aggregate pass reads a variable's staged members, computes every field its container holds,
and encodes them. Its residency budget is the ingestion container's, which already runs near its
limit with a staging semaphore sized for one decoded member dataset per in-flight item -- so
"how many member stacks does this hold at once, and what do the group producers add on top"
is a property the wiring cannot be chosen without.

This stages synthetic members through the real v1 encoder (so the reads, the reassembly and the
geometry are the ones production uses) and measures peak RSS around the real
``build_container_fields`` call, per variable. Synthetic members are used because the *values*
are irrelevant to residency: the shapes, the dtypes and the number of members are what decide it.
The one value-dependent branch is ``rose_fields``' quantile edges, and a normal speed
distribution is what the real field has.

Run:  .venv/Scripts/python.exe bench_build_residency.py
      .venv/Scripts/python.exe bench_build_residency.py --quick   # a quarter grid, 8 members
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

#: Repository root, five levels up from ``docs/investigations/numeric-encoding/benches``.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in (_REPO_ROOT, _REPO_ROOT / "packages" / "domain" / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

#: Variables whose container carries supplementary groups, which is where the residency is.
#: A distribution-only variable is one member stack plus the encoding's own working set.
TARGETS = (
    "temperature_2m",
    "precipitation_amount_3h",
    "cloud_ceiling",
    "wind_10m",
)


def rss_mb() -> float:
    """Resident set size in MB, or NaN when psutil is unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001 - a missing psutil makes the measurement unavailable
        return float("nan")


class Peak:
    """Peak RSS over a block, sampled from a watcher thread."""

    def __enter__(self) -> Peak:
        self.base = self.peak = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.002):
            self.peak = max(self.peak, rss_mb())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def delta(self) -> float:
        """Peak above the level the block started at."""
        return self.peak - self.base


def _plane(variable: str, member: int, lat: int, lon: int, rng: np.random.Generator) -> np.ndarray:
    """A member plane of a shape and distribution matching the variable's real field."""
    if variable == "temperature_2m":
        return rng.normal(280.0 + member * 0.1, 8.0, (lat, lon)).astype(np.float32)
    if variable == "precipitation_amount_3h":
        return np.abs(rng.normal(0.0, 2.0, (lat, lon))).astype(np.float32)
    if variable == "cloud_ceiling":
        return np.where(
            rng.random((lat, lon)) < 0.4,
            np.float32(20.0),
            rng.uniform(0.2, 12.0, (lat, lon)),
        ).astype(np.float32)
    if variable in ("wind_u_10m", "wind_v_10m"):
        return rng.normal(3.0, 4.0, (lat, lon)).astype(np.float32)
    if variable in ("crain", "csnow", "cfrzr", "cicep"):
        return (rng.random((lat, lon)) < 0.3).astype(np.float32)
    return rng.normal(0.0, 1.0, (lat, lon)).astype(np.float32)


#: Every variable a target's container reads, beyond the target itself.
EXTRA_INPUTS = {
    "wind_10m": ("wind_u_10m", "wind_v_10m"),
    "precipitation_amount_3h": ("crain", "csnow", "cfrzr", "cicep"),
}


def stage_inputs(
    store: str,
    variables: tuple[str, ...],
    *,
    members: int,
    lead: int,
    lat: int,
    lon: int,
) -> None:
    """Write every variable's members into the staging area through the real v1 encoder."""
    import xarray as xr

    from ingestion.core.aggregate_staging import staging_relative_key
    from ingestion.core.zarr_writer import encode_region_sharded_v1

    rng = np.random.default_rng(0)
    for variable in variables:
        for member in range(1, members + 1):
            plane = _plane(variable, member, lat, lon, rng)
            dataset = xr.Dataset(
                {
                    variable: (
                        ("member", "lead_time_hours", "latitude", "longitude"),
                        plane[None, None],
                    )
                }
            )
            blob = encode_region_sharded_v1(
                dataset, member=member, lead_time_hours=lead
            )[0][1]
            relative = staging_relative_key(variable, member, lead)
            full = os.path.join(store, *relative.split("/"))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as handle:
                handle.write(blob)
            del plane, blob, dataset


def measure(
    variable: str, *, members: int, lead: int, lat: int, lon: int, chunk: int
) -> tuple[float, float, int]:
    """Stage one variable's inputs, build its container, and report the cost.

    Returns:
        ``(seconds, peak MB above the pre-build level, field count)``.
    """
    from ingestion.core.aggregate_fields import MemberReader, build_container_fields

    store = tempfile.mkdtemp(prefix="agg_residency_")
    try:
        inputs = (variable, *EXTRA_INPUTS.get(variable, ()))
        stage_inputs(store, inputs, members=members, lead=lead, lat=lat, lon=lon)
        reader = MemberReader(
            store, grid_lat=lat, grid_lon=lon, chunk_lat=chunk, chunk_lon=chunk
        )
        gc.collect()
        with Peak() as peak:
            started = time.perf_counter()
            fields, _count = build_container_fields(
                reader, variable, lead, expected_members=members
            )
            elapsed = time.perf_counter() - started
        return elapsed, peak.delta, len(fields)
    finally:
        shutil.rmtree(store, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a quarter-size grid and 8 members")
    args = parser.parse_args()

    members = 8 if args.quick else 30
    lat, lon = (180, 360) if args.quick else (721, 1440)
    chunk = 100

    print()
    print("=" * 96)
    print(
        f"CONTAINER BUILDER RESIDENCY   ({members} members, {lat}x{lon}, "
        f"{chunk}x{chunk} chunks)"
    )
    print("=" * 96)
    print()
    print(f"  one member stack            {lat * lon * 4 / 1e6:8.1f} MB  (float32, the grid)")
    print(f"  wind's two stacks           {2 * lat * lon * 4 / 1e6:8.1f} MB")
    print()
    print(f"{'variable':<26} {'fields':>7} {'seconds':>9} {'peak above pre-build':>22}")
    print("-" * 96)
    for variable in TARGETS:
        seconds, delta, fields = measure(
            variable, members=members, lead=6, lat=lat, lon=lon, chunk=chunk
        )
        print(f"{variable:<26} {fields:>7} {seconds:>9.1f} {delta:>19.0f} MB")
    print()
    print("Reading the table:")
    print("  * a distribution-only variable holds one member stack; wind holds two, because a")
    print("    rose needs both components of the same members;")
    print("  * the peak includes the group producers' working set, which is why a 78-field")
    print("    wind container is not 78/35ths of temperature's;")
    print("  * precipitation's phase and transition groups stream one member at a time, so they")
    print("    add a bounded amount rather than another stack.")


if __name__ == "__main__":
    main()
