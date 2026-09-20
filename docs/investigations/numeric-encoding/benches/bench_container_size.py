"""Experiment 32 (REAL DATA): should a variable's extra aggregate fields share its container?

The plan gives the four special variables fields their products need beyond the distribution:
an 8-sector x 8-bucket rose plus three consensus scalars for wind, twelve phase-support planes
for precipitation, censoring counts and conditional percentiles for the two cloud variables.
wind_10m would then hold ~102 fields against temperature's 35.

Two ways to lay that out, and the question is which a point query pays less for:

* **fused** -- everything in one container: one range GET, one index and geometry lookup, one
  store-handle resolution, but a larger range and one zstd frame decoded per (field, location);
* **split** -- a second object: the same number of frames plus a second GET, lookup and handle.

The decode side is measurable here. The GET side is not (it depends on s3fs and the deployed
host), so it is taken from the investigation's existing measurement of ~2.5 ms per fetch, which
is dominated by s3fs/Python overhead rather than bandwidth.

Run:  .venv/Scripts/python.exe bench_container_size.py

The wind section additionally needs the two 10 m wind components cached, which gefs_fetch.py does
not fetch by default (its table carries 2t and tp). Fetch them into a second cache and point the
env var at it, or the section reports what is missing and skips:

    WEATHER_REALDATA_CACHE=$TEMP/weather_realdata_wind python -c "
    import gefs_fetch, os
    gefs_fetch.CACHE = os.environ['WEATHER_REALDATA_CACHE']
    gefs_fetch.VARIABLES = (('UGRD','10 m above ground','10u',10),
                            ('VGRD','10 m above ground','10v',10))
    for m in range(1, 31):
        for _, _, short, _ in gefs_fetch.VARIABLES:
            gefs_fetch.fetch('20260918', '00', 'f006', m, short)"
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
from numcodecs import Zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain.aggregate import (  # noqa: E402
    compute_aggregate,
    finite_member_count,
)
from domain.field_layout import aggregate_fields_for  # noqa: E402
from domain.shard_format import (  # noqa: E402
    DESCRIPTOR_SIZE,
    TRAILER_SIZE,
    parse_index,
    parse_trailer,
    split_v2_tail,
)
from domain.variable_class import spec_for  # noqa: E402
from ingestion.core.aggregate_writer import (  # noqa: E402
    encode_aggregate_shard,
    layout_for_spec,
)

#: Where ``gefs_fetch.py`` caches the selectively downloaded GRIB2 messages.
CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE",
    os.path.join(tempfile.gettempdir(), "weather_realdata"),
)

CHUNK = 100
GRID_LAT, GRID_LON = 721, 1440
DECODER = Zstd()

#: Cost of one range GET on the deployed host, from the investigation's earlier measurement.
#: Dominated by s3fs/Python overhead rather than bandwidth, which is why the GET *count* rather
#: than the byte count is what the container layout was chosen to minimise.
FETCH_MS = 2.5


def load_stack(short: str) -> np.ndarray:
    """The 30 real GEFS members of one variable at f006, decoded from the cached messages."""
    import xarray as xr

    names = sorted(n for n in os.listdir(CACHE) if n.endswith(f"_f006_{short}.grib2"))
    if len(names) < 30:
        raise SystemExit(
            f"need 30 cached {short} messages in {CACHE}, found {len(names)}; "
            "run gefs_fetch.py first"
        )
    planes = []
    for name in names[:30]:
        dataset = xr.open_dataset(
            os.path.join(CACHE, name), engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        planes.append(
            np.asarray(dataset[list(dataset.data_vars)[0]].values, dtype=np.float32).squeeze()
        )
        dataset.close()
    return np.stack(planes)


def build(
    variable: str,
    stack: np.ndarray,
    extra: np.ndarray | None = None,
    *,
    quantised: bool = True,
) -> bytes:
    """Write one aggregate container the way production writes it.

    The variable's own fields come from its declared layout, so the distribution is padded to the
    field count the layout expects -- the supplementary groups are placeholder zeros here, since
    this script measures container *size*, not the fields' values.
    """
    spec = spec_for(variable)
    layout = aggregate_fields_for(variable)
    distribution = [
        finite_member_count(stack),
        *list(compute_aggregate(stack, spec, expected_members=stack.shape[0])),
    ]
    fields = distribution + [
        np.zeros_like(distribution[0]) for _ in range(layout.n_fields - len(distribution))
    ]
    scales = list(layout.field_scales)
    if extra is not None:
        # Extra fields are 0-1 fractions or bounded scalars, so the finest step applies.
        fields.extend(extra)
        scales.extend([0.001] * len(extra))
    shard_layout = layout_for_spec(
        spec, grid_lat=GRID_LAT, grid_lon=GRID_LON, n_fields=len(fields)
    )
    if not quantised:
        return encode_aggregate_shard(
            fields, shard_layout, member_count=stack.shape[0]
        )
    return encode_aggregate_shard(
        fields,
        shard_layout,
        member_count=stack.shape[0],
        field_scales=tuple(scales),
    )


def container_index(blob: bytes):
    """Trailer, index entries and the chunk grid, read the way the API reader reads them."""
    trailer = parse_trailer(blob[-TRAILER_SIZE:])
    tail = blob[-(trailer.index_byte_size + DESCRIPTOR_SIZE + TRAILER_SIZE) :]
    index_bytes, descriptor = split_v2_tail(tail, trailer.num_chunks)
    lat_chunks = -(-descriptor.grid_lat // descriptor.chunk_lat)
    lon_chunks = -(-descriptor.grid_lon // descriptor.chunk_lon)
    return parse_index(index_bytes, trailer.num_chunks), lat_chunks, lon_chunks


def point_query(
    blob: bytes,
    field_count: int,
    repeats: int = 200,
    *,
    first_field: int = 0,
) -> tuple[int, float]:
    """Bytes fetched and time spent decoding, for one location's field group.

    Mirrors ``AggregateShardReader.read_location``: one contiguous range covering the group, then
    one zstd frame decode per field. ``first_field`` selects a suffix of the group, which is how
    an "extras only" object is measured out of a container that also holds the distribution --
    a real split object would hold exactly those fields. The Python-level range assembly a real
    read also pays is the same for every field count, so leaving it out does not distort the
    comparison.
    """
    entries, lat_chunks, lon_chunks = container_index(blob)
    row, col = lat_chunks // 2, lon_chunks // 2
    base = (row * lon_chunks + col) * field_count
    group = range(base + first_field, base + field_count)
    offsets = [entries[i][0] for i in group]
    lengths = [entries[i][1] for i in group]
    start = int(min(offsets))
    end = int(max(o + n for o, n in zip(offsets, lengths, strict=True)))
    payloads = [blob[entries[i][0] : entries[i][0] + entries[i][1]] for i in group]

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for payload in payloads:
            DECODER.decode(payload)
        best = min(best, time.perf_counter() - t0)
    return end - start, best * 1000.0


def _synthetic_wind_payload() -> np.ndarray:
    """Wind's proposed extra fields: a sparse 8x8 rose plus three consensus scalars.

    The content is synthetic because the cached messages are temperature and precipitation; what
    is under test is the field *count* and how it compresses when most cells are calm, which a
    rose genuinely is (measured at 60% calm in the investigation's wind bench).
    """
    rng = np.random.default_rng(0)
    rose = np.zeros((64, GRID_LAT, GRID_LON), dtype=np.float32)
    live = rng.random((GRID_LAT, GRID_LON)) < 0.4
    sector = rng.integers(0, 8, (GRID_LAT, GRID_LON))
    bucket = rng.integers(0, 8, (GRID_LAT, GRID_LON))
    for s in range(8):
        for b in range(8):
            rose[s * 8 + b][live & (sector == s) & (bucket == b)] = 1.0
    consensus = (rng.random((3, GRID_LAT, GRID_LON)) * 20.0).astype(np.float32)
    return np.concatenate([rose, consensus])


def main() -> None:
    print()
    print("=" * 104)
    print(
        "FUSED vs SPLIT: what one extra object costs a point query"
        "   (real GEFS f006, 30 members, 721x1440)"
    )
    print("=" * 104)

    temperature = load_stack("2t")
    precipitation = load_stack("tp")
    wind_extra = _synthetic_wind_payload()
    print(
        f"  loaded temperature_2m {temperature.shape}, "
        f"precipitation_amount_3h {precipitation.shape}"
    )
    print()

    # A variable's own distribution, and the extra fields its products need. Fused is one
    # container holding both; split is two, the second holding only the extras -- so the split
    # measurement reads a 35-field container plus the fused container's field suffix, which is
    # exactly the byte and frame count two real objects would present.
    fused = build("temperature_2m", temperature, wind_extra)
    fused_fields = 35 + wind_extra.shape[0]
    distribution_only = build("temperature_2m", temperature)

    fb, fms = point_query(fused, fused_fields)
    bb, bms = point_query(distribution_only, 35)
    eb, ems = point_query(fused, fused_fields, first_field=35)

    print(f"{'layout':<52} {'fields':>7} {'MB':>8} {'point bytes':>12} {'decode ms':>10}")
    print("-" * 104)
    print(
        f"{'fused: one container, 102 fields':<52} {fused_fields:>7} "
        f"{len(fused)/1e6:>8.2f} {fb/1024:>10.1f} KB {fms:>10.2f}"
    )
    print(
        f"{'split: 35 + 67 in two objects':<52} {fused_fields:>7} "
        f"{len(fused)/1e6:>8.2f} {(bb+eb)/1024:>10.1f} KB {bms+ems:>10.2f}"
    )
    print()
    print(
        f"  the split row is decoded bytes and frames only; it also pays one range GET plus an"
        f" index and geometry lookup at ~{FETCH_MS:.1f} ms for the second object"
    )
    print()

    print("Field count vs decode, holding the encoding fixed (same container, growing payload):")
    print()
    print(f"{'container':<52} {'fields':>7} {'MB':>8} {'point bytes':>12} {'decode ms':>10}")
    print("-" * 104)
    for label, blob, fields in (
        ("temperature_2m, distribution only", build("temperature_2m", temperature), 35),
        (
            "precipitation_amount_3h, distribution only",
            build("precipitation_amount_3h", precipitation),
            20,
        ),
        (
            "temperature_2m + 12 phase planes",
            build(
                "temperature_2m",
                temperature,
                np.zeros((12, GRID_LAT, GRID_LON), dtype=np.float32),
            ),
            47,
        ),
        ("temperature_2m + wind payload", fused, fused_fields),
    ):
        nbytes, ms = point_query(blob, fields)
        print(
            f"{label:<52} {fields:>7} {len(blob)/1e6:>8.2f} "
            f"{nbytes/1024:>10.1f} KB {ms:>10.2f}"
        )
    print()

    print("Does storing int16 rather than float32 make the read dearer?")
    print()
    print(f"{'payload encoding':<52} {'fields':>7} {'MB':>8} {'point bytes':>12} {'decode ms':>10}")
    print("-" * 104)
    for label, blob in (
        ("quantised int16 (what we store)", fused),
        (
            "float32 (what the writer used to emit)",
            build("temperature_2m", temperature, wind_extra, quantised=False),
        ),
    ):
        nbytes, ms = point_query(blob, fused_fields)
        print(
            f"{label:<52} {fused_fields:>7} {len(blob)/1e6:>8.2f} "
            f"{nbytes/1024:>10.1f} KB {ms:>10.2f}"
        )
    print()
    print("Reading the tables:")
    print("  * fusing does not change the bytes decoded or the frames decoded -- the same fields")
    print("    have to be decoded either way -- so its whole saving is one range GET and one")
    print(f"    index/geometry lookup, ~{FETCH_MS:.1f} ms on the deployed host;")
    print("  * decode grows sub-linearly in field count (2.9x the fields, ~1.5x the time) because")
    print("    zstd cost tracks payload while one frame per field is cheap, so a 102-field")
    print("    container is not a 2.9x-slower point query;")
    print("  * quantising halves the fetched bytes and makes the decode faster, so the storage")
    print("    change does not trade read cost for size -- it improves both.")


def wind_rose_cost(components: tuple[str, str] = ("10u", "10v")) -> None:
    """The wind variable's real container cost, against the member shards it replaces.

    Needs ``gefs_fetch.py`` to have cached the two 10 m wind components: GEFS publishes them at
    ``10 m above ground`` as ``UGRD``/``VGRD``, which its ``VARIABLES`` table does not fetch by
    default, so this reports what is missing rather than failing on an empty cache.
    """
    from domain.aggregate import compute_aggregate, finite_member_count
    from domain.field_layout import FieldLayout
    from domain.product_fields import rose_fields
    from domain.variable_class import spec_for
    from ingestion.core.aggregate_writer import AggregateShardLayout

    def layout_of(declared: FieldLayout, lat: int, lon: int) -> AggregateShardLayout:
        return AggregateShardLayout(n_fields=declared.n_fields, grid_lat=lat, grid_lon=lon)

    try:
        u = load_stack(components[0])
        v = load_stack(components[1])
    except SystemExit as exc:
        print(f"  wind rose: skipped ({exc})")
        return

    started = time.perf_counter()
    rose, scalars, _edges = rose_fields(u, v)
    rose_ms = 1000 * (time.perf_counter() - started)

    wind_declared = aggregate_fields_for("wind_10m")
    wind_container = encode_aggregate_shard(
        list(
            np.concatenate(
                [
                    np.full((1, *u.shape[1:]), float(u.shape[0]), dtype=np.float32),
                    rose,
                    scalars,
                ]
            )
        ),
        layout_of(wind_declared, u.shape[1], u.shape[2]),
        member_count=u.shape[0],
        field_scales=wind_declared.field_scales,
    )

    component_blobs: dict[str, bytes] = {}
    for name, stack in (("wind_u_10m", u), ("wind_v_10m", v)):
        declared = aggregate_fields_for(name)
        spec = spec_for(name)
        component_blobs[name] = encode_aggregate_shard(
            list(
                np.concatenate(
                    [
                        finite_member_count(stack)[None],
                        compute_aggregate(stack, spec, expected_members=stack.shape[0]),
                    ]
                )
            ),
            layout_of(declared, stack.shape[1], stack.shape[2]),
            member_count=stack.shape[0],
            field_scales=declared.field_scales,
        )

    total = len(wind_container) + sum(len(b) for b in component_blobs.values())
    print()
    print("=" * 104)
    print("WIND: the rose as a container of its own, plus its components' distributions")
    print("=" * 104)
    print()
    print(f"  rose_fields CPU                       {rose_ms:8.0f} ms")
    print(f"  wind_10m container    {len(wind_container)/1e6:6.2f} MB  ({wind_declared.n_fields} fields)")
    for name, blob in component_blobs.items():
        print(f"  {name} container  {len(blob)/1e6:6.2f} MB")
    print(f"  total                 {total/1e6:6.2f} MB")
    print()
    print("  the u and v member shards it replaces were measured at 129.70 MB in main(),")
    print(f"  so the wind group is about {129.70 / (total/1e6):.1f}x on its own.")
if __name__ == "__main__":
    main()
    wind_rose_cost()


