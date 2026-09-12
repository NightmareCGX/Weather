"""Benchmark script for Marker PUT and GET concurrency scaling (Tasks 10 & 11)."""

from __future__ import annotations

import concurrent.futures
import gc
import threading
import time
from typing import Any

from ingestion.core.coordinator import _read_marker_payloads_bounded
from ingestion.core.marker_put_scheduler import put_markers_rolling
from ingestion.core.markers import list_region_marker_keys, write_region_marker


def get_rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def generate_gefs_targets() -> list[tuple[str, int, int | None, bool]]:
    """Generate canonical 2,511 GEFS region targets (30 members * 81 leads + 81 mean leads)."""
    targets = []
    # 81 canonical leads: 0 to 240 step 3
    leads = list(range(0, 243, 3))
    # 30 ensemble members
    for lead in leads:
        # Mean
        targets.append((f"gefs_mean_L{lead:04d}", lead, None, True))
        for mem in range(1, 31):
            targets.append((f"gefs_mem{mem:02d}_L{lead:04d}", lead, mem, False))
    return targets


def run_marker_put_benchmark(store_path: str, concurrency_levels: list[int]) -> list[dict[str, Any]]:
    targets = generate_gefs_targets()
    print(f"--- MARKER PUT BENCHMARK: {len(targets)} targets ---")
    results = []

    for conc in concurrency_levels:
        gc.collect()
        latencies: list[float] = []
        lat_lock = threading.Lock()

        def put_one(region_id: str) -> None:
            # Locate target metadata
            lead = int(region_id.split("_L")[1])
            is_mean = "_mean_" in region_id
            mem = None if is_mean else int(region_id.split("_mem")[1].split("_L")[0])

            t0 = time.monotonic()
            write_region_marker(
                store_path,
                lead_time_hours=lead,
                member=mem,
                is_mean=is_mean,
                payload={
                    "protocol_version": 1,
                    "state": "updating",
                    "generation": f"bench_{conc}",
                    "logical_region": {"lead_time_hours": lead},
                },
            )
            elapsed = time.monotonic() - t0
            with lat_lock:
                latencies.append(elapsed)

        region_ids = [t[0] for t in targets]
        cancel_ev = threading.Event()

        t_start = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            res = put_markers_rolling(
                region_ids,
                put_one,
                concurrency=conc,
                cancel_event=cancel_ev,
                timeout_seconds=30.0,
                executor=executor,
            )

        t_end = time.monotonic()
        total_wall = t_end - t_start
        rss_after = get_rss_mb()

        lat_sorted = sorted(latencies)
        p50 = lat_sorted[int(len(lat_sorted) * 0.50)] * 1000.0 if lat_sorted else 0.0
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)] * 1000.0 if lat_sorted else 0.0
        p99 = lat_sorted[int(len(lat_sorted) * 0.99)] * 1000.0 if lat_sorted else 0.0
        throughput = len(res.successes) / total_wall if total_wall > 0 else 0.0

        item = {
            "concurrency": conc,
            "total_markers": len(targets),
            "wall_time_s": total_wall,
            "throughput_m_s": throughput,
            "successes": len(res.successes),
            "failures": len(res.failures),
            "cancelled": len(res.cancelled),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "rss_mb": rss_after,
        }
        results.append(item)
        print(
            f"Concurrency {conc:2d}: {total_wall:6.3f}s | {throughput:6.1f} PUTs/s | "
            f"p50={p50:5.2f}ms p95={p95:5.2f}ms p99={p99:5.2f}ms | "
            f"Success={len(res.successes)} Fail={len(res.failures)} RSS={rss_after:.1f}MB"
        )
    return results


def run_marker_get_benchmark(store_path: str, concurrency_levels: list[int]) -> list[dict[str, Any]]:
    print("\n--- MARKER GET BENCHMARK (2,511 markers) ---")
    keys = list_region_marker_keys(store_path)
    print(f"Listed {len(keys)} marker keys from {store_path}")
    results = []

    for conc in concurrency_levels:
        gc.collect()
        t_start = time.monotonic()

        payloads = _read_marker_payloads_bounded(store_path, keys, max_concurrency=conc)

        t_end = time.monotonic()
        total_wall = t_end - t_start
        rss_after = get_rss_mb()
        throughput = len(payloads) / total_wall if total_wall > 0 else 0.0

        item = {
            "concurrency": conc,
            "total_markers": len(keys),
            "wall_time_s": total_wall,
            "throughput_reads_s": throughput,
            "successes": len(payloads),
            "rss_mb": rss_after,
        }
        results.append(item)
        print(
            f"Concurrency {conc:3d}: {total_wall:6.3f}s | {throughput:6.1f} GETs/s | "
            f"Reads={len(payloads)} RSS={rss_after:.1f}MB"
        )
    return results


if __name__ == "__main__":
    store = "s3://weather-data/benchmarks/marker_scaling.zarr"
    put_res = run_marker_put_benchmark(store, [8, 16, 32, 64])
    get_res = run_marker_get_benchmark(store, [16, 32, 64, 128])
