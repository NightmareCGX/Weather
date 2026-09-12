"""Robust benchmark script for NOAA download concurrency scaling across repeated trials (Tasks 4-6)."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
import random
import shutil
import time
from typing import Any

from ingestion.providers.noaa.connector import NOAAConnector


def get_60_gefs_targets() -> list[tuple[int, int]]:
    """Return 60 representative GEFS targets: 30 members at lead 6 + 30 members at lead 12."""
    targets = []
    for lead in (6, 12):
        for mem in range(1, 31):
            targets.append((lead, mem))
    return targets


async def run_single_benchmark_trial(
    concurrency: int,
    targets: list[tuple[int, int]],
    trial_idx: int,
    tmp_base: Path,
) -> dict[str, Any]:
    cycle_date = date(2026, 9, 2)
    cycle_hour = 18
    model = "gefs"

    run_dir = tmp_base / f"trial_c{concurrency}_{trial_idx}_{random.randint(1000, 9999)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    latencies: list[float] = []
    bytes_downloaded = 0
    errors = 0
    http_429 = 0
    http_5xx = 0

    async with NOAAConnector() as connector:
        async def download_target(lead: int, mem: int) -> None:
            nonlocal bytes_downloaded, errors, http_429, http_5xx
            dest = run_dir / f"gep{mem:02d}.f{lead:03d}.grib2"
            t0 = time.monotonic()
            async with sem:
                try:
                    await connector.download(
                        model=model,
                        cycle_date=cycle_date,
                        cycle_hour=cycle_hour,
                        lead_time_hours=lead,
                        destination=dest,
                        member=mem,
                    )
                    dur = time.monotonic() - t0
                    latencies.append(dur)
                    if dest.exists():
                        bytes_downloaded += dest.stat().st_size
                except Exception as exc:
                    errors += 1
                    err_str = str(exc)
                    if "429" in err_str:
                        http_429 += 1
                    elif "50" in err_str or "503" in err_str:
                        http_5xx += 1

        t_start = time.monotonic()
        tasks = [asyncio.create_task(download_target(lead, mem)) for lead, mem in targets]
        await asyncio.gather(*tasks)
        total_wall = time.monotonic() - t_start

    mb_downloaded = bytes_downloaded / (1024 * 1024)
    throughput_targets_s = len(targets) / total_wall if total_wall > 0 else 0.0
    throughput_mb_s = mb_downloaded / total_wall if total_wall > 0 else 0.0

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(len(lat_sorted) * 0.50)] if lat_sorted else 0.0
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0

    shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "concurrency": concurrency,
        "trial": trial_idx,
        "targets": len(targets),
        "wall_time_s": total_wall,
        "targets_per_s": throughput_targets_s,
        "total_mb": mb_downloaded,
        "mb_per_s": throughput_mb_s,
        "p50_s": p50,
        "p95_s": p95,
        "errors": errors,
        "http_429": http_429,
        "http_5xx": http_5xx,
    }


async def run_robust_benchmark():
    targets = get_60_gefs_targets()
    tmp_base = Path("downloads/bench_robust")
    tmp_base.mkdir(parents=True, exist_ok=True)

    concurrency_levels = [4, 8, 12, 16, 24]
    repetitions = 3

    trials = [(conc, rep) for conc in concurrency_levels for rep in range(1, repetitions + 1)]
    # Randomize trial order to mitigate network ordering bias
    random.seed(42)
    random.shuffle(trials)

    print(f"Starting Robust NOAA Download Benchmark: {len(targets)} targets per trial across {len(trials)} trials...")
    print("Target set: 30 GEFS members @ lead 6h + 30 GEFS members @ lead 12h (Total: 60 targets)")

    raw_results = []
    for conc, rep in trials:
        print(f"Running Concurrency {conc:2d} (Trial {rep}/{repetitions})...", end="", flush=True)
        res = await run_single_benchmark_trial(conc, targets, rep, tmp_base)
        raw_results.append(res)
        print(f" Done in {res['wall_time_s']:5.2f}s | {res['targets_per_s']:4.2f} targets/s | {res['mb_per_s']:5.2f} MB/s | p50={res['p50_s']:4.2f}s | err={res['errors']}")

    shutil.rmtree(tmp_base, ignore_errors=True)

    # Summarize results
    print("\n" + "=" * 90)
    print("                    ROBUST DOWNLOAD BENCHMARK SUMMARY (3 REPETITIONS)")
    print("=" * 90)
    print(f"{'Concurrency':<12} {'Wall Time (s) [min/med/max]':<30} {'Targets/s (median)':<20} {'MB/s (median)':<16} {'Errors':<8}")
    print("-" * 90)

    for conc in concurrency_levels:
        conc_trials = [r for r in raw_results if r["concurrency"] == conc]
        wall_times = sorted([r["wall_time_s"] for r in conc_trials])
        tp_targets = sorted([r["targets_per_s"] for r in conc_trials])
        tp_mb = sorted([r["mb_per_s"] for r in conc_trials])
        total_errors = sum(r["errors"] for r in conc_trials)

        med_wall = wall_times[len(wall_times) // 2]
        min_wall = min(wall_times)
        max_wall = max(wall_times)
        med_tp_targets = tp_targets[len(tp_targets) // 2]
        med_tp_mb = tp_mb[len(tp_mb) // 2]

        print(
            f"{conc:<12} {min_wall:5.2f} / {med_wall:5.2f} / {max_wall:5.2f} s              "
            f"{med_tp_targets:5.2f} targets/s       "
            f"{med_tp_mb:5.2f} MB/s         "
            f"{total_errors:<8}"
        )


if __name__ == "__main__":
    asyncio.run(run_robust_benchmark())
