"""Benchmark script for NOAA download concurrency scaling (Task 12)."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
import shutil
import time
from typing import Any

from ingestion.providers.noaa.connector import NOAAConnector


async def benchmark_download_concurrency(concurrency_levels: list[int], members_count: int = 12) -> list[dict[str, Any]]:
    cycle_date = date(2026, 9, 2)
    cycle_hour = 18
    lead = 6
    model = "gefs"
    tmp_base = Path("downloads/bench_dl")
    tmp_base.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"--- NOAA DOWNLOAD CONCURRENCY BENCHMARK ({members_count} members, lead {lead}) ---")

    for conc in concurrency_levels:
        run_dir = tmp_base / f"conc_{conc}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(conc)

        latencies: list[float] = []
        bytes_downloaded = 0
        errors = 0

        async with NOAAConnector() as connector:
            async def download_one(mem: int) -> None:
                nonlocal bytes_downloaded, errors
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
                        print(f"  Error on member {mem}: {exc}")

            t_start = time.monotonic()
            tasks = [asyncio.create_task(download_one(m)) for m in range(1, members_count + 1)]
            await asyncio.gather(*tasks)
            total_wall = time.monotonic() - t_start

        mb_downloaded = bytes_downloaded / (1024 * 1024)
        throughput_targets_s = members_count / total_wall if total_wall > 0 else 0.0
        throughput_mb_s = mb_downloaded / total_wall if total_wall > 0 else 0.0

        lat_sorted = sorted(latencies)
        p50 = lat_sorted[int(len(lat_sorted) * 0.50)] if lat_sorted else 0.0
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0

        item = {
            "concurrency": conc,
            "targets": members_count,
            "wall_time_s": total_wall,
            "targets_per_s": throughput_targets_s,
            "total_mb": mb_downloaded,
            "mb_per_s": throughput_mb_s,
            "p50_s": p50,
            "p95_s": p95,
            "errors": errors,
        }
        results.append(item)
        print(
            f"Concurrency {conc:2d}: {total_wall:6.2f}s | {throughput_targets_s:4.2f} files/s | "
            f"{throughput_mb_s:5.2f} MB/s | p50={p50:4.2f}s p95={p95:4.2f}s | "
            f"Errors={errors} MB={mb_downloaded:.1f}"
        )

    # Clean up benchmark staging
    shutil.rmtree(tmp_base, ignore_errors=True)
    return results


if __name__ == "__main__":
    asyncio.run(benchmark_download_concurrency([4, 8, 12, 16, 24], members_count=12))
