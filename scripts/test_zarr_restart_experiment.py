"""Zarr Init Post-Restart Experiment (Tasks 9-11)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from ingestion.core.config import settings
from ingestion.core.s3 import get_s3_fs
from scripts.diagnose_zarr_init import instrumented_prepare_run_store, make_dummy_gfs_dataset


def run_experiment():
    leads = tuple(range(0, 243, 3))
    ds = make_dummy_gfs_dataset()

    print("================================================================================")
    print("               TASK 9: ZARR INIT POST-RESTART EXPERIMENT")
    print("================================================================================")

    # 1. Warm Phase
    print("\n--- PHASE 1: WARM INFRASTRUCTURE (Pre-Restart) ---")
    store_warm_a = f"s3://weather-data/benchmarks/warm_a_{uuid.uuid4().hex[:8]}.zarr"
    res_warm_a = instrumented_prepare_run_store(ds, store_warm_a, expected_leads=leads)
    print(f"Warm Run A: {res_warm_a['timings']['total_prepare_run_store_s']:.3f}s | "
          f"coords={res_warm_a['timings']['coords_to_zarr_s']:.3f}s | "
          f"vars={res_warm_a['timings']['step4_total_vars_s']:.3f}s | "
          f"consolidate={res_warm_a['timings']['consolidate_metadata_s']:.3f}s | "
          f"S3 calls: {sum(res_warm_a['s3_counts'].values())}")

    store_warm_b = f"s3://weather-data/benchmarks/warm_b_{uuid.uuid4().hex[:8]}.zarr"
    res_warm_b = instrumented_prepare_run_store(ds, store_warm_b, expected_leads=leads)
    print(f"Warm Run B: {res_warm_b['timings']['total_prepare_run_store_s']:.3f}s | "
          f"coords={res_warm_b['timings']['coords_to_zarr_s']:.3f}s | "
          f"vars={res_warm_b['timings']['step4_total_vars_s']:.3f}s | "
          f"consolidate={res_warm_b['timings']['consolidate_metadata_s']:.3f}s | "
          f"S3 calls: {sum(res_warm_b['s3_counts'].values())}")

    # 2. Restart MinIO container
    print("\n--- PHASE 2: RESTARTING MINIO CONTAINER ---")
    t0 = time.monotonic()
    subprocess.run(["docker", "restart", "weather_minio"], check=True, capture_output=True)
    restart_time = time.monotonic() - t0
    print(f"Docker restart issued in {restart_time:.2f}s; waiting for healthcheck readiness...")

    # Wait for healthcheck readiness
    import urllib.request
    healthy = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://localhost:9000/minio/health/live", timeout=1.0) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            time.sleep(0.5)

    if not healthy:
        raise RuntimeError("MinIO failed to become healthy within 30s")
    print("MinIO healthcheck reports READY.")

    # 3. First S3 Request Latency after restart
    t_first = time.monotonic()
    fs = get_s3_fs(settings, force_new=True)
    fs.info("weather-data")
    first_req_ms = (time.monotonic() - t_first) * 1000.0
    print(f"First S3 request latency immediately post-restart: {first_req_ms:.2f} ms")

    # 4. Immediate Post-Restart Fresh-Store Init
    print("\n--- PHASE 3: IMMEDIATE POST-RESTART FRESH STORE INIT (Run Cold 1) ---")
    store_cold_1 = f"s3://weather-data/benchmarks/post_restart_1_{uuid.uuid4().hex[:8]}.zarr"
    res_cold_1 = instrumented_prepare_run_store(ds, store_cold_1, expected_leads=leads)
    print(f"Post-Restart Run 1: {res_cold_1['timings']['total_prepare_run_store_s']:.3f}s | "
          f"coords={res_cold_1['timings']['coords_to_zarr_s']:.3f}s | "
          f"vars={res_cold_1['timings']['step4_total_vars_s']:.3f}s | "
          f"consolidate={res_cold_1['timings']['consolidate_metadata_s']:.3f}s | "
          f"S3 calls: {sum(res_cold_1['s3_counts'].values())}")
    print("Post-Restart Run 1 S3 Calls Breakdown:", res_cold_1["s3_counts"])

    # 5. Second Immediate Fresh Init (showing recovery / warm-up)
    print("\n--- PHASE 4: SUBSEQUENT FRESH STORE INIT (Run Cold 2) ---")
    store_cold_2 = f"s3://weather-data/benchmarks/post_restart_2_{uuid.uuid4().hex[:8]}.zarr"
    res_cold_2 = instrumented_prepare_run_store(ds, store_cold_2, expected_leads=leads)
    print(f"Post-Restart Run 2: {res_cold_2['timings']['total_prepare_run_store_s']:.3f}s | "
          f"coords={res_cold_2['timings']['coords_to_zarr_s']:.3f}s | "
          f"vars={res_cold_2['timings']['step4_total_vars_s']:.3f}s | "
          f"consolidate={res_cold_2['timings']['consolidate_metadata_s']:.3f}s | "
          f"S3 calls: {sum(res_cold_2['s3_counts'].values())}")


if __name__ == "__main__":
    run_experiment()
