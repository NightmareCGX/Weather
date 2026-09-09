"""Explicit offline utility to batch-hydrate cities.elevation_m from Open-Meteo.

Usage:
    poetry run python scripts/hydrate_city_elevations.py [--batch-size 100] [--limit 1000] [--dry-run]

This script is an explicit, offline maintenance tool only:
- It uses Open-Meteo's batch elevation endpoint (up to 100 coordinates per request);
- It updates only rows where elevation_m IS NULL;
- It is completely decoupled from application startup, migrations, and runtime queries;
- It respects provider rate limits by batching and pacing requests.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request

from sqlalchemy import func, select
from sqlalchemy.orm import Session

# Add services/api/src to path if run standalone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api" / "src"))

from api.core.config import settings
from api.core.database import SessionLocal
from api.models.entities import City

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def query_open_meteo_batch(
    coords: list[tuple[float, float]],
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[float | None]:
    """Query Open-Meteo elevation endpoint for up to 100 coordinates."""
    if not coords:
        return []
    lats = ",".join(f"{c[0]:.6f}" for c in coords)
    lons = ",".join(f"{c[1]:.6f}" for c in coords)
    url = f"{base_url}?latitude={lats}&longitude={lons}"
    if api_key:
        url += f"&apikey={api_key}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WeatherPlatform-CityHydration/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Open-Meteo returned status {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
        elevations = data.get("elevation", [])
        return [float(e) if e is not None else None for e in elevations]


def hydrate_cities(
    db: Session,
    batch_size: int = 100,
    limit: int | None = None,
    dry_run: bool = False,
    sleep_seconds: float = 0.5,
) -> int:
    """Hydrate NULL elevation_m records in the cities table."""
    base_url = str(settings.ELEVATION_BASE_URL)
    api_key = str(settings.ELEVATION_API_KEY) if settings.ELEVATION_API_KEY else None

    stmt = (
        select(City, func.ST_X(City.geom), func.ST_Y(City.geom))
        .where(City.elevation_m.is_(None))
        .order_by(City.id.asc())
    )
    if limit:
        stmt = stmt.limit(limit)

    rows = db.execute(stmt).all()
    total = len(rows)
    logger.info("Found %d cities with missing elevation_m", total)
    if total == 0:
        return 0

    updated_count = 0
    batch_size = min(max(1, batch_size), 100)  # Open-Meteo cap is 100

    for i in range(0, total, batch_size):
        chunk = rows[i : i + batch_size]
        coords: list[tuple[float, float]] = []
        for r in chunk:
            lat = float(r[2])
            lon = float(r[1])
            coords.append((lat, lon))

        logger.info("Fetching elevations for batch %d..%d of %d", i, i + len(chunk), total)
        try:
            elevations = query_open_meteo_batch(coords, base_url, api_key=api_key)
        except Exception as exc:
            logger.error("Failed to query batch: %s", exc)
            continue

        for (city_row, _, _), elev in zip(chunk, elevations):
            if elev is not None:
                if not dry_run:
                    city_row.elevation_m = elev
                updated_count += 1

        if not dry_run:
            db.commit()

        if sleep_seconds > 0 and (i + batch_size) < total:
            time.sleep(sleep_seconds)

    logger.info("Successfully hydrated %d / %d cities (dry_run=%s)", updated_count, total, dry_run)
    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-hydrate city elevations from Open-Meteo")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size (max 100)")
    parser.add_argument("--limit", type=int, default=None, help="Max cities to process")
    parser.add_argument("--dry-run", action="store_true", help="Fetch elevations without writing to DB")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between batches")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        hydrate_cities(
            db,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            sleep_seconds=args.sleep,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
