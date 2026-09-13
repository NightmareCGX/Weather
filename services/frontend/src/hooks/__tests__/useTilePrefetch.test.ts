import type { Map as MapLibreMap } from "maplibre-gl";

import { adjacentTileUrls, coveringTiles } from "@/hooks/useTilePrefetch";
import type { ValidTimeAvailability } from "@/lib/api/types";

const pinnedTemplate =
  "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z&initial_time=2026-08-13T00%3A00%3A00Z";

function entry(overrides: Partial<ValidTimeAvailability> = {}): ValidTimeAvailability {
  return {
    valid_time: "2026-08-13T15:00:00Z",
    source_cycle: "2026-08-13T06:00:00Z",
    lead_time_hours: 9,
    servable: true,
    available_members: 1,
    expected_members: 1,
    coverage_ratio: 1,
    ...overrides,
  };
}

describe("adjacentTileUrls", () => {
  it("swaps the valid time and pins the target's serving cycle per tile", () => {
    const urls = adjacentTileUrls(pinnedTemplate, entry(), 3, [
      { x: 0, y: 1 },
      { x: 2, y: 3 },
    ]);
    expect(urls).toEqual([
      "/v1/maps/gfs/temperature_2m/surface/3/0/1.png?valid_time=2026-08-13T15%3A00%3A00Z&initial_time=2026-08-13T06%3A00%3A00Z",
      "/v1/maps/gfs/temperature_2m/surface/3/2/3.png?valid_time=2026-08-13T15%3A00%3A00Z&initial_time=2026-08-13T06%3A00%3A00Z",
    ]);
  });

  it("drops the pinned cycle parameter when the target entry has no cycle", () => {
    const target = entry({ source_cycle: "" });
    const urls = adjacentTileUrls(pinnedTemplate, target, 2, [{ x: 1, y: 1 }]);
    expect(urls).toEqual([
      "/v1/maps/gfs/temperature_2m/surface/2/1/1.png?valid_time=2026-08-13T15%3A00%3A00Z",
    ]);
  });
});

describe("coveringTiles", () => {
  function fakeMap(
    west: number,
    south: number,
    east: number,
    north: number,
    zoom: number
  ): MapLibreMap {
    return {
      getBounds: () => ({
        getSouthWest: () => ({ lng: west, lat: south }),
        getNorthEast: () => ({ lng: east, lat: north }),
      }),
      getZoom: () => zoom,
    } as unknown as MapLibreMap;
  }

  it("returns the four world tiles at zoom 1", () => {
    const tiles = coveringTiles(fakeMap(-180, -85, 180, 85, 1), 1);
    expect(tiles).toEqual([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ]);
  });

  it("folds a longitude-180 east edge onto the last column instead of wrapping", () => {
    const tiles = coveringTiles(fakeMap(150, -10, 180, 10, 1), 1);
    expect(tiles).toEqual([
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ]);
  });

  it("clamps the viewport to the metatile range at deep zoom", () => {
    const tiles = coveringTiles(fakeMap(-106.9, 39.1, -106.7, 39.3, 9), 9);
    expect(tiles.length).toBeGreaterThan(0);
    for (const tile of tiles) {
      expect(tile.x).toBeGreaterThanOrEqual(0);
      expect(tile.x).toBeLessThan(512);
      expect(tile.y).toBeGreaterThanOrEqual(0);
      expect(tile.y).toBeLessThan(512);
    }
  });
});
