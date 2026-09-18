import {
  MAP_WORLD_SIZE_AT_ZOOM_0,
  STARTUP_TILE_SPAN,
  STARTUP_TILE_ZOOM,
  startupViewZoom,
} from "@/lib/map/startupView";

/** Tiles of the STARTUP_TILE_ZOOM grid a viewport of this width spans. */
function tilesAcross(viewportWidthPx: number): number {
  return (
    (viewportWidthPx / MAP_WORLD_SIZE_AT_ZOOM_0) *
    2 ** (STARTUP_TILE_ZOOM - startupViewZoom(viewportWidthPx))
  );
}

describe("startupViewZoom", () => {
  it("frames exactly the tile span across every viewport width", () => {
    for (const width of [390, 768, 900, 1150, 1400, 1920, 2560]) {
      expect(tilesAcross(width)).toBeCloseTo(STARTUP_TILE_SPAN, 9);
    }
  });

  it("keeps a typical desktop viewport inside the 5-9 tile band", () => {
    // A 1400x900 map area shows 3 x 1.93 tiles; the region grows taller with the
    // viewport aspect, so the shorter dimension is the tile ceiling.
    const across = tilesAcross(1400);
    const down = across * (900 / 1400);

    expect(Math.round(across * down)).toBeGreaterThanOrEqual(5);
    expect(Math.round(across * down)).toBeLessThanOrEqual(9);
  });

  it("scales with the viewport so the ground area stays constant", () => {
    // Region width in degrees of longitude is what the tile budget pins down.
    const regionDegrees = 3 * (360 / 2 ** STARTUP_TILE_ZOOM);

    for (const width of [900, 1400, 1920]) {
      const worldPx = MAP_WORLD_SIZE_AT_ZOOM_0 * 2 ** startupViewZoom(width);
      expect((width / worldPx) * 360).toBeCloseTo(regionDegrees, 9);
    }
  });

  it("pins the zoom for known viewport widths", () => {
    // ~360 km of ground across at Denver's latitude, versus the ~930 km the
    // previous fixed zoom 6.5 framed on the same viewport.
    expect(startupViewZoom(1400)).toBeCloseTo(7.8662, 3);
    expect(startupViewZoom(1920)).toBeCloseTo(8.3219, 3);
    expect(startupViewZoom(900)).toBeCloseTo(7.2288, 3);
  });

  it("falls back to the tile grid's nominal zoom for an unmeasurable viewport", () => {
    for (const width of [null, undefined, 0, -100, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(startupViewZoom(width)).toBe(STARTUP_TILE_ZOOM);
    }
  });
});
