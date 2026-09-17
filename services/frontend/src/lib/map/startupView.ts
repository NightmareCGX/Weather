/**
 * Startup regional view sizing.
 *
 * The startup camera is defined by a tile budget rather than a raw zoom: the
 * viewport is framed so that it spans {@link STARTUP_TILE_SPAN} tiles of the
 * {@link STARTUP_TILE_ZOOM} Web-Mercator grid horizontally. On a typical desktop
 * that is a 3x2 region (~6 tiles), staying inside the intended 5-9 tile band
 * across screen sizes.
 *
 * Sizing the *ground area* instead of the zoom is what keeps the region
 * consistent: the number of tiles on screen is a function of viewport pixels,
 * so one fixed zoom frames a different amount of ground on every display.
 *
 * MapLibre renders the world at ``MAP_WORLD_SIZE_AT_ZOOM_0 * 2^zoom`` CSS px
 * (``Transform.tileSize`` is 512), so a viewport W px wide spans
 * ``(W / 512) * 2^(tileZoom - zoom)`` tiles of the tileZoom grid.
 */

/** Tile zoom whose grid the startup region is measured in. */
export const STARTUP_TILE_ZOOM = 8;

/** Width of the startup region, in tiles of the {@link STARTUP_TILE_ZOOM} grid. */
export const STARTUP_TILE_SPAN = 3;

/** MapLibre's world size in CSS px at zoom 0 (``Transform.tileSize``). */
export const MAP_WORLD_SIZE_AT_ZOOM_0 = 512;

/**
 * Map zoom at which a viewport `viewportWidthPx` wide spans
 * {@link STARTUP_TILE_SPAN} tiles of the {@link STARTUP_TILE_ZOOM} grid.
 *
 * An unmeasurable viewport (hidden container, test environment without layout)
 * yields {@link STARTUP_TILE_ZOOM} itself, the nominal zoom for that grid.
 */
export function startupViewZoom(viewportWidthPx: number | null | undefined): number {
  if (
    viewportWidthPx === null ||
    viewportWidthPx === undefined ||
    !Number.isFinite(viewportWidthPx) ||
    viewportWidthPx <= 0
  ) {
    return STARTUP_TILE_ZOOM;
  }
  return (
    STARTUP_TILE_ZOOM + Math.log2(viewportWidthPx / (MAP_WORLD_SIZE_AT_ZOOM_0 * STARTUP_TILE_SPAN))
  );
}
