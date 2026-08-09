import type { PointLocationSpecifier } from "@/lib/api/client";
import type { SearchResult, SelectedLocation } from "@/lib/api/types";

/**
 * Conversions between backend search results / raw coordinates and the shared
 * {@link SelectedLocation} model, and back to `/v1/points` spatial
 * specifiers.
 *
 * Station results resolve by coordinates because `/v1/points` defines no
 * station spatial specifier (only `lat`/`lon`, `city_id`, `resort_id`).
 */

const COORDINATE_PRECISION = 4;

/** Format a coordinate pair as a stable label, e.g. `"38.19, -106.82"`. */
export function formatCoordinates(latitude: number, longitude: number): string {
  return `${latitude.toFixed(COORDINATE_PRECISION)}, ${longitude.toFixed(COORDINATE_PRECISION)}`;
}

/** Convert a `/v1/search` result into the shared selected-location model. */
export function searchResultToSelectedLocation(result: SearchResult): SelectedLocation {
  const base = {
    name: result.name,
    latitude: result.latitude,
    longitude: result.longitude,
    elevation_m: result.elevation_m,
    region: result.region,
    country: result.country,
  };
  switch (result.object) {
    case "city":
      return {
        ...base,
        object: "city" as const,
        id: result.id,
        resolvedVia: "city" as const,
      };
    case "ski_resort":
      return {
        ...base,
        object: "ski_resort" as const,
        id: result.id,
        resolvedVia: "resort" as const,
      };
    default:
      // Station: no /v1/points station specifier exists, so it resolves by
      // coordinates. The platform id is preserved for labeling but is not used
      // as a spatial specifier.
      return {
        ...base,
        object: "station" as const,
        id: result.id,
        resolvedVia: "coordinates" as const,
      };
  }
}

/** Convert a raw map-click coordinate into the shared selected-location model. */
export function coordinatesToSelectedLocation(
  latitude: number,
  longitude: number
): SelectedLocation {
  return {
    name: formatCoordinates(latitude, longitude),
    object: "coordinates",
    latitude,
    longitude,
    elevation_m: null,
    region: null,
    country: null,
    id: null,
    resolvedVia: "coordinates",
  };
}

/**
 * Build the `/v1/points` spatial specifier for a selected location.
 *
 * City and ski resort locations use their platform id; station and raw
 * coordinate locations use the coordinate pair.
 */
export function toPointSpecifier(location: SelectedLocation): PointLocationSpecifier {
  if (location.object === "city" && location.id !== null) {
    return { type: "city", cityId: location.id };
  }
  if (location.object === "ski_resort" && location.id !== null) {
    return { type: "resort", resortId: location.id };
  }
  return { type: "coordinates", latitude: location.latitude, longitude: location.longitude };
}

/** A compact type label for a selected location (used in badges/ARIA). */
export function locationTypeLabel(location: SelectedLocation): string {
  switch (location.object) {
    case "city":
      return "City";
    case "ski_resort":
      return "Ski resort";
    case "station":
      return "Station";
    default:
      return "Coordinates";
  }
}
