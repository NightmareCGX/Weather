import tzLookup from "@photostructure/tz-lookup";

import { canonicalizeLongitude, isValidCoordinate } from "@/lib/forecast/selection";
import type { SelectedLocation } from "@/lib/api/types";

/**
 * Resolve an IANA timezone identifier for the given latitude and longitude.
 *
 * Returns null if coordinates are invalid, out of range, or if lookup fails.
 * Never throws into the caller.
 */
export function getTimezoneForCoordinates(
  latitude: number | null | undefined,
  longitude: number | null | undefined
): string | null {
  if (
    latitude === null ||
    latitude === undefined ||
    longitude === null ||
    longitude === undefined
  ) {
    return null;
  }
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }
  const canonicalLon = canonicalizeLongitude(longitude);
  if (!isValidCoordinate(latitude, canonicalLon)) {
    return null;
  }
  try {
    return tzLookup(latitude, canonicalLon);
  } catch {
    return null;
  }
}

/**
 * Convenience helper to resolve the IANA timezone for a {@link SelectedLocation}.
 */
export function getTimezoneForLocation(
  location: SelectedLocation | null | undefined
): string | null {
  if (!location) return null;
  return getTimezoneForCoordinates(location.latitude, location.longitude);
}
