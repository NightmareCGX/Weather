import {
  canonicalizeLongitude,
  coordinatesToSelectedLocation,
  isValidCoordinate,
  searchResultToSelectedLocation,
  toPointSpecifier,
} from "@/lib/forecast/selection";
import { getEnsembleStatistics, getPointForecast } from "@/lib/api/client";
import type { SearchResult } from "@/lib/api/types";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("canonicalizeLongitude", () => {
  it("resolves confirmed positive Antarctic overflow (187.58796875 -> -172.41203125)", () => {
    expect(canonicalizeLongitude(187.58796875)).toBeCloseTo(-172.41203125, 8);
  });

  it("resolves confirmed negative Antarctic overflow (-188.6328125 -> 171.3671875)", () => {
    expect(canonicalizeLongitude(-188.6328125)).toBeCloseTo(171.3671875, 8);
  });

  it("satisfies canonical equivalence for equivalent western coordinates (181, -179, 541 -> -179)", () => {
    expect(canonicalizeLongitude(181)).toBe(-179);
    expect(canonicalizeLongitude(-179)).toBe(-179);
    expect(canonicalizeLongitude(541)).toBe(-179);
  });

  it("satisfies canonical equivalence for equivalent eastern coordinates (-181, 179, -541 -> 179)", () => {
    expect(canonicalizeLongitude(-181)).toBe(179);
    expect(canonicalizeLongitude(179)).toBe(179);
    expect(canonicalizeLongitude(-541)).toBe(179);
  });

  it("establishes antimeridian identity (+180 and -180 map to +180)", () => {
    expect(canonicalizeLongitude(180)).toBe(180);
    expect(canonicalizeLongitude(-180)).toBe(180);
    expect(canonicalizeLongitude(540)).toBe(180);
    expect(canonicalizeLongitude(-540)).toBe(180);
  });

  it("leaves ordinary coordinates numerically unchanged", () => {
    expect(canonicalizeLongitude(-155.3636)).toBeCloseTo(-155.3636, 6);
    expect(canonicalizeLongitude(-105.0)).toBe(-105.0);
    expect(canonicalizeLongitude(174.4044)).toBeCloseTo(174.4044, 6);
    expect(canonicalizeLongitude(0)).toBe(0);
  });
});

describe("isValidCoordinate", () => {
  it("accepts valid geodetic coordinates including zero", () => {
    expect(isValidCoordinate(0, 0)).toBe(true);
    expect(isValidCoordinate(39.7392, -104.9903)).toBe(true);
    expect(isValidCoordinate(-90, 180)).toBe(true);
    expect(isValidCoordinate(90, -180)).toBe(true);
    expect(isValidCoordinate(-90, -180)).toBe(true);
    expect(isValidCoordinate(90, 180)).toBe(true);
  });

  it("rejects non-numeric, NaN, and infinite values", () => {
    expect(isValidCoordinate(NaN, -105.0)).toBe(false);
    expect(isValidCoordinate(40.0, NaN)).toBe(false);
    expect(isValidCoordinate(Infinity, 0)).toBe(false);
    expect(isValidCoordinate(0, -Infinity)).toBe(false);
    expect(isValidCoordinate(null, 0)).toBe(false);
    expect(isValidCoordinate(undefined, 0)).toBe(false);
    expect(isValidCoordinate("40.0", "-105.0")).toBe(false);
  });

  it("rejects out-of-range coordinates", () => {
    expect(isValidCoordinate(90.1, 0)).toBe(false);
    expect(isValidCoordinate(-90.1, 0)).toBe(false);
    expect(isValidCoordinate(0, 180.1)).toBe(false);
    expect(isValidCoordinate(0, -180.1)).toBe(false);
  });
});

describe("searchResultToSelectedLocation direct coordinates", () => {
  it("converts direct place result into canonical SelectedLocation", () => {
    const result: SearchResult = {
      id: "place_denver",
      object: "place",
      name: "Denver",
      region: "Colorado",
      country: "United States",
      elevation_m: null,
      latitude: 39.7392,
      longitude: -104.9903,
      place_id: "51a3denver",
    };
    const selected = searchResultToSelectedLocation(result);
    expect(selected).toMatchObject({
      name: "Denver",
      object: "coordinates",
      id: "51a3denver",
      resolvedVia: "coordinates",
      latitude: 39.7392,
      elevation_m: null,
      region: "Colorado",
      country: "United States",
    });
    expect(selected.longitude).toBeCloseTo(-104.9903, 4);
  });

  it("throws on invalid or missing coordinates", () => {
    const invalidResult: SearchResult = {
      id: "place_corrupt",
      object: "place",
      name: "Corrupt Place",
      region: null,
      country: null,
      elevation_m: null,
      latitude: 100.0,
      longitude: -104.9903,
    };
    expect(() => searchResultToSelectedLocation(invalidResult)).toThrow(
      "Invalid coordinates in search result"
    );
  });
});

describe("coordinatesToSelectedLocation canonicalization", () => {
  it("canonicalizes positive overflow and updates display label", () => {
    const loc = coordinatesToSelectedLocation(-77.85, 187.58796875);
    expect(loc.latitude).toBe(-77.85);
    expect(loc.longitude).toBeCloseTo(-172.41203125, 8);
    expect(loc.name).toBe("-77.8500, -172.4120");
  });

  it("canonicalizes negative overflow and updates display label", () => {
    const loc = coordinatesToSelectedLocation(-80.0, -188.6328125);
    expect(loc.latitude).toBe(-80.0);
    expect(loc.longitude).toBeCloseTo(171.3671875, 8);
    expect(loc.name).toBe("-80.0000, 171.3672");
  });

  it("preserves identical identity for equivalent representations", () => {
    const locA = coordinatesToSelectedLocation(-75.0, 181);
    const locB = coordinatesToSelectedLocation(-75.0, -179);
    const locC = coordinatesToSelectedLocation(-75.0, 541);

    expect(locA.longitude).toBe(-179);
    expect(locB.longitude).toBe(-179);
    expect(locC.longitude).toBe(-179);
    expect(locA.name).toBe(locB.name);
    expect(locB.name).toBe(locC.name);
  });

  it("preserves identical identity for antimeridian representations (+180 and -180)", () => {
    const locPos = coordinatesToSelectedLocation(-78.0, 180);
    const locNeg = coordinatesToSelectedLocation(-78.0, -180);

    expect(locPos.longitude).toBe(180);
    expect(locNeg.longitude).toBe(180);
    expect(locPos.name).toBe(locNeg.name);
  });

  it("canonicalizes search results entering selected-location state", () => {
    const searchRes: SearchResult = {
      id: "place_antimeridian",
      object: "place",
      name: "Taveuni Island",
      latitude: -16.8,
      longitude: -180,
      elevation_m: null,
      region: "Northern Division",
      country: "Fiji",
      place_id: "place_123",
    };
    const loc = searchResultToSelectedLocation(searchRes);
    expect(loc.longitude).toBe(180);
    expect(loc.name).toBe("Taveuni Island");
  });
});

describe("product request propagation", () => {
  it("propagates canonical longitude into getPointForecast (/v1/points)", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "point_forecast",
        data: {
          location: {
            latitude: -77.85,
            longitude: -172.412,
            elevation_m: null,
            resolved_via: "coordinates",
          },
          generated_at: "2026-07-21T00:00:00Z",
          model: "gfs",
          forecasts: [],
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const location = coordinatesToSelectedLocation(-77.85, 187.58796875);
    expect(location.longitude).toBeCloseTo(-172.41203125, 8);

    await getPointForecast({
      location: toPointSpecifier(location),
      model: "gfs",
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const requestedUrl = mockFetch.mock.calls[0][0] as string;
    const url = new URL(requestedUrl, "http://localhost");
    const lat = Number(url.searchParams.get("lat"));
    const lon = Number(url.searchParams.get("lon"));

    expect(lat).toBeCloseTo(-77.85, 4);
    expect(lon).toBeCloseTo(-172.41203125, 4);
    expect(lon).toBeGreaterThanOrEqual(-180);
    expect(lon).toBeLessThanOrEqual(180);
  });

  it("propagates canonical longitude into getEnsembleStatistics (/v1/ensembles)", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "ensemble_statistics",
        data: {
          model: "gefs",
          lead_time_hours: 0,
          member_count: 30,
          statistics: {},
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const location = coordinatesToSelectedLocation(-80.0, -188.6328125);
    expect(location.longitude).toBeCloseTo(171.3671875, 8);

    await getEnsembleStatistics({
      latitude: location.latitude,
      longitude: location.longitude,
      variable: "temperature_2m",
      model: "gefs",
      leadTimeHours: 0,
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const requestedUrl = mockFetch.mock.calls[0][0] as string;
    const url = new URL(requestedUrl, "http://localhost");
    const lat = Number(url.searchParams.get("lat"));
    const lon = Number(url.searchParams.get("lon"));

    expect(lat).toBeCloseTo(-80.0, 4);
    expect(lon).toBeCloseTo(171.3671875, 4);
    expect(lon).toBeGreaterThanOrEqual(-180);
    expect(lon).toBeLessThanOrEqual(180);
  });
});
