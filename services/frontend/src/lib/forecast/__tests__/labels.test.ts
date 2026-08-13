import type { VariableResource } from "@/lib/api/types";
import {
  buildVariableMeta,
  formatPercent,
  formatProbabilityRange,
  formatValue,
} from "@/lib/forecast/labels";
import {
  coordinatesToSelectedLocation,
  formatCoordinates,
  locationTypeLabel,
  searchResultToSelectedLocation,
  toPointSpecifier,
} from "@/lib/forecast/selection";
import { formatDayHourUtc, formatLeadTimeHours, formatTimeUtc } from "@/lib/forecast/time";
import type { SelectedLocation } from "@/lib/api/types";

describe("labels", () => {
  it("builds variable metadata from the catalog with fallbacks", () => {
    const catalog: VariableResource[] = [
      { id: "temperature_2m", object: "variable", name: "2-Meter Temperature", unit: "°C" },
    ];
    const meta = buildVariableMeta(catalog);
    expect(meta.temperature_2m).toEqual({ name: "2-Meter Temperature", unit: "°C" });
    // Fallback applies when the catalog does not carry a known default.
    expect(meta.precipitation_rate).toEqual({ name: "Precipitation Rate", unit: "mm/h" });
  });

  it("falls back for a missing or empty catalog", () => {
    expect(buildVariableMeta(null).temperature_2m.unit).toBe("°C");
    expect(buildVariableMeta([]).precipitation_rate.name).toBe("Precipitation Rate");
  });

  it("formats values with units", () => {
    expect(formatValue(15, "°C")).toBe("15 °C");
    expect(formatValue(15.234, "mm/h")).toBe("15.23 mm/h");
    expect(formatValue(3, "")).toBe("3");
  });

  it("formats percents and probability ranges", () => {
    expect(formatPercent(0.42)).toBe("42%");
    expect(formatProbabilityRange(0.42, [0.38, 0.46])).toBe("42% (95% CI 38%–46%)");
  });
});

describe("selection", () => {
  it("converts a city search result to a city selection with a platform id", () => {
    const result = {
      id: "city_aspen",
      object: "city",
      name: "Aspen",
      region: "Colorado",
      country: "USA",
      elevation_m: null,
      latitude: 38.19,
      longitude: -106.82,
    } as const;
    const selected = searchResultToSelectedLocation(result);
    expect(selected).toEqual({
      object: "city",
      resolvedVia: "city",
      id: "city_aspen",
      name: "Aspen",
      region: "Colorado",
      country: "USA",
      elevation_m: null,
      latitude: 38.19,
      longitude: -106.82,
    });
    expect(toPointSpecifier(selected)).toEqual({ type: "city", cityId: "city_aspen" });
  });

  it("converts a ski resort to a resort selection", () => {
    const result = {
      id: "resort_aspen_mountain",
      object: "ski_resort",
      name: "Aspen Mountain",
      region: "Colorado",
      country: "USA",
      elevation_m: 3417,
      latitude: 38.19,
      longitude: -106.82,
    } as const;
    const selected = searchResultToSelectedLocation(result);
    expect(selected).toMatchObject({
      object: "ski_resort",
      resolvedVia: "resort",
      elevation_m: 3417,
    });
    expect(toPointSpecifier(selected)).toEqual({
      type: "resort",
      resortId: "resort_aspen_mountain",
    });
  });

  it("converts a station to a coordinate resolution (no station specifier exists)", () => {
    const result = {
      id: "station_aspen_co",
      object: "station",
      name: "Aspen Station",
      region: null,
      country: null,
      elevation_m: 2380,
      latitude: 38.19,
      longitude: -106.82,
    } as const;
    const selected = searchResultToSelectedLocation(result);
    expect(selected).toMatchObject({
      object: "station",
      resolvedVia: "coordinates",
      id: "station_aspen_co",
    });
    expect(toPointSpecifier(selected)).toEqual({
      type: "coordinates",
      latitude: 38.19,
      longitude: -106.82,
    });
  });

  it("converts raw coordinates to a coordinate selection", () => {
    const selected = coordinatesToSelectedLocation(38.1911, -106.8175);
    expect(selected).toMatchObject({
      object: "coordinates",
      resolvedVia: "coordinates",
      id: null,
      elevation_m: null,
    });
    expect(selected.name).toBe(formatCoordinates(38.1911, -106.8175));
  });

  it("labels location types", () => {
    const city: SelectedLocation = coordinatesToSelectedLocation(1, 2);
    expect(locationTypeLabel({ ...city, object: "city" })).toBe("City");
    expect(locationTypeLabel({ ...city, object: "ski_resort" })).toBe("Ski resort");
    expect(locationTypeLabel({ ...city, object: "station" })).toBe("Station");
    expect(locationTypeLabel(city)).toBe("Coordinates");
  });
});

describe("time", () => {
  it("formats timestamps in UTC", () => {
    expect(formatTimeUtc("2026-07-21T06:00:00Z")).toBe("06:00");
    expect(formatDayHourUtc("2026-07-21T06:00:00Z")).toContain("Jul 21");
    expect(formatLeadTimeHours(6)).toBe("+6h");
  });
});
