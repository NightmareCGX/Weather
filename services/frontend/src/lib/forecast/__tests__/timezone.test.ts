import { getTimezoneForCoordinates, getTimezoneForLocation } from "@/lib/forecast/timezone";
import type { SelectedLocation } from "@/lib/api/types";

describe("getTimezoneForCoordinates", () => {
  it("resolves IANA timezone for major world locations", () => {
    // Denver
    expect(getTimezoneForCoordinates(39.7392, -104.9903)).toBe("America/Denver");
    // Tokyo
    expect(getTimezoneForCoordinates(35.6762, 139.6503)).toBe("Asia/Tokyo");
    // London
    expect(getTimezoneForCoordinates(51.5074, -0.1278)).toBe("Europe/London");
    // Kathmandu
    expect(getTimezoneForCoordinates(27.7172, 85.324)).toBe("Asia/Kathmandu");
    // Adelaide
    expect(getTimezoneForCoordinates(-34.9285, 138.6007)).toBe("Australia/Adelaide");
  });

  it("handles antimeridian longitude canonicalization safely", () => {
    // 180 and -180
    const tzPos = getTimezoneForCoordinates(0, 180);
    expect(typeof tzPos).toBe("string");
    const tzWrapped = getTimezoneForCoordinates(0, 540); // 540 maps to 180
    expect(tzWrapped).toBe(tzPos);
  });

  it("returns null for invalid or out-of-range coordinates without throwing", () => {
    expect(getTimezoneForCoordinates(NaN, 0)).toBeNull();
    expect(getTimezoneForCoordinates(0, NaN)).toBeNull();
    expect(getTimezoneForCoordinates(Infinity, 0)).toBeNull();
    expect(getTimezoneForCoordinates(0, -Infinity)).toBeNull();
    expect(getTimezoneForCoordinates(95, 0)).toBeNull();
    expect(getTimezoneForCoordinates(-95, 0)).toBeNull();
    expect(getTimezoneForCoordinates(null, 0)).toBeNull();
    expect(getTimezoneForCoordinates(0, undefined)).toBeNull();
  });
});

describe("getTimezoneForLocation", () => {
  it("derives timezone from a SelectedLocation object", () => {
    const denver: SelectedLocation = {
      name: "Denver",
      object: "city",
      latitude: 39.7392,
      longitude: -104.9903,
      elevation_m: 1609,
      region: "Colorado",
      country: "USA",
      id: "city_denver",
      resolvedVia: "city",
    };
    expect(getTimezoneForLocation(denver)).toBe("America/Denver");
  });

  it("returns null when location is null or undefined", () => {
    expect(getTimezoneForLocation(null)).toBeNull();
    expect(getTimezoneForLocation(undefined)).toBeNull();
  });
});
