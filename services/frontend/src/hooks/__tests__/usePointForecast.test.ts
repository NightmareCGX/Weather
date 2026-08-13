import { act, renderHook, waitFor } from "@testing-library/react";
import { usePointForecast } from "@/hooks/usePointForecast";
import type { SelectedLocation } from "@/lib/api/types";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function forecastResponse(lat: number, lon: number) {
  return jsonResponse({
    object: "point_forecast",
    data: {
      location: { latitude: lat, longitude: lon, elevation_m: null, resolved_via: "coordinates" },
      generated_at: "2026-07-21T00:00:00Z",
      model: "gfs",
      forecasts: [{ lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 15 }],
    },
    has_more: false,
    next_cursor: null,
  });
}

const aspen: SelectedLocation = {
  name: "Aspen",
  object: "city",
  latitude: 38.19,
  longitude: -106.82,
  elevation_m: null,
  region: "Colorado",
  country: "USA",
  id: "city_aspen",
  resolvedVia: "city",
};

const other: SelectedLocation = {
  name: "Other",
  object: "coordinates",
  latitude: 40,
  longitude: -105,
  elevation_m: null,
  region: null,
  country: null,
  id: null,
  resolvedVia: "coordinates",
};

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("usePointForecast", () => {
  it("is idle with no selection", () => {
    const { result } = renderHook(() => usePointForecast(null, { model: "gfs" }));
    expect(result.current.status).toBe("idle");
    expect(result.current.forecast).toBeNull();
  });

  it("stays idle when no deterministic model is available", () => {
    const { result } = renderHook(() => usePointForecast(aspen, { model: null }));
    expect(result.current.status).toBe("idle");
    expect(result.current.forecast).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("fetches via the city id specifier", async () => {
    mockFetch.mockResolvedValueOnce(forecastResponse(38.19, -106.82));
    const { result } = renderHook(() => usePointForecast(aspen, { model: "gfs" }));

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/points?models=gfs&units=metric&city_id=city_aspen",
      expect.any(Object)
    );
    expect(result.current.forecast?.model).toBe("gfs");
  });

  it("never lets a stale selection's response overwrite a newer one", async () => {
    let resolveAspen!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveAspen = resolve))
    );
    mockFetch.mockResolvedValueOnce(forecastResponse(40, -105));

    const { result, rerender } = renderHook(({ loc }) => usePointForecast(loc, { model: "gfs" }), {
      initialProps: { loc: aspen },
    });

    expect(result.current.status).toBe("loading");

    // Selection changes before the Aspen response resolves.
    rerender({ loc: other });
    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.forecast?.location.latitude).toBe(40);

    // The stale Aspen response resolves late; it must be ignored.
    await act(async () => {
      resolveAspen(forecastResponse(38.19, -106.82));
    });

    // The newer "other" forecast is still what renders.
    expect(result.current.forecast?.location.latitude).toBe(40);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("surfaces an error when the fetch fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const { result } = renderHook(() => usePointForecast(aspen, { model: "gfs" }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.forecast).toBeNull();
  });
});
