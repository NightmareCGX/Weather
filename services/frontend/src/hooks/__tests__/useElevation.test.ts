import { act, renderHook, waitFor } from "@testing-library/react";
import { useElevation } from "@/hooks/useElevation";
import type { SelectedLocation } from "@/lib/api/types";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

const knownResort: SelectedLocation = {
  name: "Aspen Mountain",
  object: "ski_resort",
  latitude: 38.19,
  longitude: -106.82,
  elevation_m: 3417.0,
  region: "Colorado",
  country: "USA",
  id: "resort_aspen_mountain",
  resolvedVia: "resort",
};

const dynamicCoords: SelectedLocation = {
  name: "38.19, -106.82",
  object: "coordinates",
  latitude: 38.19,
  longitude: -106.82,
  elevation_m: null,
  region: null,
  country: null,
  id: null,
  resolvedVia: "coordinates",
};

const otherCoords: SelectedLocation = {
  name: "40.0, -105.0",
  object: "coordinates",
  latitude: 40.0,
  longitude: -105.0,
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

describe("useElevation", () => {
  it("is idle when location is null", () => {
    const { result } = renderHook(() => useElevation(null));
    expect(result.current.status).toBe("idle");
    expect(result.current.elevation_m).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("returns known elevation immediately without firing an API call", () => {
    const { result } = renderHook(() => useElevation(knownResort));
    expect(result.current.status).toBe("success");
    expect(result.current.elevation_m).toBe(3417.0);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("fetches elevation for dynamic coordinates when elevation_m is null", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        latitude: 38.19,
        longitude: -106.82,
        elevation_m: 2404.0,
      })
    );

    const { result } = renderHook(() => useElevation(dynamicCoords));

    await waitFor(() => {
      expect(result.current.status).toBe("success");
    });

    expect(result.current.elevation_m).toBe(2404.0);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(String(mockFetch.mock.calls[0][0])).toContain("/v1/elevation?lat=38.19&lon=-106.82");
  });

  it("handles null elevation from API gracefully as success with null", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        latitude: 38.19,
        longitude: -106.82,
        elevation_m: null,
      })
    );

    const { result } = renderHook(() => useElevation(dynamicCoords));

    await waitFor(() => {
      expect(result.current.status).toBe("success");
    });

    expect(result.current.elevation_m).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("handles network failure gracefully as error status with null elevation", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network connection failed"));

    const { result } = renderHook(() => useElevation(dynamicCoords));

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });

    expect(result.current.elevation_m).toBeNull();
    expect(result.current.error).toContain("Network request failed.");
  });

  it("cancels and discards previous in-flight request when location changes rapidly", async () => {
    let resolveFirst!: (value: Response) => void;
    const firstPromise = new Promise<Response>((resolve) => {
      resolveFirst = resolve;
    });

    mockFetch
      .mockImplementationOnce(() => firstPromise)
      .mockResolvedValueOnce(
        jsonResponse({
          latitude: 40.0,
          longitude: -105.0,
          elevation_m: 1600.0,
        })
      );

    const { result, rerender } = renderHook(
      ({ loc }: { loc: SelectedLocation }) => useElevation(loc),
      { initialProps: { loc: dynamicCoords } }
    );

    expect(result.current.status).toBe("loading");

    // Switch location while first is still pending
    act(() => {
      rerender({ loc: otherCoords });
    });

    // Resolve second request
    await waitFor(() => {
      expect(result.current.status).toBe("success");
      expect(result.current.elevation_m).toBe(1600.0);
    });

    // Now resolve the late first request
    act(() => {
      resolveFirst(
        jsonResponse({
          latitude: 38.19,
          longitude: -106.82,
          elevation_m: 9999.0,
        })
      );
    });

    // Elevation must NOT be overwritten by the stale first response!
    expect(result.current.elevation_m).toBe(1600.0);
  });
});
