import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";

import { SelectedLocationProvider, useSelectedLocation } from "@/context/selected-location";
import { useGeolocation } from "@/hooks/useGeolocation";
import { getApproximateLocation } from "@/lib/api/client";

jest.mock("../../lib/api/client", () => {
  const actual = jest.requireActual("../../lib/api/client");
  return {
    ...actual,
    getApproximateLocation: jest.fn(),
  };
});

const mockGetApproximateLocation = getApproximateLocation as jest.MockedFunction<
  typeof getApproximateLocation
>;

const wrapper = ({ children }: { children: ReactNode }) => (
  <SelectedLocationProvider>{children}</SelectedLocationProvider>
);

describe("useGeolocation", () => {
  let mockGetCurrentPosition: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetCurrentPosition = jest.fn();
    Object.defineProperty(globalThis, "navigator", {
      value: {
        geolocation: {
          getCurrentPosition: mockGetCurrentPosition,
        },
      },
      writable: true,
      configurable: true,
    });
  });

  it("does not request browser geolocation on mount (Rule 1)", () => {
    renderHook(() => useGeolocation(), { wrapper });
    expect(mockGetCurrentPosition).not.toHaveBeenCalled();
    expect(mockGetApproximateLocation).not.toHaveBeenCalled();
  });

  it("requests geolocation with low-power options only on explicit locateMe() call", () => {
    const { result } = renderHook(() => useGeolocation(), { wrapper });

    act(() => {
      result.current.locateMe();
    });

    expect(mockGetCurrentPosition).toHaveBeenCalledTimes(1);
    expect(mockGetCurrentPosition).toHaveBeenCalledWith(
      expect.any(Function),
      expect.any(Function),
      {
        enableHighAccuracy: false,
        timeout: 10000,
        maximumAge: 300000,
      }
    );
    expect(result.current.isLocating).toBe(true);
  });

  it("commits valid coordinates and calls onLocateSuccess on geolocation success", () => {
    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    const [successCb] = mockGetCurrentPosition.mock.calls[0];

    act(() => {
      successCb({
        coords: {
          latitude: 39.7392,
          longitude: -104.9903,
        },
      });
    });

    expect(result.current.geo.isLocating).toBe(false);
    expect(result.current.loc.selectedLocation).toMatchObject({
      name: "39.7392, -104.9903",
      object: "coordinates",
      resolvedVia: "coordinates",
      latitude: 39.7392,
    });
    expect(result.current.loc.selectedLocation?.longitude).toBeCloseTo(-104.9903, 4);
    expect(onLocateSuccess).toHaveBeenCalledTimes(1);
    expect(result.current.geo.notice).toBeNull();
  });

  it("canonicalizes -180 longitude to +180 on success", () => {
    const { result } = renderHook(
      () => {
        const geo = useGeolocation();
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    const [successCb] = mockGetCurrentPosition.mock.calls[0];
    act(() => {
      successCb({
        coords: {
          latitude: 15.0,
          longitude: -180.0,
        },
      });
    });

    expect(result.current.loc.selectedLocation?.longitude).toBe(180.0);
  });

  it("PERMISSION_DENIED: sets alert notice, does NOT call IP fallback, does not update location", () => {
    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    const [, errorCb] = mockGetCurrentPosition.mock.calls[0];

    act(() => {
      errorCb({
        code: 1, // PERMISSION_DENIED
        PERMISSION_DENIED: 1,
        message: "User denied Geolocation",
      });
    });

    expect(result.current.geo.isLocating).toBe(false);
    expect(result.current.loc.selectedLocation).toBeNull();
    expect(onLocateSuccess).not.toHaveBeenCalled();
    // Strict privacy invariant: NEVER call IP fallback on permission denial!
    expect(mockGetApproximateLocation).not.toHaveBeenCalled();
    expect(result.current.geo.notice).toEqual({
      type: "alert",
      message: "Location access denied. Please enable location permissions in your browser.",
    });
  });

  it("POSITION_UNAVAILABLE: triggers coarse IP fallback, commits approximate location with status notice", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce({
      latitude: 40.015,
      longitude: -105.2705,
      city: "Boulder",
      region: "Colorado",
      country: "US",
      approximate: true,
    });

    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    const [, errorCb] = mockGetCurrentPosition.mock.calls[0];

    await act(async () => {
      errorCb({
        code: 2, // POSITION_UNAVAILABLE
        POSITION_UNAVAILABLE: 2,
        message: "Position unavailable",
      });
    });

    expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);

    await waitFor(() => {
      expect(result.current.geo.isLocating).toBe(false);
      expect(result.current.loc.selectedLocation).toMatchObject({
        name: "Boulder, Colorado, US",
        object: "coordinates",
        latitude: 40.015,
        region: "Colorado",
        country: "US",
      });
      expect(result.current.loc.selectedLocation?.longitude).toBeCloseTo(-105.2705, 4);
      expect(onLocateSuccess).toHaveBeenCalledTimes(1);
      expect(result.current.geo.notice).toEqual({
        type: "status",
        message: "Using approximate location",
      });
    });
  });

  it("TIMEOUT: triggers coarse IP fallback, sets alert if IP fallback returns null", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce(null);

    const { result } = renderHook(
      () => {
        const geo = useGeolocation();
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    const [, errorCb] = mockGetCurrentPosition.mock.calls[0];

    await act(async () => {
      errorCb({
        code: 3, // TIMEOUT
        TIMEOUT: 3,
        message: "Timeout expired",
      });
    });

    expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);

    await waitFor(() => {
      expect(result.current.geo.isLocating).toBe(false);
      expect(result.current.loc.selectedLocation).toBeNull();
      expect(result.current.geo.notice).toEqual({
        type: "alert",
        message: "Location unavailable. Please search for a city or click the map.",
      });
    });
  });

  it("stale browser geolocation result rejected when map click intervenes", () => {
    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    // 1. Locate Me clicked
    act(() => {
      result.current.geo.locateMe();
    });
    const [successCb] = mockGetCurrentPosition.mock.calls[0];

    // 2. User clicks map before geolocation finishes
    act(() => {
      result.current.loc.selectLocation({
        name: "Map Click Point",
        object: "coordinates",
        latitude: 38.0,
        longitude: -106.0,
        elevation_m: null,
        region: null,
        country: null,
        id: null,
        resolvedVia: "coordinates",
      });
    });
    expect(result.current.loc.selectedLocation?.name).toBe("Map Click Point");

    // 3. Delayed geolocation fix arrives
    act(() => {
      successCb({
        coords: {
          latitude: 39.7392,
          longitude: -104.9903,
        },
      });
    });

    // Stale fix must be discarded; Map Click point remains selected!
    expect(result.current.loc.selectedLocation?.name).toBe("Map Click Point");
    expect(onLocateSuccess).not.toHaveBeenCalled();
  });

  it("stale IP fallback result rejected when user selects a search result", async () => {
    let resolveIp!: (value: any) => void;
    mockGetApproximateLocation.mockImplementationOnce(
      () => new Promise((resolve) => (resolveIp = resolve))
    );

    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    act(() => {
      result.current.geo.locateMe();
    });

    // Geolocation times out -> triggers IP fallback
    const [, errorCb] = mockGetCurrentPosition.mock.calls[0];
    act(() => {
      errorCb({ code: 3, TIMEOUT: 3 });
    });

    // User selects a search result while IP fallback is still in-flight
    act(() => {
      result.current.loc.selectLocation({
        name: "Vail Ski Resort",
        object: "ski_resort",
        latitude: 39.64,
        longitude: -106.37,
        elevation_m: 3500,
        region: "Colorado",
        country: "USA",
        id: "resort_vail",
        resolvedVia: "resort",
      });
    });
    expect(result.current.loc.selectedLocation?.name).toBe("Vail Ski Resort");

    // Delayed IP fallback arrives
    await act(async () => {
      resolveIp({
        latitude: 40.0,
        longitude: -105.0,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      });
    });

    // Stale IP fallback must NOT overwrite Vail Ski Resort!
    expect(result.current.loc.selectedLocation?.name).toBe("Vail Ski Resort");
    expect(onLocateSuccess).not.toHaveBeenCalled();
  });

  it("repeated Locate Me clicks: newest token wins, earlier callback rejected", () => {
    const onLocateSuccess = jest.fn();
    const { result } = renderHook(
      () => {
        const geo = useGeolocation({ onLocateSuccess });
        const loc = useSelectedLocation();
        return { geo, loc };
      },
      { wrapper }
    );

    // Click 1
    act(() => {
      result.current.geo.locateMe();
    });
    const [successCb1] = mockGetCurrentPosition.mock.calls[0];

    // Click 2 before 1 finishes
    act(() => {
      result.current.geo.locateMe();
    });
    const [successCb2] = mockGetCurrentPosition.mock.calls[1];

    // Response 2 arrives and commits
    act(() => {
      successCb2({
        coords: {
          latitude: 40.0,
          longitude: -105.0,
        },
      });
    });
    expect(result.current.loc.selectedLocation?.latitude).toBe(40.0);
    expect(onLocateSuccess).toHaveBeenCalledTimes(1);

    // Delayed response 1 arrives
    act(() => {
      successCb1({
        coords: {
          latitude: 30.0,
          longitude: -90.0,
        },
      });
    });

    // Response 1 rejected; response 2 remains!
    expect(result.current.loc.selectedLocation?.latitude).toBe(40.0);
    expect(onLocateSuccess).toHaveBeenCalledTimes(1);
  });
});
