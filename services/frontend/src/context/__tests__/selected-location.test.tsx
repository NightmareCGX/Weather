import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";

import {
  SelectedLocationProvider,
  _resetStartupLocationPromiseForTesting,
  useDisplayTimezone,
  useSelectedLocation,
  useSelectedLocationTimezone,
  useStartupLocation,
} from "@/context/selected-location";
import { getApproximateLocation } from "@/lib/api/client";
import type { ApproximateStartupLocation, SelectedLocation } from "@/lib/api/types";

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

const mockLocationA: SelectedLocation = {
  name: "Denver",
  object: "city",
  id: "city_denver",
  resolvedVia: "city",
  latitude: 39.7392,
  longitude: -104.9903,
  elevation_m: 1609,
  region: "Colorado",
  country: "United States",
};

const mockLocationB: SelectedLocation = {
  name: "Boulder",
  object: "city",
  id: "city_boulder",
  resolvedVia: "city",
  latitude: 40.015,
  longitude: -105.2705,
  elevation_m: 1655,
  region: "Colorado",
  country: "United States",
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <SelectedLocationProvider>{children}</SelectedLocationProvider>
);

beforeEach(() => {
  jest.clearAllMocks();
  _resetStartupLocationPromiseForTesting();
  mockGetApproximateLocation.mockResolvedValue(null);
});

describe("SelectedLocationContext selectionGeneration guard", () => {
  it("initializes with null location and generation 0", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });
    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectionGeneration).toBe(0);
  });

  it("increments selectionGeneration on synchronous selectLocation", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });

    expect(result.current.selectedLocation).toEqual(mockLocationA);
    expect(result.current.selectionGeneration).toBe(1);

    act(() => {
      result.current.selectLocation(mockLocationB);
    });

    expect(result.current.selectedLocation).toEqual(mockLocationB);
    expect(result.current.selectionGeneration).toBe(2);
  });

  it("increments selectionGeneration on clearSelection", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectionGeneration).toBe(1);

    act(() => {
      result.current.clearSelection();
    });

    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectionGeneration).toBe(2);
  });

  it("commits an async selection when no newer action intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });
    expect(token).toBe(1);
    expect(result.current.selectionGeneration).toBe(1);

    let committed = false;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    expect(committed).toBe(true);
    expect(result.current.selectedLocation).toEqual(mockLocationA);
  });

  it("rejects an async selection if a map click or search selection intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    // Step 1: Start async selection (e.g. Locate Me)
    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });
    expect(token).toBe(1);

    // Step 2: User explicitly clicks the map or selects a search result while async is pending
    act(() => {
      result.current.selectLocation(mockLocationB);
    });
    expect(result.current.selectionGeneration).toBe(2);
    expect(result.current.selectedLocation).toEqual(mockLocationB);

    // Step 3: Old async response arrives late
    let committed = true;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    // Stale async selection must be rejected and must NOT overwrite the explicit selection
    expect(committed).toBe(false);
    expect(result.current.selectedLocation).toEqual(mockLocationB);
  });

  it("rejects an async selection if clearSelection intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });

    act(() => {
      result.current.clearSelection();
    });

    let committed = true;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    expect(committed).toBe(false);
    expect(result.current.selectedLocation).toBeNull();
  });
});

describe("SelectedLocationContext selectedTimezone derivation", () => {
  const mockTokyo: SelectedLocation = {
    name: "Tokyo",
    object: "city",
    id: "city_tokyo",
    resolvedVia: "city",
    latitude: 35.6762,
    longitude: 139.6503,
    elevation_m: 40,
    region: "Tokyo",
    country: "Japan",
  };

  it("initializes with null selectedTimezone", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });
    expect(result.current.selectedTimezone).toBeNull();
  });

  it("derives IANA timezone when a location is selected", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });

    expect(result.current.selectedTimezone).toBe("America/Denver");
  });

  it("updates timezone when selected location changes", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectedTimezone).toBe("America/Denver");

    act(() => {
      result.current.selectLocation(mockTokyo);
    });
    expect(result.current.selectedTimezone).toBe("Asia/Tokyo");
  });

  it("resets selectedTimezone to null when selection is cleared", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectedTimezone).toBe("America/Denver");

    act(() => {
      result.current.clearSelection();
    });
    expect(result.current.selectedTimezone).toBeNull();
  });

  it("useSelectedLocationTimezone hook returns timezone and safely falls back to null", () => {
    // Within provider
    const { result } = renderHook(
      () => {
        const loc = useSelectedLocation();
        const tz = useSelectedLocationTimezone();
        return { loc, tz };
      },
      { wrapper }
    );

    expect(result.current.tz).toBeNull();

    act(() => {
      result.current.loc.selectLocation(mockLocationA);
    });
    expect(result.current.tz).toBe("America/Denver");

    // Outside provider
    const outside = renderHook(() => useSelectedLocationTimezone());
    expect(outside.result.current).toBeNull();
  });
});

describe("SelectedLocationContext startupLocation and displayTimezone precedence", () => {
  const mockTokyo: SelectedLocation = {
    name: "Tokyo",
    object: "city",
    id: "city_tokyo",
    resolvedVia: "city",
    latitude: 35.6762,
    longitude: 139.6503,
    elevation_m: 40,
    region: "Tokyo",
    country: "Japan",
  };

  function useLocationHarness() {
    const startup = useStartupLocation();
    const displayTz = useDisplayTimezone();
    const loc = useSelectedLocation();
    return {
      startup,
      displayTz,
      ...loc,
    };
  }

  it("initializes with null startupLocation and null displayTimezone when IP lookup returns null", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce(null);

    let result!: ReturnType<
      typeof renderHook<ReturnType<typeof useLocationHarness>, unknown>
    >["result"];
    await act(async () => {
      const rendered = renderHook(() => useLocationHarness(), { wrapper });
      result = rendered.result;
    });

    await waitFor(() => {
      expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);
    });

    expect(result.current.startupLocation).toBeNull();
    expect(result.current.startupTimezone).toBeNull();
    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectedTimezone).toBeNull();
    expect(result.current.displayTimezone).toBeNull();
    expect(result.current.displayTz).toBeNull();
  });

  it("resolves startupLocation and derives startupTimezone / displayTimezone on IP lookup success", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce({
      latitude: 39.7392,
      longitude: -104.9903,
      city: "Denver",
      region: "Colorado",
      country: "US",
      approximate: true,
    });

    let result!: ReturnType<
      typeof renderHook<ReturnType<typeof useLocationHarness>, unknown>
    >["result"];
    await act(async () => {
      const rendered = renderHook(() => useLocationHarness(), { wrapper });
      result = rendered.result;
    });

    await waitFor(() => {
      expect(result.current.startupLocation).not.toBeNull();
    });

    // Invariant: startup IP location is ambient context and NEVER sets selectedLocation
    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectedTimezone).toBeNull();
    expect(result.current.selectionGeneration).toBe(0);

    // Startup context is derived
    expect(result.current.startupLocation).toMatchObject({
      latitude: 39.7392,
      city: "Denver",
      region: "Colorado",
      country: "US",
      source: "ip",
    });
    expect(result.current.startupLocation?.longitude).toBeCloseTo(-104.9903, 4);
    expect(result.current.startupTimezone).toBe("America/Denver");
    expect(result.current.displayTimezone).toBe("America/Denver");
    expect(result.current.displayTz).toBe("America/Denver");
  });

  it("explicit location selection overrides startupTimezone in displayTimezone", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce({
      latitude: 39.7392,
      longitude: -104.9903,
      city: "Denver",
      region: "Colorado",
      country: "US",
      approximate: true,
    });

    let result!: ReturnType<
      typeof renderHook<ReturnType<typeof useLocationHarness>, unknown>
    >["result"];
    await act(async () => {
      const rendered = renderHook(() => useLocationHarness(), { wrapper });
      result = rendered.result;
    });

    await waitFor(() => {
      expect(result.current.displayTimezone).toBe("America/Denver");
    });

    act(() => {
      result.current.selectLocation(mockTokyo);
    });

    // Tokyo selection overrides Denver startup timezone
    expect(result.current.selectedLocation).toEqual(mockTokyo);
    expect(result.current.selectedTimezone).toBe("Asia/Tokyo");
    expect(result.current.displayTimezone).toBe("Asia/Tokyo");
    expect(result.current.displayTz).toBe("Asia/Tokyo");
  });

  it("clearing selection restores displayTimezone to startupTimezone", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce({
      latitude: 39.7392,
      longitude: -104.9903,
      city: "Denver",
      region: "Colorado",
      country: "US",
      approximate: true,
    });

    let result!: ReturnType<
      typeof renderHook<ReturnType<typeof useLocationHarness>, unknown>
    >["result"];
    await act(async () => {
      const rendered = renderHook(() => useLocationHarness(), { wrapper });
      result = rendered.result;
    });

    await waitFor(() => {
      expect(result.current.displayTimezone).toBe("America/Denver");
    });

    act(() => {
      result.current.selectLocation(mockTokyo);
    });
    expect(result.current.displayTimezone).toBe("Asia/Tokyo");

    act(() => {
      result.current.clearSelection();
    });

    // Selected location is cleared, displayTimezone falls back to startup Denver timezone
    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectedTimezone).toBeNull();
    expect(result.current.displayTimezone).toBe("America/Denver");
    expect(result.current.displayTz).toBe("America/Denver");
  });

  it("gracefully handles IP lookup rejection/error without throwing", async () => {
    mockGetApproximateLocation.mockRejectedValueOnce(new Error("Network connection lost"));

    let result!: ReturnType<
      typeof renderHook<ReturnType<typeof useLocationHarness>, unknown>
    >["result"];
    await act(async () => {
      const rendered = renderHook(() => useLocationHarness(), { wrapper });
      result = rendered.result;
    });

    await waitFor(() => {
      expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);
    });

    expect(result.current.startupLocation).toBeNull();
    expect(result.current.startupTimezone).toBeNull();
    expect(result.current.displayTimezone).toBeNull();
    expect(result.current.displayTz).toBeNull();
  });

  it("useStartupLocation and useDisplayTimezone hooks work within and outside provider", async () => {
    mockGetApproximateLocation.mockResolvedValueOnce({
      latitude: 39.7392,
      longitude: -104.9903,
      city: "Denver",
      region: "Colorado",
      country: "US",
      approximate: true,
    });

    let result!: ReturnType<
      typeof renderHook<
        { startup: ApproximateStartupLocation | null; displayTz: string | null; loc: any },
        unknown
      >
    >["result"];
    await act(async () => {
      const rendered = renderHook(
        () => {
          const startup = useStartupLocation();
          const displayTz = useDisplayTimezone();
          const loc = useSelectedLocation();
          return { startup, displayTz, loc };
        },
        { wrapper }
      );
      result = rendered.result;
    });

    await waitFor(() => {
      expect(result.current.displayTz).toBe("America/Denver");
    });

    expect(result.current.startup).toMatchObject({
      latitude: 39.7392,
      source: "ip",
    });
    expect(result.current.startup?.longitude).toBeCloseTo(-104.9903, 4);
    expect(result.current.displayTz).toBe("America/Denver");

    // Outside provider
    const outsideStartup = renderHook(() => useStartupLocation());
    expect(outsideStartup.result.current).toBeNull();

    const outsideDisplayTz = renderHook(() => useDisplayTimezone());
    expect(outsideDisplayTz.result.current).toBeNull();
  });

  it("Strict Mode safety: multiple mounts share the cached startup promise without duplicate fetch", async () => {
    let resolveLocation!: (value: any) => void;
    const slowPromise = new Promise((resolve) => {
      resolveLocation = resolve;
    });
    mockGetApproximateLocation.mockReturnValueOnce(slowPromise as any);

    // Mount 1 (first mount)
    const { unmount } = renderHook(() => useLocationHarness(), { wrapper });
    await waitFor(() => {
      expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);
    });

    // Simulate Strict Mode unmount before promise resolves
    unmount();

    // Mount 2 (remount)
    const { result: result2 } = renderHook(() => useLocationHarness(), { wrapper });
    // Must NOT fire another request
    expect(mockGetApproximateLocation).toHaveBeenCalledTimes(1);

    // Resolve the single in-flight request
    await act(async () => {
      resolveLocation({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      });
    });

    await waitFor(() => {
      expect(result2.current.startupLocation).not.toBeNull();
    });

    expect(result2.current.startupLocation?.city).toBe("Denver");
    expect(result2.current.displayTimezone).toBe("America/Denver");
  });
});
