import { act, renderHook, waitFor } from "@testing-library/react";

import { useMapLayer } from "@/hooks/useMapLayer";
import { useForecastSelection } from "@/context/forecast-selection";
import type { ForecastSelection } from "@/lib/forecast/availability";
import type { SpatialLayer } from "@/lib/api/types";

jest.mock("../../context/forecast-selection");

const mockUseForecastSelection = useForecastSelection as jest.MockedFunction<
  typeof useForecastSelection
>;

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function layerResponse(layer: SpatialLayer): Response {
  return jsonResponse({
    object: "spatial_layer",
    data: layer,
    has_more: false,
    next_cursor: null,
  });
}

const selectionA: ForecastSelection = {
  model: "gfs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 6,
};

const layerA: SpatialLayer = {
  tile_url_template:
    "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=6&initial_time=2026-08-13T00:00:00Z",
  min_zoom: 0,
  max_zoom: 9,
  lead_time_hours: 6,
  legend: {
    unit: "°C",
    stops: [
      [-40, "#0000ff"],
      [40, "#ff0000"],
    ],
  },
};

const selectionB: ForecastSelection = {
  model: "gfs",
  variable: "precipitation_rate",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 6,
};

const layerB: SpatialLayer = {
  tile_url_template:
    "/v1/maps/gfs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours=6&initial_time=2026-08-13T00:00:00Z",
  min_zoom: 0,
  max_zoom: 9,
  lead_time_hours: 6,
  legend: {
    unit: "mm/h",
    stops: [
      [0, "#ffffff"],
      [50, "#0000ff"],
    ],
  },
};

const selectionC: ForecastSelection = {
  model: "gefs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 12,
};

const layerC: SpatialLayer = {
  tile_url_template:
    "/v1/maps/gefs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12&initial_time=2026-08-13T00:00:00Z",
  min_zoom: 0,
  max_zoom: 9,
  lead_time_hours: 12,
  legend: {
    unit: "°C",
    stops: [
      [-40, "#313695"],
      [45, "#a50026"],
    ],
  },
};

function mockContextValue(selection: ForecastSelection | null) {
  return {
    availability: null,
    status: "success" as const,
    error: null,
    selection,
    validTime: null,
    options: {
      models: [],
      model: null,
      variables: [],
      initialTimes: [],
      variable: null,
      initialTime: null,
      leadTimes: [],
    },
    setModel: jest.fn(),
    setVariable: jest.fn(),
    setInitialTime: jest.fn(),
    setLeadTimeHours: jest.fn(),
    retry: jest.fn(),
  };
}

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("useMapLayer", () => {
  it("is idle with null selection", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(null));
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.layer).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("fetches metadata for active selection", async () => {
    mockFetch.mockResolvedValueOnce(layerResponse(layerA));
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));

    const { result } = renderHook(() => useMapLayer());

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining(
        "/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=6&initial_time=2026-08-13T00%3A00%3A00Z"
      ),
      expect.any(Object)
    );
    expect(result.current.layer).toEqual(layerA);
    expect(result.current.error).toBeNull();
  });

  it("starts request B immediately while request A is still pending", async () => {
    let resolveA!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveA = resolve))
    );
    mockFetch.mockResolvedValueOnce(layerResponse(layerB));

    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.loading).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Switch selection to B before A resolves
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionB));
    rerender();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch).toHaveBeenLastCalledWith(
      expect.stringContaining("variable=precipitation_rate"),
      expect.any(Object)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.layer).toEqual(layerB);

    // Clean up unresolved promise
    resolveA(layerResponse(layerA));
  });

  it("never allows a stale A success to overwrite B", async () => {
    let resolveA!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveA = resolve))
    );
    mockFetch.mockResolvedValueOnce(layerResponse(layerB));

    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));
    const { result, rerender } = renderHook(() => useMapLayer());

    // Switch to B
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionB));
    rerender();

    // B succeeds
    await waitFor(() => expect(result.current.layer).toEqual(layerB));
    expect(result.current.loading).toBe(false);

    // Stale A resolves late
    await act(async () => {
      resolveA(layerResponse(layerA));
    });

    // Layer must remain B
    expect(result.current.layer).toEqual(layerB);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("never allows a stale A error to affect B", async () => {
    let rejectA!: (reason: unknown) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((_, reject) => (rejectA = reject))
    );
    mockFetch.mockResolvedValueOnce(layerResponse(layerB));

    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));
    const { result, rerender } = renderHook(() => useMapLayer());

    // Switch to B
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionB));
    rerender();

    // B succeeds
    await waitFor(() => expect(result.current.layer).toEqual(layerB));
    expect(result.current.error).toBeNull();

    // Stale A rejects with a non-abort network error
    await act(async () => {
      rejectA(new TypeError("Network error"));
    });

    // Layer remains B and no error is surfaced
    expect(result.current.layer).toEqual(layerB);
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("never allows a stale A completion to clear B loading state", async () => {
    let resolveA!: (value: Response) => void;
    let resolveB!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveA = resolve))
    );
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveB = resolve))
    );

    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));
    const { result, rerender } = renderHook(() => useMapLayer());

    // Switch to B; both A and B are now pending
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionB));
    rerender();

    expect(result.current.loading).toBe(true);

    // A completes while B is still pending
    await act(async () => {
      resolveA(layerResponse(layerA));
    });

    // Loading must still be true because B is pending
    expect(result.current.loading).toBe(true);

    // Now B completes
    await act(async () => {
      resolveB(layerResponse(layerB));
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.layer).toEqual(layerB);
  });

  it("leaves only C authoritative in a rapid A -> B -> C sequence", async () => {
    let resolveA!: (value: Response) => void;
    let rejectB!: (reason: unknown) => void;
    let resolveC!: (value: Response) => void;

    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveA = resolve))
    );
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((_, reject) => (rejectB = reject))
    );
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveC = resolve))
    );

    // A starts
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));
    const { result, rerender } = renderHook(() => useMapLayer());

    // Switch to B
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionB));
    rerender();

    // Switch to C
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionC));
    rerender();

    expect(result.current.loading).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(3);

    // Out of order: A resolves, B rejects with 500 error, C resolves
    await act(async () => {
      resolveA(layerResponse(layerA));
    });
    expect(result.current.layer).toBeNull();
    expect(result.current.loading).toBe(true);

    await act(async () => {
      rejectB(new TypeError("Server error 500"));
    });
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(true);

    await act(async () => {
      resolveC(layerResponse(layerC));
    });

    // Only C's result commits
    expect(result.current.loading).toBe(false);
    expect(result.current.layer).toEqual(layerC);
    expect(result.current.error).toBeNull();
  });

  it("surfaces an error message when the active request fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionA));

    const { result } = renderHook(() => useMapLayer());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.layer).toBeNull();
    expect(result.current.error).toBe("Network request failed.");
  });
});
