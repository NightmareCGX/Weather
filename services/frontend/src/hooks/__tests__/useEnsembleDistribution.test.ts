import { renderHook, waitFor } from "@testing-library/react";
import { useEnsembleDistribution } from "@/hooks/useEnsembleDistribution";
import type { SelectedLocation } from "@/lib/api/types";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function distResponse(
  lead: number,
  members?: number[],
  pdf?: { x: number[]; density: number[] } | null
) {
  return jsonResponse({
    object: "ensemble_statistics",
    data: {
      model: "gefs",
      lead_time_hours: lead,
      member_count: members?.length ?? 5,
      statistics: {
        mean: 10 + lead,
        median: 10 + lead,
        spread: 2,
        p10: 8 + lead,
        p25: 9 + lead,
        p50: 10 + lead,
        p75: 11 + lead,
        p90: 12 + lead,
      },
      ...(members !== undefined ? { members } : {}),
      ...(pdf !== undefined ? { pdf } : {}),
    },
    has_more: false,
    next_cursor: null,
  });
}

const location: SelectedLocation = {
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

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("useEnsembleDistribution", () => {
  it("requests a single lead with include_members=true and returns the data", async () => {
    const samplePdf = {
      x: [10.0, 15.0, 20.0, 25.0, 30.0],
      density: [0.01, 0.05, 0.2, 0.05, 0.01],
    };
    mockFetch.mockResolvedValueOnce(distResponse(6, [15.5, 17.5, 19.5, 21.5, 23.5], samplePdf));

    const { result } = renderHook(() =>
      useEnsembleDistribution(location, 6, "temperature_2m", { model: "gefs" })
    );

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&lead_time_hours=6&include_members=true",
      expect.any(Object)
    );
    expect(result.current.data?.members).toEqual([15.5, 17.5, 19.5, 21.5, 23.5]);
    expect(result.current.data?.member_count).toBe(5);
    expect(result.current.data?.pdf).toEqual(samplePdf);
  });

  it("is idle with no selection", () => {
    const { result } = renderHook(() =>
      useEnsembleDistribution(null, 0, "temperature_2m", { model: "gefs" })
    );
    expect(result.current.status).toBe("idle");
    expect(result.current.data).toBeNull();
  });

  it("stays idle when no ensemble model is selected", () => {
    const { result } = renderHook(() =>
      useEnsembleDistribution(location, 6, "temperature_2m", { model: null })
    );
    expect(result.current.status).toBe("idle");
    expect(result.current.data).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("surfaces an error when the request fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const { result } = renderHook(() =>
      useEnsembleDistribution(location, 6, "temperature_2m", { model: "gefs" })
    );
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.data).toBeNull();
  });

  it("refetches with the new lead when leadTimeHours changes (0 -> 24)", async () => {
    mockFetch.mockResolvedValueOnce(distResponse(0, [10, 11, 12, 13, 14]));
    mockFetch.mockResolvedValueOnce(distResponse(24, [34, 35, 36, 37, 38]));

    const { result, rerender } = renderHook(
      ({ lead }: { lead: number }) =>
        useEnsembleDistribution(location, lead, "temperature_2m", { model: "gefs" }),
      { initialProps: { lead: 0 } }
    );

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      "/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&lead_time_hours=0&include_members=true",
      expect.any(Object)
    );
    expect(result.current.data?.lead_time_hours).toBe(0);
    expect(result.current.data?.members).toEqual([10, 11, 12, 13, 14]);

    // Rerender with non-zero lead (+24h)
    rerender({ lead: 24 });
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      "/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&lead_time_hours=24&include_members=true",
      expect.any(Object)
    );
    expect(result.current.data?.lead_time_hours).toBe(24);
    expect(result.current.data?.members).toEqual([34, 35, 36, 37, 38]);
  });

  it("discards a slow response from an older lead time so it cannot overwrite newer lead state", async () => {
    let resolveLead0!: (res: Response) => void;
    const lead0Promise = new Promise<Response>((resolve) => {
      resolveLead0 = resolve;
    });

    // Lead 0 hangs initially
    mockFetch.mockImplementationOnce(() => lead0Promise);
    // Lead 24 resolves immediately
    mockFetch.mockResolvedValueOnce(distResponse(24, [34, 35, 36, 37, 38]));

    const { result, rerender } = renderHook(
      ({ lead }: { lead: number }) =>
        useEnsembleDistribution(location, lead, "temperature_2m", { model: "gefs" }),
      { initialProps: { lead: 0 } }
    );

    expect(result.current.status).toBe("loading");

    // Switch lead to +24h while +0h is still pending
    rerender({ lead: 24 });
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data?.lead_time_hours).toBe(24);
    expect(result.current.data?.members).toEqual([34, 35, 36, 37, 38]);

    // Now resolve the older +0h request
    resolveLead0(distResponse(0, [10, 11, 12, 13, 14]));
    // Wait a tick to ensure any pending promises settle
    await new Promise((r) => setTimeout(r, 20));

    // Stale +0h response must not overwrite current +24h state
    expect(result.current.data?.lead_time_hours).toBe(24);
    expect(result.current.data?.members).toEqual([34, 35, 36, 37, 38]);
  });

  it("Test F: clears previous distribution data immediately upon parameter transition", async () => {
    mockFetch.mockResolvedValueOnce(distResponse(0, [10, 11, 12]));

    let resolveLead12!: (value: Response) => void;
    mockFetch.mockImplementationOnce(() => new Promise<Response>((r) => (resolveLead12 = r)));

    const { result, rerender } = renderHook(
      ({ lead }: { lead: number }) =>
        useEnsembleDistribution(location, lead, "temperature_2m", { model: "gefs" }),
      { initialProps: { lead: 0 } }
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).not.toBeNull();

    // Rerender with lead: 12
    rerender({ lead: 12 });

    // Assert previous data is cleared immediately and status is loading
    expect(result.current.status).toBe("loading");
    expect(result.current.data).toBeNull();

    // Settle the second request
    resolveLead12(distResponse(12, [20, 21, 22]));

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data?.lead_time_hours).toBe(12);
  });

  it("requests with valid_time parameter when a string timestamp is passed", async () => {
    mockFetch.mockResolvedValueOnce(distResponse(6, [15, 16, 17]));

    const { result } = renderHook(() =>
      useEnsembleDistribution(location, "2026-09-04T06:00:00Z", "temperature_2m", { model: "gefs" })
    );

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("valid_time=2026-09-04T06%3A00%3A00Z"),
      expect.any(Object)
    );
  });
});
