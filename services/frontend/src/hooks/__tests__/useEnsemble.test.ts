import { renderHook, waitFor } from "@testing-library/react";
import { useEnsemble } from "@/hooks/useEnsemble";
import type { SelectedLocation } from "@/lib/api/types";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function batchStatsResponse(leads: number[]) {
  return jsonResponse({
    object: "ensemble_statistics",
    data: leads.map((lead) => ({
      model: "gefs",
      lead_time_hours: lead,
      member_count: 5,
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
    })),
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

describe("useEnsemble", () => {
  it("requests batch series for leads and returns statistics by lead", async () => {
    mockFetch.mockResolvedValueOnce(batchStatsResponse([0, 6, 12]));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(result.current.byLead.has(0)).toBe(true);
    expect(result.current.byLead.get(6)?.statistics.mean).toBe(16);
    expect(result.current.byLead.get(12)?.statistics.mean).toBe(22);
  });

  it("tolerates individual missing leads in batch and still succeeds with the rest", async () => {
    mockFetch.mockResolvedValueOnce(batchStatsResponse([0, 12]));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(result.current.byLead.has(0)).toBe(true);
    expect(result.current.byLead.has(6)).toBe(false);
    expect(result.current.byLead.has(12)).toBe(true);
  });

  it("errors when request fails", async () => {
    mockFetch.mockRejectedValue(new TypeError("Failed to fetch"));

    const { result } = renderHook(() =>
      useEnsemble(location, [0], "temperature_2m", { model: "gefs", retryDelaysMs: [] })
    );

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.byLead.size).toBe(0);
    expect(result.current.error).not.toBeNull();
  });

  it("splits the series into batches and accumulates every batch into byLead", async () => {
    mockFetch
      .mockResolvedValueOnce(batchStatsResponse([0, 6]))
      .mockResolvedValueOnce(batchStatsResponse([12, 18]));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12, 18], "temperature_2m", { model: "gefs", batchSize: 2 })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(Array.from(result.current.byLead.keys()).sort((a, b) => a - b)).toEqual([0, 6, 12, 18]);

    // Each request must carry only its own slice of leads.
    const firstUrl = new URL(String(mockFetch.mock.calls[0][0]), "http://localhost");
    const secondUrl = new URL(String(mockFetch.mock.calls[1][0]), "http://localhost");
    expect(firstUrl.searchParams.get("leads")).toBe("0,6");
    expect(secondUrl.searchParams.get("leads")).toBe("12,18");
  });

  it("publishes the first batch while later batches are still in flight", async () => {
    let resolveSecond!: (value: Response) => void;
    mockFetch
      .mockResolvedValueOnce(batchStatsResponse([0, 6]))
      .mockImplementationOnce(() => new Promise<Response>((r) => (resolveSecond = r)));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12, 18], "temperature_2m", { model: "gefs", batchSize: 2 })
    );

    // The leading days become renderable before the tail finishes.
    await waitFor(() => expect(result.current.byLead.size).toBe(2));
    expect(result.current.status).toBe("loading");
    expect(result.current.byLead.has(0)).toBe(true);
    expect(result.current.byLead.has(12)).toBe(false);

    resolveSecond(batchStatsResponse([12, 18]));

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.byLead.size).toBe(4);
  });

  it("retries a failed batch and keeps the recovered result", async () => {
    mockFetch
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(batchStatsResponse([0, 6]));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6], "temperature_2m", { model: "gefs", retryDelaysMs: [0] })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result.current.byLead.size).toBe(2);
  });

  it("keeps the partial series when a later batch fails after its retries", async () => {
    mockFetch
      .mockResolvedValueOnce(batchStatsResponse([0, 6]))
      .mockRejectedValue(new TypeError("Failed to fetch"));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12, 18], "temperature_2m", {
        model: "gefs",
        batchSize: 2,
        retryDelaysMs: [],
      })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(Array.from(result.current.byLead.keys()).sort((a, b) => a - b)).toEqual([0, 6]);
    expect(result.current.error).toBeNull();
  });

  it("stops issuing later batches when the first batch fails with nothing on screen", async () => {
    mockFetch.mockRejectedValue(new TypeError("Failed to fetch"));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12, 18], "temperature_2m", {
        model: "gefs",
        batchSize: 2,
        retryDelaysMs: [],
      })
    );

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(result.current.byLead.size).toBe(0);
  });

  it("stays idle and does not issue premature requests when leads array is empty", () => {
    const { result } = renderHook(() =>
      useEnsemble(location, [], "temperature_2m", { model: "gefs" })
    );

    expect(result.current.status).toBe("idle");
    expect(result.current.byLead.size).toBe(0);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("never issues a request with a metadata/coordinate field as the variable", async () => {
    mockFetch.mockResolvedValueOnce(batchStatsResponse([0, 6, 12]));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [input] = mockFetch.mock.calls[0];
    const url = new URL(String(input), "http://localhost");
    expect(url.searchParams.get("variable")).toBe("temperature_2m");
    expect(url.searchParams.get("leads")).toBe("0,6,12");
    expect(url.searchParams.has("cycle_time")).toBe(false);
    expect(url.searchParams.has("valid_time")).toBe(false);
  });

  it("stays idle when no ensemble model is selected", async () => {
    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6], "temperature_2m", { model: null })
    );

    expect(result.current.status).toBe("idle");
    expect(result.current.byLead.size).toBe(0);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("Test F: clears previous byLead map immediately upon variable parameter transition", async () => {
    mockFetch.mockResolvedValueOnce(batchStatsResponse([0, 6]));

    let resolveTemp!: (value: Response) => void;
    mockFetch.mockImplementationOnce(() => new Promise<Response>((r) => (resolveTemp = r)));

    const { result, rerender } = renderHook(
      ({ variable }: { variable: string }) =>
        useEnsemble(location, [0, 6], variable, { model: "gefs" }),
      { initialProps: { variable: "precipitation_rate" } }
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.byLead.size).toBe(2);

    // Rerender with variable: "temperature_2m"
    rerender({ variable: "temperature_2m" });

    // Assert previous data is cleared immediately and status is loading
    expect(result.current.status).toBe("loading");
    expect(result.current.byLead.size).toBe(0);

    // Resolve the pending temperature responses
    resolveTemp(batchStatsResponse([0, 6]));

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.byLead.size).toBe(2);
  });
});
