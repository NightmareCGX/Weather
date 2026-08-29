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

function statsResponse(lead: number, members?: number[]) {
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

describe("useEnsemble", () => {
  it("fans out one request per lead and returns statistics by lead", async () => {
    mockFetch.mockResolvedValueOnce(statsResponse(0));
    mockFetch.mockResolvedValueOnce(statsResponse(6));
    mockFetch.mockResolvedValueOnce(statsResponse(12));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(mockFetch).toHaveBeenCalledTimes(3);
    expect(result.current.byLead.has(0)).toBe(true);
    expect(result.current.byLead.get(6)?.statistics.mean).toBe(16);
    expect(result.current.byLead.get(12)?.statistics.mean).toBe(22);
  });

  it("tolerates individual lead failures and still succeeds with the rest", async () => {
    mockFetch.mockResolvedValueOnce(statsResponse(0));
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    mockFetch.mockResolvedValueOnce(statsResponse(12));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.byLead.has(0)).toBe(true);
    expect(result.current.byLead.has(6)).toBe(false);
    expect(result.current.byLead.has(12)).toBe(true);
  });

  it("errors when every lead fails", async () => {
    mockFetch.mockRejectedValue(new TypeError("Failed to fetch"));

    const { result } = renderHook(() =>
      useEnsemble(location, [0], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.byLead.size).toBe(0);
    expect(result.current.error).not.toBeNull();
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
    // Regression: a metadata/coordinate field such as `cycle_time` must never
    // reach the API as `variable=…`. A single polluted `variable` fans out
    // invalid 404 requests across EVERY lead time, so assert the exact query
    // string carried by every request in a multi-lead fan-out.
    mockFetch.mockResolvedValueOnce(statsResponse(0));
    mockFetch.mockResolvedValueOnce(statsResponse(6));
    mockFetch.mockResolvedValueOnce(statsResponse(12));

    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6, 12], "temperature_2m", { model: "gefs" })
    );

    await waitFor(() => expect(result.current.status).toBe("success"));

    expect(mockFetch).toHaveBeenCalledTimes(3);
    const queriedVariables = mockFetch.mock.calls.map(([input]) => {
      const url = new URL(String(input), "http://localhost");
      return url.searchParams.get("variable");
    });
    expect(queriedVariables).toEqual(["temperature_2m", "temperature_2m", "temperature_2m"]);
    expect(queriedVariables).not.toContain("cycle_time");
    expect(queriedVariables).not.toContain("lead_time_hours");
    expect(queriedVariables).not.toContain("valid_time");
  });

  it("stays idle when no ensemble model is selected", async () => {
    const { result } = renderHook(() =>
      useEnsemble(location, [0, 6], "temperature_2m", { model: null })
    );

    expect(result.current.status).toBe("idle");
    expect(result.current.byLead.size).toBe(0);
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
