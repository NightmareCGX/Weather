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

function distResponse(lead: number, members?: number[]) {
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

describe("useEnsembleDistribution", () => {
  it("requests a single lead with include_members=true and returns the data", async () => {
    mockFetch.mockResolvedValueOnce(distResponse(6, [15.5, 17.5, 19.5, 21.5, 23.5]));

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
});
