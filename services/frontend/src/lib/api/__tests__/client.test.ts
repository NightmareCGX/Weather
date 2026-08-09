import { ApiError, getMapLayer, listModels } from "@/lib/api/client";

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("listModels", () => {
  it("requests /v1/models and returns typed models from the envelope", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "list",
        data: [
          {
            id: "gfs",
            object: "model",
            name: "Global Forecast System",
            center_id: "noaa",
            is_ensemble: false,
            resolution_km: 25,
          },
        ],
        has_more: false,
        next_cursor: null,
      })
    );

    const models = await listModels();

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Accept: "application/json" }),
      })
    );
    expect(models).toHaveLength(1);
    expect(models[0].id).toBe("gfs");
    expect(models[0].is_ensemble).toBe(false);
  });

  it("passes filter query params", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ object: "list", data: [], has_more: false, next_cursor: null })
    );

    await listModels({ center_id: "noaa", is_ensemble: true });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/models?center_id=noaa&is_ensemble=true",
      expect.any(Object)
    );
  });

  it("throws ApiError with status 0 on network failure", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await expect(listModels()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      type: "network_error",
    });
  });
});

describe("getMapLayer", () => {
  it("requests /v1/maps with the required query string", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "spatial_layer",
        data: {
          tile_url_template:
            "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12",
          min_zoom: 0,
          max_zoom: 9,
          lead_time_hours: 12,
          legend: { unit: "°C", stops: [[-40, "#0000ff"]] },
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const layer = await getMapLayer({
      model: "gfs",
      variable: "temperature_2m",
      leadTimeHours: 12,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=12",
      expect.any(Object)
    );
    expect(layer.lead_time_hours).toBe(12);
    expect(layer.legend.unit).toBe("°C");
  });

  it("throws ApiError carrying RFC 7807 fields on non-2xx", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        {
          error: {
            code: "invalid_request_error",
            type: "validation_error",
            message: "Invalid level",
            param: "level",
            request_id: "req_123",
          },
        },
        422
      )
    );

    const error = await getMapLayer({
      model: "gfs",
      variable: "temperature_2m",
      leadTimeHours: 12,
    }).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      name: "ApiError",
      status: 422,
      code: "invalid_request_error",
      type: "validation_error",
      message: "Invalid level",
      param: "level",
      requestId: "req_123",
    });
  });
});
