import {
  ApiError,
  getEnsembleStatistics,
  getForecastAvailability,
  getMapLayer,
  getPointForecast,
  listModels,
  listVariables,
  RequestAbortedError,
  searchLocations,
} from "@/lib/api/client";

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

function envelopeList(body: unknown) {
  return jsonResponse({ object: "list", data: body, has_more: false, next_cursor: null });
}

describe("searchLocations", () => {
  it("requests /v1/search with q, default type=all, and returns results", async () => {
    mockFetch.mockResolvedValueOnce(
      envelopeList([
        {
          id: "city_aspen",
          object: "city",
          name: "Aspen",
          region: "Colorado",
          country: "USA",
          elevation_m: null,
          latitude: 38.19,
          longitude: -106.82,
        },
      ])
    );

    const results = await searchLocations({ q: "Aspen" });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/search?q=Aspen&type=all",
      expect.objectContaining({ headers: expect.objectContaining({ Accept: "application/json" }) })
    );
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe("city_aspen");
    expect(results[0].object).toBe("city");
  });

  it("passes type and limit and an abort signal", async () => {
    mockFetch.mockResolvedValueOnce(envelopeList([]));
    const controller = new AbortController();

    await searchLocations({ q: "Aspen", type: "resort", limit: 5, signal: controller.signal });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/search?q=Aspen&type=resort&limit=5",
      expect.objectContaining({ signal: controller.signal })
    );
  });
});

describe("getPointForecast", () => {
  it("queries by coordinates with models and units", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "point_forecast",
        data: {
          location: {
            latitude: 38.19,
            longitude: -106.82,
            elevation_m: null,
            resolved_via: "coordinates",
          },
          generated_at: "2026-07-21T00:00:00Z",
          model: "gfs",
          forecasts: [
            { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 15.0 },
          ],
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const forecast = await getPointForecast({
      location: { type: "coordinates", latitude: 38.19, longitude: -106.82 },
      variables: ["temperature_2m"],
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/points?models=gfs&units=metric&lat=38.19&lon=-106.82&variables=temperature_2m",
      expect.any(Object)
    );
    expect(forecast.model).toBe("gfs");
    expect(forecast.forecasts[0].temperature_2m).toBe(15.0);
  });

  it("uses city_id / resort_id specifiers and lead-time window", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "point_forecast",
        data: {
          location: {
            latitude: 38.19,
            longitude: -106.82,
            elevation_m: 3417,
            resolved_via: "resort",
          },
          generated_at: "2026-07-21T00:00:00Z",
          model: "gfs",
          forecasts: [],
        },
        has_more: false,
        next_cursor: null,
      })
    );

    await getPointForecast({
      location: { type: "resort", resortId: "resort_aspen_mountain" },
      startLeadTimeHours: 0,
      endLeadTimeHours: 18,
    });
    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/points?models=gfs&units=metric&resort_id=resort_aspen_mountain&start_lead_time_hours=0&end_lead_time_hours=18",
      expect.any(Object)
    );

    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "point_forecast",
        data: {
          location: {
            latitude: 38.19,
            longitude: -106.82,
            elevation_m: null,
            resolved_via: "city",
          },
          generated_at: "2026-07-21T00:00:00Z",
          model: "gfs",
          forecasts: [],
        },
        has_more: false,
        next_cursor: null,
      })
    );
    await getPointForecast({ location: { type: "city", cityId: "city_aspen" } });
    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/points?models=gfs&units=metric&city_id=city_aspen",
      expect.any(Object)
    );
  });
});

describe("getEnsembleStatistics", () => {
  it("requests /v1/ensembles with lat/lon/variable/model/lead and returns stats", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "ensemble_statistics",
        data: {
          model: "gefs",
          lead_time_hours: 6,
          member_count: 5,
          statistics: {
            mean: 17.5,
            median: 17.5,
            spread: 3.16,
            p10: 13.9,
            p25: 15.5,
            p50: 17.5,
            p75: 19.5,
            p90: 21.1,
          },
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const data = await getEnsembleStatistics({
      latitude: 38.19,
      longitude: -106.82,
      variable: "temperature_2m",
      leadTimeHours: 6,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&lead_time_hours=6",
      expect.any(Object)
    );
    expect(data.member_count).toBe(5);
    expect(data.statistics.p50).toBe(17.5);
    // The statistics-only default omits members.
    expect(data.members).toBeUndefined();
  });

  it("requests include_members=true when opted in and returns members", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "ensemble_statistics",
        data: {
          model: "gefs",
          lead_time_hours: 6,
          member_count: 5,
          statistics: {
            mean: 17.5,
            median: 17.5,
            spread: 3.16,
            p10: 13.9,
            p25: 15.5,
            p50: 17.5,
            p75: 19.5,
            p90: 21.1,
          },
          members: [15.5, 17.5, 19.5, 21.5, 23.5],
        },
        has_more: false,
        next_cursor: null,
      })
    );

    const data = await getEnsembleStatistics({
      latitude: 38.19,
      longitude: -106.82,
      variable: "temperature_2m",
      leadTimeHours: 6,
      includeMembers: true,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/ensembles?lat=38.19&lon=-106.82&variable=temperature_2m&model=gefs&lead_time_hours=6&include_members=true",
      expect.any(Object)
    );
    expect(data.members).toEqual([15.5, 17.5, 19.5, 21.5, 23.5]);
    expect(data.members?.length).toBe(5);
  });
});

describe("listVariables", () => {
  it("requests /v1/variables and returns catalog resources", async () => {
    mockFetch.mockResolvedValueOnce(
      envelopeList([
        { id: "temperature_2m", object: "variable", name: "2-Meter Temperature", unit: "°C" },
      ])
    );

    const variables = await listVariables();

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/variables",
      expect.objectContaining({ headers: expect.objectContaining({ Accept: "application/json" }) })
    );
    expect(variables[0].unit).toBe("°C");
  });
});

describe("getForecastAvailability", () => {
  it("requests /v1/forecast/availability with cache: 'no-cache' for fresh discovery", async () => {
    const availabilityData = {
      models: [
        {
          id: "gfs",
          name: "Global Forecast System",
          is_ensemble: false,
          variables: [
            {
              id: "temperature_2m",
              name: "2-Meter Temperature",
              unit: "°C",
              initial_times: [
                {
                  value: "2026-07-21T00:00:00Z",
                  lead_time_hours: [0, 6, 12, 18],
                },
              ],
            },
          ],
        },
      ],
    };

    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        object: "forecast_availability",
        data: availabilityData,
        has_more: false,
        next_cursor: null,
      })
    );

    const controller = new AbortController();
    const result = await getForecastAvailability(controller.signal);

    expect(mockFetch).toHaveBeenCalledWith(
      "/v1/forecast/availability",
      expect.objectContaining({
        cache: "no-cache",
        signal: controller.signal,
        headers: expect.objectContaining({ Accept: "application/json" }),
      })
    );
    expect(result).toEqual(availabilityData);
  });
});

describe("request cancellation", () => {
  it("throws RequestAbortedError when the fetch aborts", async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.reject(new DOMException("Aborted", "AbortError"))
    );

    const error = await searchLocations({ q: "x", signal: new AbortController().signal }).catch(
      (err: unknown) => err
    );

    expect(error).toBeInstanceOf(RequestAbortedError);
  });

  it("throws ApiError with status 0 on a genuine network failure", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const error = await searchLocations({ q: "x" }).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 0, type: "network_error" });
  });
});
