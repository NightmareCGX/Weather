import type { Page } from "@playwright/test";

/**
 * Deterministic `/v1/*` API mocks for Playwright E2E tests.
 *
 * The fixtures mirror the repository's API test fixtures
 * (`services/api/tests/fixtures`): GFS (deterministic `temperature_2m`,
 * `precipitation_rate`, leads [0, 6, 12, 18]) and GEFS (the same variables
 * with a `member` axis). Responses use the universal envelope and RFC 7807
 * error shapes.
 */

export interface MockOptions {
  /** When true, requests reach the real backend instead of mocks. */
  live?: boolean;
}

export const LEAD_TIMES = [0, 6, 12, 18];

export function temperatureAt(lat: number, lon: number, lead: number): number {
  return 10 + 10 * (lat - 38) + 10 * (lon - -107) + 0.5 * lead;
}

export function precipitationAt(lead: number): number {
  return 0.5 * lead;
}

export function ensembleTemperatureAt(
  member: number,
  lat: number,
  lon: number,
  lead: number
): number {
  return temperatureAt(lat, lon, lead) + 2 * member;
}

export const GEFS_MEMBERS = [0, 1, 2, 3, 4];

function envelope(data: unknown, object: string) {
  return { object, data, has_more: false, next_cursor: null };
}

export async function installApiMocks(page: Page, options: MockOptions = {}): Promise<void> {
  if (options.live) {
    return;
  }

  await page.route("**/v1/models", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          [
            {
              id: "gfs",
              object: "model",
              name: "Global Forecast System",
              center_id: "noaa",
              is_ensemble: false,
              resolution_km: 25,
            },
            {
              id: "gefs",
              object: "model",
              name: "Global Ensemble Forecast System",
              center_id: "noaa",
              is_ensemble: true,
              resolution_km: 25,
            },
          ],
          "list"
        )
      ),
    })
  );

  await page.route("**/v1/maps?*", (route) => {
    const url = new URL(route.request().url());
    const lead = url.searchParams.get("lead_time_hours") ?? "12";
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          {
            tile_url_template: `/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=${lead}`,
            min_zoom: 0,
            max_zoom: 9,
            lead_time_hours: Number(lead),
            legend: {
              unit: "°C",
              stops: [
                [-40, "#0000ff"],
                [0, "#00ff00"],
                [40, "#ff0000"],
              ],
            },
          },
          "spatial_layer"
        )
      ),
    });
  });

  // The weather layer's tile template points at `/v1/maps/{model}/{variable}/
  // {level}/{z}/{x}/{y}.png`. The backend serves no tile imagery, so fulfill
  // those image requests with a 204 so MapLibre does not log tile errors or
  // cascade coordinate-resolution reads while the base map keeps rendering.
  await page.route("**/v1/maps/**/*.png?*", (route) => route.fulfill({ status: 204, body: "" }));
  await page.route("**/v1/maps/**/*.png", (route) => route.fulfill({ status: 204, body: "" }));

  // The base style uses OSM raster tiles; block them offline too.
  await page.route("https://tile.openstreetmap.org/**", (route) =>
    route.fulfill({ status: 204, body: "" })
  );

  await page.route("**/v1/variables", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          [
            { id: "temperature_2m", object: "variable", name: "2-Meter Temperature", unit: "°C" },
            {
              id: "precipitation_rate",
              object: "variable",
              name: "Precipitation Rate",
              unit: "mm/h",
            },
          ],
          "list"
        )
      ),
    })
  );

  await page.route("**/v1/search?*", (route) => {
    const url = new URL(route.request().url());
    const q = (url.searchParams.get("q") ?? "").toLowerCase();
    const all = [
      {
        id: "city_denver",
        object: "city",
        name: "Denver",
        region: "Colorado",
        country: "USA",
        elevation_m: null,
        latitude: 39.7392,
        longitude: -104.9903,
      },
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
      {
        id: "resort_aspen_mountain",
        object: "ski_resort",
        name: "Aspen Mountain",
        region: "Colorado",
        country: "USA",
        elevation_m: 3417,
        latitude: 38.19,
        longitude: -106.82,
      },
    ];
    const matches = all.filter((item) => item.name.toLowerCase().includes(q));
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(envelope(matches, "list")),
    });
  });

  await page.route("**/v1/points?*", (route) => {
    const url = new URL(route.request().url());
    const lat = Number(url.searchParams.get("lat") ?? 38.19);
    const lon = Number(url.searchParams.get("lon") ?? -106.82);
    const forecasts = LEAD_TIMES.map((lead) => ({
      lead_time_hours: lead,
      valid_time: `2026-07-21T${String(lead).padStart(2, "0")}:00:00Z`,
      temperature_2m: temperatureAt(lat, lon, lead),
      precipitation_rate: precipitationAt(lead),
    }));
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          {
            location: {
              latitude: lat,
              longitude: lon,
              elevation_m: null,
              resolved_via: url.searchParams.get("city_id")
                ? "city"
                : url.searchParams.get("resort_id")
                  ? "resort"
                  : "coordinates",
            },
            generated_at: "2026-07-21T00:00:00Z",
            model: url.searchParams.get("models") ?? "gfs",
            forecasts,
          },
          "point_forecast"
        )
      ),
    });
  });

  await page.route("**/v1/ensembles?*", (route) => {
    const url = new URL(route.request().url());
    const lat = Number(url.searchParams.get("lat") ?? 38.19);
    const lon = Number(url.searchParams.get("lon") ?? -106.82);
    const lead = Number(url.searchParams.get("lead_time_hours") ?? 0);
    const includeMembers = url.searchParams.get("include_members") === "true";
    const members = GEFS_MEMBERS.map((member) => ensembleTemperatureAt(member, lat, lon, lead));
    const sorted = [...members].sort((a, b) => a - b);
    const stats = {
      mean: members.reduce((s, v) => s + v, 0) / members.length,
      median: sorted[2],
      spread: 2,
      p10: sorted[0],
      p25: sorted[1],
      p50: sorted[2],
      p75: sorted[3],
      p90: sorted[4],
    };
    // Production-contract-faithful: `members` is returned only when the
    // request opts in with `include_members=true`.
    const payload: Record<string, unknown> = {
      model: url.searchParams.get("model") ?? "gefs",
      lead_time_hours: lead,
      member_count: members.length,
      statistics: stats,
    };
    if (includeMembers) {
      payload.members = members;
    }
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(envelope(payload, "ensemble_statistics")),
    });
  });
}

/** Install a mock that makes every `/v1/*` call fail with a 500. */
export async function installApiFailureMocks(page: Page): Promise<void> {
  await page.route("**/v1/**", (route) =>
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: {
          code: "api_error",
          type: "server_error",
          message: "Backend unavailable",
          param: null,
          request_id: "req_e2e",
        },
      }),
    })
  );
}
