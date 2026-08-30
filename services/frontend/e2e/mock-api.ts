import zlib from "zlib";
import type { Page } from "@playwright/test";

/**
 * Pure in-memory 256x256 solid-color RGBA PNG generator for MapLibre raster tile testing.
 */
export function createPngTile(r: number, g: number, b: number, a: number = 255): Buffer {
  const width = 256;
  const height = 256;
  const rowSize = 1 + width * 4;
  const raw = Buffer.alloc(height * rowSize);
  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowSize;
    raw[rowOffset] = 0;
    for (let x = 0; x < width; x++) {
      const pxOffset = rowOffset + 1 + x * 4;
      raw[pxOffset] = r;
      raw[pxOffset + 1] = g;
      raw[pxOffset + 2] = b;
      raw[pxOffset + 3] = a;
    }
  }
  const deflated = zlib.deflateSync(raw);
  function crc32(buf: Buffer): number {
    let crc = -1;
    for (let i = 0; i < buf.length; i++) {
      let byte = buf[i];
      for (let j = 0; j < 8; j++) {
        const bit = (crc ^ byte) & 1;
        crc = (crc >>> 1) ^ (bit ? 0xedb88320 : 0);
        byte >>>= 1;
      }
    }
    return (crc ^ -1) >>> 0;
  }
  function chunk(type: string, data: Buffer): Buffer {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const typeAndData = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(typeAndData), 0);
    return Buffer.concat([len, typeAndData, crc]);
  }
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflated),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const DEFAULT_WEATHER_TILE = createPngTile(50, 150, 250, 255);
const DEFAULT_OSM_TILE = createPngTile(240, 240, 240, 255);

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

  // The data-driven forecast explorer sources every selector option from
  // /v1/forecast/availability. The mock mirrors the repository's API fixtures:
  // GFS (temperature_2m, precipitation_rate, initial time 2026-08-13T00:00:00Z,
  // lead [0, 6, 12, 18]).
  await page.route("**/v1/forecast/availability", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          {
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
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "°C",
                        stops: [
                          [-40, "#313695"],
                          [45, "#a50026"],
                        ],
                      },
                    },
                  },
                  {
                    id: "precipitation_rate",
                    name: "Precipitation Rate",
                    unit: "mm/h",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "mm/h",
                        stops: [
                          [0, "#ffffff"],
                          [40, "#54278f"],
                        ],
                      },
                    },
                  },
                ],
              },
              {
                id: "gefs",
                name: "Global Ensemble Forecast System",
                is_ensemble: true,
                variables: [
                  {
                    id: "temperature_2m",
                    name: "2-Meter Temperature",
                    unit: "°C",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "°C",
                        stops: [
                          [-40, "#313695"],
                          [45, "#a50026"],
                        ],
                      },
                    },
                  },
                ],
              },
            ],
          },
          "forecast_availability"
        )
      ),
    })
  );

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
    const initial = url.searchParams.get("initial_time") ?? "2026-08-13T00:00:00Z";
    const variable = url.searchParams.get("variable") ?? "temperature_2m";
    const isPrecip = variable === "precipitation_rate";
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          {
            tile_url_template: `/v1/maps/gfs/${variable}/surface/{z}/{x}/{y}.png?lead_time_hours=${lead}&initial_time=${initial}`,
            min_zoom: 0,
            max_zoom: 9,
            lead_time_hours: Number(lead),
            legend: isPrecip
              ? {
                  unit: "mm/h",
                  stops: [
                    [0, "#ffffff"],
                    [0.5, "#c2e699"],
                    [2.5, "#31a354"],
                    [10, "#314e8f"],
                    [40, "#54278f"],
                  ],
                }
              : {
                  unit: "°C",
                  stops: [
                    [-40, "#313695"],
                    [-5, "#74add1"],
                    [5, "#f0f9e8"],
                    [15, "#fed976"],
                    [25, "#fe9929"],
                    [45, "#a50026"],
                  ],
                },
          },
          "spatial_layer"
        )
      ),
    });
  });

  // Serve valid PNG tiles so MapLibre GL JS successfully decodes raster textures.
  await page.route("**/v1/maps/**/*.png?*", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: DEFAULT_WEATHER_TILE })
  );
  await page.route("**/v1/maps/**/*.png", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: DEFAULT_WEATHER_TILE })
  );

  // The base style uses OSM raster tiles.
  await page.route("https://tile.openstreetmap.org/**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: DEFAULT_OSM_TILE })
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
      {
        id: "place_boulder",
        object: "place",
        name: "Boulder, CO",
        place_id: "ChIJ_boulder_mock_place",
        region: "Colorado",
        country: "USA",
        elevation_m: null,
        latitude: null as unknown as number,
        longitude: null as unknown as number,
      },
    ];
    const matches = all.filter((item) => item.name.toLowerCase().includes(q));
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(envelope(matches, "list")),
    });
  });

  await page.route("**/v1/search/places/*", (route) => {
    const url = new URL(route.request().url());
    const placeId = decodeURIComponent(url.pathname.split("/").pop() ?? "");
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        envelope(
          {
            id: `resolved_${placeId}`,
            object: "place",
            name: "Boulder, CO",
            place_id: placeId,
            region: "Colorado",
            country: "USA",
            elevation_m: 1655,
            latitude: 40.015,
            longitude: -105.2705,
          },
          "search_result"
        )
      ),
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
    // Production-contract-faithful: `members` and `pdf` are returned only when the
    // request opts in with `include_members=true`.
    const payload: Record<string, unknown> = {
      model: url.searchParams.get("model") ?? "gefs",
      lead_time_hours: lead,
      member_count: members.length,
      statistics: stats,
    };
    if (includeMembers) {
      payload.members = members;
      payload.pdf = {
        x: [10.0, 15.0, 20.0, 25.0, 30.0],
        density: [0.01, 0.05, 0.2, 0.05, 0.01],
      };
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
