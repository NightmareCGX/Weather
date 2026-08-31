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
 * Pure in-memory quantized Int16 (WNDQ) binary payload generator for vector field testing.
 */
export function createMockVectorFieldBuffer(latCount = 4, lonCount = 4): Buffer {
  const numPoints = latCount * lonCount;
  const buf = Buffer.alloc(36 + numPoints * 4);
  buf.write("WNDQ", 0, 4, "ascii");
  buf.writeUInt8(1, 4); // version
  buf.writeUInt8(1, 5); // flags
  buf.writeUInt16LE(0, 6); // reserved
  buf.writeFloatLE(0.01, 8); // scale
  buf.writeFloatLE(90.0, 12); // lat_start
  buf.writeFloatLE(-0.5, 16); // lat_step
  buf.writeUInt32LE(latCount, 20);
  buf.writeFloatLE(0.0, 24); // lon_start
  buf.writeFloatLE(0.5, 28); // lon_step
  buf.writeUInt32LE(lonCount, 32);

  const uOffset = 36;
  const vOffset = 36 + numPoints * 2;
  for (let i = 0; i < numPoints; i++) {
    buf.writeInt16LE(1000, uOffset + i * 2); // 10 m/s
    buf.writeInt16LE(-500, vOffset + i * 2); // -5 m/s
  }
  return buf;
}

const DEFAULT_VECTOR_FIELD_PAYLOAD = createMockVectorFieldBuffer(10, 10);

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
                  {
                    id: "precipitation_amount_3h",
                    name: "3-Hour Precipitation",
                    unit: "mm",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/precipitation_amount_3h/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "mm",
                        stops: [
                          [0, "#ffffff"],
                          [0.5, "#c2e699"],
                          [1.0, "#78c679"],
                          [2.5, "#31a354"],
                          [5.0, "#197278"],
                          [10.0, "#314e8f"],
                          [20.0, "#7b4173"],
                          [40.0, "#542788"],
                        ],
                      },
                    },
                  },
                  {
                    id: "relative_humidity_2m",
                    name: "Relative Humidity",
                    unit: "%",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/relative_humidity_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "%",
                        stops: [
                          [0, "#8c510a"],
                          [100, "#01665e"],
                        ],
                      },
                    },
                  },
                  {
                    id: "wind_gust",
                    name: "Wind Gust",
                    unit: "km/h",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/wind_gust/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "km/h",
                        stops: [
                          [0, "#f7f7f7"],
                          [150, "#49006a"],
                        ],
                      },
                    },
                  },
                  {
                    id: "visibility",
                    name: "Visibility",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/visibility/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#49006a"],
                          [24000, "#ffffff"],
                        ],
                      },
                    },
                  },
                  {
                    id: "snow_depth",
                    name: "Snow Depth",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/snow_depth/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#ffffff"],
                          [2.5, "#1a0040"],
                        ],
                      },
                    },
                  },
                  {
                    id: "wind_10m",
                    name: "10-Meter Wind",
                    unit: "km/h",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/wind_10m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "km/h",
                        stops: [
                          [0, "#ffffff"],
                          [140, "#49006a"],
                        ],
                      },
                      vector_field_url_template:
                        "/v1/maps/gfs/wind_10m/vector-field?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                    },
                  },
                  {
                    id: "cloud_cover_3h",
                    name: "3-Hour Cloud Cover",
                    unit: "%",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/cloud_cover_3h/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "%",
                        stops: [
                          [0, "#ffffff"],
                          [100, "#323c4b"],
                        ],
                      },
                    },
                  },
                  {
                    id: "cloud_ceiling",
                    name: "Cloud Ceiling Height",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gfs/cloud_ceiling/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#a50026"],
                          [3000, "#ffffff"],
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
                  {
                    id: "relative_humidity_2m",
                    name: "Relative Humidity",
                    unit: "%",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/relative_humidity_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "%",
                        stops: [
                          [0, "#8c510a"],
                          [100, "#01665e"],
                        ],
                      },
                    },
                  },
                  {
                    id: "wind_gust",
                    name: "Wind Gust",
                    unit: "km/h",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/wind_gust/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "km/h",
                        stops: [
                          [0, "#f7f7f7"],
                          [150, "#49006a"],
                        ],
                      },
                    },
                  },
                  {
                    id: "visibility",
                    name: "Visibility",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/visibility/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#49006a"],
                          [24000, "#ffffff"],
                        ],
                      },
                    },
                  },
                  {
                    id: "snow_depth",
                    name: "Snow Depth",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/snow_depth/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#ffffff"],
                          [2.5, "#1a0040"],
                        ],
                      },
                    },
                  },
                  {
                    id: "wind_10m",
                    name: "10-Meter Wind",
                    unit: "km/h",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/wind_10m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "km/h",
                        stops: [
                          [0, "#ffffff"],
                          [140, "#49006a"],
                        ],
                      },
                      vector_field_url_template:
                        "/v1/maps/gefs/wind_10m/vector-field?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                    },
                  },
                  {
                    id: "precipitation_amount_3h",
                    name: "3-Hour Precipitation",
                    unit: "mm",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/precipitation_amount_3h/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "mm",
                        stops: [
                          [0, "#ffffff"],
                          [0.5, "#c2e699"],
                          [1.0, "#78c679"],
                          [2.5, "#31a354"],
                          [5.0, "#197278"],
                          [10.0, "#314e8f"],
                          [20.0, "#7b4173"],
                          [40.0, "#542788"],
                        ],
                      },
                    },
                  },
                  {
                    id: "cloud_cover_3h",
                    name: "3-Hour Cloud Cover",
                    unit: "%",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/cloud_cover_3h/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "%",
                        stops: [
                          [0, "#ffffff"],
                          [100, "#323c4b"],
                        ],
                      },
                    },
                  },
                  {
                    id: "cloud_ceiling",
                    name: "Cloud Ceiling Height",
                    unit: "m",
                    initial_times: [
                      {
                        value: "2026-08-13T00:00:00Z",
                        lead_time_hours: LEAD_TIMES,
                      },
                    ],
                    layer: {
                      tile_url_template:
                        "/v1/maps/gefs/cloud_ceiling/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                      min_zoom: 0,
                      max_zoom: 9,
                      legend: {
                        unit: "m",
                        stops: [
                          [0, "#a50026"],
                          [3000, "#ffffff"],
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

  // Serve quantized binary vector field for flow animation
  await page.route("**/v1/maps/**/vector-field*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/octet-stream",
      body: DEFAULT_VECTOR_FIELD_PAYLOAD,
    })
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
            {
              id: "precipitation_amount_3h",
              object: "variable",
              name: "3-Hour Precipitation",
              unit: "mm",
            },
            {
              id: "relative_humidity_2m",
              object: "variable",
              name: "Relative Humidity",
              unit: "%",
            },
            {
              id: "wind_gust",
              object: "variable",
              name: "Wind Gust",
              unit: "km/h",
            },
            {
              id: "visibility",
              object: "variable",
              name: "Visibility",
              unit: "m",
            },
            {
              id: "snow_depth",
              object: "variable",
              name: "Snow Depth",
              unit: "m",
            },
            {
              id: "wind_10m",
              object: "variable",
              name: "10-Meter Wind",
              unit: "km/h",
            },
            {
              id: "cloud_cover_3h",
              object: "variable",
              name: "3-Hour Cloud Cover",
              unit: "%",
            },
            {
              id: "cloud_ceiling",
              object: "variable",
              name: "Cloud Ceiling Height",
              unit: "m",
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
      precipitation_amount_3h: lead === 0 ? undefined : lead === 6 ? 4.2 : 5.1,
      precipitation_type: lead === 0 ? "none" : lead === 6 ? "rain" : "mixed",
      precipitation_transition:
        lead === 0 ? "none" : lead === 6 ? "persistent_rain" : "rain_to_snow",
      precipitation_start_type: lead === 0 ? "none" : lead === 6 ? "rain" : "rain",
      precipitation_end_type: lead === 0 ? "none" : lead === 6 ? "rain" : "snow",
      precipitation_evidence: "exact",
      wind_10m: 25.4,
      wind_direction_10m: 225.0,
      wind_cardinal_10m: "SW",
      cloud_cover_3h: lead === 0 ? null : 65.0,
      cloud_ceiling: lead === 18 ? null : 1200.0,
      cloud_ceiling_unlimited: lead === 18,
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
    const variable = url.searchParams.get("variable") ?? "temperature_2m";
    // Production-contract-faithful: `members` and `pdf` are returned only when the
    // request opts in with `include_members=true`.
    const payload: Record<string, unknown> = {
      model: url.searchParams.get("model") ?? "gefs",
      lead_time_hours: lead,
      member_count: members.length,
      statistics: stats,
    };
    if (variable === "wind_10m") {
      payload.consensus_vector = {
        speed: 24.5,
        direction: 220.0,
        cardinal: "SW",
        coherence: 0.95,
      };
      payload.wind_rose = {
        calm_percentage: 10.0,
        calm_count: 3,
        sectors: [
          {
            sector: "N",
            count: 3,
            probability: 0.1,
            bins: { light: 0.05, moderate: 0.05, strong: 0.0, gale: 0.0 },
          },
          {
            sector: "NE",
            count: 0,
            probability: 0.0,
            bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
          },
          {
            sector: "E",
            count: 0,
            probability: 0.0,
            bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
          },
          {
            sector: "SE",
            count: 0,
            probability: 0.0,
            bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
          },
          {
            sector: "S",
            count: 6,
            probability: 0.2,
            bins: { light: 0.05, moderate: 0.1, strong: 0.05, gale: 0.0 },
          },
          {
            sector: "SW",
            count: 18,
            probability: 0.6,
            bins: { light: 0.1, moderate: 0.3, strong: 0.15, gale: 0.05 },
          },
          {
            sector: "W",
            count: 0,
            probability: 0.0,
            bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
          },
          {
            sector: "NW",
            count: 0,
            probability: 0.0,
            bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
          },
        ],
      };
    }
    if (variable === "precipitation_amount_3h") {
      payload.phase_support = {
        dry: 0.1,
        rain: 0.52,
        snow: 0.26,
        freezing_rain: 0.08,
        ice_pellets: 0.03,
        unknown: 0.01,
      };
      payload.transition_frequency = {
        rain_to_snow: 0.27,
        persistent_rain: 0.25,
      };
    }
    if (variable === "cloud_cover_3h") {
      payload.valid_member_count = 30;
      payload.statistics = {
        mean: 65.0,
        median: 65.0,
        spread: 10.0,
        p10: 50.0,
        p25: 58.0,
        p50: 65.0,
        p75: 72.0,
        p90: 80.0,
      };
    }
    if (variable === "cloud_ceiling") {
      payload.unlimited_probability = 0.4;
      payload.valid_member_count = 30;
      payload.finite_member_count = 18;
      payload.unlimited_member_count = 12;
      payload.statistics = {
        mean: 2100.0,
        median: 2100.0,
        spread: 500.0,
        p10: 1200.0,
        p25: 1600.0,
        p50: 2100.0,
        p75: 2800.0,
        p90: 3500.0,
      };
    }
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
