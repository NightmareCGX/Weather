import fs from "fs";
import path from "path";
import type { ForecastEntry } from "@/lib/api/types";
import {
  FORECAST_ENTRY_METADATA_FIELDS,
  isForecastDataVariable,
  isForecastEntryMetadataField,
} from "@/lib/api/types";
import { forecastVariableCodes, toMeteogramSeries } from "@/lib/forecast/transform";

/**
 * Stage 7D-A — Cross-Package OpenAPI ↔ Frontend Contract Suite
 *
 * Validates that the actual OpenAPI specification emitted by FastAPI in
 * `services/api` matches the endpoints, parameters, envelopes, and resource
 * properties consumed by the Next.js frontend client.
 */

describe("OpenAPI ↔ Frontend Client Contract (7D-A)", () => {
  const openApiPath = path.resolve(__dirname, "../../../../openapi.json");
  let openapi: any;

  beforeAll(() => {
    expect(fs.existsSync(openApiPath)).toBe(true);
    const raw = fs.readFileSync(openApiPath, "utf-8");
    openapi = JSON.parse(raw);
    expect(openapi.openapi).toBeDefined();
  });

  describe("Endpoint Paths & Methods", () => {
    const requiredEndpoints = [
      { path: "/v1/models", method: "get" },
      { path: "/v1/variables", method: "get" },
      { path: "/v1/forecast/availability", method: "get" },
      { path: "/v1/search", method: "get" },
      { path: "/v1/points", method: "get" },
      { path: "/v1/ensembles", method: "get" },
      { path: "/v1/maps", method: "get" },
      { path: "/v1/maps/{model}/wind_10m/vector-field", method: "get" },
      { path: "/v1/health", method: "get" },
    ];

    test.each(requiredEndpoints)(
      "OpenAPI defines $method $path",
      ({ path: endpointPath, method }) => {
        expect(openapi.paths).toHaveProperty(endpointPath);
        expect(openapi.paths[endpointPath]).toHaveProperty(method);
      }
    );
  });

  describe("Query Parameter Contracts", () => {
    test("/v1/points defines expected query parameters", () => {
      const op = openapi.paths["/v1/points"].get;
      const paramNames = op.parameters.map((p: any) => p.name);
      expect(paramNames).toEqual(
        expect.arrayContaining([
          "lat",
          "lon",
          "city_id",
          "resort_id",
          "models",
          "variables",
          "units",
          "start_lead_time_hours",
          "end_lead_time_hours",
        ])
      );
    });

    test("/v1/search defines query parameters", () => {
      const op = openapi.paths["/v1/search"].get;
      const paramNames = op.parameters.map((p: any) => p.name);
      expect(paramNames).toEqual(expect.arrayContaining(["q"]));
    });

    test("/v1/ensembles defines location and variable parameters", () => {
      const op = openapi.paths["/v1/ensembles"].get;
      const paramNames = op.parameters.map((p: any) => p.name);
      expect(paramNames).toEqual(expect.arrayContaining(["model", "variable", "lead_time_hours"]));
    });
  });

  describe("Resource Schema Properties", () => {
    const schemas = () => openapi.components.schemas;

    test("PointForecastData schema contains core fields", () => {
      const schema = schemas()["PointForecastData"];
      expect(schema).toBeDefined();
      expect(schema.properties).toHaveProperty("location");
      expect(schema.properties).toHaveProperty("generated_at");
      expect(schema.properties).toHaveProperty("model");
      expect(schema.properties).toHaveProperty("forecasts");
    });

    test("ForecastLocationOut schema contains spatial fields", () => {
      const schema = schemas()["ForecastLocationOut"];
      expect(schema).toBeDefined();
      expect(schema.properties).toHaveProperty("latitude");
      expect(schema.properties).toHaveProperty("longitude");
      expect(schema.properties).toHaveProperty("elevation_m");
      expect(schema.properties).toHaveProperty("resolved_via");
    });

    test("ForecastAvailabilityData schema contains availability hierarchy", () => {
      const schema = schemas()["ForecastAvailabilityData"];
      expect(schema).toBeDefined();
      expect(schema.properties).toHaveProperty("models");
      const modelSchema = schemas()["ModelAvailability"];
      expect(modelSchema).toBeDefined();
      expect(modelSchema.properties).toHaveProperty("variables");
      const variableSchema = schemas()["VariableAvailability"];
      expect(variableSchema).toBeDefined();
      expect(variableSchema.properties).toHaveProperty("initial_times");
      const initialTimeSchema = schemas()["InitialTimeAvailability"];
      expect(initialTimeSchema).toBeDefined();
      expect(initialTimeSchema.properties).toHaveProperty("value");
      expect(initialTimeSchema.properties).toHaveProperty("lead_time_hours");
      expect(initialTimeSchema.properties).toHaveProperty("leads");
    });

    test("SpatialLayerData schema contains spatial layer descriptor fields", () => {
      const schema = schemas()["SpatialLayerData"];
      expect(schema).toBeDefined();
      expect(schema.properties).toHaveProperty("tile_url_template");
      expect(schema.properties).toHaveProperty("min_zoom");
      expect(schema.properties).toHaveProperty("max_zoom");
      expect(schema.properties).toHaveProperty("lead_time_hours");
      expect(schema.properties).toHaveProperty("legend");
    });

    test("HealthCheckData schema defines dependency statuses", () => {
      const schema = schemas()["HealthCheckData"];
      expect(schema).toBeDefined();
      expect(schema.properties).toHaveProperty("status");
      expect(schema.properties).toHaveProperty("version");
      expect(schema.properties).toHaveProperty("database");
      expect(schema.properties).toHaveProperty("redis");
      expect(schema.properties).toHaveProperty("object_storage");
    });
  });

  describe("Dynamic Point-Forecast Metadata Contract", () => {
    test("excludes all backend structural/metadata fields from plottable variable set", () => {
      const mockBackendForecastEntry: ForecastEntry = {
        lead_time_hours: 6,
        valid_time: "2026-09-03T06:00:00Z",
        cycle_time: "2026-09-03T00:00:00Z",
        wind_direction_10m: 180.5,
        wind_cardinal_10m: "S",
        precipitation_type: "rain",
        precipitation_transition: "none",
        precipitation_start_type: "none",
        precipitation_end_type: "none",
        precipitation_evidence: "exact",
        cloud_ceiling_unlimited: false,
        // Actual weather variables:
        temperature_2m: 18.2,
        precipitation_rate: 1.5,
        precipitation_amount_3h: 4.2,
      };

      // Ensure every declared metadata field is identified
      const declaredMetadata = Array.from(FORECAST_ENTRY_METADATA_FIELDS);
      for (const field of declaredMetadata) {
        expect(isForecastEntryMetadataField(field)).toBe(true);
        expect(isForecastDataVariable(field)).toBe(false);
      }

      // Ensure variable code extraction ignores all metadata
      const extractedCodes = forecastVariableCodes([mockBackendForecastEntry]);
      expect(extractedCodes.sort()).toEqual(
        ["precipitation_amount_3h", "precipitation_rate", "temperature_2m"].sort()
      );

      // Verify meteogram series extraction
      const tempSeries = toMeteogramSeries([mockBackendForecastEntry], "temperature_2m");
      expect(tempSeries).toEqual([
        {
          lead_time_hours: 6,
          valid_time: "2026-09-03T06:00:00Z",
          value: 18.2,
        },
      ]);
    });
  });
});
