import { deriveCenterHealth, STALE_CYCLE_MULTIPLIER } from "../centerHealth";
import type {
  ForecastAvailability,
  InitialTimeAvailability,
  ModelAvailability,
  ValidTimeAvailability,
  VariableAvailability,
} from "@/lib/api/types";

/** A cycle instant that is `hours` old relative to {@link NOW}. */
const NOW = Date.parse("2026-09-17T12:00:00Z");

function cycle(hoursAgo: number): string {
  return new Date(NOW - hoursAgo * 3_600_000).toISOString();
}

function initialTime(
  hoursAgo: number,
  status: string | null,
  leadTimeHours: number[] = [0]
): InitialTimeAvailability {
  return { value: cycle(hoursAgo), lead_time_hours: leadTimeHours, status };
}

function validTime(hoursAgo: number, servable: boolean): ValidTimeAvailability {
  return {
    valid_time: cycle(hoursAgo),
    source_cycle: cycle(hoursAgo),
    lead_time_hours: 0,
    servable,
    available_members: 1,
    expected_members: 1,
    coverage_ratio: 1,
  };
}

function variable(
  id: string,
  initialTimes: InitialTimeAvailability[],
  validTimes: ValidTimeAvailability[] = [validTime(0, true)]
): VariableAvailability {
  return { id, name: id, unit: "°C", initial_times: initialTimes, valid_times: validTimes };
}

function model(overrides: Partial<ModelAvailability> & { id: string }): ModelAvailability {
  return {
    name: overrides.id.toUpperCase(),
    is_ensemble: false,
    variables: [variable("temperature_2m", [initialTime(0, "ready")])],
    center_id: "noaa",
    center_name: "National Oceanic and Atmospheric Administration",
    cycle_cadence_hours: 6,
    ...overrides,
  };
}

function availability(models: ModelAvailability[]): ForecastAvailability {
  return {
    models,
    serving_start_valid_time: cycle(0),
    generated_at: cycle(0),
  };
}

describe("deriveCenterHealth", () => {
  it("returns an empty list when availability has not loaded", () => {
    expect(deriveCenterHealth(null, NOW)).toEqual([]);
  });

  it("groups models under their issuing center and sorts by model id", () => {
    const centers = deriveCenterHealth(
      availability([model({ id: "gefs", is_ensemble: true }), model({ id: "gfs" })]),
      NOW
    );

    expect(centers).toHaveLength(1);
    expect(centers[0].id).toBe("noaa");
    expect(centers[0].name).toBe("National Oceanic and Atmospheric Administration");
    expect(centers[0].models.map((m) => m.id)).toEqual(["gefs", "gfs"]);
    expect(centers[0].readyCount).toBe(2);
    expect(centers[0].totalCount).toBe(2);
  });

  it("splits models across centers when more than one center is present", () => {
    const centers = deriveCenterHealth(
      availability([
        model({ id: "gfs", center_id: "noaa", center_name: "NOAA" }),
        model({ id: "ifs", center_id: "ecmwf", center_name: "ECMWF" }),
      ]),
      NOW
    );

    expect(centers.map((c) => c.id)).toEqual(["ecmwf", "noaa"]);
  });

  it("classifies a committed cycle with a servable valid time as ready", () => {
    const [center] = deriveCenterHealth(availability([model({ id: "gfs" })]), NOW);
    expect(center.models[0].status).toBe("ready");
    expect(center.status).toBe("healthy");
  });

  it("classifies an in-flight cycle as syncing rather than down", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [variable("temperature_2m", [initialTime(0, "processing")])],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("syncing");
    expect(center.status).toBe("degraded");
  });

  it("classifies a partial cycle as syncing", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [variable("temperature_2m", [initialTime(0, "partial")])],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("syncing");
  });

  it("classifies a ready cycle with no servable valid time as down", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [variable("temperature_2m", [initialTime(0, "ready")], [validTime(0, false)])],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("down");
  });

  it("classifies a cycle older than the stale threshold as stale even when marked ready", () => {
    const staleHoursAgo = 6 * STALE_CYCLE_MULTIPLIER + 1;
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [variable("temperature_2m", [initialTime(staleHoursAgo, "ready")])],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("stale");
  });

  it("keeps a cycle inside the stale threshold as ready", () => {
    const freshHoursAgo = 6 * STALE_CYCLE_MULTIPLIER - 1;
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [variable("temperature_2m", [initialTime(freshHoursAgo, "ready")])],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("ready");
  });

  it("honors a model's own cadence when judging staleness", () => {
    // 25h is stale for a 6-hourly model (>12h) but the same instant is within
    // the window for a 24-hourly model (>48h required).
    const variables = [variable("temperature_2m", [initialTime(25, "ready")])];
    const [sixHourly] = deriveCenterHealth(
      availability([model({ id: "gfs", cycle_cadence_hours: 6, variables })]),
      NOW
    );
    const [daily] = deriveCenterHealth(
      availability([model({ id: "other", cycle_cadence_hours: 24, variables })]),
      NOW
    );

    expect(sixHourly.models[0].status).toBe("stale");
    expect(daily.models[0].status).toBe("ready");
  });

  it("falls back to a 6-hour cadence when the field is absent", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "legacy",
          cycle_cadence_hours: null,
          variables: [
            variable("temperature_2m", [initialTime(6 * STALE_CYCLE_MULTIPLIER + 1, "ready")]),
          ],
        }),
      ]),
      NOW
    );
    expect(center.models[0].status).toBe("stale");
  });

  it("classifies a model with no initial times as down", () => {
    const [center] = deriveCenterHealth(
      availability([model({ id: "gfs", variables: [variable("temperature_2m", [], [])] })]),
      NOW
    );
    expect(center.models[0].status).toBe("down");
    expect(center.models[0].latestCycle).toBeNull();
  });

  it("rolls a center up to degraded when only some models are ready", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({ id: "gfs" }),
        model({
          id: "gefs",
          center_id: "noaa",
          variables: [variable("temperature_2m", [initialTime(0, "processing")])],
        }),
      ]),
      NOW
    );
    expect(center.status).toBe("degraded");
    expect(center.readyCount).toBe(1);
    expect(center.totalCount).toBe(2);
  });

  it("rolls a center up to down when every model is down", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({ id: "gfs", variables: [variable("temperature_2m", [], [])] }),
        model({ id: "gefs", variables: [variable("temperature_2m", [], [])] }),
      ]),
      NOW
    );
    expect(center.status).toBe("down");
  });

  it("keeps models without a center id visible under a synthetic center", () => {
    const centers = deriveCenterHealth(
      availability([model({ id: "gfs", center_id: null, center_name: null })]),
      NOW
    );
    expect(centers).toHaveLength(1);
    expect(centers[0].id).toBe("unknown");
    expect(centers[0].models).toHaveLength(1);
  });

  it("takes the newest cycle across all of a model's variables", () => {
    const [center] = deriveCenterHealth(
      availability([
        model({
          id: "gfs",
          variables: [
            variable("temperature_2m", [initialTime(3, "ready")]),
            variable("precipitation_rate", [initialTime(0, "ready")]),
          ],
        }),
      ]),
      NOW
    );
    expect(center.models[0].latestCycle).toBe(cycle(0));
  });
});
