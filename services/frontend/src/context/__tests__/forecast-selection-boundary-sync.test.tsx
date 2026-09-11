import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";

import { ForecastSelectionProvider, useForecastSelection } from "@/context/forecast-selection";
import { getForecastAvailability } from "@/lib/api/client";
import type { ForecastAvailability } from "@/lib/api/types";

jest.mock("../../lib/api/client", () => ({
  getForecastAvailability: jest.fn(),
  RequestAbortedError: class RequestAbortedError extends Error {},
}));

const mockGetForecastAvailability = getForecastAvailability as jest.MockedFunction<
  typeof getForecastAvailability
>;

const wrapper = ({ children }: { children: ReactNode }) => (
  <ForecastSelectionProvider>{children}</ForecastSelectionProvider>
);

describe("Forecast Selection Cadence-Boundary Synchronization", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("triggers boundary revalidation at the server-authoritative cadence boundary and reconciles selection", async () => {
    // Initial backend availability generated at 08:59:20Z (40s before 09:00Z boundary)
    const availability0859: ForecastAvailability = {
      generated_at: "2026-09-11T08:59:20.000Z",
      serving_start_valid_time: "2026-09-11T06:00:00.000Z",
      models: [
        {
          id: "gfs",
          name: "Global Forecast System",
          is_ensemble: false,
          variables: [
            {
              id: "temperature_2m",
              name: "Temperature",
              unit: "°C",
              initial_times: [],
              valid_times: [
                {
                  valid_time: "2026-09-11T06:00:00.000Z",
                  source_cycle: "2026-09-11T06:00:00.000Z",
                  lead_time_hours: 0,
                  servable: true,
                  available_members: 1,
                  expected_members: 1,
                  coverage_ratio: 1.0,
                },
                {
                  valid_time: "2026-09-11T09:00:00.000Z",
                  source_cycle: "2026-09-11T06:00:00.000Z",
                  lead_time_hours: 3,
                  servable: true,
                  available_members: 1,
                  expected_members: 1,
                  coverage_ratio: 1.0,
                },
              ],
            },
          ],
        },
      ],
    };

    // Refreshed backend availability generated at 09:00:00.100Z (boundary has rolled to 09Z)
    const availability0900: ForecastAvailability = {
      generated_at: "2026-09-11T09:00:00.100Z",
      serving_start_valid_time: "2026-09-11T09:00:00.000Z",
      models: [
        {
          id: "gfs",
          name: "Global Forecast System",
          is_ensemble: false,
          variables: [
            {
              id: "temperature_2m",
              name: "Temperature",
              unit: "°C",
              initial_times: [],
              valid_times: [
                {
                  valid_time: "2026-09-11T09:00:00.000Z",
                  source_cycle: "2026-09-11T06:00:00.000Z",
                  lead_time_hours: 3,
                  servable: true,
                  available_members: 1,
                  expected_members: 1,
                  coverage_ratio: 1.0,
                },
              ],
            },
          ],
        },
      ],
    };

    mockGetForecastAvailability.mockResolvedValueOnce(availability0859);

    const { result } = renderHook(() => useForecastSelection(), { wrapper });

    // Wait for initial availability load
    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.status).toBe("success");
    expect(mockGetForecastAvailability).toHaveBeenCalledTimes(1);

    // Selected valid time should initially be 06Z (the earliest available)
    expect(result.current.validTime).toBe("2026-09-11T06:00:00.000Z");
    expect(result.current.options.validTimes).toEqual([
      "2026-09-11T06:00:00.000Z",
      "2026-09-11T09:00:00.000Z",
    ]);

    // Setup next availability response for boundary rollover
    mockGetForecastAvailability.mockResolvedValueOnce(availability0900);

    // Fast-forward by 40,150 ms (just past the 40,100ms server-calculated boundary delay)
    // Notice: 40.15s is well before the arbitrary 60-second polling interval!
    await act(async () => {
      jest.advanceTimersByTime(40_150);
      await Promise.resolve();
    });

    // Proactive boundary revalidation must have fired at 40s (not 60s)
    expect(mockGetForecastAvailability).toHaveBeenCalledTimes(2);

    // 06Z must now be expired and dropped
    expect(result.current.options.validTimes).toEqual(["2026-09-11T09:00:00.000Z"]);

    // Active selection must be reconciled to 09Z
    expect(result.current.validTime).toBe("2026-09-11T09:00:00.000Z");
  });

  it("is immune to browser clock skew and uses backend server timing for the boundary delay", async () => {
    // Client clock is artificially set to 2 hours in the future (10:59:20Z)
    // But backend payload shows server generated at 08:59:20Z with serving_start = 06Z
    const skewedNow = new Date("2026-09-11T10:59:20.000Z").getTime();
    jest.setSystemTime(skewedNow);

    const availabilityWithSkew: ForecastAvailability = {
      generated_at: "2026-09-11T08:59:20.000Z",
      serving_start_valid_time: "2026-09-11T06:00:00.000Z",
      models: [
        {
          id: "gfs",
          name: "Global Forecast System",
          is_ensemble: false,
          variables: [
            {
              id: "temperature_2m",
              name: "Temperature",
              unit: "°C",
              initial_times: [],
              valid_times: [
                {
                  valid_time: "2026-09-11T06:00:00.000Z",
                  source_cycle: "2026-09-11T06:00:00.000Z",
                  lead_time_hours: 0,
                  servable: true,
                  available_members: 1,
                  expected_members: 1,
                  coverage_ratio: 1.0,
                },
              ],
            },
          ],
        },
      ],
    };

    mockGetForecastAvailability.mockResolvedValueOnce(availabilityWithSkew);

    const { result } = renderHook(() => useForecastSelection(), { wrapper });

    await act(async () => {
      await Promise.resolve();
    });

    // The fast client clock MUST NOT hide 06Z: backend serving_start_valid_time is authoritative!
    expect(result.current.validTime).toBe("2026-09-11T06:00:00.000Z");
    expect(result.current.options.validTimes).toContain("2026-09-11T06:00:00.000Z");
  });
});
