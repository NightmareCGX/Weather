/**
 * Cross-language fixture consistency for the serving-window boundary (I20).
 *
 * The JSON case table at packages/domain/tests/fixtures/serving_window_cases.json
 * is the single shared fixture consumed by BOTH this suite and the Python domain
 * suite (packages/domain/tests/test_temporal_serving_window_fixture.py), locking
 * the TS fallback implementation (computeServingStartValidTime /
 * isServableValidTime) against the authoritative Python primitive
 * (domain/temporal.py serving_start_valid_time / is_valid_time_protected).
 *
 * This is the interim §11-11 lock: once the frontend consumes only the
 * backend-provided serving_start_valid_time, the TS boundary math and this
 * fixture test can be retired together.
 */

import { readFileSync } from "fs";
import path from "path";

import {
  computeServingStartValidTime,
  isServableValidTime,
} from "@/lib/forecast/availability";

const fixturePath = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "..",
  "..",
  "..",
  "packages",
  "domain",
  "tests",
  "fixtures",
  "serving_window_cases.json"
);

interface ServingStartCase {
  now: string;
  expected_serving_start: string;
}

interface ProtectionCase {
  now: string;
  valid_time: string;
  expected_protected: boolean;
}

const fixture = JSON.parse(readFileSync(fixturePath, "utf-8")) as {
  serving_start_cases: ServingStartCase[];
  protection_cases: ProtectionCase[];
};

describe("Serving-window fixture consistency with domain/temporal.py (I20)", () => {
  describe("computeServingStartValidTime matches the shared fixture", () => {
    for (const c of fixture.serving_start_cases) {
      it(`now=${c.now} -> ${c.expected_serving_start}`, () => {
        const actual = computeServingStartValidTime(new Date(c.now));
        expect(new Date(actual).getTime()).toBe(
          new Date(c.expected_serving_start).getTime()
        );
      });
    }
  });

  describe("isServableValidTime matches the shared fixture", () => {
    for (const c of fixture.protection_cases) {
      it(`valid_time=${c.valid_time} at now=${c.now} -> ${c.expected_protected}`, () => {
        const servingStart = computeServingStartValidTime(new Date(c.now));
        expect(isServableValidTime(c.valid_time, servingStart)).toBe(
          c.expected_protected
        );
      });
    }
  });
});
