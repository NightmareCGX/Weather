import type { VectorFieldData } from "@/lib/api/types";
import { MockMap } from "@/test-utils/maplibre";
import {
  advectParticle,
  CALM_WIND_THRESHOLD_MPS,
  DEFAULT_LAT_CLAMP,
  EARTH_RADIUS_METERS,
  isCalmWind,
  sampleVectorBilinear,
  WindParticleAnimation,
} from "../windParticles";

describe("windParticles math", () => {
  const sampleField: VectorFieldData = {
    meta: {
      lat_start: 90.0,
      lat_step: -1.0,
      lat_count: 3, // 90, 89, 88
      lon_start: 0.0,
      lon_step: 1.0,
      lon_count: 4, // 0, 1, 2, 3
      scale: 0.01,
    },
    u: new Float32Array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
    v: new Float32Array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
  };

  describe("sampleVectorBilinear", () => {
    it("samples exact grid nodes", () => {
      const [u, v] = sampleVectorBilinear(sampleField, 90.0, 0.0);
      expect(u).toBeCloseTo(0.0);
      expect(v).toBeCloseTo(10.0);
    });

    it("interpolates cell midpoints bilinearly", () => {
      const [u, v] = sampleVectorBilinear(sampleField, 89.5, 1.5);
      expect(u).toBeCloseTo(1.5);
      expect(v).toBeCloseTo(7.5);
    });

    it("handles global longitude wrapping across 360", () => {
      const [u] = sampleVectorBilinear(sampleField, 90.0, 3.5);
      // Midway between col 3 (u=3.0) and col 0 (u=0.0) -> 1.5
      expect(u).toBeCloseTo(1.5);
    });

    it("handles negative longitude wrapping", () => {
      const [u] = sampleVectorBilinear(sampleField, 90.0, -0.5);
      expect(u).toBeCloseTo(1.5);
    });

    it("clamps out-of-bounds latitude to edge nodes", () => {
      const [u, v] = sampleVectorBilinear(sampleField, 95.0, 0.0);
      expect(u).toBeCloseTo(0.0);
      expect(v).toBeCloseTo(10.0);
    });

    it("handles [-180, 180] lon_start convention", () => {
      const negField: VectorFieldData = {
        meta: {
          ...sampleField.meta,
          lon_start: -180.0,
        },
        u: sampleField.u,
        v: sampleField.v,
      };
      const [u] = sampleVectorBilinear(negField, 90.0, -180.0);
      expect(u).toBeCloseTo(0.0);
    });

    it("returns NaN for NaN coordinates", () => {
      expect(sampleVectorBilinear(sampleField, NaN, 0.0)).toEqual([NaN, NaN]);
      expect(sampleVectorBilinear(sampleField, 0.0, NaN)).toEqual([NaN, NaN]);
    });
  });

  describe("advectParticle", () => {
    it("advects purely northward from the equator", () => {
      const dt = 3600.0; // 1 hour
      const v = 100.0; // 100 m/s
      const [newLat, newLon] = advectParticle(0.0, 0.0, 0.0, v, dt);
      const expectedDlat = ((100.0 * 3600.0) / EARTH_RADIUS_METERS) * (180.0 / Math.PI);
      expect(newLat).toBeCloseTo(expectedDlat);
      expect(newLon).toBeCloseTo(0.0);
    });

    it("advects across the antimeridian dateline (+180)", () => {
      const dt = 3600.0;
      const u = 100.0;
      const [newLat, newLon] = advectParticle(0.0, 179.0, u, 0.0, dt);
      const expectedDlon = ((100.0 * 3600.0) / EARTH_RADIUS_METERS) * (180.0 / Math.PI);
      expect(newLat).toBeCloseTo(0.0);
      expect(newLon).toBeCloseTo(179.0 + expectedDlon - 360.0);
    });

    it("advects across the antimeridian dateline (-180)", () => {
      const dt = 3600.0;
      const u = -100.0;
      const [newLat, newLon] = advectParticle(0.0, -179.0, u, 0.0, dt);
      const expectedDlon = ((100.0 * 3600.0) / EARTH_RADIUS_METERS) * (180.0 / Math.PI);
      expect(newLon).toBeCloseTo(-179.0 - expectedDlon + 360.0);
    });

    it("clamps at high latitude poles", () => {
      const [northLat] = advectParticle(88.0, 0.0, 10.0, 10.0, 3600.0, DEFAULT_LAT_CLAMP);
      expect(northLat).toBe(DEFAULT_LAT_CLAMP);

      const [southLat] = advectParticle(-88.0, 0.0, -10.0, -10.0, 3600.0, DEFAULT_LAT_CLAMP);
      expect(southLat).toBe(-DEFAULT_LAT_CLAMP);
    });

    it("preserves exact 180.0 longitude boundary", () => {
      const [, lon] = advectParticle(0.0, 180.0, 0.0, 0.0);
      expect(lon).toBe(180.0);
    });

    it("returns NaN for NaN inputs", () => {
      expect(advectParticle(NaN, 0, 1, 1)).toEqual([NaN, NaN]);
      expect(advectParticle(0, NaN, 1, 1)).toEqual([NaN, NaN]);
      expect(advectParticle(0, 0, NaN, 1)).toEqual([NaN, NaN]);
      expect(advectParticle(0, 0, 1, NaN)).toEqual([NaN, NaN]);
    });
  });

  describe("isCalmWind", () => {
    it("returns true below the 0.5 m/s calm threshold", () => {
      expect(isCalmWind(0.1, 0.1)).toBe(true);
      expect(isCalmWind(0.3, 0.3)).toBe(true);
    });

    it("returns false at or above 0.5 m/s", () => {
      expect(isCalmWind(0.3, 0.4)).toBe(false); // hypot(0.3, 0.4) = 0.5
      expect(isCalmWind(3.0, 4.0)).toBe(false);
    });

    it("returns true for NaN inputs", () => {
      expect(isCalmWind(NaN, 1.0)).toBe(true);
      expect(isCalmWind(1.0, NaN)).toBe(true);
    });
  });
});

describe("WindParticleAnimation", () => {
  let canvas: HTMLCanvasElement;
  let map: MockMap;

  beforeEach(() => {
    canvas = document.createElement("canvas");
    canvas.getBoundingClientRect = jest.fn(() => ({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      bottom: 600,
      right: 800,
      x: 0,
      y: 0,
      toJSON: () => {},
    }));
    map = new MockMap();
  });

  it("initializes and handles setField(null) / clear", () => {
    const anim = new WindParticleAnimation(canvas, map as any);
    expect(canvas.width).toBeGreaterThan(0);
    expect(canvas.height).toBeGreaterThan(0);

    anim.setField(null);
    anim.clear();
    anim.destroy();
  });

  it("starts and stops animation loop smoothly", () => {
    jest.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      return setTimeout(() => cb(performance.now()), 16) as any;
    });
    jest.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      clearTimeout(id);
    });

    const anim = new WindParticleAnimation(canvas, map as any);
    const sampleField: VectorFieldData = {
      meta: {
        lat_start: 90.0,
        lat_step: -0.5,
        lat_count: 361,
        lon_start: 0.0,
        lon_step: 0.5,
        lon_count: 720,
        scale: 0.01,
      },
      u: new Float32Array(361 * 720).fill(5.0),
      v: new Float32Array(361 * 720).fill(2.0),
    };

    anim.setField(sampleField);
    anim.start();
    anim.stop();
    anim.destroy();
  });
});
