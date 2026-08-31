import type { VectorFieldData } from "@/lib/api/types";
import {
  clearVectorFieldCache,
  getCachedVectorField,
  setCachedVectorField,
} from "../vectorFieldCache";

describe("vectorFieldCache", () => {
  beforeEach(() => {
    clearVectorFieldCache();
  });

  const dummyData: VectorFieldData = {
    meta: {
      lat_start: 90,
      lat_step: -0.5,
      lat_count: 2,
      lon_start: 0,
      lon_step: 0.5,
      lon_count: 2,
      scale: 0.01,
    },
    u: new Float32Array([1, 2, 3, 4]),
    v: new Float32Array([5, 6, 7, 8]),
  };

  it("stores and retrieves cached vector field payloads", () => {
    expect(getCachedVectorField("/url-1")).toBeUndefined();

    setCachedVectorField("/url-1", dummyData);
    expect(getCachedVectorField("/url-1")).toBe(dummyData);
  });

  it("evicts oldest entries when cache size exceeds 6", () => {
    for (let i = 1; i <= 6; i++) {
      setCachedVectorField(`/url-${i}`, { ...dummyData });
    }
    for (let i = 1; i <= 6; i++) {
      expect(getCachedVectorField(`/url-${i}`)).toBeDefined();
    }

    // Adding 7th item should evict url-1 (the oldest)
    setCachedVectorField("/url-7", dummyData);
    expect(getCachedVectorField("/url-1")).toBeUndefined();
    expect(getCachedVectorField("/url-7")).toBeDefined();
  });

  it("clears cache completely", () => {
    setCachedVectorField("/url-1", dummyData);
    clearVectorFieldCache();
    expect(getCachedVectorField("/url-1")).toBeUndefined();
  });
});
