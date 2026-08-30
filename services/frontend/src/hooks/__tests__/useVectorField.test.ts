import { renderHook, waitFor } from "@testing-library/react";
import { clearVectorFieldCache } from "@/lib/map/vectorFieldCache";
import { useVectorField } from "../useVectorField";

function createTestBinary(): ArrayBuffer {
  const num_pts = 4;
  const buf = new ArrayBuffer(36 + num_pts * 4);
  const view = new DataView(buf);
  const magic = "WNDQ";
  for (let i = 0; i < 4; i++) {
    view.setUint8(i, magic.charCodeAt(i));
  }
  view.setUint8(4, 1);
  view.setUint8(5, 1);
  view.setUint16(6, 0, true);
  view.setFloat32(8, 0.01, true);
  view.setFloat32(12, 90.0, true);
  view.setFloat32(16, -0.5, true);
  view.setUint32(20, 2, true);
  view.setFloat32(24, 0.0, true);
  view.setFloat32(28, 0.5, true);
  view.setUint32(32, 2, true);

  const u_i16 = new Int16Array(buf, 36, num_pts);
  const v_i16 = new Int16Array(buf, 36 + num_pts * 2, num_pts);
  for (let i = 0; i < num_pts; i++) {
    u_i16[i] = 500; // 5.0 m/s
    v_i16[i] = -200; // -2.0 m/s
  }
  return buf;
}

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

beforeEach(() => {
  clearVectorFieldCache();
  mockFetch.mockReset();
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("useVectorField", () => {
  it("returns null field when disabled or url is null", () => {
    const { result } = renderHook(() =>
      useVectorField({
        vectorFieldUrl: null,
        enabled: false,
      })
    );

    expect(result.current.field).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("fetches and loads vector field on valid url", async () => {
    const testBuf = createTestBinary();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(testBuf),
    } as Response);

    const { result } = renderHook(() =>
      useVectorField({
        vectorFieldUrl: "/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=6",
        enabled: true,
      })
    );

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.field).not.toBeNull();
    expect(result.current.field?.meta.lat_count).toBe(2);
    expect(result.current.field?.u[0]).toBeCloseTo(5.0);
    expect(result.current.error).toBeNull();
  });

  it("prefetches adjacent lead times opportunistically", async () => {
    const testBuf = createTestBinary();
    mockFetch.mockImplementation(async () => {
      return {
        ok: true,
        status: 200,
        arrayBuffer: () => Promise.resolve(testBuf),
      } as Response;
    });

    const { result } = renderHook(() =>
      useVectorField({
        vectorFieldUrl: "/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=6",
        availableLeads: [0, 6, 12, 18],
        currentLead: 6,
        enabled: true,
      })
    );

    await waitFor(() => {
      expect(result.current.field).not.toBeNull();
    });

    // Foreground request for lead 6 plus prefetch for 0 and 12
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("lead_time_hours=0"),
        expect.any(Object)
      );
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("lead_time_hours=12"),
        expect.any(Object)
      );
    });
  });

  it("handles fetch errors gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network failure"));

    const { result } = renderHook(() =>
      useVectorField({
        vectorFieldUrl: "/v1/maps/gfs/wind_10m/vector-field?lead_time_hours=6",
        enabled: true,
      })
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.field).toBeNull();
    expect(result.current.error).toContain("Network request failed");
  });
});
