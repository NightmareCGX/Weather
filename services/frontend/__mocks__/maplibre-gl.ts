/**
 * Manual mock for `maplibre-gl` (swapped in via jest `moduleNameMapper`).
 *
 * jsdom has no WebGL/canvas, so the real library cannot initialize in tests.
 * The `Map`/`NavigationControl`/`Marker` facades and their test helpers live
 * in `src/test-utils/maplibre.ts` (a real, type-checked module that tests can
 * import directly); this file re-exports them so `import maplibregl from
 * "maplibre-gl"` resolves to the mock at runtime.
 */
import { MockMap, MockMarker, NavigationControlMock } from "../src/test-utils/maplibre";

export { MockMap, MockMarker };
export { clearInstances, getInstances, clearMarkers, getMarkers } from "../src/test-utils/maplibre";

export function supported(): boolean {
  return true;
}

const maplibregl = {
  Map: MockMap,
  Marker: MockMarker,
  NavigationControl: NavigationControlMock,
  supported,
};

export default maplibregl;
