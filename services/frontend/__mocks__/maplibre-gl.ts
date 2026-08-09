/**
 * Manual mock for `maplibre-gl` (swapped in via jest `moduleNameMapper`).
 *
 * jsdom has no WebGL/canvas, so the real library cannot initialize in tests.
 * The `Map`/`NavigationControl` facades and their test helpers live in
 * `src/test-utils/maplibre.ts` (a real, type-checked module that tests can
 * import directly); this file re-exports them so `import maplibregl from
 * "maplibre-gl"` resolves to the mock at runtime.
 */
import { MockMap, NavigationControlMock } from "../src/test-utils/maplibre";

export { MockMap };
export { clearInstances, getInstances } from "../src/test-utils/maplibre";

export function supported(): boolean {
  return true;
}

const maplibregl = {
  Map: MockMap,
  NavigationControl: NavigationControlMock,
  supported,
};

export default maplibregl;
