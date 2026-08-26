/**
 * Test helpers for the maplibre-gl manual mock.
 *
 * Jest swaps `maplibre-gl` for `__mocks__/maplibre-gl.ts` at runtime via
 * `moduleNameMapper`, but `tsc` resolves the real package's type declarations,
 * which neither export these helpers nor treat `MockMap` as assignable to the
 * real `Map` class. Tests therefore import the helpers from this real,
 * type-checked module and cast the mock to the real `Map` type only at the
 * `applyWeatherLayer`/component call boundary.
 */

/** Instances of {@link MockMap} created since the last {@link clearInstances}. */
const instances: MockMap[] = [];

/** Markers created since the last {@link clearMarkers}. */
const markers: MockMarker[] = [];

export function clearInstances(): void {
  instances.length = 0;
  MockMap.defaultIsStyleLoaded = true;
  MockMap.defaultIsLoaded = true;
}

export function getInstances(): MockMap[] {
  return instances;
}

export function clearMarkers(): void {
  markers.length = 0;
}

export function getMarkers(): MockMarker[] {
  return markers;
}

export class MockMap {
  static defaultIsStyleLoaded = true;
  static defaultIsLoaded = true;

  options: Record<string, unknown>;
  handlers = new Map<string, (payload?: unknown) => void>();
  sources = new Set<string>();
  layers = new Set<string>();
  removed = false;
  isLoaded = MockMap.defaultIsLoaded;
  _isStyleLoaded = MockMap.defaultIsStyleLoaded;

  on = jest.fn((event: string, handler: (payload?: unknown) => void) => {
    this.handlers.set(event, handler);
  });
  off = jest.fn((event: string) => {
    this.handlers.delete(event);
  });
  addControl = jest.fn();
  addSource = jest.fn((id: string) => {
    this.sources.add(id);
  });
  addLayer = jest.fn((layer: { id: string }) => {
    this.layers.add(layer.id);
  });
  removeSource = jest.fn((id: string) => {
    this.sources.delete(id);
  });
  removeLayer = jest.fn((id: string) => {
    this.layers.delete(id);
  });
  getSource = jest.fn((id: string) => (this.sources.has(id) ? { id } : undefined));
  getLayer = jest.fn((id: string) => (this.layers.has(id) ? { id } : undefined));
  loaded = jest.fn(() => this.isLoaded);
  isStyleLoaded = jest.fn(() => this._isStyleLoaded);
  remove = jest.fn(() => {
    this.removed = true;
  });

  constructor(options: Record<string, unknown> = {}) {
    this.options = options;
    instances.push(this);
  }

  /**
   * Invoke a registered event handler with an optional payload (e.g. a click
   * event carrying `lngLat`).
   */
  fire(event: string, payload?: unknown): void {
    const handler = this.handlers.get(event);
    handler?.(payload);
  }
}

/** A mock of the MapLibre `Marker` class (used by `WeatherMap`). */
export class MockMarker {
  options: Record<string, unknown>;
  coordinates: unknown = null;
  addedTo: MockMap | null = null;
  removed = false;

  setLngLat = jest.fn((lngLat: unknown) => {
    this.coordinates = lngLat;
    return this;
  });
  addTo = jest.fn((map: MockMap) => {
    this.addedTo = map;
    return this;
  });
  remove = jest.fn(() => {
    this.removed = true;
  });

  constructor(options: Record<string, unknown> = {}) {
    this.options = options;
    markers.push(this);
  }
}

export class NavigationControlMock {}
