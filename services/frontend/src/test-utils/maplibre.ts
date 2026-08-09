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

export function clearInstances(): void {
  instances.length = 0;
}

export function getInstances(): MockMap[] {
  return instances;
}

export class MockMap {
  options: Record<string, unknown>;
  handlers = new Map<string, () => void>();
  sources = new Set<string>();
  layers = new Set<string>();
  removed = false;
  isLoaded = true;

  on = jest.fn((event: string, handler: () => void) => {
    this.handlers.set(event, handler);
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
  remove = jest.fn(() => {
    this.removed = true;
  });

  constructor(options: Record<string, unknown> = {}) {
    this.options = options;
    instances.push(this);
  }

  /** Invoke a registered event handler (e.g. the `load` handler). */
  fire(event: string): void {
    const handler = this.handlers.get(event);
    handler?.();
  }
}

export class NavigationControlMock {}
