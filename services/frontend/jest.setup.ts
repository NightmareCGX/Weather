import "@testing-library/jest-dom";

// jsdom does not implement ResizeObserver, which Recharts' ResponsiveContainer
// requires. Provide a minimal no-op implementation so chart components render
// in component tests.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  };
}
