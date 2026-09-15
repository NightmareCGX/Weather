import { act, renderHook } from "@testing-library/react";
import { PARTICLES_ENABLED_STORAGE_KEY, useParticlePreference } from "../useParticlePreference";

describe("useParticlePreference", () => {
  let matchMediaListeners: Array<(e: { matches: boolean }) => void> = [];
  let matchesReducedMotion = false;

  beforeEach(() => {
    window.localStorage.clear();
    matchMediaListeners = [];
    matchesReducedMotion = false;

    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes("prefers-reduced-motion: reduce") ? matchesReducedMotion : false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn((event: string, cb: (e: { matches: boolean }) => void) => {
        if (event === "change") {
          matchMediaListeners.push(cb);
        }
      }),
      removeEventListener: jest.fn((event: string, cb: (e: { matches: boolean }) => void) => {
        matchMediaListeners = matchMediaListeners.filter((l) => l !== cb);
      }),
      dispatchEvent: jest.fn(),
    }));
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("defaults to true when system does not prefer reduced motion and no storage exists", () => {
    matchesReducedMotion = false;
    const { result } = renderHook(() => useParticlePreference());

    expect(result.current.particlesEnabled).toBe(true);
    expect(result.current.hasExplicitOverride).toBe(false);
  });

  it("defaults to false when system prefers reduced motion and no storage exists", () => {
    matchesReducedMotion = true;
    const { result } = renderHook(() => useParticlePreference());

    expect(result.current.particlesEnabled).toBe(false);
    expect(result.current.hasExplicitOverride).toBe(false);
  });

  it("respects explicit localStorage override regardless of system reduced-motion", () => {
    matchesReducedMotion = true;
    window.localStorage.setItem(PARTICLES_ENABLED_STORAGE_KEY, "true");

    const { result } = renderHook(() => useParticlePreference());

    expect(result.current.particlesEnabled).toBe(true);
    expect(result.current.hasExplicitOverride).toBe(true);
  });

  it("toggles preference and persists to localStorage", () => {
    matchesReducedMotion = false;
    const { result } = renderHook(() => useParticlePreference());

    expect(result.current.particlesEnabled).toBe(true);

    act(() => {
      result.current.toggleParticles();
    });

    expect(result.current.particlesEnabled).toBe(false);
    expect(result.current.hasExplicitOverride).toBe(true);
    expect(window.localStorage.getItem(PARTICLES_ENABLED_STORAGE_KEY)).toBe("false");

    act(() => {
      result.current.toggleParticles();
    });

    expect(result.current.particlesEnabled).toBe(true);
    expect(window.localStorage.getItem(PARTICLES_ENABLED_STORAGE_KEY)).toBe("true");
  });

  it("dynamically reacts to media query changes when unconfigured", () => {
    matchesReducedMotion = false;
    const { result } = renderHook(() => useParticlePreference());

    expect(result.current.particlesEnabled).toBe(true);

    act(() => {
      for (const listener of matchMediaListeners) {
        listener({ matches: true });
      }
    });

    expect(result.current.particlesEnabled).toBe(false);
  });
});
