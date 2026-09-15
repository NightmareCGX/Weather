"use client";

import { useCallback, useEffect, useState } from "react";

export const PARTICLES_ENABLED_STORAGE_KEY = "zeus_weather_particles_enabled";

/**
 * Inspect whether the client system currently prefers reduced motion.
 */
function getSystemReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Read the user's explicit preference override from localStorage safely.
 * Returns boolean if a valid override is saved, or null if unconfigured.
 */
function getStoredPreference(): boolean | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const item = window.localStorage.getItem(PARTICLES_ENABLED_STORAGE_KEY);
    if (item === "true") return true;
    if (item === "false") return false;
  } catch {
    // Storage access may throw in restricted sandboxes or private browsing.
  }
  return null;
}

/**
 * Determines whether wind particle animation should be enabled.
 *
 * Rules:
 * 1. An explicit user override in localStorage always takes precedence.
 * 2. If no user override exists, defaults to enabled EXCEPT when the client
 *    prefers reduced motion (prefers-reduced-motion: reduce).
 * 3. Dynamically reacts to system reduced-motion preference changes when no
 *    user override is stored.
 */
export function useParticlePreference() {
  const [particlesEnabled, setParticlesEnabledState] = useState<boolean>(() => {
    const stored = getStoredPreference();
    if (stored !== null) {
      return stored;
    }
    return !getSystemReducedMotion();
  });

  const [hasExplicitOverride, setHasExplicitOverride] = useState<boolean>(() => {
    return getStoredPreference() !== null;
  });

  // Listen for media query changes if the user hasn't explicitly set a preference
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia || hasExplicitOverride) {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handleChange = (e: MediaQueryListEvent) => {
      // Re-check storage just in case another tab set it
      const stored = getStoredPreference();
      if (stored !== null) {
        setHasExplicitOverride(true);
        setParticlesEnabledState(stored);
        return;
      }
      setParticlesEnabledState(!e.matches);
    };

    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    } else if (typeof mediaQuery.addListener === "function") {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, [hasExplicitOverride]);

  const setParticlesEnabled = useCallback((enabled: boolean) => {
    try {
      window.localStorage.setItem(PARTICLES_ENABLED_STORAGE_KEY, String(enabled));
    } catch {
      // Ignore localStorage write failures
    }
    setHasExplicitOverride(true);
    setParticlesEnabledState(enabled);
  }, []);

  const toggleParticles = useCallback(() => {
    setParticlesEnabled(!particlesEnabled);
  }, [particlesEnabled, setParticlesEnabled]);

  return {
    particlesEnabled,
    hasExplicitOverride,
    setParticlesEnabled,
    toggleParticles,
  };
}
