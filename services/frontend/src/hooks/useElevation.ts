"use client";

import { useEffect, useState } from "react";

import { getElevation, RequestAbortedError } from "@/lib/api/client";
import type { SelectedLocation } from "@/lib/api/types";

export type ElevationFetchStatus = "idle" | "loading" | "success" | "error";

export interface UseElevationResult {
  elevation_m: number | null;
  status: ElevationFetchStatus;
  error: string | null;
}

/**
 * Resolve terrain elevation for the selected location (Tier 1 known or Tier 2 dynamic).
 *
 * If the selected location already carries an authoritative elevation (cities,
 * ski resorts, stations), it is returned immediately without firing an API request.
 *
 * For dynamic coordinates (map clicks, place autocomplete without elevation),
 * this hook fires GET /v1/elevation independently of point forecast loading.
 *
 * Stale responses from rapid location switching are cancelled via AbortController
 * and discarded via an active flag.
 */
export function useElevation(location: SelectedLocation | null): UseElevationResult {
  const knownElevation = location?.elevation_m ?? null;

  const [state, setState] = useState<{
    elevation_m: number | null;
    status: ElevationFetchStatus;
    error: string | null;
  }>(() => {
    if (location === null) {
      return { elevation_m: null, status: "idle", error: null };
    }
    if (knownElevation !== null) {
      return { elevation_m: knownElevation, status: "success", error: null };
    }
    return { elevation_m: null, status: "loading", error: null };
  });

  useEffect(() => {
    if (location === null) {
      setState({ elevation_m: null, status: "idle", error: null });
      return;
    }

    // Tier 1: Known location already has elevation in metadata -> no API call
    if (location.elevation_m !== null) {
      setState({
        elevation_m: location.elevation_m,
        status: "success",
        error: null,
      });
      return;
    }

    // Tier 2: Dynamic coordinates -> query /v1/elevation independently
    const controller = new AbortController();
    let active = true;

    setState({ elevation_m: null, status: "loading", error: null });

    getElevation({
      latitude: location.latitude,
      longitude: location.longitude,
      signal: controller.signal,
    })
      .then((res) => {
        if (!active) return;
        setState({
          elevation_m: res.elevation_m,
          status: "success",
          error: null,
        });
      })
      .catch((err: unknown) => {
        if (!active || err instanceof RequestAbortedError) return;
        setState({
          elevation_m: null,
          status: "error",
          error: err instanceof Error ? err.message : "Failed to load elevation.",
        });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [location]);

  return state;
}
