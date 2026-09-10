"use client";

import { useEffect, useRef, useState } from "react";

import { RequestAbortedError, searchLocations } from "@/lib/api/client";
import type { SearchResult } from "@/lib/api/types";

/**
 * Debounced, abortable `/v1/search` hook for the autocomplete combobox.
 *
 * - Queries shorter than the minimum length (2, per the approved provider
 *   guidance) never fire.
 * - Each keystroke debounces by {@link DEBOUNCE_MS} and aborts the previous
 *   in-flight request, so a slow stale response can never clobber a newer one.
 * - `status` distinguishes idle / loading / success / error for the UI.
 * - A search **session token** (Google billing semantics) is generated when
 *   the query first becomes active and reused for the whole session, so the
 *   provider bills one Autocomplete request instead of one per keystroke.
 */

export const MIN_QUERY_LENGTH = 2;
export const DEBOUNCE_MS = 300;
export const SEARCH_LIMIT = 20;

export type SearchStatus = "idle" | "loading" | "success" | "error";

export interface SearchBiasCoordinates {
  latitude: number;
  longitude: number;
}

export interface UseSearchOptions {
  type?: "city" | "resort" | "station" | "all" | "place";
  bias?: SearchBiasCoordinates | null;
  getBias?: () => SearchBiasCoordinates | undefined;
}

export interface UseSearchResult {
  results: SearchResult[];
  status: SearchStatus;
  error: string | null;
  /** The active search-session token (for place resolution). */
  sessionToken: string;
}

function newSessionToken(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `tok-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function useSearch(query: string, options?: UseSearchOptions): UseSearchResult {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  // A fresh session token per search session. It is stable while the user
  // types (so every keystroke shares one billing session) and regenerated
  // when the query returns to empty (a new search session begins).
  const [sessionToken, setSessionToken] = useState<string>(() => newSessionToken());

  // Track whether the previous render had an active (>= MIN) query, so we can
  // detect the empty->active transition and mint a new session token without a
  // setState-in-effect loop.
  const wasActiveRef = useRef(false);
  // Keep the current session token available to the search effect without
  // making it a dependency (which would re-fire the search on every token
  // change).
  const sessionTokenRef = useRef(sessionToken);
  sessionTokenRef.current = sessionToken;

  // Keep bias and bias getter refs fresh without triggering query refetches when
  // map movement updates the map center (Rule B: map movement alone must NOT refetch).
  const getBiasRef = useRef(options?.getBias);
  getBiasRef.current = options?.getBias;
  const biasRef = useRef(options?.bias);
  biasRef.current = options?.bias;
  const searchTypeRef = useRef(options?.type ?? "place");
  searchTypeRef.current = options?.type ?? "place";

  useEffect(() => {
    let active = true;
    const trimmed = query.trim();
    const isQueryActive = trimmed.length >= MIN_QUERY_LENGTH;
    if (isQueryActive && !wasActiveRef.current) {
      // A new search session begins: fresh token for Google billing.
      setSessionToken(newSessionToken());
    }
    wasActiveRef.current = isQueryActive;

    if (!isQueryActive) {
      setResults([]);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setStatus("loading");
      setError(null);

      // Sample latest map-center bias at debounce fire time
      const activeBias = getBiasRef.current ? getBiasRef.current() : biasRef.current;
      const biasLat =
        activeBias && typeof activeBias.latitude === "number" ? activeBias.latitude : undefined;
      const biasLon =
        activeBias && typeof activeBias.longitude === "number" ? activeBias.longitude : undefined;

      searchLocations({
        q: trimmed,
        type: searchTypeRef.current,
        limit: SEARCH_LIMIT,
        sessionToken: sessionTokenRef.current,
        biasLat,
        biasLon,
        signal: controller.signal,
      })
        .then((next) => {
          if (!active) return;
          setResults(next);
          setStatus("success");
        })
        .catch((err: unknown) => {
          if (!active || err instanceof RequestAbortedError) {
            return; // A newer query superseded this one; stay silent.
          }
          setResults([]);
          setError(err instanceof Error ? err.message : "Failed to search locations.");
          setStatus("error");
        });
    }, DEBOUNCE_MS);

    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  return { results, status, error, sessionToken };
}
