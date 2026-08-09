"use client";

import { useEffect, useRef, useState } from "react";

import { RequestAbortedError, searchLocations } from "@/lib/api/client";
import type { SearchResult } from "@/lib/api/types";

/**
 * Debounced, abortable `/v1/search` hook for the autocomplete combobox.
 *
 * - Queries shorter than the minimum length (1, matching the backend's
 *   `min_length=1`) never fire.
 * - Each keystroke debounces by {@link DEBOUNCE_MS} and aborts the previous
 *   in-flight request, so a slow stale response can never clobber a newer one.
 * - `status` distinguishes idle / loading / success / error for the UI.
 */

export const MIN_QUERY_LENGTH = 1;
export const DEBOUNCE_MS = 300;
export const SEARCH_LIMIT = 20;

export type SearchStatus = "idle" | "loading" | "success" | "error";

export interface UseSearchResult {
  results: SearchResult[];
  status: SearchStatus;
  error: string | null;
}

export function useSearch(query: string): UseSearchResult {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  // Track the latest query for a terminal-state guard (avoid setting state on
  // a stale timer after unmount or a newer query).
  const queryRef = useRef(query);
  queryRef.current = query;

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < MIN_QUERY_LENGTH) {
      setResults([]);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setStatus("loading");
      setError(null);
      searchLocations({ q: trimmed, type: "all", limit: SEARCH_LIMIT, signal: controller.signal })
        .then((next) => {
          // Only the most recent query may publish results.
          if (queryRef.current.trim() === trimmed) {
            setResults(next);
            setStatus("success");
          }
        })
        .catch((err: unknown) => {
          if (err instanceof RequestAbortedError) {
            return; // A newer query superseded this one; stay silent.
          }
          if (queryRef.current.trim() === trimmed) {
            setResults([]);
            setError(err instanceof Error ? err.message : "Failed to search locations.");
            setStatus("error");
          }
        });
    }, DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  return { results, status, error };
}
