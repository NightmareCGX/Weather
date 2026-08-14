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

export function useSearch(query: string): UseSearchResult {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  // A fresh session token per search session. It is stable while the user
  // types (so every keystroke shares one billing session) and regenerated
  // when the query returns to empty (a new search session begins).
  const [sessionToken, setSessionToken] = useState<string>(() => newSessionToken());

  // Track the latest query for a terminal-state guard (avoid setting state on
  // a stale timer after unmount or a newer query).
  const queryRef = useRef(query);
  queryRef.current = query;
  // Track whether the previous render had an active (>= MIN) query, so we can
  // detect the empty->active transition and mint a new session token without a
  // setState-in-effect loop.
  const wasActiveRef = useRef(false);
  // Keep the current session token available to the search effect without
  // making it a dependency (which would re-fire the search on every token
  // change).
  const sessionTokenRef = useRef(sessionToken);
  sessionTokenRef.current = sessionToken;

  useEffect(() => {
    const trimmed = query.trim();
    const active = trimmed.length >= MIN_QUERY_LENGTH;
    if (active && !wasActiveRef.current) {
      // A new search session begins: fresh token for Google billing.
      setSessionToken(newSessionToken());
    }
    wasActiveRef.current = active;

    if (!active) {
      setResults([]);
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setStatus("loading");
      setError(null);
      searchLocations({
        q: trimmed,
        type: "place",
        limit: SEARCH_LIMIT,
        sessionToken: sessionTokenRef.current,
        signal: controller.signal,
      })
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

  return { results, status, error, sessionToken };
}
