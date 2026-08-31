"use client";

import { useEffect, useState } from "react";
import { getVectorField, RequestAbortedError } from "@/lib/api/client";
import type { VectorFieldData } from "@/lib/api/types";
import { getCachedVectorField, setCachedVectorField } from "@/lib/map/vectorFieldCache";

export interface UseVectorFieldOptions {
  vectorFieldUrl: string | null;
  availableLeads?: number[];
  currentLead?: number;
  enabled?: boolean;
}

export interface UseVectorFieldResult {
  field: VectorFieldData | null;
  loading: boolean;
  error: string | null;
}

function shouldSkipPrefetch(): boolean {
  if (typeof navigator === "undefined") return false;
  const nav = navigator as Navigator & {
    connection?: {
      saveData?: boolean;
      effectiveType?: string;
    };
  };
  if (nav.connection?.saveData === true) return true;
  if (nav.connection?.effectiveType === "2g" || nav.connection?.effectiveType === "slow-2g") {
    return true;
  }
  return false;
}

/**
 * Manages foreground fetching, progressive loading, in-memory caching,
 * and adjacent-lead prefetching for the 10 m wind vector field.
 */
export function useVectorField({
  vectorFieldUrl,
  availableLeads = [],
  currentLead,
  enabled = true,
}: UseVectorFieldOptions): UseVectorFieldResult {
  const [field, setField] = useState<VectorFieldData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Foreground fetch effect
  useEffect(() => {
    if (!enabled || !vectorFieldUrl) {
      setField(null);
      setLoading(false);
      setError(null);
      return;
    }

    const cached = getCachedVectorField(vectorFieldUrl);
    if (cached !== undefined) {
      setField(cached);
      setLoading(false);
      setError(null);
      return;
    }

    // Unset current field to immediately stop stale animation frames during transition
    setField(null);
    setLoading(true);
    setError(null);

    const controller = new AbortController();

    getVectorField(vectorFieldUrl, controller.signal)
      .then((data) => {
        setCachedVectorField(vectorFieldUrl, data);
        setField(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (err instanceof RequestAbortedError) {
          return;
        }
        setField(null);
        setError(err instanceof Error ? err.message : "Failed to load wind vector field.");
        setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [vectorFieldUrl, enabled]);

  // 2. Adjacent lead prefetch effect
  useEffect(() => {
    if (!enabled || !vectorFieldUrl || field === null || currentLead === undefined) {
      return;
    }
    if (shouldSkipPrefetch()) {
      return;
    }

    // Find adjacent leads (e.g. lead index - 1 and lead index + 1)
    const sortedLeads = [...availableLeads].sort((a, b) => a - b);
    const currIdx = sortedLeads.indexOf(currentLead);
    if (currIdx === -1) return;

    const adjacentLeads: number[] = [];
    if (currIdx > 0) adjacentLeads.push(sortedLeads[currIdx - 1]);
    if (currIdx < sortedLeads.length - 1) adjacentLeads.push(sortedLeads[currIdx + 1]);

    const prefetchControllers: AbortController[] = [];

    for (const lead of adjacentLeads) {
      const prefetchUrl = vectorFieldUrl.replace(
        `lead_time_hours=${currentLead}`,
        `lead_time_hours=${lead}`
      );
      if (getCachedVectorField(prefetchUrl) !== undefined) {
        continue;
      }

      const controller = new AbortController();
      prefetchControllers.push(controller);

      getVectorField(prefetchUrl, controller.signal)
        .then((data) => {
          setCachedVectorField(prefetchUrl, data);
        })
        .catch(() => {
          // Prefetch failures are silently ignored and never affect active state
        });
    }

    return () => {
      for (const ctrl of prefetchControllers) {
        ctrl.abort();
      }
    };
  }, [vectorFieldUrl, field, currentLead, availableLeads, enabled]);

  return { field, loading, error };
}
