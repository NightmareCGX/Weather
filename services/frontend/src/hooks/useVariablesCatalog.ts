"use client";

import { useEffect, useState } from "react";

import { listVariables, RequestAbortedError } from "@/lib/api/client";
import type { VariableResource } from "@/lib/api/types";

export interface UseVariablesCatalogResult {
  variables: VariableResource[];
  status: "idle" | "loading" | "success" | "error";
  error: string | null;
}

/**
 * Load the `/v1/variables` catalog once for chart axis labels and units.
 *
 * The catalog is a low-churn resource (24-hour cache), so it is fetched once
 * on mount. A failure degrades gracefully: the dashboard falls back to
 * {@link FALLBACK_VARIABLE_META} for the default variables, so a catalog
 * outage does not block chart rendering.
 */
export function useVariablesCatalog(): UseVariablesCatalogResult {
  const [variables, setVariables] = useState<VariableResource[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    listVariables(controller.signal)
      .then((next) => {
        setVariables(next);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (err instanceof RequestAbortedError) return;
        setVariables([]);
        setError(err instanceof Error ? err.message : "Failed to load the variable catalog.");
        setStatus("error");
      });
    return () => controller.abort();
  }, []);

  return { variables, status, error };
}
