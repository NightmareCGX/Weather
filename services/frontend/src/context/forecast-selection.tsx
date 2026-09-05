"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getForecastAvailability, RequestAbortedError } from "@/lib/api/client";
import type { ForecastAvailability } from "@/lib/api/types";
import {
  buildForecastOptions,
  defaultModel,
  defaultValidTime,
  defaultVariable,
  extractVariableValidTimes,
  findModel,
  findVariable,
  isWithinGraceWindow,
  type ForecastOptions,
  type ForecastSelection,
} from "@/lib/forecast/availability";

export type AvailabilityStatus = "idle" | "loading" | "success" | "error";

export interface ForecastSelectionContextValue {
  /** Raw availability payload (database-driven), or null before/after failure. */
  availability: ForecastAvailability | null;
  status: AvailabilityStatus;
  error: string | null;
  /** The current forecast selection (or null when nothing is available). */
  selection: ForecastSelection | null;
  /** The currently selected valid time (ISO 8601 UTC). */
  validTime: string | null;
  /** The options available at each level for the current selection. */
  options: ForecastOptions;
  setModel: (model: string) => void;
  setVariable: (variable: string) => void;
  setValidTime?: (validTime: string) => void;
  /** Legacy compatibility setters. */
  setInitialTime?: (initialTime: string) => void;
  setLeadTimeHours?: (leadTimeHours: number) => void;
  retry: () => void;
}

const ForecastSelectionContext = createContext<ForecastSelectionContextValue | null>(null);

/**
 * Provides the database-driven forecast selection state for the map and dashboard (Lifecycle V2).
 *
 * User-facing selection is model -> variable -> validTime.
 * Initial Time and Lead Time are removed as primary selectors.
 * Selectable valid times are filtered by the 3-hour UI grace window (valid_time >= now - 3h).
 * A periodic 60-second timer automatically re-evaluates the grace window so expired times
 * age out in long-lived browser sessions without a page reload.
 */
export function ForecastSelectionProvider({ children }: { children: ReactNode }) {
  const [availability, setAvailability] = useState<ForecastAvailability | null>(null);
  const [status, setStatus] = useState<AvailabilityStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<ForecastSelection | null>(null);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());

  // 60-second periodic timer to advance nowMs for the 3h UI grace window
  useEffect(() => {
    const timer = setInterval(() => {
      setNowMs(Date.now());
    }, 60_000);
    return () => clearInterval(timer);
  }, []);

  const load = useCallback((signal?: AbortSignal) => {
    setStatus("loading");
    setError(null);
    getForecastAvailability(signal)
      .then((next) => {
        setAvailability(next);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (err instanceof RequestAbortedError) return;
        setAvailability(null);
        setError(err instanceof Error ? err.message : "Unable to load forecast data.");
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const retry = useCallback(() => {
    load();
  }, [load]);

  const options = useMemo(
    () => buildForecastOptions(availability, selection, nowMs),
    [availability, selection, nowMs]
  );

  const setModel = useCallback(
    (modelId: string) => {
      setSelection((current) => {
        const model = findModel(availability, modelId);
        const variable = defaultVariable(model);
        const variableEntry = findVariable(model, variable);
        const validTimes = extractVariableValidTimes(variableEntry);
        // Preserve currently selected validTime if still available and within grace window
        const nextVt =
          current !== null &&
          current.validTime !== undefined &&
          validTimes.includes(current.validTime) &&
          isWithinGraceWindow(current.validTime, nowMs)
            ? current.validTime
            : defaultValidTime(validTimes, nowMs);

        if (variable === null || nextVt === null) {
          return current !== null && current.model === modelId ? current : null;
        }
        return { model: modelId, variable, validTime: nextVt };
      });
    },
    [availability, nowMs]
  );

  const setVariable = useCallback(
    (variableId: string) => {
      setSelection((current) => {
        const model = current !== null ? findModel(availability, current.model) : null;
        const variableEntry = findVariable(model, variableId);
        const validTimes = extractVariableValidTimes(variableEntry);
        // Preserve currently selected validTime if available in new variable
        const nextVt =
          current !== null &&
          current.validTime !== undefined &&
          validTimes.includes(current.validTime) &&
          isWithinGraceWindow(current.validTime, nowMs)
            ? current.validTime
            : defaultValidTime(validTimes, nowMs);

        if (current === null || variableEntry === null || nextVt === null) {
          return current;
        }
        return {
          model: current.model,
          variable: variableId,
          validTime: nextVt,
        };
      });
    },
    [availability, nowMs]
  );

  const setValidTime = useCallback((validTime: string) => {
    setSelection((current) => {
      if (current === null) {
        return current;
      }
      return { ...current, validTime };
    });
  }, []);

  // Legacy setters for backwards compatibility with any remaining callers
  const setInitialTime = useCallback((initialTimeValue: string) => {
    setSelection((current) => {
      if (current === null) return current;
      return { ...current, initialTime: initialTimeValue };
    });
  }, []);

  const setLeadTimeHours = useCallback((leadTimeHours: number) => {
    setSelection((current) => {
      if (current === null) return current;
      return { ...current, leadTimeHours };
    });
  }, []);

  // Synchronize selection with availability and grace window
  useEffect(() => {
    if (status !== "success" || availability === null) {
      return;
    }
    setSelection((current) => {
      const modelId = current !== null ? current.model : null;
      const model = findModel(availability, modelId);
      const variableCode = current !== null ? current.variable : null;
      const variable = model !== null ? findVariable(model, variableCode) : null;
      const allValidTimes = extractVariableValidTimes(variable);

      // If the current selection is still fully valid and within grace window, keep it
      if (
        model !== null &&
        variable !== null &&
        current !== null &&
        current.validTime !== undefined &&
        allValidTimes.includes(current.validTime) &&
        isWithinGraceWindow(current.validTime, nowMs)
      ) {
        return current;
      }

      const nextModel = model ?? findModel(availability, defaultModel(availability));
      const nextVariable = defaultVariable(nextModel);
      const nextVariableEntry = findVariable(nextModel, nextVariable);
      const nextValidTimes = extractVariableValidTimes(nextVariableEntry);
      const nextVt = defaultValidTime(nextValidTimes, nowMs);

      if (nextModel === null || nextVariable === null || nextVt === null) {
        return null;
      }
      return {
        model: nextModel.id,
        variable: nextVariable,
        validTime: nextVt,
      };
    });
  }, [availability, status, nowMs]);

  const validTime = selection?.validTime ?? null;

  const value = useMemo<ForecastSelectionContextValue>(
    () => ({
      availability,
      status,
      error,
      selection,
      validTime,
      options,
      setModel,
      setVariable,
      setValidTime,
      setInitialTime,
      setLeadTimeHours,
      retry,
    }),
    [
      availability,
      status,
      error,
      selection,
      validTime,
      options,
      setModel,
      setVariable,
      setValidTime,
      setInitialTime,
      setLeadTimeHours,
      retry,
    ]
  );

  return (
    <ForecastSelectionContext.Provider value={value}>{children}</ForecastSelectionContext.Provider>
  );
}

export function useForecastSelection(): ForecastSelectionContextValue {
  const context = useContext(ForecastSelectionContext);
  if (context === null) {
    throw new Error("useForecastSelection must be used within a ForecastSelectionProvider");
  }
  return context;
}
