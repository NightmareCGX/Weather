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
  defaultInitialTime,
  defaultLeadTime,
  defaultModel,
  defaultVariable,
  findInitialTime,
  findModel,
  findVariable,
  resolveValidTime,
  type ForecastOptions,
  type ForecastSelection,
} from "@/lib/forecast/availability";

export type AvailabilityStatus = "idle" | "loading" | "success" | "error";

export interface ForecastSelectionContextValue {
  /** Raw availability payload (database-driven), or null before/after failure. */
  availability: ForecastAvailability | null;
  status: AvailabilityStatus;
  error: string | null;
  /** The current cascading selection (or null when nothing is available). */
  selection: ForecastSelection | null;
  /** The computed valid time for the current selection. */
  validTime: string | null;
  /** The options available at each cascade level for the current selection. */
  options: ForecastOptions;
  setModel: (model: string) => void;
  setVariable: (variable: string) => void;
  setInitialTime: (initialTime: string) => void;
  setLeadTimeHours: (leadTimeHours: number) => void;
  retry: () => void;
}

const ForecastSelectionContext = createContext<ForecastSelectionContextValue | null>(null);

/**
 * Provides the database-driven forecast selection state for the map and
 * dashboard.
 *
 * The provider fetches `/v1/forecast/availability` once and derives every
 * selector option from it. When the payload arrives, a valid default selection
 * is chosen from real available values (model -> variable -> initial time ->
 * lead time). Changing an upstream selection resets stale downstream
 * selections to the first value actually available for the new upstream.
 *
 * No model/variable/initial-time/lead-time is ever hard-coded: an empty
 * database yields an empty selection and the UI renders an empty state.
 */
export function ForecastSelectionProvider({ children }: { children: ReactNode }) {
  const [availability, setAvailability] = useState<ForecastAvailability | null>(null);
  const [status, setStatus] = useState<AvailabilityStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<ForecastSelection | null>(null);

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
    () => buildForecastOptions(availability, selection),
    [availability, selection]
  );

  const setModel = useCallback(
    (modelId: string) => {
      setSelection((current) => {
        const model = findModel(availability, modelId);
        const variable = defaultVariable(model);
        const variableEntry = findVariable(model, variable);
        const initialTime = defaultInitialTime(variableEntry);
        const initialTimeEntry = findInitialTime(variableEntry, initialTime);
        const lead = defaultLeadTime(initialTimeEntry);
        if (variable === null || initialTime === null || lead === null) {
          return current !== null && current.model === modelId ? current : null;
        }
        return { model: modelId, variable, initialTime, leadTimeHours: lead };
      });
    },
    [availability]
  );

  const setVariable = useCallback(
    (variableId: string) => {
      setSelection((current) => {
        const model = current !== null ? findModel(availability, current.model) : null;
        const variable = findVariable(model, variableId);
        const initialTime = defaultInitialTime(variable);
        const initialTimeEntry = findInitialTime(variable, initialTime);
        const lead = defaultLeadTime(initialTimeEntry);
        if (current === null || variable === null || initialTime === null || lead === null) {
          return current;
        }
        return {
          model: current.model,
          variable: variableId,
          initialTime,
          leadTimeHours: lead,
        };
      });
    },
    [availability]
  );

  const setInitialTime = useCallback(
    (initialTimeValue: string) => {
      setSelection((current) => {
        if (current === null) {
          return current;
        }
        const model = findModel(availability, current.model);
        const variable = findVariable(model, current.variable);
        const entry = findInitialTime(variable, initialTimeValue);
        const lead = defaultLeadTime(entry);
        if (entry === null || lead === null) {
          return current;
        }
        return {
          model: current.model,
          variable: current.variable,
          initialTime: initialTimeValue,
          leadTimeHours: lead,
        };
      });
    },
    [availability]
  );

  const setLeadTimeHours = useCallback((leadTimeHours: number) => {
    setSelection((current) => {
      if (current === null) {
        return current;
      }
      return { ...current, leadTimeHours };
    });
  }, []);

  // Derive the default selection when availability first resolves and no
  // selection exists yet (or the current selection is no longer valid).
  useEffect(() => {
    if (status !== "success" || availability === null) {
      return;
    }
    setSelection((current) => {
      const modelId = current !== null ? current.model : null;
      const model = findModel(availability, modelId);
      const variableCode = current !== null ? current.variable : null;
      const variable = model !== null ? findVariable(model, variableCode) : null;
      const initialValue = current !== null ? current.initialTime : null;
      const initial = variable !== null ? findInitialTime(variable, initialValue) : null;
      const lead = current !== null ? current.leadTimeHours : null;

      // If the current selection is still fully valid, keep it.
      if (
        model !== null &&
        variable !== null &&
        initial !== null &&
        initial.lead_time_hours.includes(lead ?? -1)
      ) {
        return current;
      }

      const nextModel = model ?? findModel(availability, defaultModel(availability));
      const nextVariable = defaultVariable(nextModel);
      const nextVariableEntry = findVariable(nextModel, nextVariable);
      const nextInitial = defaultInitialTime(nextVariableEntry);
      const nextInitialEntry = findInitialTime(nextVariableEntry, nextInitial);
      const nextLead = defaultLeadTime(nextInitialEntry);
      if (
        nextModel === null ||
        nextVariable === null ||
        nextInitial === null ||
        nextLead === null
      ) {
        return null;
      }
      return {
        model: nextModel.id,
        variable: nextVariable,
        initialTime: nextInitial,
        leadTimeHours: nextLead,
      };
    });
  }, [availability, status]);

  const validTime = useMemo(
    () => resolveValidTime(selection?.initialTime ?? null, selection?.leadTimeHours ?? null),
    [selection]
  );

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
