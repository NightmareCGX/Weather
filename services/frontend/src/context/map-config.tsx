"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

export interface MapConfig {
  model: string;
  variable: string;
  leadTimeHours: number;
}

interface MapConfigContextValue extends MapConfig {
  setModel: (model: string) => void;
  setVariable: (variable: string) => void;
  setLeadTimeHours: (leadTimeHours: number) => void;
}

const MapConfigContext = createContext<MapConfigContextValue | null>(null);

const DEFAULT_MODEL = "gfs";
const DEFAULT_VARIABLE = "temperature_2m";
const DEFAULT_LEAD_TIME_HOURS = 12;

export function MapConfigProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [variable, setVariable] = useState(DEFAULT_VARIABLE);
  const [leadTimeHours, setLeadTimeHours] = useState(DEFAULT_LEAD_TIME_HOURS);

  const value = useMemo<MapConfigContextValue>(
    () => ({ model, variable, leadTimeHours, setModel, setVariable, setLeadTimeHours }),
    [model, variable, leadTimeHours]
  );

  return <MapConfigContext.Provider value={value}>{children}</MapConfigContext.Provider>;
}

export function useMapConfig(): MapConfigContextValue {
  const context = useContext(MapConfigContext);
  if (context === null) {
    throw new Error("useMapConfig must be used within a MapConfigProvider");
  }
  return context;
}
