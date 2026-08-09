"use client";

import type { Model } from "@/lib/api/types";

interface LayerControlsProps {
  models: Model[];
  model: string;
  variable: string;
  leadTimeHours: number;
  onModelChange: (model: string) => void;
  onVariableChange: (variable: string) => void;
  onLeadTimeChange: (leadTimeHours: number) => void;
}

const VARIABLES = [
  { value: "temperature_2m", label: "Temperature (2 m)" },
  { value: "precipitation_rate", label: "Precipitation Rate" },
];

const LEAD_TIMES = [0, 6, 12, 24, 48, 72];

/**
 * Presentation-only controls for the map layer configuration. All state and
 * callbacks are passed as props (ENGINEERING_CONTRACT section 7).
 */
export function LayerControls({
  models,
  model,
  variable,
  leadTimeHours,
  onModelChange,
  onVariableChange,
  onLeadTimeChange,
}: LayerControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-slate-200 bg-white px-4 py-2">
      <label className="flex items-center gap-2 text-sm text-slate-700">
        Model
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={model}
          onChange={(event) => onModelChange(event.target.value)}
          aria-label="Model"
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        Variable
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={variable}
          onChange={(event) => onVariableChange(event.target.value)}
          aria-label="Variable"
        >
          {VARIABLES.map((v) => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        Lead time
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={leadTimeHours}
          onChange={(event) => onLeadTimeChange(Number(event.target.value))}
          aria-label="Lead time"
        >
          {LEAD_TIMES.map((lead) => (
            <option key={lead} value={lead}>
              +{lead}h
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
