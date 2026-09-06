"use client";

import React, { useState } from "react";
import {
  GEFS_PHYSICAL_PHASES,
  GEFS_PHASE_LABELS,
  PRECIPITATION_PHASE_TOKENS,
  formatTransitionName,
} from "@/lib/forecast/precipitation";
import { formatPercent } from "@/lib/forecast/labels";
import { formatDayHourUtc, formatLeadTimeHours } from "@/lib/forecast/time";
import type { PhysicalPhase } from "@/lib/api/types";

interface EnsemblePhaseSupportProps {
  phaseSupport: Record<string, number>;
  transitionFrequency?: Record<string, number> | null;
  selectedLead: number | string;
  memberCount?: number;
}

/**
 * GEFS Ensemble Phase Support Visualization.
 *
 * Renders a 100% stacked composition of normalized ensemble support across
 * physical precipitation phases:
 * - Dry
 * - Rain
 * - Snow
 * - Freezing Rain
 * - Ice Pellets
 * - Unknown
 *
 * Invariants:
 * 1. There is NO "Mixed" segment (transition members distribute 1.0 support
 *    equally across physical phases).
 * 2. Unknown is preserved and rendered if non-zero.
 * 3. Segments visually and mathematically sum to 100%.
 * 4. Phase support represents ensemble support fraction, not liquid mass
 *    fraction or duration probability.
 * 5. Secondary transition frequencies are rendered cleanly as context.
 */
export function EnsemblePhaseSupport({
  phaseSupport,
  transitionFrequency,
  selectedLead,
  memberCount = 30,
}: EnsemblePhaseSupportProps) {
  const [hoveredPhase, setHoveredPhase] = useState<PhysicalPhase | null>(null);

  // Compute percentages for each physical phase
  const phaseEntries = GEFS_PHYSICAL_PHASES.map((phase) => {
    const rawVal = phaseSupport[phase] ?? 0;
    const pct = rawVal * 100;
    return {
      phase,
      label: GEFS_PHASE_LABELS[phase],
      raw: rawVal,
      percentage: pct,
      color: PRECIPITATION_PHASE_TOKENS[phase]?.color ?? "#64748b",
      meta: PRECIPITATION_PHASE_TOKENS[phase],
    };
  });

  // Filter non-zero transitions sorted by frequency descending
  const transitions = Object.entries(transitionFrequency ?? {})
    .filter(([_, freq]) => freq > 0)
    .sort((a, b) => b[1] - a[1]);

  const leadLabel =
    typeof selectedLead === "string"
      ? `${formatDayHourUtc(selectedLead)} UTC`
      : formatLeadTimeHours(selectedLead);
  const ariaLead = typeof selectedLead === "number" ? `lead ${selectedLead}h` : leadLabel;

  return (
    <div className="mt-4 rounded border border-slate-200 bg-slate-50/50 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h4 className="text-xs font-semibold text-slate-800">
          Ensemble Phase Support ({leadLabel})
        </h4>
        <span className="text-[11px] text-slate-500">{memberCount} members · 100% total</span>
      </div>

      <p className="mb-3 text-[11px] text-slate-500">
        Normalized ensemble member support across physical precipitation phases.
      </p>

      {/* 100% Stacked Bar */}
      <div
        className="relative flex h-7 w-full overflow-hidden rounded-md border border-slate-200 bg-slate-100 shadow-inner"
        role="img"
        aria-label={`Ensemble phase support composition at ${ariaLead}`}
      >
        {phaseEntries.map(({ phase, label, raw, percentage, color }) => {
          if (percentage <= 0) return null;
          const isHovered = hoveredPhase === phase;

          return (
            <div
              key={phase}
              style={{
                width: `${percentage}%`,
                backgroundColor: color,
              }}
              onMouseEnter={() => setHoveredPhase(phase)}
              onMouseLeave={() => setHoveredPhase(null)}
              className={`relative h-full transition-opacity cursor-pointer ${
                isHovered ? "brightness-110 ring-2 ring-white/80 z-10" : ""
              }`}
              title={`${label}: ${formatPercent(raw)}`}
              data-testid={`phase-segment-${phase}`}
            >
              {percentage >= 12 && (
                <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-white drop-shadow-sm truncate px-1">
                  {formatPercent(raw)}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Phase Support Breakdown Badges */}
      <div className="mt-3 grid grid-cols-2 gap-1.5 text-xs">
        {phaseEntries.map(({ phase, label, raw, color }) => {
          const isNonZero = raw > 0;
          return (
            <div
              key={phase}
              onMouseEnter={() => setHoveredPhase(phase)}
              onMouseLeave={() => setHoveredPhase(null)}
              className={`flex items-center justify-between rounded px-2 py-1 transition-colors ${
                hoveredPhase === phase
                  ? "bg-slate-200"
                  : isNonZero
                    ? "bg-white border border-slate-200"
                    : "bg-transparent text-slate-400"
              }`}
            >
              <div className="flex items-center gap-1.5 truncate">
                <span
                  className="h-2.5 w-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: color }}
                />
                <span className="truncate">{label}</span>
              </div>
              <span className="font-medium tabular-nums ml-2 shrink-0">{formatPercent(raw)}</span>
            </div>
          );
        })}
      </div>

      {/* Secondary Transition Frequency */}
      {transitions.length > 0 && (
        <div className="mt-3.5 border-t border-slate-200 pt-2.5">
          <h5 className="text-[11px] font-medium text-slate-700 mb-1.5">
            Member Phase Transitions
          </h5>
          <div className="flex flex-wrap gap-1.5">
            {transitions.map(([trCode, freq]) => (
              <span
                key={trCode}
                className="inline-flex items-center gap-1 rounded bg-slate-100 px-2 py-0.5 text-[11px] text-slate-700 border border-slate-200"
                title={`${formatTransitionName(trCode)}: ${formatPercent(freq)} of members`}
              >
                <span className="font-medium">{formatTransitionName(trCode)}</span>
                <span className="text-slate-500 tabular-nums">· {formatPercent(freq)}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
