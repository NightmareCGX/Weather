"use client";

import React, { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  GEFS_PHYSICAL_PHASES,
  GEFS_PHASE_LABELS,
  PRECIPITATION_PHASE_TOKENS,
  formatTransitionName,
} from "@/lib/forecast/precipitation";
import { formatPercent } from "@/lib/forecast/labels";
import { formatDayHourInTimeZone, formatDayHourWithTimeZone } from "@/lib/forecast/time";
import {
  toEnsemblePhaseSupportData,
  type EnsemblePhaseSupportPoint,
} from "@/lib/forecast/transform";
import type { EnsembleStatisticsData } from "@/lib/api/types";

export interface EnsemblePhaseSupportProps {
  byLead?: ReadonlyMap<number, EnsembleStatisticsData>;
  validTimesByLead?: ReadonlyMap<number, string>;
  timezone?: string | null;
  phaseSupport?: Record<string, number>;
  transitionFrequency?: Record<string, number> | null;
  selectedLead?: number | string;
  validTime?: string | null;
  memberCount?: number;
}

function renderBarLabel(props: any) {
  const { x, y, width, height, value } = props;
  if (value == null || typeof value !== "number" || value < 12 || height < 14) {
    return null;
  }
  return (
    <text
      x={x + width / 2}
      y={y + height / 2}
      fill="#ffffff"
      textAnchor="middle"
      dominantBaseline="central"
      fontSize={10}
      fontWeight={600}
      style={{ pointerEvents: "none", textShadow: "0 1px 2px rgba(0,0,0,0.5)" }}
    >
      {Math.round(value)}%
    </text>
  );
}

export function EnsemblePhaseSupportTooltip({ active, payload, label, timezone }: any) {
  if (!active || !payload || payload.length === 0) return null;

  const point = payload[0]?.payload as EnsemblePhaseSupportPoint | undefined;
  const validTime = point?.valid_time ?? label ?? "";
  const formattedTime = validTime ? formatDayHourWithTimeZone(validTime, timezone) : "";
  const memberCount = point?.valid_member_count ?? point?.member_count ?? 30;

  if (!point || !point.has_data) {
    return (
      <div className="rounded border border-slate-200 bg-white p-2.5 shadow-md text-xs">
        <p className="font-semibold text-slate-800">{formattedTime}</p>
        <p className="text-slate-500 mt-1">No ensemble phase data available</p>
      </div>
    );
  }

  return (
    <div className="rounded border border-slate-200 bg-white p-2.5 shadow-md text-xs min-w-[190px]">
      <div className="border-b border-slate-100 pb-1 mb-1.5 flex items-baseline justify-between gap-2">
        <p className="font-semibold text-slate-800">{formattedTime}</p>
        <span className="text-[10px] text-slate-500">{memberCount} members</span>
      </div>
      <div className="space-y-1">
        {GEFS_PHYSICAL_PHASES.map((phase) => {
          const val = point[phase];
          if (val == null || val <= 0) return null;
          const token = PRECIPITATION_PHASE_TOKENS[phase];
          return (
            <div key={phase} className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-1.5">
                <span
                  className="h-2 w-2 rounded-sm shrink-0"
                  style={{ backgroundColor: token?.color ?? "#64748b" }}
                />
                <span className="text-slate-700">{GEFS_PHASE_LABELS[phase]}</span>
              </div>
              <span className="font-medium text-slate-900 tabular-nums">
                {Math.round(val * 10) / 10}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * GEFS Ensemble Phase Support Visualization.
 *
 * Renders a TIME-VARYING 100% stacked bar chart of normalized ensemble support
 * across forecast valid times, representing the established physical phase taxonomy:
 * - Dry
 * - Rain
 * - Snow
 * - Freezing Rain
 * - Ice Pellets
 * - Unknown
 *
 * Invariants:
 * 1. There is NO "Mixed" category (members distribute support across physical phases).
 * 2. Unknown is preserved and rendered if non-zero.
 * 3. Each valid-time stack normalizes to 100% using available members.
 * 4. Missing/unusable phase data at t[n] does not truncate later times t[n+1].
 * 5. Timezone localization aligns with display timezone without secondary implementations.
 */
export function EnsemblePhaseSupport({
  byLead,
  validTimesByLead,
  timezone,
  phaseSupport,
  transitionFrequency,
  selectedLead,
  validTime,
  memberCount = 30,
}: EnsemblePhaseSupportProps) {
  // Construct effective byLead map, supporting single-lead snapshot fallback
  const effectiveByLead = useMemo(() => {
    if (byLead && byLead.size > 0) return byLead;
    if (phaseSupport) {
      const map = new Map<number, EnsembleStatisticsData>();
      const lead = typeof selectedLead === "number" ? selectedLead : 0;
      map.set(lead, {
        model: "gefs",
        lead_time_hours: lead,
        valid_time: validTime ?? "2026-09-10T12:00:00Z",
        member_count: memberCount,
        valid_member_count: memberCount,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: phaseSupport,
        transition_frequency: transitionFrequency,
      });
      return map;
    }
    return new Map<number, EnsembleStatisticsData>();
  }, [byLead, phaseSupport, selectedLead, validTime, memberCount, transitionFrequency]);

  const chartData = useMemo(() => {
    return toEnsemblePhaseSupportData(effectiveByLead, validTimesByLead, timezone);
  }, [effectiveByLead, validTimesByLead, timezone]);

  // Aggregate transition frequencies if available
  const transitions = useMemo(() => {
    const map = new Map<string, number>();
    if (transitionFrequency) {
      for (const [k, v] of Object.entries(transitionFrequency)) {
        const val = typeof v === "number" ? v : Number(v);
        if (val > 0) map.set(k, val);
      }
    }
    effectiveByLead.forEach((entry) => {
      if (entry.transition_frequency) {
        for (const [k, v] of Object.entries(entry.transition_frequency)) {
          const val = typeof v === "number" ? v : Number(v);
          if (val > 0 && !map.has(k)) map.set(k, val);
        }
      }
    });
    return Array.from(map.entries()).sort((a, b) => b[1] - a[1]);
  }, [transitionFrequency, effectiveByLead]);

  return (
    <div className="mt-4 rounded border border-slate-200 bg-slate-50/50 p-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-xs font-semibold text-slate-800">
          Ensemble Phase Support — time-varying support (0–100%)
        </h4>
        <span className="text-[11px] text-slate-500">
          Normalized support across physical phases
        </span>
      </div>

      <p className="mb-2 text-[11px] text-slate-500">
        Per-valid-time ensemble member classification normalized to 100% across available members.
      </p>

      {/* Legend / Phase taxonomy badges */}
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-600">
        <span className="font-medium text-slate-500">Phases:</span>
        {GEFS_PHYSICAL_PHASES.map((phase) => {
          const token = PRECIPITATION_PHASE_TOKENS[phase];
          return (
            <span
              key={phase}
              className="inline-flex items-center gap-1"
              data-testid={`phase-badge-${phase}`}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm shrink-0"
                style={{ backgroundColor: token?.color ?? "#64748b" }}
              />
              {GEFS_PHASE_LABELS[phase]}
            </span>
          );
        })}
      </div>

      {/* 100% Stacked Bar Chart across Valid Times */}
      <div role="img" aria-label="Ensemble phase support over time" className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="valid_time"
              tickFormatter={(vt: string) => (vt ? formatDayHourInTimeZone(vt, timezone) : "")}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={40}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip content={<EnsemblePhaseSupportTooltip timezone={timezone} />} />
            {GEFS_PHYSICAL_PHASES.map((phase) => (
              <Bar
                key={phase}
                dataKey={phase}
                stackId="phase"
                fill={PRECIPITATION_PHASE_TOKENS[phase]?.color ?? "#64748b"}
                isAnimationActive={false}
                name={GEFS_PHASE_LABELS[phase]}
              >
                <LabelList dataKey={phase} content={renderBarLabel} />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
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
