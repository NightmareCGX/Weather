"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { toEnsembleFanData } from "@/lib/forecast/transform";
import { formatValue } from "@/lib/forecast/labels";
import { formatDayHourInTimeZone, formatDayHourWithTimeZone } from "@/lib/forecast/time";
import type { EnsembleStatisticsData } from "@/lib/api/types";

interface EnsembleChartProps {
  byLead: ReadonlyMap<number, EnsembleStatisticsData>;
  variableLabel: string;
  timezone?: string | null;
  validTimesByLead?: ReadonlyMap<number, string>;
  unit?: string;
}

export interface EnsembleChartTooltipProps {
  active?: boolean;
  payload?: any[];
  timezone?: string | null;
  unit?: string;
}

/**
 * Custom tooltip for the ensemble percentile fan chart.
 *
 * Renders meaningful weather statistics:
 * - Median (P50) & Mean
 * - P25–P75 central range (50% ensemble members)
 * - P10–P90 outer percentile range (80% ensemble members)
 *
 * Hides internal stacked-area base coordinates (p10Base, p25Base) entirely.
 */
export function EnsembleChartTooltip({
  active,
  payload,
  timezone,
  unit = "",
}: EnsembleChartTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const point = payload[0]?.payload;
  if (!point) {
    return null;
  }

  const validTime = point.valid_time;
  const timeLabel = validTime ? formatDayHourWithTimeZone(validTime, timezone) : "";

  const formatVal = (v: number | null | undefined) =>
    v !== null && v !== undefined && typeof v === "number" && Number.isFinite(v)
      ? formatValue(v, unit)
      : "—";

  const formatRange = (low: number | null | undefined, high: number | null | undefined) => {
    const hasLow =
      low !== null && low !== undefined && typeof low === "number" && Number.isFinite(low);
    const hasHigh =
      high !== null && high !== undefined && typeof high === "number" && Number.isFinite(high);
    if (!hasLow && !hasHigh) return "—";
    const lowStr = hasLow ? formatValue(low as number, "") : "—";
    const highStr = hasHigh ? formatValue(high as number, unit) : "—";
    return `${lowStr} – ${highStr}`;
  };

  return (
    <div
      className="recharts-default-tooltip rounded-lg border border-slate-700 bg-slate-900/95 p-2.5 text-xs text-slate-200 shadow-2xl backdrop-blur-md"
      style={{ whiteSpace: "nowrap" }}
    >
      {timeLabel && <p className="mb-1.5 font-semibold text-slate-100">{timeLabel}</p>}
      <div className="space-y-1 font-mono">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-400 font-sans">
            <span className="h-2 w-2 rounded-full bg-[#38bdf8]" />
            Median (P50)
          </span>
          <span className="font-medium text-slate-100 tabular-nums">{formatVal(point.median)}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-400 font-sans">
            <span className="h-2 w-2 rounded-full bg-[#f59e0b]" />
            Mean
          </span>
          <span className="font-medium text-slate-100 tabular-nums">{formatVal(point.mean)}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-400 font-sans">
            <span className="h-2 w-2 rounded-sm bg-[#0284c7]" />
            P25–P75
          </span>
          <span className="font-medium text-slate-100 tabular-nums">
            {formatRange(point.p25, point.p75)}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-400 font-sans">
            <span className="h-2 w-2 rounded-sm bg-[#38bdf8]" />
            P10–P90
          </span>
          <span className="font-medium text-slate-100 tabular-nums">
            {formatRange(point.p10, point.p90)}
          </span>
        </div>
      </div>
    </div>
  );
}

/**
 * Ensemble statistics / spread over forecast valid time.
 *
 * Renders the P10–P90 outer percentile band and the P25–P75 central range as a
 * fan (two stacked areas), with the median (P50) and mean as lines. This is a
 * mathematically honest summary of the `/v1/ensembles` statistics — it is
 * explicitly labeled "percentile range", never a min/max boxplot, because the
 * backend does not expose min/max or raw members.
 */
export function EnsembleChart({
  byLead,
  variableLabel,
  timezone,
  validTimesByLead,
  unit,
}: EnsembleChartProps) {
  const data = toEnsembleFanData(byLead, validTimesByLead).map((point) => {
    const validTime =
      point.valid_time ??
      validTimesByLead?.get(point.lead_time_hours) ??
      byLead.get(point.lead_time_hours)?.valid_time ??
      "";
    return {
      lead_time_hours: point.lead_time_hours,
      valid_time: validTime,
      label: validTime ? formatDayHourInTimeZone(validTime, timezone) : "",
      p10Base: point.p10Base,
      p90Height: point.p90Height,
      p25Base: point.p25Base,
      p75Height: point.p75Height,
      median: point.median,
      mean: point.mean,
      p10: point.p10,
      p25: point.p25,
      p50: point.p50,
      p75: point.p75,
      p90: point.p90,
    };
  });

  return (
    <div className="mb-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-sm font-medium text-slate-800">{variableLabel} — percentile range</h4>
        <span className="text-xs text-slate-500">P10–P90 band · P25–P75 box · median · mean</span>
      </div>
      <div
        role="img"
        aria-label={`${variableLabel} ensemble percentile fan over time`}
        className="h-48 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="valid_time"
              tickFormatter={(vt: string) => (vt ? formatDayHourInTimeZone(vt, timezone) : "")}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              width={46}
              domain={["auto", "auto"]}
            />
            <Tooltip
              content={<EnsembleChartTooltip timezone={timezone} unit={unit} />}
              formatter={(value: any, name: string, item: any) => {
                if (
                  name === undefined ||
                  item?.dataKey === "p10Base" ||
                  item?.dataKey === "p25Base"
                ) {
                  return null;
                }
                if (
                  value === null ||
                  value === undefined ||
                  typeof value !== "number" ||
                  !Number.isFinite(value)
                ) {
                  return ["—", name];
                }
                return [formatValue(value, unit ?? ""), name];
              }}
              labelFormatter={(label: string, payload: any[]) => {
                const validTime = payload?.[0]?.payload?.valid_time ?? label;
                return validTime ? formatDayHourWithTimeZone(validTime, timezone) : "";
              }}
              contentStyle={{ fontSize: 12 }}
            />
            {/* Transparent base stacks so the colored heights render as bands. */}
            <Area
              dataKey="p10Base"
              stackId="p10"
              stroke="none"
              fill="none"
              isAnimationActive={false}
              connectNulls={false}
              tooltipType="none"
            />
            <Area
              dataKey="p90Height"
              stackId="p10"
              stroke="none"
              fill="#38bdf8"
              fillOpacity={0.25}
              isAnimationActive={false}
              connectNulls={false}
              name="P10–P90"
            />
            <Area
              dataKey="p25Base"
              stackId="p25"
              stroke="none"
              fill="none"
              isAnimationActive={false}
              connectNulls={false}
              tooltipType="none"
            />
            <Area
              dataKey="p75Height"
              stackId="p25"
              stroke="none"
              fill="#0284c7"
              fillOpacity={0.4}
              isAnimationActive={false}
              connectNulls={false}
              name="P25–P75"
            />
            <Line
              dataKey="median"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
              name="Median (P50)"
            />
            <Line
              dataKey="mean"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
              name="Mean"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
