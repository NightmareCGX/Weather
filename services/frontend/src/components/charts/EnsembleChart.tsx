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
import { formatLeadTimeHours } from "@/lib/forecast/time";
import type { EnsembleStatisticsData } from "@/lib/api/types";

interface EnsembleChartProps {
  byLead: ReadonlyMap<number, EnsembleStatisticsData>;
  variableLabel: string;
}

/**
 * Ensemble statistics / spread over forecast lead time.
 *
 * Renders the P10–P90 outer percentile band and the P25–P75 central range as a
 * fan (two stacked areas), with the median (P50) and mean as lines. This is a
 * mathematically honest summary of the `/v1/ensembles` statistics — it is
 * explicitly labeled "percentile range", never a min/max boxplot, because the
 * backend does not expose min/max or raw members.
 */
export function EnsembleChart({ byLead, variableLabel }: EnsembleChartProps) {
  const data = toEnsembleFanData(byLead).map((point) => ({
    lead: formatLeadTimeHours(point.lead_time_hours),
    p10Base: point.p10Base,
    p90Height: point.p90Height,
    p25Base: point.p25Base,
    p75Height: point.p75Height,
    median: point.median,
    mean: point.mean,
  }));

  return (
    <div className="mb-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-sm font-medium text-slate-800">{variableLabel} — percentile range</h4>
        <span className="text-xs text-slate-500">P10–P90 band · P25–P75 box · median · mean</span>
      </div>
      <div
        role="img"
        aria-label={`${variableLabel} ensemble percentile fan over lead time`}
        className="h-48 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="lead" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
            <YAxis
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={46}
              domain={["auto", "auto"]}
            />
            <Tooltip
              formatter={(value: number, name: string) => [formatValue(value, ""), name]}
              contentStyle={{ fontSize: 12 }}
            />
            {/* Transparent base stacks so the colored heights render as bands. */}
            <Area
              dataKey="p10Base"
              stackId="p10"
              stroke="none"
              fill="none"
              isAnimationActive={false}
            />
            <Area
              dataKey="p90Height"
              stackId="p10"
              stroke="none"
              fill="#93c5fd"
              fillOpacity={0.35}
              isAnimationActive={false}
              name="P10–P90"
            />
            <Area
              dataKey="p25Base"
              stackId="p25"
              stroke="none"
              fill="none"
              isAnimationActive={false}
            />
            <Area
              dataKey="p75Height"
              stackId="p25"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.45}
              isAnimationActive={false}
              name="P25–P75"
            />
            <Line
              dataKey="median"
              stroke="#1e3a8a"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="Median (P50)"
            />
            <Line
              dataKey="mean"
              stroke="#b45309"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
              name="Mean"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
