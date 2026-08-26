"use client";

import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  distributionSummary,
  distributionXDomain,
  histogramBins,
  toMemberDots,
  toPdfPoints,
} from "@/lib/forecast/transform";
import { formatValue } from "@/lib/forecast/labels";
import { formatLeadTimeHours } from "@/lib/forecast/time";
import type { DistributionStatus } from "@/hooks/useEnsembleDistribution";
import type { EnsembleStatisticsData } from "@/lib/api/types";

interface EnsembleDistributionProps {
  /** The `include_members=true` response for the selected lead, or null. */
  data: EnsembleStatisticsData | null;
  status: DistributionStatus;
  error: string | null;
  /** The lead time whose distribution to show. */
  selectedLead: number;
  variableLabel: string;
}

const ACCENT = "#1d4ed8";
const HISTOGRAM_COLORS = ["#93c5fd", "#3b82f6", "#1d4ed8"];

/**
 * Ensemble Distribution View — the raw member-level distribution and canonical
 * Probability Density Function (PDF) for a selected location / variable / lead.
 *
 * This visualization overlays a 1-D Gaussian Kernel Density Estimate (PDF)
 * directly over the discrete ensemble histogram within a single plot area using
 * dual Y-axes:
 *
 * - Left Y-axis: Member count (histogram bar frequency)
 * - Right Y-axis: Probability density (continuous PDF estimate, 1/variable-unit)
 * - Shared numeric X-axis: Physical variable values
 * - Member rug/dot plot: Discrete raw ensemble member markers aligned below
 *
 * When `members` is absent (loading, error, or statistics-only response), an
 * honest "not yet available" state is shown without fabricating values. When
 * `pdf` is null (degenerate spread / identical members), the histogram and rug
 * are shown with an informative note.
 */
export function EnsembleDistribution({
  data,
  status,
  error,
  selectedLead,
  variableLabel,
}: EnsembleDistributionProps) {
  if (status === "loading") {
    return (
      <p role="status" className="text-sm text-slate-500">
        Loading ensemble distribution…
      </p>
    );
  }

  if (status === "error") {
    return (
      <p role="alert" className="text-sm text-red-700">
        {error ?? "Failed to load the ensemble distribution."}
      </p>
    );
  }

  if (data === null) {
    return (
      <div className="rounded border border-slate-200 bg-slate-50 px-3 py-3">
        <p className="text-xs text-slate-600">
          No ensemble distribution available for {formatLeadTimeHours(selectedLead)}.
        </p>
      </div>
    );
  }

  const members = data.members;
  if (members === undefined || members.length === 0) {
    return (
      <div className="rounded border border-slate-200 bg-slate-50 px-3 py-3">
        <p className="text-xs text-slate-600">
          Ensemble distribution for {variableLabel} at {formatLeadTimeHours(selectedLead)} is not
          yet available: the API returned no raw member values. The summary above is shown instead.
        </p>
      </div>
    );
  }

  const bins = histogramBins(members);
  const dots = toMemberDots(members);
  const summary = distributionSummary(members);
  const pdfPoints = toPdfPoints(data.pdf);
  const [xMin, xMax] = distributionXDomain(summary, data.pdf);
  const memberCount = data.member_count;

  // Prepare bin data points with explicit x coordinate for numeric XAxis
  const binChartData = bins.map((bin) => ({
    x: bin.mid,
    count: bin.count,
    start: bin.start,
    end: bin.end,
  }));

  return (
    <div className="mt-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-sm font-medium text-slate-800">
          Member distribution · {formatLeadTimeHours(selectedLead)}
        </h4>
        <span className="text-xs text-slate-500">{memberCount} members</span>
      </div>

      <dl className="mb-2 grid grid-cols-4 gap-2 text-center text-xs">
        <StatCell label="Min" value={summary.min} />
        <StatCell label="Max" value={summary.max} />
        <StatCell label="Mean" value={summary.mean} />
        <StatCell label="StdDev" value={summary.stdDev} />
      </dl>

      <div
        role="img"
        aria-label={`Histogram and PDF of ${memberCount} ensemble members for ${variableLabel}`}
        className="h-44 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={binChartData} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[xMin, xMax]}
              tickFormatter={(value: number) => value.toFixed(1)}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
            />
            <YAxis
              yAxisId="left"
              allowDecimals={false}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={32}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              allowDecimals={true}
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={38}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                name === "Probability density" ? value.toFixed(4) : value,
                name === "Probability density" ? "Probability density" : "Members",
              ]}
              labelFormatter={(label) => `Value ≈ ${Number(label).toFixed(2)}`}
              contentStyle={{ fontSize: 12 }}
            />
            <Bar
              yAxisId="left"
              dataKey="count"
              isAnimationActive={false}
              radius={[2, 2, 0, 0]}
              name="Member count"
              barSize={32}
            >
              {bins.map((bin, index) => (
                <Cell
                  key={`${bin.start}-${bin.end}`}
                  fill={HISTOGRAM_COLORS[index % HISTOGRAM_COLORS.length]}
                />
              ))}
            </Bar>
            {pdfPoints.length > 0 && (
              <Line
                yAxisId="right"
                data={pdfPoints}
                dataKey="density"
                type="linear"
                stroke="#d97706"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="Probability density"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div
        role="img"
        aria-label={`Member values for ${variableLabel} at ${formatLeadTimeHours(selectedLead)}`}
        className="mt-2 h-12 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <XAxis
              type="number"
              dataKey="value"
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              domain={[xMin, xMax]}
            />
            <YAxis hide domain={[0, 1]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              formatter={(value: number) => [formatValue(value, ""), "Member value"]}
              labelFormatter={() => ""}
              contentStyle={{ fontSize: 12 }}
            />
            <Scatter data={dots} dataKey="value" fill={ACCENT} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {data.pdf === null && (
        <p className="mt-1 text-[11px] text-amber-700">
          Continuous probability density is unavailable for this lead time (insufficient spread
          across members).
        </p>
      )}

      <p className="mt-1 text-[11px] text-slate-400">
        Histogram bars and dots show the discrete ensemble member sample. The continuous curve shows
        the canonical Gaussian kernel density estimate (probability density).
      </p>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded bg-slate-50 px-2 py-1">
      <dt className="text-[10px] uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="tabular-nums text-slate-800">
        {Number.isFinite(value) ? value.toFixed(1) : "—"}
      </dd>
    </div>
  );
}
