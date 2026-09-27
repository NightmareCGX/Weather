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
  niceDistributionTicks,
  tickDecimals,
  toMemberDots,
  toPdfPoints,
} from "@/lib/forecast/transform";
import { formatPercent, formatValue } from "@/lib/forecast/labels";
import { formatDayHourWithTimeZone } from "@/lib/forecast/time";
import type { DistributionStatus } from "@/hooks/useEnsembleDistribution";
import type { EnsembleStatisticsData } from "@/lib/api/types";

interface EnsembleDistributionProps {
  /** The `include_members=true` response for the selected lead, or null. */
  data: EnsembleStatisticsData | null;
  status: DistributionStatus;
  error: string | null;
  /** The lead time or valid time whose distribution to show. */
  selectedLead?: number | string;
  validTime?: string | null;
  timezone?: string | null;
  variableLabel: string;
}

const ACCENT = "#38bdf8";
const HISTOGRAM_COLORS = ["#0284c7", "#0ea5e9", "#38bdf8"];

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
  validTime,
  timezone,
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
      <p role="alert" className="text-sm text-red-700" data-testid="distribution-error">
        {error ?? "Failed to load the ensemble distribution."}
      </p>
    );
  }

  const resolvedValidTime =
    validTime ?? (typeof selectedLead === "string" ? selectedLead : (data?.valid_time ?? null));

  const timeLabel = resolvedValidTime ? formatDayHourWithTimeZone(resolvedValidTime, timezone) : "";

  if (data === null) {
    return (
      <div
        className="rounded border border-slate-200 bg-slate-50 px-3 py-3"
        data-testid="distribution-empty"
      >
        <p className="text-xs text-slate-600">
          No ensemble distribution available{timeLabel ? ` for ${timeLabel}` : ""}.
        </p>
      </div>
    );
  }

  const isCeiling = data.unlimited_probability !== undefined && data.unlimited_probability !== null;
  const rawMembers = data.members;
  const members = isCeiling ? (rawMembers?.filter((m) => m < 19.99) ?? []) : (rawMembers ?? []);

  if (rawMembers === undefined || rawMembers.length === 0) {
    return (
      <div
        className="rounded border border-slate-200 bg-slate-50 px-3 py-3"
        data-testid="distribution-empty"
      >
        <p className="text-xs text-slate-600">
          Ensemble distribution for {variableLabel}
          {timeLabel ? ` at ${timeLabel}` : ""} is not yet available: the API returned no raw member
          values. The summary above is shown instead.
        </p>
      </div>
    );
  }

  const memberCount = data.member_count;
  const unlimitedProb = data.unlimited_probability;
  const finiteCount = data.finite_member_count ?? members.length;

  if (isCeiling && finiteCount < 10) {
    return (
      <div className="mt-4 space-y-3" data-testid="ensemble-distribution">
        <div
          className="flex items-center justify-between rounded-lg border border-sky-200 bg-sky-50 px-4 py-3"
          data-testid="unlimited-probability"
        >
          <div>
            <div className="text-xs font-semibold text-sky-800">Unlimited Ceiling Probability</div>
            <div className="text-xl font-bold text-sky-950">
              {unlimitedProb !== undefined && unlimitedProb !== null
                ? formatPercent(unlimitedProb)
                : "—"}
            </div>
          </div>
          <div className="text-right text-xs text-sky-700">
            {data.unlimited_member_count ?? memberCount - finiteCount} /{" "}
            {data.valid_member_count ?? memberCount} members
          </div>
        </div>
        <div
          className="rounded border border-slate-200 bg-slate-50 px-3 py-3"
          data-testid="distribution-empty"
        >
          <p className="text-xs text-slate-600">
            Only {finiteCount} members predict a finite ceiling (minimum 10 required for continuous
            distribution).
          </p>
        </div>
      </div>
    );
  }

  const bins = histogramBins(members);
  const dots = toMemberDots(members);
  const summary = distributionSummary(members);
  const pdfPoints = toPdfPoints(data.pdf);
  const [xMin, xMax] = distributionXDomain(summary, data.pdf);
  // Pin both stacked charts to the same fixed nice ticks so labels stay short,
  // evenly spaced, and vertically aligned with the histogram gridlines.
  const xTicks = niceDistributionTicks(xMin, xMax);
  const formatXTick = (value: number) => value.toFixed(tickDecimals(xTicks));

  // Prepare bin data points with explicit x coordinate for numeric XAxis
  const binChartData = bins.map((bin) => ({
    x: bin.mid,
    count: bin.count,
    start: bin.start,
    end: bin.end,
  }));

  return (
    <div className="mt-4 space-y-3" data-testid="ensemble-distribution">
      {isCeiling && unlimitedProb !== undefined && unlimitedProb !== null && (
        <div
          className="flex items-center justify-between rounded-lg border border-sky-200 bg-sky-50 px-4 py-3"
          data-testid="unlimited-probability"
        >
          <div>
            <div className="text-xs font-semibold text-sky-800">Unlimited Ceiling Probability</div>
            <div className="text-xl font-bold text-sky-950">{formatPercent(unlimitedProb)}</div>
          </div>
          <div className="text-right text-xs text-sky-700">
            {data.unlimited_member_count ?? memberCount - finiteCount} /{" "}
            {data.valid_member_count ?? memberCount} members
          </div>
        </div>
      )}

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <h4 className="text-sm font-semibold text-slate-200" data-testid="distribution-subtitle">
            {isCeiling
              ? `Conditional finite distribution · ${timeLabel}`
              : `Member distribution · ${timeLabel}`}
          </h4>
          <span className="text-xs font-mono text-cyan-400">
            {isCeiling ? `${finiteCount} finite members` : `${memberCount} members`}
          </span>
        </div>

        <dl className="mb-2 grid grid-cols-4 gap-2 text-center text-xs">
          <StatCell label="Min" value={summary.min} />
          <StatCell label="Max" value={summary.max} />
          <StatCell label="Mean" value={summary.mean} />
          <StatCell label="StdDev" value={summary.stdDev} />
        </dl>

        <div
          role="img"
          aria-label={`Histogram and PDF of ${members.length} ensemble members for ${variableLabel}`}
          className="h-44 w-full"
          data-testid="distribution-chart"
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={binChartData} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[xMin, xMax]}
                ticks={xTicks.length > 0 ? xTicks : undefined}
                tickFormatter={formatXTick}
                interval={0}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                tickLine={false}
              />
              <YAxis
                yAxisId="left"
                allowDecimals={false}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                tickLine={false}
                axisLine={false}
                width={32}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                allowDecimals={true}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
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
                contentStyle={{
                  fontSize: 12,
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  color: "#f8fafc",
                  borderRadius: 8,
                }}
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
                  stroke="#fbbf24"
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
          aria-label={`Member values for ${variableLabel} at ${timeLabel}`}
          className="mt-2 h-12 w-full"
        >
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              // Mirror the histogram's horizontal geometry (left Y-axis width 32
              // + right Y-axis width 38 + margins) so tick columns and member
              // dots line up column-for-column with the chart above.
              margin={{ top: 4, right: 38 + 8, bottom: 0, left: 32 + 0 }}
            >
              <XAxis
                type="number"
                dataKey="value"
                domain={[xMin, xMax]}
                ticks={xTicks.length > 0 ? xTicks : undefined}
                tickFormatter={formatXTick}
                interval={0}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                tickLine={false}
              />
              <YAxis hide domain={[0, 1]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                formatter={(value: number) => [formatValue(value, ""), "Member value"]}
                labelFormatter={() => ""}
                contentStyle={{
                  fontSize: 12,
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  color: "#f8fafc",
                  borderRadius: 8,
                }}
              />
              <Scatter data={dots} dataKey="value" fill={ACCENT} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {data.pdf === null && (
          <p className="mt-1 text-[11px] text-amber-400">
            Continuous probability density is unavailable for this lead time (insufficient spread
            across members).
          </p>
        )}

        <p className="mt-1 text-[11px] text-slate-500">
          Histogram bars and dots show the discrete ensemble member sample. The continuous curve
          shows the canonical Gaussian kernel density estimate (probability density).
        </p>
      </div>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border border-slate-700/60 bg-slate-800/80 px-2 py-1">
      <dt className="text-[10px] uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="font-mono tabular-nums text-slate-100 font-semibold">
        {Number.isFinite(value) ? value.toFixed(1) : "—"}
      </dd>
    </div>
  );
}
