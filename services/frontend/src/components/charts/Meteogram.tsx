"use client";

import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { toMeteogramSeries } from "@/lib/forecast/transform";
import { formatCloudCeiling, formatValue, formatWindDirection } from "@/lib/forecast/labels";
import { formatDayHourUtc } from "@/lib/forecast/time";
import {
  getPointForecastPhaseLabel,
  getBarColorForEntry,
  PRECIPITATION_PHASE_TOKENS,
  PointForecastPrecipEntry,
} from "@/lib/forecast/precipitation";
import type { ForecastEntry } from "@/lib/api/types";

interface VariableMeta {
  name: string;
  unit: string;
}

interface MeteogramProps {
  forecasts: ForecastEntry[];
  variableCode: string;
  meta: VariableMeta;
}

const KNOWN_TRANSITION_GRADIENTS = [
  { id: "precip-grad-rain_to_snow", start: "#0284c7", end: "#6366f1" },
  { id: "precip-grad-snow_to_rain", start: "#6366f1", end: "#0284c7" },
  { id: "precip-grad-rain_to_freezing_rain", start: "#0284c7", end: "#e11d48" },
  { id: "precip-grad-freezing_rain_to_rain", start: "#e11d48", end: "#0284c7" },
  { id: "precip-grad-snow_to_freezing_rain", start: "#6366f1", end: "#e11d48" },
  { id: "precip-grad-freezing_rain_to_snow", start: "#e11d48", end: "#6366f1" },
  { id: "precip-grad-snow_to_ice_pellets", start: "#6366f1", end: "#0d9488" },
  { id: "precip-grad-ice_pellets_to_snow", start: "#0d9488", end: "#6366f1" },
  { id: "precip-grad-rain_to_ice_pellets", start: "#0284c7", end: "#0d9488" },
  { id: "precip-grad-ice_pellets_to_rain", start: "#0d9488", end: "#0284c7" },
];

/**
 * Hourly meteogram for a single forecast variable from `/v1/points`.
 *
 * Line series are used for continuous atmospheric variables (temperature,
 * wind, humidity, visibility). Bars are used for precipitation variables
 * (precipitation_rate, precipitation_amount_3h).
 *
 * For `precipitation_amount_3h`:
 * - Bar height represents accumulated liquid-equivalent precipitation (mm / in).
 * - Bar fill reflects domain-classified precipitation phase (Rain, Snow,
 *   Freezing Rain, Ice Pellets, Mixed, Unclassified).
 * - Two-phase transitions render semantic gradients without implying exact timing.
 * - Lead 0 (null accumulation) renders no bar and formats as "—" rather than 0 mm.
 */
export function Meteogram({ forecasts, variableCode, meta }: MeteogramProps) {
  const isPrecipAmount3h = variableCode === "precipitation_amount_3h";
  const isCloudCover = variableCode === "cloud_cover_3h";
  const isCloudCeiling = variableCode === "cloud_ceiling";
  const isPrecipitation =
    meta.unit === "mm/h" ||
    meta.unit === "in/h" ||
    meta.unit === "mm" ||
    meta.unit === "in" ||
    variableCode.startsWith("precip");

  const data = toMeteogramSeries(forecasts, variableCode).map((point, index) => {
    const entry = forecasts[index];
    const precipEntry: PointForecastPrecipEntry = {
      precipitation_amount_3h: point.value,
      precipitation_type: entry?.precipitation_type as string | undefined,
      precipitation_transition: entry?.precipitation_transition as string | undefined,
      precipitation_start_type: entry?.precipitation_start_type as string | undefined,
      precipitation_end_type: entry?.precipitation_end_type as string | undefined,
      precipitation_evidence: entry?.precipitation_evidence as string | undefined,
      lead_time_hours: point.lead_time_hours,
    };

    const rawVal = point.value;
    const isUnlimitedCeiling =
      isCloudCeiling &&
      (rawVal === null ||
        rawVal === undefined ||
        rawVal >= 19990 ||
        Boolean(entry?.cloud_ceiling_unlimited));
    const plotValue = isCloudCeiling && isUnlimitedCeiling ? null : rawVal;

    return {
      lead_time_hours: point.lead_time_hours,
      valid_time: point.valid_time,
      label: formatDayHourUtc(point.valid_time),
      value: plotValue,
      rawValue: rawVal,
      isUnlimitedCeiling,
      wind_direction_10m: entry?.wind_direction_10m as number | null | undefined,
      wind_cardinal_10m: entry?.wind_cardinal_10m as string | null | undefined,
      precipEntry,
    };
  });

  function getBarFill(item: (typeof data)[number]): string {
    if (!isPrecipAmount3h) {
      return "#0ea5e9";
    }
    const { precipEntry, value } = item;
    if (value === null || value <= 0.05) {
      return "transparent";
    }
    const transition = precipEntry.precipitation_transition;
    if (
      transition &&
      KNOWN_TRANSITION_GRADIENTS.some((g) => g.id === `precip-grad-${transition}`)
    ) {
      return `url(#precip-grad-${transition})`;
    }
    return getBarColorForEntry(precipEntry);
  }

  return (
    <div className="mb-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-sm font-medium text-slate-800">{meta.name}</h4>
        <span className="text-xs text-slate-500">{meta.unit}</span>
      </div>

      {isPrecipAmount3h && (
        <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-600">
          <span className="font-medium text-slate-500">Phases:</span>
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: PRECIPITATION_PHASE_TOKENS.rain.color }}
            />
            Rain
          </span>
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: PRECIPITATION_PHASE_TOKENS.snow.color }}
            />
            Snow
          </span>
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: PRECIPITATION_PHASE_TOKENS.freezing_rain.color }}
            />
            Freezing Rain
          </span>
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: PRECIPITATION_PHASE_TOKENS.ice_pellets.color }}
            />
            Ice Pellets
          </span>
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: PRECIPITATION_PHASE_TOKENS.mixed.color }}
            />
            Mixed
          </span>
        </div>
      )}

      <div
        role="img"
        aria-label={`${meta.name} hourly forecast over lead time`}
        className="h-48 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <defs>
              {KNOWN_TRANSITION_GRADIENTS.map((grad) => (
                <linearGradient key={grad.id} id={grad.id} x1="0" y1="1" x2="0" y2="0">
                  <stop offset="0%" stopColor={grad.start} />
                  <stop offset="100%" stopColor={grad.end} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={46}
              domain={isCloudCover ? [0, 100] : isPrecipitation ? [0, "auto"] : ["auto", "auto"]}
            />
            <Tooltip
              formatter={(value: any, _name: string, item: any) => {
                if (isPrecipAmount3h) {
                  const precipEntry = item?.payload?.precipEntry as
                    PointForecastPrecipEntry | undefined;
                  if (value === null || typeof value !== "number") {
                    return ["— (Analysis / f000)", meta.name];
                  }
                  const phaseLabel = getPointForecastPhaseLabel(precipEntry);
                  return [`${formatValue(value, meta.unit)} · ${phaseLabel}`, meta.name];
                }

                if (isCloudCover) {
                  if (value === null || typeof value !== "number") {
                    return ["— (Analysis / f000)", meta.name];
                  }
                  return [`${formatValue(value, "%")} (3h avg)`, meta.name];
                }

                if (isCloudCeiling) {
                  const isUnlim = item?.payload?.isUnlimitedCeiling;
                  if (isUnlim || value === null || typeof value !== "number") {
                    return ["Unlimited", meta.name];
                  }
                  return [formatCloudCeiling(value, meta.unit), meta.name];
                }

                if (typeof value !== "number") return ["n/a", meta.name];
                if (variableCode === "wind_10m") {
                  const dir = item?.payload?.wind_direction_10m;
                  const card = item?.payload?.wind_cardinal_10m;
                  const dirLabel = formatWindDirection(dir, card);
                  return [`${formatValue(value, meta.unit)} (${dirLabel})`, meta.name];
                }
                return [formatValue(value, meta.unit), meta.name];
              }}
              labelFormatter={(label: string) => `${label} UTC`}
              contentStyle={{ fontSize: 12 }}
            />
            {isPrecipitation ? (
              <Bar
                dataKey="value"
                fill="#0ea5e9"
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
                name={meta.name}
              >
                {data.map((entry, idx) => (
                  <Cell key={`precip-cell-${idx}`} fill={getBarFill(entry)} />
                ))}
              </Bar>
            ) : (
              <Line
                dataKey="value"
                type="monotone"
                stroke="#1d4ed8"
                strokeWidth={2}
                dot={{ r: 2, fill: "#1d4ed8" }}
                isAnimationActive={false}
                connectNulls
                name={meta.name}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
