"use client";

import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { toMeteogramSeries } from "@/lib/forecast/transform";
import { formatValue, formatWindDirection } from "@/lib/forecast/labels";
import { formatDayHourUtc } from "@/lib/forecast/time";
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

/**
 * Hourly meteogram for a single forecast variable from `/v1/points`.
 *
 * Line/area series are used for temperature-style variables; bars are used
 * for precipitation-style variables (values `>= 0`), which reads as a rain
 * accumulations chart. Missing values (`null`) are connected across so the
 * series stays continuous; the tooltip reports the raw value or "n/a".
 */
export function Meteogram({ forecasts, variableCode, meta }: MeteogramProps) {
  const data = toMeteogramSeries(forecasts, variableCode).map((point, index) => {
    const entry = forecasts[index];
    return {
      lead_time_hours: point.lead_time_hours,
      valid_time: point.valid_time,
      label: formatDayHourUtc(point.valid_time),
      value: point.value,
      wind_direction_10m: entry?.wind_direction_10m as number | null | undefined,
      wind_cardinal_10m: entry?.wind_cardinal_10m as string | null | undefined,
    };
  });

  const isPrecipitation =
    meta.unit === "mm/h" || meta.unit === "in/h" || variableCode.startsWith("precip");

  return (
    <div className="mb-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-sm font-medium text-slate-800">{meta.name}</h4>
        <span className="text-xs text-slate-500">{meta.unit}</span>
      </div>
      <div
        role="img"
        aria-label={`${meta.name} hourly forecast over lead time`}
        className="h-48 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
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
              domain={isPrecipitation ? [0, "auto"] : ["auto", "auto"]}
            />
            <Tooltip
              formatter={(value: number | string, _name: string, item: any) => {
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
              />
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
