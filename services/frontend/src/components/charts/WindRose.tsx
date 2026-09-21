"use client";

import React, { useState } from "react";
import type { WindRose as WindRoseType } from "@/lib/api/types";
import { formatPercent } from "@/lib/forecast/labels";

interface WindRoseProps {
  windRose: WindRoseType;
}

const SECTOR_ANGLES: Record<string, number> = {
  N: 0,
  NE: 45,
  E: 90,
  SE: 135,
  S: 180,
  SW: 225,
  W: 270,
  NW: 315,
};

/**
 * The member path's four speed bins, by name and by physical range.
 *
 * The stored rose bins differently -- its buckets are quantile edges of the cycle's own member
 * set, so one bucket holds as much of the distribution as the next -- and it sends the edges
 * alongside. Both shapes are drawn here: a sector's `bins` are looked up by these names when the
 * store was read from members, and by `bucket_N` when it was read from a container, in which case
 * the labels come from the edges instead of from this table.
 */
const MEMBER_BIN_COLORS: Record<string, string> = {
  light: "#74c476",
  moderate: "#41ab5d",
  strong: "#4292c6",
  gale: "#7a0177",
};

const BIN_COLORS: Record<string, { label: string; color: string }> = {
  light: { label: "1.8–20 km/h (Light)", color: MEMBER_BIN_COLORS.light },
  moderate: { label: "20–40 km/h (Moderate)", color: MEMBER_BIN_COLORS.moderate },
  strong: { label: "40–60 km/h (Strong)", color: MEMBER_BIN_COLORS.strong },
  gale: { label: "≥60 km/h (Gale+)", color: MEMBER_BIN_COLORS.gale },
};

/** The stored rose's bucket keys, in order. */
const STORED_BUCKET_KEYS = Array.from({ length: 8 }, (_, index) => `bucket_${index}`);

const BIN_KEYS = ["light", "moderate", "strong", "gale"] as const;
const STORED_BUCKET_COLORS = [
  "#74c476",
  "#a1d99b",
  "#41ab5d",
  "#4292c6",
  "#6baed6",
  "#7a0177",
  "#c51b8a",
  "#f768a1",
];

/**
 * How this rose bins its speeds, and what each bin is called.
 *
 * Decided by the payload, not by the variable: a stored rose carries its own edges and bucket
 * keys, a member-derived one carries fixed physical names. Reading the wrong one would draw the
 * right heights under the wrong legend.
 */
interface RoseBinning {
  keys: string[];
  label: (key: string, index: number) => string;
  color: (key: string, index: number) => string;
}

function roseBinning(windRose: WindRoseType): RoseBinning {
  const edges = windRose.bucket_edges_mps;
  const storedKeys =
    windRose.bins && Object.keys(windRose.bins).length > 0
      ? Object.keys(windRose.bins)
      : windRose.sectors[0]?.bins
        ? Object.keys(windRose.sectors[0].bins)
        : [];
  const isStored =
    storedKeys.length > 0 &&
    storedKeys.every((key) => key.startsWith("bucket_")) &&
    Array.isArray(edges) &&
    edges.length === storedKeys.length + 1;

  if (!isStored) {
    return {
      keys: [...BIN_KEYS],
      label: (key) => BIN_COLORS[key]?.label ?? key,
      color: (key) => BIN_COLORS[key]?.color ?? "#94a3b8",
    };
  }
  // The edges are the only source of a label: the buckets are quantiles of this cycle's member
  // set, so "bucket 3" of one cycle is not "bucket 3" of the next.
  const ms = (value: number) => (value * 3.6).toFixed(0);
  return {
    keys: storedKeys.length > 0 ? storedKeys : STORED_BUCKET_KEYS,
    label: (key, index) => {
      const lower = edges[index];
      const upper = edges[index + 1];
      return index === edges.length - 2 ? `≥${ms(lower)} km/h` : `${ms(lower)}–${ms(upper)} km/h`;
    },
    color: (_key, index) => STORED_BUCKET_COLORS[index % STORED_BUCKET_COLORS.length],
  };
}

export function WindRose({ windRose }: WindRoseProps) {
  const [hoveredSector, setHoveredSector] = useState<string | null>(null);
  const binning = roseBinning(windRose);

  const centerRadius = 24;
  const maxRadius = 110;
  const center = 140;

  // Find max sector probability to scale lengths
  const maxProb = Math.max(0.2, ...windRose.sectors.map((s) => s.probability));

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-72 w-72">
        <svg
          viewBox="0 0 280 280"
          className="h-full w-full"
          role="img"
          aria-label="Ensemble Wind Rose chart"
        >
          {/* Circular grid rings */}
          {[0.25, 0.5, 0.75, 1.0].map((frac) => {
            const r = centerRadius + (maxRadius - centerRadius) * frac;
            return (
              <circle
                key={frac}
                cx={center}
                cy={center}
                r={r}
                fill="none"
                stroke="#334155"
                strokeDasharray="2 2"
              />
            );
          })}

          {/* Compass crosshairs */}
          <line
            x1={center}
            y1={center - maxRadius}
            x2={center}
            y2={center + maxRadius}
            stroke="#334155"
          />
          <line
            x1={center - maxRadius}
            y1={center}
            x2={center + maxRadius}
            y2={center}
            stroke="#334155"
          />

          {/* Direction wedges */}
          {windRose.sectors.map((sectorData) => {
            const angleDeg = SECTOR_ANGLES[sectorData.sector] ?? 0;
            const isHovered = hoveredSector === sectorData.sector;
            const halfWedge = 20; // 40 degree wedge width

            // Stacked speed bins
            let currentRadius = centerRadius;
            return (
              <g
                key={sectorData.sector}
                onMouseEnter={() => setHoveredSector(sectorData.sector)}
                onMouseLeave={() => setHoveredSector(null)}
                className="cursor-pointer transition-opacity"
                opacity={hoveredSector && !isHovered ? 0.4 : 1}
              >
                {binning.keys.map((binKey) => {
                  const binProb = sectorData.bins[binKey] ?? 0;
                  if (binProb <= 0) return null;

                  const binRadialHeight = (maxRadius - centerRadius) * (binProb / maxProb);
                  const rInner = currentRadius;
                  const rOuter = currentRadius + binRadialHeight;
                  currentRadius = rOuter;

                  // Create SVG arc path
                  const startRad = ((angleDeg - halfWedge - 90) * Math.PI) / 180;
                  const endRad = ((angleDeg + halfWedge - 90) * Math.PI) / 180;

                  const x1 = center + rInner * Math.cos(startRad);
                  const y1 = center + rInner * Math.sin(startRad);
                  const x2 = center + rOuter * Math.cos(startRad);
                  const y2 = center + rOuter * Math.sin(startRad);
                  const x3 = center + rOuter * Math.cos(endRad);
                  const y3 = center + rOuter * Math.sin(endRad);
                  const x4 = center + rInner * Math.cos(endRad);
                  const y4 = center + rInner * Math.sin(endRad);

                  const pathData = `
                    M ${x1} ${y1}
                    L ${x2} ${y2}
                    A ${rOuter} ${rOuter} 0 0 1 ${x3} ${y3}
                    L ${x4} ${y4}
                    A ${rInner} ${rInner} 0 0 0 ${x1} ${y1}
                    Z
                  `;

                  return (
                    <path
                      key={binKey}
                      d={pathData}
                      fill={binning.color(binKey, binning.keys.indexOf(binKey))}
                      stroke="#ffffff"
                      strokeWidth={0.5}
                    />
                  );
                })}

                {/* Cardinal Label on outer edge */}
                {(() => {
                  const labelRad = ((angleDeg - 90) * Math.PI) / 180;
                  const lx = center + (maxRadius + 14) * Math.cos(labelRad);
                  const ly = center + (maxRadius + 14) * Math.sin(labelRad);
                  return (
                    <text
                      x={lx}
                      y={ly}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="text-[10px] font-bold font-mono fill-slate-400"
                    >
                      {sectorData.sector}
                    </text>
                  );
                })()}
              </g>
            );
          })}

          {/* Center Calm Circle */}
          <circle
            cx={center}
            cy={center}
            r={centerRadius}
            fill="#0f172a"
            stroke="#334155"
            strokeWidth={1.5}
          />
          <text
            x={center}
            y={center - 4}
            textAnchor="middle"
            dominantBaseline="central"
            className="text-[8px] font-bold fill-slate-400 font-mono"
          >
            CALM
          </text>
          <text
            x={center}
            y={center + 6}
            textAnchor="middle"
            dominantBaseline="central"
            className="text-[9px] font-bold fill-slate-200 font-mono"
          >
            {Math.round(windRose.calm_percentage)}%
          </text>
        </svg>
      </div>

      {/* Hover Information / Summary */}
      <div className="h-6 text-xs text-slate-300 text-center font-mono">
        {hoveredSector ? (
          (() => {
            const sec = windRose.sectors.find((s) => s.sector === hoveredSector);
            if (!sec) return null;
            const totalMembers =
              windRose.member_count ??
              windRose.calm_count + windRose.sectors.reduce((sum, s) => sum + s.count, 0);
            return (
              <span>
                <strong className="text-cyan-400">{sec.sector}</strong>:{" "}
                {formatPercent(sec.probability)} ({sec.count}/{totalMembers} members)
              </span>
            );
          })()
        ) : (
          <span className="text-slate-400 font-sans">
            Hover a sector to view member probability
          </span>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap justify-center gap-3 text-[10px] text-slate-400">
        {binning.keys.map((binKey, index) => (
          <div key={binKey} className="flex items-center gap-1 font-mono">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: binning.color(binKey, index) }}
            />
            <span>{binning.label(binKey, index)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
