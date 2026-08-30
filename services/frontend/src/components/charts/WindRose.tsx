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

const BIN_COLORS: Record<string, { label: string; color: string }> = {
  light: { label: "1.8–20 km/h (Light)", color: "#74c476" },
  moderate: { label: "20–40 km/h (Moderate)", color: "#41ab5d" },
  strong: { label: "40–60 km/h (Strong)", color: "#4292c6" },
  gale: { label: "≥60 km/h (Gale+)", color: "#7a0177" },
};

const BIN_KEYS = ["light", "moderate", "strong", "gale"] as const;

export function WindRose({ windRose }: WindRoseProps) {
  const [hoveredSector, setHoveredSector] = useState<string | null>(null);

  const centerRadius = 24;
  const maxRadius = 110;
  const center = 140;

  // Find max sector probability to scale lengths
  const maxProb = Math.max(
    0.2,
    ...windRose.sectors.map((s) => s.probability)
  );

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
                stroke="#e2e8f0"
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
            stroke="#cbd5e1"
          />
          <line
            x1={center - maxRadius}
            y1={center}
            x2={center + maxRadius}
            y2={center}
            stroke="#cbd5e1"
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
                {BIN_KEYS.map((binKey) => {
                  const binProb = sectorData.bins[binKey] ?? 0;
                  if (binProb <= 0) return null;

                  const binRadialHeight =
                    ((maxRadius - centerRadius) * (binProb / maxProb));
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
                      fill={BIN_COLORS[binKey].color}
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
                      className="text-[10px] font-semibold fill-slate-600"
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
            fill="#f8fafc"
            stroke="#94a3b8"
            strokeWidth={1.5}
          />
          <text
            x={center}
            y={center - 4}
            textAnchor="middle"
            dominantBaseline="central"
            className="text-[8px] font-bold fill-slate-500"
          >
            CALM
          </text>
          <text
            x={center}
            y={center + 6}
            textAnchor="middle"
            dominantBaseline="central"
            className="text-[9px] font-bold fill-slate-700"
          >
            {Math.round(windRose.calm_percentage)}%
          </text>
        </svg>
      </div>

      {/* Hover Information / Summary */}
      <div className="h-6 text-xs text-slate-600 text-center">
        {hoveredSector ? (
          (() => {
            const sec = windRose.sectors.find((s) => s.sector === hoveredSector);
            if (!sec) return null;
            const totalMembers =
              windRose.member_count ??
              (windRose.calm_count + windRose.sectors.reduce((sum, s) => sum + s.count, 0));
            return (
              <span>
                <strong>{sec.sector}</strong>: {formatPercent(sec.probability)} ({sec.count}/
                {totalMembers} members)
              </span>
            );
          })()
        ) : (
          <span>Hover a sector to view member probability</span>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap justify-center gap-3 text-[10px] text-slate-600">
        {BIN_KEYS.map((binKey) => (
          <div key={binKey} className="flex items-center gap-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: BIN_COLORS[binKey].color }}
            />
            <span>{BIN_COLORS[binKey].label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
