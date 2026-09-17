"use client";

import { useEffect, useRef, useState } from "react";

import type {
  CenterHealth,
  CenterHealthStatus,
  ModelHealthStatus,
} from "@/lib/forecast/centerHealth";

/**
 * Read-only forecast-center status badges for the header.
 *
 * Each badge summarizes one issuing center (e.g. NOAA) and expands on hover
 * or on focus/click to list that center's models with their individual
 * status. Nothing here is selectable: model selection stays in the layer
 * controls below.
 */

/** How long the pointer may leave before the panel closes, in ms. */
const CLOSE_DELAY_MS = 120;

/**
 * Window in which a click is attributed to the hover that a tap synthesizes
 * rather than to a deliberate toggle. Well above the ~0ms gap a touch browser
 * uses between its synthetic mouseenter and click, and well below the time a
 * desktop user takes to move onto the badge and click it.
 */
const TAP_HOVER_GUARD_MS = 300;

interface StatusTone {
  dot: string;
  text: string;
  chip: string;
}

const CENTER_TONES: Record<CenterHealthStatus, StatusTone> = {
  healthy: {
    dot: "bg-emerald-500 shadow-[0_0_8px_#10b981]",
    text: "text-emerald-400",
    chip: "border-emerald-500/30 bg-emerald-950/40",
  },
  degraded: {
    dot: "bg-amber-400 shadow-[0_0_8px_#f59e0b]",
    text: "text-amber-300",
    chip: "border-amber-500/30 bg-amber-950/40",
  },
  down: {
    dot: "bg-rose-500 shadow-[0_0_8px_#f43f5e]",
    text: "text-rose-400",
    chip: "border-rose-500/30 bg-rose-950/40",
  },
};

const MODEL_TONES: Record<ModelHealthStatus, StatusTone> = {
  ready: {
    dot: "bg-emerald-500 shadow-[0_0_8px_#10b981]",
    text: "text-emerald-400",
    chip: "border-emerald-500/30 bg-emerald-950/40",
  },
  syncing: {
    dot: "bg-amber-400 animate-pulse shadow-[0_0_8px_#f59e0b]",
    text: "text-amber-300",
    chip: "border-amber-500/30 bg-amber-950/40",
  },
  stale: {
    dot: "bg-amber-600 shadow-[0_0_8px_#d97706]",
    text: "text-amber-500",
    chip: "border-amber-700/30 bg-amber-950/40",
  },
  down: {
    dot: "bg-rose-500 shadow-[0_0_8px_#f43f5e]",
    text: "text-rose-400",
    chip: "border-rose-500/30 bg-rose-950/40",
  },
};

const CENTER_LABELS: Record<CenterHealthStatus, string> = {
  healthy: "all feeds ready",
  degraded: "partially available",
  down: "no feeds available",
};

const MODEL_LABELS: Record<ModelHealthStatus, string> = {
  ready: "Ready",
  syncing: "Syncing",
  stale: "Stale",
  down: "Unavailable",
};

function formatCycle(cycle: string | null): string {
  if (cycle === null) return "no cycle";
  const parsed = new Date(cycle);
  if (Number.isNaN(parsed.getTime())) return "no cycle";
  return `${String(parsed.getUTCHours()).padStart(2, "0")}Z`;
}

function CenterBadge({ center }: { center: CenterHealth }) {
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  /** When hover last opened the panel, used to disambiguate synthetic taps. */
  const hoverOpenedAt = useRef(0);
  const tone = CENTER_TONES[center.status];

  const cancelClose = () => {
    if (closeTimer.current !== null) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };

  const scheduleClose = () => {
    cancelClose();
    closeTimer.current = setTimeout(() => setOpen(false), CLOSE_DELAY_MS);
  };

  useEffect(() => cancelClose, []);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    // Touch devices have no hover-out, so tapping anywhere else is the only
    // way to dismiss the panel there.
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  return (
    <div
      ref={containerRef}
      className="relative"
      onMouseEnter={() => {
        cancelClose();
        hoverOpenedAt.current = Date.now();
        setOpen(true);
      }}
      onMouseLeave={scheduleClose}
    >
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => {
          // Touch browsers synthesize mouseenter immediately before click for
          // a single tap, so an unguarded toggle would open and instantly
          // re-close. Only a click that did not follow a just-fired hover is a
          // deliberate toggle.
          if (Date.now() - hoverOpenedAt.current < TAP_HOVER_GUARD_MS) return;
          setOpen((current) => !current);
        }}
        onFocus={() => {
          cancelClose();
          setOpen(true);
        }}
        onBlur={scheduleClose}
        className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold transition-colors duration-200 ${tone.chip} ${tone.text}`}
      >
        <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
        <span>{center.name}</span>
        <span className="font-mono text-[10px] opacity-80">
          {center.readyCount}/{center.totalCount}
        </span>
      </button>

      {open && (
        // The transparent top padding is the pointer bridge between the badge
        // and the panel: without it the pointer crosses a dead gap on its way
        // down and the panel closes under the cursor.
        <div className="absolute right-0 top-full z-50 pt-2">
          <div className="w-72 rounded-lg border border-slate-700/80 bg-slate-950/95 p-3 shadow-2xl backdrop-blur-md">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
              {center.name} feeds — {CENTER_LABELS[center.status]}
            </p>
            <ul className="flex flex-col gap-1.5">
              {center.models.map((model) => {
                const modelTone = MODEL_TONES[model.status];
                return (
                  <li key={model.id} className="flex items-center gap-2 text-xs">
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${modelTone.dot}`} />
                    <span className="truncate font-medium text-slate-200">{model.name}</span>
                    {model.isEnsemble && (
                      <span className="shrink-0 rounded border border-slate-600 px-1 text-[9px] font-bold text-slate-400">
                        ENS
                      </span>
                    )}
                    <span className={`ml-auto shrink-0 font-mono text-[10px] ${modelTone.text}`}>
                      {MODEL_LABELS[model.status]} · {formatCycle(model.latestCycle)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

/** Neutral pill used while availability is loading or has failed. */
function PlaceholderBadge({ tone, label }: { tone: StatusTone; label: string }) {
  return (
    <div
      className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${tone.chip} ${tone.text}`}
      role="status"
    >
      <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
      <span>{label}</span>
    </div>
  );
}

export function CenterStatusBadges({
  centers,
  isLoading,
  isError,
}: {
  centers: CenterHealth[];
  isLoading: boolean;
  isError: boolean;
}) {
  if (centers.length > 0) {
    return (
      <>
        {centers.map((center) => (
          <CenterBadge key={center.id} center={center} />
        ))}
      </>
    );
  }

  if (isError) {
    return (
      <PlaceholderBadge
        tone={{ ...CENTER_TONES.down, dot: "bg-rose-500 shadow-[0_0_8px_#f43f5e]" }}
        label="Forecast feeds offline"
      />
    );
  }

  return (
    <PlaceholderBadge
      tone={{
        ...CENTER_TONES.degraded,
        dot: "bg-amber-400 animate-pulse shadow-[0_0_8px_#f59e0b]",
      }}
      label={isLoading ? "Syncing forecast feeds…" : "Forecast feeds"}
    />
  );
}
