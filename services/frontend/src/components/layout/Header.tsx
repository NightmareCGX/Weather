"use client";

import { useForecastSelection } from "@/context/forecast-selection";
import { deriveCenterHealth } from "@/lib/forecast/centerHealth";

import { CenterStatusBadges } from "./CenterStatusBadge";
import { ZeusWxBrand } from "./ZeusLogo";

export function Header() {
  const { availability, status } = useForecastSelection();

  // Intentionally not memoized on `availability` alone: the provider re-renders
  // on its 60s heartbeat, and recomputing here keeps the staleness check within
  // a minute of the clock. The derivation is a walk over a handful of models.
  const centers = deriveCenterHealth(availability, Date.now());
  const isLoading = status === "loading" || status === "idle";

  return (
    <header className="relative z-40 flex h-14 items-center justify-between border-b border-slate-800/80 bg-slate-950/85 px-4 backdrop-blur-md">
      <h1 className="flex items-center" aria-label="Zeus Wx">
        <ZeusWxBrand />
      </h1>
      <div className="flex items-center gap-3">
        <CenterStatusBadges centers={centers} isLoading={isLoading} isError={status === "error"} />
      </div>
    </header>
  );
}
