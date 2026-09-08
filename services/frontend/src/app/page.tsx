"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

import { Header } from "@/components/layout/Header";
import { LayerControls } from "@/components/map/LayerControls";
import { Legend } from "@/components/map/Legend";
import { LocationSearch } from "@/components/search/LocationSearch";
import { ForecastDashboard } from "@/components/forecast/ForecastDashboard";
import { useForecastSelection } from "@/context/forecast-selection";
import { useSelectedLocation } from "@/context/selected-location";
import { useMapLayer } from "@/hooks/useMapLayer";
import type { SelectedLocation } from "@/lib/api/types";

const WeatherMap = dynamic(() => import("@/components/map/WeatherMap").then((m) => m.WeatherMap), {
  ssr: false,
});

/**
 * Frontend shell for the data-driven forecast explorer.
 *
 * The map layer configuration (model / variable / initial time / lead time)
 * is owned by {@link ForecastSelectionProvider}; this page reads the shared
 * selection and fetches the map layer metadata for it. Selecting a location
 * opens the point forecast dashboard. The base map, search, and legend are
 * preserved.
 */
export default function HomePage() {
  const { validTime, options } = useForecastSelection();
  const { layer, loading, error } = useMapLayer();
  const { selectedLocation, selectLocation, clearSelection } = useSelectedLocation();
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  const handleSelectLocation = (location: SelectedLocation) => {
    selectLocation(location);
    setIsPanelCollapsed(false);
  };

  const handleCloseForecastPanel = () => {
    clearSelection();
    setIsPanelCollapsed(false);
  };

  return (
    <div className="flex h-full flex-col">
      <Header />
      <LayerControls />

      <div className="flex min-h-0 flex-1">
        <main className="relative min-w-0 flex-1">
          <div className="absolute left-4 top-4 z-20 w-72 max-w-[calc(100%-2rem)]">
            <LocationSearch onSelect={handleSelectLocation} />
          </div>

          {error !== null && (
            <div
              className="absolute left-1/2 top-4 z-20 -translate-x-1/2 rounded border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700"
              role="alert"
            >
              {error}
            </div>
          )}

          {loading && layer === null ? (
            <div className="flex h-full items-center justify-center text-slate-500">
              Loading map layer…
            </div>
          ) : (
            <WeatherMap
              layer={layer}
              selectedLocation={selectedLocation}
              validTime={validTime}
              availableLeads={options.leadTimes}
              onSelect={handleSelectLocation}
            />
          )}

          <Legend layer={layer} variableName={options.variable?.name} />
        </main>

        {selectedLocation !== null && (
          <aside
            id="forecast-panel"
            aria-label="Forecast panel"
            className={
              isPanelCollapsed
                ? "relative w-0 shrink-0 bg-white"
                : "relative w-96 max-w-[calc(100%-2rem)] shrink-0 border-l border-slate-200 bg-white lg:w-[26rem]"
            }
          >
            <button
              type="button"
              aria-label={isPanelCollapsed ? "Expand forecast panel" : "Collapse forecast panel"}
              aria-expanded={!isPanelCollapsed}
              aria-controls="forecast-panel-content"
              title={isPanelCollapsed ? "Expand forecast panel" : "Collapse forecast panel"}
              onClick={() => setIsPanelCollapsed((prev) => !prev)}
              className="absolute -left-7 top-1/2 z-20 flex h-11 w-7 -translate-y-1/2 items-center justify-center rounded-l-md border border-r-0 border-slate-300 bg-white text-slate-500 shadow-sm hover:bg-slate-50 hover:text-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
            >
              <svg
                className="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
                focusable="false"
              >
                {isPanelCollapsed ? (
                  <polyline points="15 18 9 12 15 6" />
                ) : (
                  <polyline points="9 18 15 12 9 6" />
                )}
              </svg>
            </button>

            <div
              id="forecast-panel-content"
              className={isPanelCollapsed ? "hidden" : "flex h-full flex-col"}
            >
              <ForecastDashboard location={selectedLocation} onClose={handleCloseForecastPanel} />
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
