"use client";

import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";

import { Header } from "@/components/layout/Header";
import { LayerControls } from "@/components/map/LayerControls";
import { Legend } from "@/components/map/Legend";
import { LocationSearch } from "@/components/search/LocationSearch";
import { ForecastDashboard } from "@/components/forecast/ForecastDashboard";
import { useForecastSelection } from "@/context/forecast-selection";
import { useSelectedLocation, useStartupLocation } from "@/context/selected-location";
import { useGeolocation } from "@/hooks/useGeolocation";
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
  const startupLocation = useStartupLocation();
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // Maintain latest canonical map center in a ref so map movement does NOT trigger
  // component re-renders or premature search refetches (Rule B: map move alone does not refetch).
  const mapCenterRef = useRef<{ latitude: number; longitude: number }>({
    latitude: 39.2,
    longitude: -106.8,
  });

  const handleCenterChange = useCallback((center: { latitude: number; longitude: number }) => {
    mapCenterRef.current = center;
  }, []);

  const getMapBias = useCallback(() => {
    return mapCenterRef.current;
  }, []);

  const handleSelectLocation = (location: SelectedLocation) => {
    selectLocation(location);
    setIsPanelCollapsed(false);
  };

  const { isLocating, notice, clearNotice, locateMe } = useGeolocation({
    onLocateSuccess: () => {
      setIsPanelCollapsed(false);
    },
  });

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
            <LocationSearch onSelect={handleSelectLocation} getBias={getMapBias} />
          </div>

          {error !== null && (
            <div
              className="absolute left-1/2 top-4 z-20 -translate-x-1/2 rounded-lg border border-red-900/60 bg-red-950/80 px-4 py-2 text-sm text-red-300 shadow-xl backdrop-blur-md"
              role="alert"
            >
              {error}
            </div>
          )}

          {notice !== null && (
            <div
              className={`absolute left-1/2 ${
                error !== null ? "top-16" : "top-4"
              } z-30 flex max-w-[90vw] -translate-x-1/2 items-center gap-2 rounded-lg border px-3.5 py-1.5 text-xs shadow-xl backdrop-blur-md ${
                notice.type === "alert"
                  ? "border-amber-500/40 bg-amber-950/90 text-amber-200"
                  : "border-sky-500/40 bg-slate-900/90 text-sky-300"
              }`}
              role={notice.type === "alert" ? "alert" : "status"}
              data-testid="geo-notice"
            >
              <span>{notice.message}</span>
              <button
                type="button"
                onClick={clearNotice}
                aria-label="Dismiss notice"
                className="ml-1 rounded p-0.5 hover:bg-white/10 focus:outline-none focus:ring-1 focus:ring-slate-400"
              >
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          )}

          {loading && layer === null ? (
            <div className="flex h-full items-center justify-center font-mono text-sm text-slate-400">
              Loading map layer…
            </div>
          ) : (
            <WeatherMap
              layer={layer}
              selectedLocation={selectedLocation}
              approximateLocation={startupLocation}
              validTime={validTime}
              availableLeads={options.leadTimes}
              availableValidTimes={options.variable?.valid_times ?? []}
              onSelect={handleSelectLocation}
              onCenterChange={handleCenterChange}
              onLocate={locateMe}
              isLocating={isLocating}
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
                ? "relative w-0 shrink-0 bg-slate-900"
                : "relative w-96 max-w-[calc(100%-2rem)] shrink-0 border-l border-slate-800 bg-slate-900/95 shadow-2xl backdrop-blur-md lg:w-[28rem]"
            }
          >
            <button
              type="button"
              aria-label={isPanelCollapsed ? "Expand forecast panel" : "Collapse forecast panel"}
              aria-expanded={!isPanelCollapsed}
              aria-controls="forecast-panel-content"
              title={isPanelCollapsed ? "Expand forecast panel" : "Collapse forecast panel"}
              onClick={() => setIsPanelCollapsed((prev) => !prev)}
              className="absolute -left-7 top-1/2 z-20 flex h-11 w-7 -translate-y-1/2 items-center justify-center rounded-l-md border border-r-0 border-slate-700 bg-slate-900 text-slate-400 shadow-md hover:bg-slate-800 hover:text-cyan-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
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
