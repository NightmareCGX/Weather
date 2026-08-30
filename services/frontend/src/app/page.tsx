"use client";

import dynamic from "next/dynamic";

import { Header } from "@/components/layout/Header";
import { LayerControls } from "@/components/map/LayerControls";
import { Legend } from "@/components/map/Legend";
import { LocationSearch } from "@/components/search/LocationSearch";
import { ForecastDashboard } from "@/components/forecast/ForecastDashboard";
import { useForecastSelection } from "@/context/forecast-selection";
import { useSelectedLocation } from "@/context/selected-location";
import { useMapLayer } from "@/hooks/useMapLayer";

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
  const { selectedLocation, selectLocation } = useSelectedLocation();

  return (
    <div className="flex h-full flex-col">
      <Header />
      <LayerControls />

      <div className="flex min-h-0 flex-1">
        <main className="relative min-w-0 flex-1">
          <div className="absolute left-4 top-4 z-20 w-72 max-w-[calc(100%-2rem)]">
            <LocationSearch onSelect={selectLocation} />
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
              onSelect={selectLocation}
            />
          )}

          <Legend layer={layer} />
        </main>

        {selectedLocation !== null && (
          <aside className="w-96 max-w-full shrink-0 border-l border-slate-200 bg-white lg:w-[26rem]">
            <ForecastDashboard location={selectedLocation} />
          </aside>
        )}
      </div>
    </div>
  );
}
