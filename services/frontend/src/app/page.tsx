"use client";

import dynamic from "next/dynamic";

import { Header } from "@/components/layout/Header";
import { LayerControls } from "@/components/map/LayerControls";
import { Legend } from "@/components/map/Legend";
import { LocationSearch } from "@/components/search/LocationSearch";
import { ForecastDashboard } from "@/components/forecast/ForecastDashboard";
import { useMapConfig } from "@/context/map-config";
import { useSelectedLocation } from "@/context/selected-location";
import { useMapData } from "@/hooks/useMapData";

const WeatherMap = dynamic(() => import("@/components/map/WeatherMap").then((m) => m.WeatherMap), {
  ssr: false,
});

/**
 * Frontend shell for Milestone 12 + Milestone 13.
 *
 * Milestone 12 provided the MapLibre map, model/variable/lead-time controls,
 * and legend, driven by `/v1/models` and `/v1/maps` metadata. Milestone 13
 * adds location search, map point selection, and the point forecast dashboard
 * — all sharing one selected-location state.
 */
export default function HomePage() {
  const { model, variable, leadTimeHours, setModel, setVariable, setLeadTimeHours } =
    useMapConfig();
  const { models, layer, loading, error } = useMapData();
  const { selectedLocation, selectLocation } = useSelectedLocation();

  return (
    <div className="flex h-full flex-col">
      <Header />
      <LayerControls
        models={models}
        model={model}
        variable={variable}
        leadTimeHours={leadTimeHours}
        onModelChange={setModel}
        onVariableChange={setVariable}
        onLeadTimeChange={setLeadTimeHours}
      />

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
