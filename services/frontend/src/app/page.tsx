"use client";

import dynamic from "next/dynamic";

import { Header } from "@/components/layout/Header";
import { LayerControls } from "@/components/map/LayerControls";
import { Legend } from "@/components/map/Legend";
import { useMapConfig } from "@/context/map-config";
import { useMapData } from "@/hooks/useMapData";

const WeatherMap = dynamic(() => import("@/components/map/WeatherMap").then((m) => m.WeatherMap), {
  ssr: false,
});

/**
 * Frontend shell for Milestone 12: a MapLibre map with a model/variable/lead
 * time control surface and a legend, all driven by `/v1/models` and
 * `/v1/maps` metadata.
 */
export default function HomePage() {
  const { model, variable, leadTimeHours, setModel, setVariable, setLeadTimeHours } =
    useMapConfig();
  const { models, layer, loading, error } = useMapData();

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

      <main className="relative flex-1">
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
          <WeatherMap layer={layer} />
        )}

        <Legend layer={layer} />
      </main>
    </div>
  );
}
