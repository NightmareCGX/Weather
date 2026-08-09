"use client";

import { useMemo, useState } from "react";

import { SelectedLocationSummary } from "@/components/forecast/SelectedLocationSummary";
import { Meteogram } from "@/components/charts/Meteogram";
import { EnsembleChart } from "@/components/charts/EnsembleChart";
import { EnsembleDistribution } from "@/components/charts/EnsembleDistribution";
import { usePointForecast } from "@/hooks/usePointForecast";
import { useEnsemble } from "@/hooks/useEnsemble";
import { useEnsembleDistribution } from "@/hooks/useEnsembleDistribution";
import { useVariablesCatalog } from "@/hooks/useVariablesCatalog";
import { forecastLeadTimes, forecastVariableCodes } from "@/lib/forecast/transform";
import { buildVariableMeta } from "@/lib/forecast/labels";
import type { SelectedLocation } from "@/lib/api/types";

interface ForecastDashboardProps {
  location: SelectedLocation;
}

/**
 * Milestone 13 forecast dashboard for the shared selected location.
 *
 * The dashboard owns the forecast/ensemble request lifecycle and renders three
 * independently-failing panels:
 *
 * 1. **Point forecast meteograms** from `/v1/points` (deterministic model).
 * 2. **Ensemble statistics / spread** from `/v1/ensembles` (fan chart over
 *    lead time), required independently of the distribution view.
 * 3. **Ensemble distribution view** from raw member values when the backend
 *    exposes them; an honest "not yet available" state otherwise.
 *
 * A failed secondary panel degrades in place without destroying the core
 * point forecast.
 */
export function ForecastDashboard({ location }: ForecastDashboardProps) {
  const { forecast, status: pointStatus, error: pointError } = usePointForecast(location);

  const variableMeta = useVariablesCatalog();
  const meta = useMemo(() => buildVariableMeta(variableMeta.variables), [variableMeta.variables]);

  const leads = useMemo(
    () => (forecast !== null ? forecastLeadTimes(forecast.forecasts) : []),
    [forecast]
  );
  const variableCodes = useMemo(
    () => (forecast !== null ? forecastVariableCodes(forecast.forecasts) : []),
    [forecast]
  );

  // The ensemble view is anchored to the first forecast variable so it stays
  // meaningful even when multiple variables are present. A future milestone may
  // add a variable switcher here.
  const ensembleVariable = variableCodes[0] ?? "temperature_2m";

  const ensemble = useEnsemble(location, leads, ensembleVariable);

  // The Ensemble Distribution View is a focused, opt-in request for the
  // selected lead only (`include_members=true`), kept separate from the
  // statistics timeline so the fan chart never pays for raw members.
  const distributionLead = leads[0] ?? 0;
  const distribution = useEnsembleDistribution(location, distributionLead, ensembleVariable);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <SelectedLocationSummary location={location} />

      <section aria-label="Point forecast" className="border-b border-slate-200 px-4 py-4">
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Hourly Forecast</h3>
        {pointStatus === "loading" && (
          <p role="status" className="text-sm text-slate-500">
            Loading forecast…
          </p>
        )}
        {pointStatus === "error" && (
          <p role="alert" className="text-sm text-red-700">
            {pointError}
          </p>
        )}
        {pointStatus === "success" && forecast !== null && (
          <>
            {variableCodes.map((code) => (
              <Meteogram
                key={code}
                forecasts={forecast.forecasts}
                variableCode={code}
                meta={meta[code] ?? { name: code, unit: "" }}
              />
            ))}
          </>
        )}
      </section>

      <section aria-label="Ensemble statistics" className="border-b border-slate-200 px-4 py-4">
        <h3 className="mb-1 text-sm font-semibold text-slate-900">Ensemble Statistics (GEFS)</h3>
        <p className="mb-2 text-xs text-slate-500">
          {ensembleVariable} · percentile range over lead time
        </p>
        {ensemble.status === "loading" && (
          <p role="status" className="text-sm text-slate-500">
            Loading ensemble statistics…
          </p>
        )}
        {ensemble.status === "error" && (
          <p role="alert" className="text-sm text-red-700">
            {ensemble.error}
          </p>
        )}
        {ensemble.status === "success" && ensemble.byLead.size > 0 && (
          <EnsembleChart
            byLead={ensemble.byLead}
            variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
          />
        )}

        <EnsembleDistribution
          data={distribution.data}
          status={distribution.status}
          error={distribution.error}
          selectedLead={distributionLead}
          variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
        />
      </section>
    </div>
  );
}
