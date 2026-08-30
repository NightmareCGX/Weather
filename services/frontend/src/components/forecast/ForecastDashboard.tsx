"use client";

import { useMemo } from "react";

import { SelectedLocationSummary } from "@/components/forecast/SelectedLocationSummary";
import { Meteogram } from "@/components/charts/Meteogram";
import { EnsembleChart } from "@/components/charts/EnsembleChart";
import { EnsembleDistribution } from "@/components/charts/EnsembleDistribution";
import { usePointForecast } from "@/hooks/usePointForecast";
import { useEnsemble } from "@/hooks/useEnsemble";
import { useEnsembleDistribution } from "@/hooks/useEnsembleDistribution";
import { useVariablesCatalog } from "@/hooks/useVariablesCatalog";
import { useForecastSelection } from "@/context/forecast-selection";
import { forecastLeadTimes, forecastVariableCodes } from "@/lib/forecast/transform";
import { buildVariableMeta } from "@/lib/forecast/labels";
import type { SelectedLocation } from "@/lib/api/types";

interface ForecastDashboardProps {
  location: SelectedLocation;
}

/**
 * Point forecast dashboard for the shared selected location, driven by the
 * shared forecast selection.
 *
 * The dashboard follows the selected model:
 *
 * 1. **Point forecast meteograms** from `/v1/points` use a deterministic
 *    model: the selected model when it is deterministic, otherwise the first
 *    deterministic model in availability. Ensemble models have no single-
 *    valued 2-D field for a point, so they are never passed to `/v1/points`.
 * 2. **Ensemble statistics / spread** from `/v1/ensembles` run only when the
 *    selected model is an ensemble model. For a deterministic model, the
 *    panel renders an honest "No ensemble data available for the selected
 *    forecast." empty state instead of requesting a hard-coded ensemble model
 *    that may not exist.
 * 3. **Ensemble distribution view** follows the same model.
 *
 * Every model here is derived from the database-driven availability response;
 * nothing is hard-coded. A failed secondary panel degrades in place without
 * destroying the core point forecast.
 */
export function ForecastDashboard({ location }: ForecastDashboardProps) {
  const { selection, options } = useForecastSelection();
  const selectedModel = selection?.model ?? null;
  const selectedModelIsEnsemble = options.model?.is_ensemble ?? false;
  // The Hourly Forecast tracks the UI selection. With the backend's
  // ensemble-mean reduction for a ``member`` dimension, ``/v1/points`` serves
  // the ensemble mean for an ensemble model (GEFS) exactly as it serves the
  // deterministic field for GFS — so the selected model is the source model for
  // both cases. Only when the selection predates a model availability change
  // (the selected model is not in the current availability) do we fall back to
  // the first available model. This preserves the invariant "UI selected model =
  // GEFS ⇒ Hourly Forecast source model = GEFS" (previously the dashboard
  // silently routed an ensemble selection to the first deterministic model).
  const selectedModelAvailable =
    selectedModel !== null && options.models.some((m) => m.id === selectedModel);
  const pointModel = selectedModelAvailable
    ? selectedModel
    : (options.models.find((model) => !model.is_ensemble)?.id ?? null);

  const {
    forecast,
    status: pointStatus,
    error: pointError,
  } = usePointForecast(location, { model: pointModel });

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
  // meaningful even when multiple variables are present.
  const ensembleVariable = variableCodes[0] ?? "temperature_2m";

  // Only run ensemble requests for an actual ensemble model. For a
  // deterministic model there is no member axis, so the panel shows the empty
  // state rather than requesting a hard-coded model.
  const ensembleModel = selectedModelIsEnsemble ? selectedModel : null;
  const ensemble = useEnsemble(location, leads, ensembleVariable, {
    model: ensembleModel,
  });

  const distributionLead = selection?.leadTimeHours ?? leads[0] ?? 0;
  const distribution = useEnsembleDistribution(location, distributionLead, ensembleVariable, {
    model: ensembleModel,
  });

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <SelectedLocationSummary location={location} />

      <section aria-label="Point forecast" className="border-b border-slate-200 px-4 py-4">
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Hourly Forecast</h3>
        {pointModel === null && (
          <p className="text-sm text-slate-500">
            No forecast model is available for this selection.
          </p>
        )}
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

      {selectedModelIsEnsemble && (
        <section aria-label="Ensemble statistics" className="border-b border-slate-200 px-4 py-4">
          <h3 className="mb-1 text-sm font-semibold text-slate-900">
            Ensemble Statistics{selectedModel !== null ? ` (${selectedModel.toUpperCase()})` : ""}
          </h3>
          {/* ensembleModel !== null here (the selected model is ensemble-capable).
              The body distinguishes "ensemble data unavailable" from a genuine
              request error, and never renders a misleading heading for a
              deterministic model. */}
          {(pointStatus === "loading" || ensemble.status === "loading") && (
            <p role="status" className="text-sm text-slate-500">
              Loading ensemble statistics…
            </p>
          )}
          {ensemble.status === "error" && (
            <p role="alert" className="text-sm text-red-700">
              {ensemble.error}
            </p>
          )}
          {ensemble.status === "success" && ensemble.byLead.size === 0 && (
            <p className="text-sm text-slate-500">
              Ensemble data is not yet available for this forecast.
            </p>
          )}
          {ensemble.status === "success" && ensemble.byLead.size > 0 && (
            <>
              <p className="mb-2 text-xs text-slate-500">
                {ensembleVariable} · percentile range over lead time
              </p>
              <EnsembleChart
                byLead={ensemble.byLead}
                variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
              />
            </>
          )}

          <EnsembleDistribution
            data={distribution.data}
            status={distribution.status}
            error={distribution.error}
            selectedLead={distributionLead}
            variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
          />
        </section>
      )}
    </div>
  );
}
