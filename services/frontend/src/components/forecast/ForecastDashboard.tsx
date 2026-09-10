"use client";

import { useMemo } from "react";

import { SelectedLocationSummary } from "@/components/forecast/SelectedLocationSummary";
import { Meteogram } from "@/components/charts/Meteogram";
import { EnsembleChart } from "@/components/charts/EnsembleChart";
import { EnsembleDistribution } from "@/components/charts/EnsembleDistribution";
import { EnsemblePhaseSupport } from "@/components/charts/EnsemblePhaseSupport";
import { WindRose } from "@/components/charts/WindRose";
import { usePointForecast } from "@/hooks/usePointForecast";
import { useElevation } from "@/hooks/useElevation";
import { useEnsemble } from "@/hooks/useEnsemble";
import { useEnsembleDistribution } from "@/hooks/useEnsembleDistribution";
import { useVariablesCatalog } from "@/hooks/useVariablesCatalog";
import { useForecastSelection } from "@/context/forecast-selection";
import { useDisplayTimezone } from "@/context/selected-location";
import { resolveValidTime } from "@/lib/forecast/availability";
import { formatDayHourWithTimeZone } from "@/lib/forecast/time";
import { forecastVariableCodes } from "@/lib/forecast/transform";
import { buildVariableMeta } from "@/lib/forecast/labels";
import type { SelectedLocation } from "@/lib/api/types";

interface ForecastDashboardProps {
  location: SelectedLocation;
  onClose?: () => void;
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
export function ForecastDashboard({ location, onClose }: ForecastDashboardProps) {
  const { selection, options } = useForecastSelection();
  const displayTimezone = useDisplayTimezone();
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

  const elevation = useElevation(location);

  const {
    forecast,
    status: pointStatus,
    error: pointError,
  } = usePointForecast(location, { model: pointModel });

  const variableMeta = useVariablesCatalog();
  const meta = useMemo(() => buildVariableMeta(variableMeta.variables), [variableMeta.variables]);

  const variableCodes = useMemo(
    () => (forecast !== null ? forecastVariableCodes(forecast.forecasts) : []),
    [forecast]
  );

  // The ensemble view derives its variable and lead parameters from the
  // authoritative normalized forecast selection (and availability options),
  // NEVER from the asynchronous point-forecast response. This ensures model,
  // variable, and cycle switches immediately update the ensemble requests
  // rather than requesting variables from stale point payloads.
  const ensembleVariable =
    selection?.variable ?? options.variable?.id ?? options.variables[0]?.id ?? "temperature_2m";

  const ensembleLeads = options.leadTimes;

  // Only run ensemble requests for an actual ensemble model. For a
  // deterministic model there is no member axis, so the panel shows the empty
  // state rather than requesting a hard-coded model.
  const ensembleModel = selectedModelIsEnsemble ? selectedModel : null;
  const ensemble = useEnsemble(location, ensembleLeads, ensembleVariable, {
    model: ensembleModel,
  });

  const distributionLead =
    selection?.validTime ??
    options.validTimes?.[0] ??
    selection?.leadTimeHours ??
    options.leadTimes[0] ??
    0;
  const distribution = useEnsembleDistribution(location, distributionLead, ensembleVariable, {
    model: ensembleModel,
  });

  // Canonical mapping from lead_time_hours -> valid_time (ISO string)
  const validTimesByLead = useMemo(() => {
    const map = new Map<number, string>();
    const targetCycle = options.initialTime?.value ?? selection?.initialTime ?? null;

    // 1. From variable availability valid_times matching the active source cycle (authoritative V2 mapping)
    if (options.variable?.valid_times) {
      for (const vt of options.variable.valid_times) {
        if (!targetCycle || vt.source_cycle === targetCycle) {
          map.set(vt.lead_time_hours, vt.valid_time);
        }
      }
    }

    // 2. From point forecast forecasts if available
    if (forecast?.forecasts) {
      for (const entry of forecast.forecasts) {
        if (!map.has(entry.lead_time_hours)) {
          if (!targetCycle || !entry.cycle_time || entry.cycle_time === targetCycle) {
            map.set(entry.lead_time_hours, entry.valid_time);
          }
        }
      }
    }

    // 3. Fallback: derive from initial_time + lead_time_hours only when canonical valid_times are absent
    if (targetCycle !== null) {
      for (const lead of options.leadTimes) {
        if (!map.has(lead)) {
          const resolved = resolveValidTime(targetCycle, lead);
          if (resolved) map.set(lead, resolved);
        }
      }
    }

    return map;
  }, [
    options.variable?.valid_times,
    options.initialTime?.value,
    options.leadTimes,
    selection?.initialTime,
    forecast?.forecasts,
  ]);

  const distributionValidTime = useMemo(() => {
    if (typeof distributionLead === "string") {
      return distributionLead;
    }
    return (
      validTimesByLead.get(distributionLead) ??
      options.validTimes?.[0] ??
      distribution.data?.valid_time ??
      null
    );
  }, [distributionLead, validTimesByLead, options.validTimes, distribution.data?.valid_time]);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <SelectedLocationSummary
        location={location}
        elevation_m={location.elevation_m ?? elevation.elevation_m}
        elevationStatus={location.elevation_m !== null ? "success" : elevation.status}
        onClose={onClose}
      />

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
                timezone={displayTimezone}
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
              <p className="mb-2 text-xs text-slate-500">{ensembleVariable} · percentile range</p>
              <EnsembleChart
                byLead={ensemble.byLead}
                variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
                timezone={displayTimezone}
                validTimesByLead={validTimesByLead}
              />
            </>
          )}

          <EnsembleDistribution
            data={distribution.data}
            status={distribution.status}
            error={distribution.error}
            selectedLead={distributionLead}
            validTime={distributionValidTime}
            timezone={displayTimezone}
            variableLabel={meta[ensembleVariable]?.name ?? ensembleVariable}
          />

          {ensembleVariable === "wind_10m" && distribution.data?.wind_rose && (
            <div className="mt-4 rounded border border-slate-200 bg-slate-50/50 p-3">
              <h4 className="mb-1 text-center text-xs font-semibold text-slate-700">
                10m Wind Direction & Speed Distribution (Wind Rose)
              </h4>
              <p className="mb-2 text-center text-[11px] text-slate-500">
                {distributionValidTime
                  ? `${formatDayHourWithTimeZone(distributionValidTime, displayTimezone)} · `
                  : ""}
                30 ensemble members
              </p>
              <WindRose windRose={distribution.data.wind_rose} />
            </div>
          )}

          {ensembleVariable === "precipitation_amount_3h" &&
            (distribution.data?.phase_support ||
              (ensemble.status === "success" && ensemble.byLead.size > 0)) && (
              <EnsemblePhaseSupport
                byLead={ensemble.byLead}
                validTimesByLead={validTimesByLead}
                timezone={displayTimezone}
                phaseSupport={distribution.data?.phase_support ?? undefined}
                transitionFrequency={distribution.data?.transition_frequency}
                selectedLead={distributionLead}
                validTime={distributionValidTime}
                memberCount={distribution.data?.member_count ?? 30}
              />
            )}
        </section>
      )}
    </div>
  );
}
