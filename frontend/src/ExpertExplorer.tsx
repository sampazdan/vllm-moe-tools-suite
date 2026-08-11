import { useEffect, useMemo, useState } from "react";

import { ExpertHeatmap } from "./ExpertHeatmap";
import {
  observedMassRetained,
  profileSelectionCount,
  replaceProfileLayer,
  selectByCumulativeMass,
  selectGlobalTopExperts,
  selectTopExperts,
  toggleProfileExpert,
} from "./expertSelection";
import { expertsForLayer } from "./profileEditor";
import type { ExpertProfile, RoutingExploreResponse, RoutingSummary } from "./types";

export type ExpertExplorerMode = "references" | "selection" | "delta";

interface ExpertExplorerProps {
  data: RoutingExploreResponse;
  profile?: ExpertProfile | null;
  onProfileChange?: (profile: ExpertProfile) => void;
  defaultMode?: ExpertExplorerMode;
  initialMetric?: "routing_mass" | "selection_count";
  onMetricChange?: (metric: "routing_mass" | "selection_count") => void;
  title?: string;
  referenceProfile?: ExpertProfile | null;
  onStrategyChange?: (
    strategy: string,
    config: Record<string, unknown>,
  ) => void;
}

export function ExpertExplorer({
  data,
  profile = null,
  onProfileChange,
  defaultMode = "references",
  initialMetric = "routing_mass",
  onMetricChange,
  title = "Expert Explorer",
  referenceProfile = null,
  onStrategyChange,
}: ExpertExplorerProps) {
  const [mode, setMode] = useState<ExpertExplorerMode>(defaultMode);
  const [metric, setMetric] = useState<"routing_mass" | "selection_count">(
    initialMetric,
  );
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [assistedKeep, setAssistedKeep] = useState(64);
  const [selectionMethod, setSelectionMethod] = useState<
    "fixed_per_layer" | "global_budget" | "cumulative_mass"
  >("fixed_per_layer");
  const [globalBudget, setGlobalBudget] = useState(
    Math.max(data.top_k * data.layer_ids.length, 64 * data.layer_ids.length),
  );
  const [massThreshold, setMassThreshold] = useState(0.9);
  useEffect(() => setMetric(initialMetric), [initialMetric]);
  const matrix = metric === "routing_mass" ? data.routing_mass : data.selection_counts;
  const comparisonMatrix = metric === "routing_mass"
    ? data.comparison_routing_mass
    : data.comparison_selection_counts;
  const visibleLayerIndex = selectedLayer == null
    ? null
    : data.layer_ids.indexOf(selectedLayer);
  const visibleValues = visibleLayerIndex == null
    ? []
    : matrix[visibleLayerIndex] ?? [];
  const visibleComparison = visibleLayerIndex == null
    ? []
    : comparisonMatrix?.[visibleLayerIndex] ?? [];
  const selectedExperts = useMemo(
    () =>
      profile && selectedLayer != null
        ? new Set(expertsForLayer(profile, selectedLayer, data.num_experts))
        : new Set<number>(),
    [data.num_experts, profile, selectedLayer],
  );
  const selectedCount = profile
    ? profileSelectionCount(profile, data.layer_ids, data.num_experts)
    : data.layer_ids.length * data.num_experts;
  const retainedMass = profile
    ? observedMassRetained(
        profile,
        data.layer_ids,
        data.num_experts,
        data.routing_mass,
      )
    : 1;
  const canEdit = Boolean(profile && onProfileChange);
  const comparisonAvailable = Boolean(comparisonMatrix);

  function toggleExpert(layerId: number, expertId: number) {
    if (!profile || !onProfileChange) return;
    onProfileChange(
      toggleProfileExpert(profile, layerId, expertId, data.num_experts, data.top_k),
    );
    onStrategyChange?.("manual", {
      metric,
      layer_id: layerId,
      expert_id: expertId,
    });
  }

  function applyAssistedSelection() {
    if (!profile || !onProfileChange) return;
    if (selectionMethod === "global_budget") {
      onProfileChange(
        selectGlobalTopExperts(
          profile,
          data.layer_ids,
          matrix,
          globalBudget,
          data.top_k,
        ),
      );
      onStrategyChange?.("global_budget", {
        metric,
        global_budget: globalBudget,
        minimum_per_layer: data.top_k,
      });
      return;
    }
    if (selectionMethod === "cumulative_mass") {
      onProfileChange(
        selectByCumulativeMass(
          profile,
          data.layer_ids,
          matrix,
          massThreshold,
          data.top_k,
        ),
      );
      onStrategyChange?.("cumulative_mass", {
        metric,
        threshold: massThreshold,
        minimum_per_layer: data.top_k,
      });
      return;
    }
    const layerIds = selectedLayer == null ? data.layer_ids : [selectedLayer];
    const rows = selectedLayer == null
      ? matrix
      : [matrix[data.layer_ids.indexOf(selectedLayer)] ?? []];
    onProfileChange(selectTopExperts(profile, layerIds, rows, assistedKeep));
    onStrategyChange?.("fixed_per_layer", {
      metric,
      keep_per_layer: assistedKeep,
      layer_id: selectedLayer,
      minimum_per_layer: data.top_k,
    });
  }

  function updateSelectedLayer(action: "minimum" | "full" | "revert" | "copy_previous") {
    if (!profile || !onProfileChange || selectedLayer == null) return;
    const layerIndex = data.layer_ids.indexOf(selectedLayer);
    let next = profile;
    if (action === "minimum") {
      next = selectTopExperts(
        profile,
        [selectedLayer],
        [matrix[layerIndex] ?? []],
        data.top_k,
      );
    } else if (action === "full") {
      next = replaceProfileLayer(
        profile,
        selectedLayer,
        Array.from({ length: data.num_experts }, (_, expertId) => expertId),
      );
    } else if (action === "revert") {
      next = replaceProfileLayer(
        profile,
        selectedLayer,
        referenceProfile
          ? expertsForLayer(referenceProfile, selectedLayer, data.num_experts)
          : Array.from({ length: data.num_experts }, (_, expertId) => expertId),
      );
    } else {
      const previousLayer = data.layer_ids[layerIndex - 1];
      if (previousLayer == null) return;
      next = replaceProfileLayer(
        profile,
        selectedLayer,
        expertsForLayer(profile, previousLayer, data.num_experts),
      );
    }
    onProfileChange(next);
    onStrategyChange?.("manual", {
      metric,
      layer_id: selectedLayer,
      operation: action,
    });
  }

  const heatmapSummary: RoutingSummary = {
    run_id: data.fingerprint,
    layer_ids: data.layer_ids,
    selection_counts: data.selection_counts,
    routing_mass: data.routing_mass,
    total_routed_slots: data.total_routed_slots,
  };

  return (
    <section className="expert-explorer">
      <header className="expert-explorer-header">
        <div>
          <span className="section-label">Routing telemetry</span>
          <h2>{title}</h2>
          <p>
            Aggregate preserves every layer × expert coordinate. Each source is
            normalized before its explicit weight is applied, so run length does
            not silently change workload influence.
          </p>
        </div>
        <div className="expert-capture-facts" aria-label="Routing capture summary">
          <span><strong>{formatCompact(data.served_tokens)}</strong> tokens</span>
          <span>
            <strong>{formatCapture(data)}</strong> calls
          </span>
          <span><strong>{data.sources.length}</strong> sources</span>
        </div>
      </header>

      <div className="expert-explorer-toolbar">
        <div className="segmented-control" aria-label="Expert explorer mode">
          {(["references", "selection", "delta"] as const).map((value) => (
            <button
              key={value}
              className={mode === value ? "selected" : ""}
              disabled={value === "delta" && !comparisonAvailable}
              title={
                value === "delta" && !comparisonAvailable
                  ? "Add a comparison source to inspect routing deltas"
                  : undefined
              }
              onClick={() => setMode(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <label>
          Layer
          <select
            value={selectedLayer ?? "aggregate"}
            onChange={(event) =>
              setSelectedLayer(
                event.target.value === "aggregate"
                  ? null
                  : Number(event.target.value),
              )
            }
          >
            <option value="aggregate">Aggregate · all layers</option>
            {data.layer_ids.map((layerId) => (
              <option key={layerId} value={layerId}>Layer {layerId}</option>
            ))}
          </select>
        </label>
        <div className="segmented-control" aria-label="Routing metric">
          <button
            className={metric === "routing_mass" ? "selected" : ""}
            onClick={() => {
              setMetric("routing_mass");
              onMetricChange?.("routing_mass");
            }}
          >
            Probability mass
          </button>
          <button
            className={metric === "selection_count" ? "selected" : ""}
            onClick={() => {
              setMetric("selection_count");
              onMetricChange?.("selection_count");
            }}
          >
            Reference share
          </button>
        </div>
      </div>

      {mode === "selection" && (
        <div className="expert-selection-toolbar">
          <div>
            <strong>{selectedCount.toLocaleString()}</strong>
            <span>eligible experts</span>
          </div>
          <div>
            <strong>{Math.round(retainedMass * 1000) / 10}%</strong>
            <span>observed mass retained</span>
          </div>
          <label>
            Assisted strategy
            <select
              value={selectionMethod}
              onChange={(event) =>
                setSelectionMethod(event.target.value as typeof selectionMethod)
              }
            >
              <option value="fixed_per_layer">Fixed per layer</option>
              <option value="global_budget">Global expert budget</option>
              <option value="cumulative_mass">Cumulative mass</option>
            </select>
          </label>
          {selectionMethod === "fixed_per_layer" && (
            <label>
              Top experts / layer
              <input
                type="number"
                min={data.top_k}
                max={data.num_experts}
                value={assistedKeep}
                onChange={(event) => setAssistedKeep(Number(event.target.value))}
              />
            </label>
          )}
          {selectionMethod === "global_budget" && (
            <label>
              Global budget
              <input
                type="number"
                min={data.top_k * data.layer_ids.length}
                max={data.num_experts * data.layer_ids.length}
                value={globalBudget}
                onChange={(event) => setGlobalBudget(Number(event.target.value))}
              />
            </label>
          )}
          {selectionMethod === "cumulative_mass" && (
            <label>
              Mass target
              <span className="selection-percent-input">
                <input
                  type="number"
                  min={1}
                  max={100}
                  step={0.5}
                  value={massThreshold * 100}
                  onChange={(event) =>
                    setMassThreshold(Number(event.target.value) / 100)
                  }
                />
                %
              </span>
            </label>
          )}
          <button
            className="text-button"
            disabled={
              !canEdit ||
              (selectionMethod === "fixed_per_layer" && assistedKeep < data.top_k) ||
              (selectionMethod === "global_budget" &&
                globalBudget < data.top_k * data.layer_ids.length)
            }
            onClick={applyAssistedSelection}
          >
            Apply {selectionMethod === "fixed_per_layer" && selectedLayer != null
              ? `to layer ${selectedLayer}`
              : "to aggregate"}
          </button>
          {!canEdit && <small>Open Profile Studio to edit this selection.</small>}
        </div>
      )}

      {mode === "selection" && selectedLayer != null && canEdit && (
        <div className="expert-layer-actions">
          <span>Layer {selectedLayer} editing</span>
          <button className="text-button" onClick={() => updateSelectedLayer("minimum")}>Keep top-{data.top_k}</button>
          <button className="text-button" onClick={() => updateSelectedLayer("full")}>Full layer</button>
          <button className="text-button" onClick={() => updateSelectedLayer("revert")}>Revert layer</button>
          <button
            className="text-button"
            disabled={data.layer_ids.indexOf(selectedLayer) === 0}
            onClick={() => updateSelectedLayer("copy_previous")}
          >
            Copy previous layer
          </button>
        </div>
      )}

      <div className="expert-canvas">
        {selectedLayer == null ? (
          <ExpertHeatmap
            summary={heatmapSummary}
            metric={metric === "selection_count" ? "selection_counts" : metric}
            comparisonValues={comparisonMatrix}
            mode={mode}
            profile={profile}
            onExpertClick={mode === "selection" && canEdit ? toggleExpert : undefined}
          />
        ) : (
          <div
            className={`expert-layer-grid mode-${mode}`}
            aria-label={`Experts for layer ${selectedLayer}`}
          >
            {visibleValues.map((value, expertId) => {
              const delta = value - (visibleComparison[expertId] ?? 0);
              const displayValue = mode === "delta" ? delta : value;
              const intensity = normalizedIntensity(
                displayValue,
                mode === "delta" ? visibleValues.map((entry, index) =>
                  entry - (visibleComparison[index] ?? 0)) : visibleValues,
              );
              const selected = !profile || selectedExperts.has(expertId);
              return (
                <button
                  key={expertId}
                  className={`${selected ? "selected" : "masked"} ${displayValue < 0 ? "negative" : "positive"}`}
                  style={{ "--heat": intensity } as React.CSSProperties}
                  aria-pressed={mode === "selection" ? selected : undefined}
                  disabled={mode === "selection" && !canEdit}
                  title={`Layer ${selectedLayer} · Expert ${expertId} · ${formatMetric(displayValue, metric, mode)}`}
                  onClick={() => mode === "selection" && toggleExpert(selectedLayer, expertId)}
                >
                  <span>E{expertId}</span>
                  <strong>{formatTileValue(displayValue, metric)}</strong>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <footer className="expert-explorer-footer">
        <div>
          <span className="section-label">Sources</span>
          {data.sources.map((source) => (
            <span className="expert-source" key={`${source.kind}:${source.id}`}>
              {source.label || source.id.slice(0, 8)}
              {source.status ? ` · ${source.status}` : ""}
            </span>
          ))}
        </div>
        <div className="filter-capabilities" aria-label="Available routing filters">
          {Object.entries(data.filter_capabilities).map(([filter, supported]) => (
            <span className={supported ? "supported" : "unavailable"} key={filter}>
              {filter.replaceAll("_", " ")} {supported ? "✓" : "—"}
            </span>
          ))}
        </div>
      </footer>
    </section>
  );
}

function formatCompact(value: number | null | undefined) {
  if (value == null) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

function formatCapture(data: RoutingExploreResponse) {
  if (data.captured_inference_calls == null) return "—";
  return data.total_inference_calls == null
    ? String(data.captured_inference_calls)
    : `${data.captured_inference_calls}/${data.total_inference_calls}`;
}

function normalizedIntensity(value: number, values: number[]) {
  const maximum = Math.max(...values.map((candidate) => Math.abs(candidate)), 0);
  return maximum > 0 ? String(Math.max(0.06, Math.abs(value) / maximum)) : "0.06";
}

function formatMetric(
  value: number,
  metric: "routing_mass" | "selection_count",
  mode: ExpertExplorerMode,
) {
  const formatted = value.toFixed(5);
  return mode === "delta" && value > 0 ? `+${formatted}` : formatted;
}

function formatTileValue(value: number, metric: "routing_mass" | "selection_count") {
  if (metric === "selection_count") {
    return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
  }
  if (Math.abs(value) >= 100) return Math.round(value).toLocaleString();
  if (Math.abs(value) >= 1) return value.toFixed(1);
  return value.toFixed(3);
}
