import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "./api";
import { ExpertHeatmap } from "./ExpertHeatmap";
import type {
  BenchmarkInfo,
  BenchmarkItem,
  BenchmarkRun,
  ModelRegistryEntry,
  ProfileProposal,
  RoutingSummary,
  SystemStatus,
} from "./types";

const modelId = "Qwen/Qwen3.6-35B-A3B-FP8";

export default function App() {
  const queryClient = useQueryClient();
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [proposal, setProposal] = useState<ProfileProposal | null>(null);
  const [keepPerLayer, setKeepPerLayer] = useState(64);
  const [metric, setMetric] = useState<"routing_mass" | "selection_counts">(
    "routing_mass",
  );

  const statusQuery = useQuery({
    queryKey: ["status"],
    queryFn: () => api<SystemStatus>("/api/system/status"),
  });
  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: () => api<ModelRegistryEntry[]>("/api/models"),
  });
  const benchmarksQuery = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api<BenchmarkInfo[]>("/api/benchmarks"),
  });
  const itemsQuery = useQuery({
    queryKey: ["benchmark-items"],
    queryFn: () =>
      api<BenchmarkItem[]>("/api/benchmarks/fixture-arithmetic/items"),
  });
  const routingQuery = useQuery({
    queryKey: ["routing", run?.id],
    queryFn: () => api<RoutingSummary>(`/api/runs/${run?.id}/routing`),
    enabled: Boolean(run),
  });

  useEffect(() => {
    if (itemsQuery.data && selectedItems.size === 0) {
      setSelectedItems(new Set(itemsQuery.data.map((item) => item.id)));
    }
  }, [itemsQuery.data, selectedItems.size]);

  const loadModel = useMutation({
    mutationFn: () =>
      api("/api/model-sessions", {
        method: "POST",
        body: JSON.stringify({ model_id: modelId }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["status"] }),
  });

  const runBenchmark = useMutation({
    mutationFn: () =>
      api<BenchmarkRun>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          benchmark_id: "fixture-arithmetic",
          item_ids: [...selectedItems],
        }),
      }),
    onSuccess: (nextRun) => {
      setRun(nextRun);
      setProposal(null);
    },
  });

  const proposeProfile = useMutation({
    mutationFn: () =>
      api<ProfileProposal>("/api/profiles/propose", {
        method: "POST",
        body: JSON.stringify({
          run_id: run?.id,
          keep_per_layer: keepPerLayer,
          metric: metric === "selection_counts" ? "selection_count" : metric,
        }),
      }),
    onSuccess: setProposal,
  });

  const model = modelsQuery.data?.[0];
  const benchmark = benchmarksQuery.data?.[0];
  const selectedCount = selectedItems.size;
  const allSelected = selectedCount === itemsQuery.data?.length;
  const error =
    statusQuery.error ||
    modelsQuery.error ||
    itemsQuery.error ||
    loadModel.error ||
    runBenchmark.error ||
    proposeProfile.error;
  const selectedCategories = useMemo(() => {
    if (!itemsQuery.data) return [];
    return [...new Set(itemsQuery.data.map((item) => item.category))];
  }, [itemsQuery.data]);

  function toggleItem(itemId: string) {
    setSelectedItems((current) => {
      const next = new Set(current);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  }

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand-mark">M</div>
        <nav aria-label="Primary navigation">
          <a className="nav-item active" href="#model" aria-label="Model">
            ◇<span>Model</span>
          </a>
          <a className="nav-item" href="#benchmark" aria-label="Benchmarks">
            ◫<span>Bench</span>
          </a>
          <a className="nav-item" href="#engagement" aria-label="Profiles">
            ⌁<span>Experts</span>
          </a>
        </nav>
        <span className="version">v{statusQuery.data?.version ?? "0.1"}</span>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <span className="eyebrow">MoE Atelier</span>
            <span className="topbar-note">Research workbench</span>
          </div>
          <div className={`status-pill ${statusQuery.data?.model_state ?? "unloaded"}`}>
            <span />
            {statusQuery.data?.mode === "mock" ? "Mock laboratory" : "GPU runtime"}
          </div>
        </header>

        <section className="hero" id="model">
          <p className="kicker">A smaller model, shaped by the work.</p>
          <h1>Understand where the model thinks.</h1>
          <p className="lede">
            Measure expert engagement, form a deliberate mask, and compare what
            remains—without losing sight of the workflow that made it useful.
          </p>
        </section>

        {error && <div className="error-banner">{(error as Error).message}</div>}

        <section className="overview-grid">
          <article className="card model-card">
            <div className="card-heading">
              <div>
                <span className="section-label">01 · Model</span>
                <h2>{model?.display_name ?? "Loading registry…"}</h2>
              </div>
              <span className="monogram">Q</span>
            </div>
            <p>{model?.notes}</p>
            <div className="fact-row">
              <Fact value={model?.topology.num_layers ?? "—"} label="MoE layers" />
              <Fact value={model?.topology.num_experts ?? "—"} label="Experts / layer" />
              <Fact value={model?.topology.top_k ?? "—"} label="Active / token" />
            </div>
            <button
              className="primary-button"
              onClick={() => loadModel.mutate()}
              disabled={loadModel.isPending || statusQuery.data?.model_state === "ready"}
            >
              {statusQuery.data?.model_state === "ready"
                ? "Model ready"
                : loadModel.isPending
                  ? "Loading…"
                  : "Load model"}
            </button>
          </article>

          <article className="card benchmark-card" id="benchmark">
            <div className="card-heading">
              <div>
                <span className="section-label">02 · Baseline</span>
                <h2>{benchmark?.name ?? "Benchmark fixture"}</h2>
              </div>
              <span className="count-badge">{selectedCount}/{itemsQuery.data?.length ?? 0}</span>
            </div>
            <p>{benchmark?.description}</p>
            <div className="tag-row">
              {selectedCategories.map((category) => (
                <span className="tag" key={category}>{category}</span>
              ))}
            </div>
            <div className="benchmark-actions">
              <button
                className="text-button"
                onClick={() =>
                  setSelectedItems(
                    allSelected
                      ? new Set()
                      : new Set(itemsQuery.data?.map((item) => item.id)),
                  )
                }
              >
                {allSelected ? "Clear selection" : "Select all"}
              </button>
              <button
                className="primary-button"
                disabled={
                  statusQuery.data?.model_state !== "ready" ||
                  selectedCount === 0 ||
                  runBenchmark.isPending
                }
                onClick={() => runBenchmark.mutate()}
              >
                {runBenchmark.isPending ? "Running…" : "Run baseline"}
              </button>
            </div>
          </article>
        </section>

        <details className="item-drawer">
          <summary>Choose benchmark items <span>{selectedCount} selected</span></summary>
          <div className="item-list">
            {itemsQuery.data?.map((item) => (
              <label className="item-row" key={item.id}>
                <input
                  type="checkbox"
                  checked={selectedItems.has(item.id)}
                  onChange={() => toggleItem(item.id)}
                />
                <span className="item-copy">
                  <strong>{item.prompt}</strong>
                  <small>{item.category} · expected {item.expected}</small>
                </span>
              </label>
            ))}
          </div>
        </details>

        {run && (
          <section className="run-summary">
            <div>
              <span className="section-label">Latest run</span>
              <h2>{Math.round(run.score * 100)}% passed</h2>
              <p>{run.completed_items} items completed with routing capture.</p>
            </div>
            <div className="score-ring" style={{ "--score": `${run.score * 360}deg` } as React.CSSProperties}>
              <span>{run.completed_items}/{run.total_items}</span>
            </div>
            <div className="transition-list">
              {run.items.filter((item) => !item.passed).map((item) => (
                <div key={item.item_id}>
                  <span>{item.item_id}</span>
                  <strong>{item.output}</strong>
                  <small>expected {item.expected}</small>
                </div>
              ))}
            </div>
          </section>
        )}

        {routingQuery.data && (
          <section className="engagement-section" id="engagement">
            <div className="section-intro">
              <div>
                <span className="section-label">03 · Expert engagement</span>
                <h2>The shape of this workload</h2>
              </div>
              <div className="segmented-control" aria-label="Heatmap metric">
                <button
                  className={metric === "routing_mass" ? "selected" : ""}
                  onClick={() => setMetric("routing_mass")}
                >Mass</button>
                <button
                  className={metric === "selection_counts" ? "selected" : ""}
                  onClick={() => setMetric("selection_counts")}
                >Count</button>
              </div>
            </div>
            <div className="heatmap-frame">
              <ExpertHeatmap summary={routingQuery.data} metric={metric} />
            </div>
            <div className="profile-workbench">
              <div>
                <span className="section-label">Profile proposal</span>
                <h3>Keep {keepPerLayer} experts per layer</h3>
                <p>
                  Rank experts by observed {metric === "routing_mass" ? "routing mass" : "selection count"},
                  while preserving the router’s top-k floor.
                </p>
              </div>
              <div className="range-field">
                <input
                  type="range"
                  min={model?.topology.top_k ?? 8}
                  max={model?.topology.num_experts ?? 256}
                  step="8"
                  value={keepPerLayer}
                  onChange={(event) => setKeepPerLayer(Number(event.target.value))}
                />
                <div><span>More selective</span><span>More coverage</span></div>
              </div>
              <button
                className="primary-button"
                disabled={proposeProfile.isPending}
                onClick={() => proposeProfile.mutate()}
              >
                {proposeProfile.isPending ? "Calculating…" : "Create proposal"}
              </button>
            </div>
            {proposal && (
              <div className="proposal-result">
                <div>
                  <span className="proposal-value">
                    {Math.round(proposal.observed_mass_retained * 1000) / 10}%
                  </span>
                  <span>observed routing mass retained</span>
                </div>
                <div>
                  <span className="proposal-value">
                    {Math.round(proposal.validation.retained_fraction * 100)}%
                  </span>
                  <span>experts eligible</span>
                </div>
                <div className={`validation-state ${proposal.validation.valid ? "valid" : "invalid"}`}>
                  {proposal.validation.valid ? "Fork-compatible profile" : proposal.validation.errors.join(", ")}
                </div>
              </div>
            )}
          </section>
        )}

        <footer>
          <span>MoE Atelier · local research mode</span>
          <span>Raw routing, honest comparisons.</span>
        </footer>
      </main>
    </div>
  );
}

function Fact({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="fact">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
