import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, setCsrfToken, waitForJob } from "./api";
import { ExpertHeatmap } from "./ExpertHeatmap";
import { ResearchArchive } from "./ResearchArchive";
import type {
  BenchmarkDatasetRecord,
  BenchmarkInfo,
  BenchmarkItemPage,
  BenchmarkRun,
  ComparisonRecord,
  CreateExpertProfileRequest,
  JobRecord,
  ModelRegistryEntry,
  ModelSession,
  ProfileProposal,
  RoutingSummary,
  RuntimeStatus,
  SavedExpertProfile,
  SessionStatus,
  SystemStatus,
} from "./types";

const modelId = "Qwen/Qwen3.6-35B-A3B-FP8";

export default function App() {
  const queryClient = useQueryClient();
  const customBenchmarkInput = useRef<HTMLInputElement>(null);
  const initializedSelection = useRef<string | null>(null);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState(
    "fixture-arithmetic",
  );
  const [benchmarkSearch, setBenchmarkSearch] = useState("");
  const [itemOffset, setItemOffset] = useState(0);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [baselineRun, setBaselineRun] = useState<BenchmarkRun | null>(null);
  const [maskedRun, setMaskedRun] = useState<BenchmarkRun | null>(null);
  const [routingVariant, setRoutingVariant] = useState<"baseline" | "masked">(
    "baseline",
  );
  const [proposal, setProposal] = useState<ProfileProposal | null>(null);
  const [keepPerLayer, setKeepPerLayer] = useState(64);
  const [profileName, setProfileName] = useState("Workload profile · 64/layer");
  const [activeJob, setActiveJob] = useState<JobRecord | null>(null);
  const [metric, setMetric] = useState<"routing_mass" | "selection_counts">(
    "routing_mass",
  );
  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: () => api<SessionStatus>("/api/session"),
    staleTime: 30_000,
  });
  const accessReady = sessionQuery.data?.authenticated === true;

  const statusQuery = useQuery({
    queryKey: ["status"],
    queryFn: () => api<SystemStatus>("/api/system/status"),
    enabled: accessReady,
  });
  const runtimeQuery = useQuery({
    queryKey: ["runtime-status"],
    queryFn: () => api<RuntimeStatus>("/api/runtime/status"),
    enabled: accessReady,
    refetchInterval: activeJob?.kind === "model_load" ? 2_000 : false,
  });
  const currentModelQuery = useQuery({
    queryKey: ["model-session-current"],
    queryFn: () => api<ModelSession>("/api/model-sessions/current"),
    enabled:
      accessReady &&
      statusQuery.data?.model_state !== undefined &&
      statusQuery.data.model_state !== "unloaded",
    retry: false,
  });
  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: () => api<ModelRegistryEntry[]>("/api/models"),
    enabled: accessReady,
  });
  const benchmarksQuery = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api<BenchmarkInfo[]>("/api/benchmarks"),
    enabled: accessReady,
  });
  const selectedBenchmark = benchmarksQuery.data?.find(
    (benchmark) => benchmark.id === selectedBenchmarkId,
  );
  const itemsQuery = useQuery({
    queryKey: [
      "benchmark-items",
      selectedBenchmarkId,
      benchmarkSearch,
      itemOffset,
    ],
    queryFn: () => {
      const params = new URLSearchParams({
        offset: String(itemOffset),
        limit: "100",
      });
      if (benchmarkSearch.trim()) params.set("search", benchmarkSearch.trim());
      return api<BenchmarkItemPage>(
        `/api/benchmarks/${encodeURIComponent(selectedBenchmarkId)}/items?${params}`,
      );
    },
    enabled: accessReady && selectedBenchmark?.ready === true,
  });
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<BenchmarkRun[]>("/api/runs"),
    enabled: accessReady,
  });
  const modelSessionsQuery = useQuery({
    queryKey: ["model-sessions"],
    queryFn: () => api<ModelSession[]>("/api/model-sessions"),
    enabled: accessReady,
  });
  const profilesQuery = useQuery({
    queryKey: ["profiles"],
    queryFn: () => api<SavedExpertProfile[]>("/api/profiles"),
    enabled: accessReady,
  });
  const comparisonsQuery = useQuery({
    queryKey: ["comparisons"],
    queryFn: () => api<ComparisonRecord[]>("/api/comparisons"),
    enabled: accessReady,
  });
  const routingRun =
    routingVariant === "masked" && maskedRun ? maskedRun : baselineRun;
  const routingQuery = useQuery({
    queryKey: ["routing", routingRun?.id],
    queryFn: () => api<RoutingSummary>(`/api/runs/${routingRun?.id}/routing`),
    enabled: Boolean(routingRun),
  });

  useEffect(() => {
    setCsrfToken(sessionQuery.data?.csrf_token ?? null);
  }, [sessionQuery.data]);

  useEffect(() => {
    if (
      itemsQuery.data &&
      initializedSelection.current !== selectedBenchmarkId
    ) {
      setSelectedItems(
        new Set(itemsQuery.data.items.map((item) => item.id)),
      );
      initializedSelection.current = selectedBenchmarkId;
    }
  }, [itemsQuery.data, selectedBenchmarkId]);

  const loadModel = useMutation({
    mutationFn: async () => {
      const job = await api<JobRecord>("/api/model-sessions", {
        method: "POST",
        body: JSON.stringify({ model_id: modelId }),
      });
      await waitForJob(job, setActiveJob);
      return api<ModelSession>("/api/model-sessions/current");
    },
    onSuccess: (modelSession) => {
      queryClient.setQueryData(["model-session-current"], modelSession);
      queryClient.invalidateQueries({ queryKey: ["status"] });
      queryClient.invalidateQueries({ queryKey: ["runtime-status"] });
      queryClient.invalidateQueries({ queryKey: ["model-sessions"] });
    },
    onSettled: () => setActiveJob(null),
  });

  const login = useMutation({
    mutationFn: (token: string) =>
      api<SessionStatus>("/api/session/login", {
        method: "POST",
        body: JSON.stringify({ token }),
      }),
    onSuccess: (session) => {
      setCsrfToken(session.csrf_token);
      queryClient.setQueryData(["session"], session);
    },
  });

  const prepareBenchmark = useMutation({
    mutationFn: async () => {
      const submitted = await api<JobRecord>(
        `/api/benchmarks/${encodeURIComponent(selectedBenchmarkId)}/prepare`,
        { method: "POST" },
      );
      return waitForJob(submitted, setActiveJob);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
      queryClient.invalidateQueries({ queryKey: ["benchmark-items"] });
    },
    onSettled: () => setActiveJob(null),
  });

  const importCustomBenchmark = useMutation({
    mutationFn: async (file: File) => {
      const content = await file.text();
      const name = file.name.replace(/\.jsonl$/i, "").trim() || "Custom benchmark";
      return api<BenchmarkDatasetRecord>("/api/benchmarks/custom", {
        method: "POST",
        body: JSON.stringify({
          name,
          description: `Imported from ${file.name}`,
          content,
        }),
      });
    },
    onSuccess: (record) => {
      initializedSelection.current = null;
      setSelectedItems(new Set());
      setSelectedBenchmarkId(record.id);
      setBenchmarkSearch("");
      setItemOffset(0);
      setBaselineRun(null);
      setMaskedRun(null);
      setProposal(null);
      queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
    },
    onSettled: () => {
      if (customBenchmarkInput.current) {
        customBenchmarkInput.current.value = "";
      }
    },
  });

  const cancelActiveJob = useMutation({
    mutationFn: (jobId: string) =>
      api<JobRecord>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
    onSuccess: (job) => setActiveJob(job),
  });

  const runBaseline = useMutation({
    mutationFn: async () => {
      const submitted = await api<JobRecord>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          benchmark_id: selectedBenchmarkId,
          item_ids: [...selectedItems],
        }),
      });
      const job = await waitForJob(submitted, setActiveJob);
      if (!job.result_id) throw new Error("Benchmark job has no result");
      return api<BenchmarkRun>(`/api/runs/${job.result_id}`);
    },
    onSuccess: (nextRun) => {
      setBaselineRun(nextRun);
      setMaskedRun(null);
      setRoutingVariant("baseline");
      setProposal(null);
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
    onSettled: () => setActiveJob(null),
  });

  const proposeProfile = useMutation({
    mutationFn: () =>
      api<ProfileProposal>("/api/profiles/propose", {
        method: "POST",
        body: JSON.stringify({
          run_id: baselineRun?.id,
          keep_per_layer: keepPerLayer,
          metric: metric === "selection_counts" ? "selection_count" : metric,
        }),
      }),
    onSuccess: (nextProposal) => {
      setProposal(nextProposal);
      setProfileName(`Workload profile · ${keepPerLayer}/layer`);
    },
  });

  const createProfile = useMutation({
    mutationFn: (request: CreateExpertProfileRequest) =>
      api<SavedExpertProfile>("/api/profiles", {
        method: "POST",
        body: JSON.stringify(request),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["profiles"] });
    },
  });

  const loadProfile = useMutation({
    mutationFn: async (savedProfile: SavedExpertProfile) => {
      const job = await api<JobRecord>("/api/model-sessions", {
        method: "POST",
        body: JSON.stringify({
          model_id: savedProfile.model_id,
          profile: savedProfile.profile,
          profile_id: savedProfile.id,
        }),
      });
      await waitForJob(job, setActiveJob);
      return api<ModelSession>("/api/model-sessions/current");
    },
    onSuccess: (modelSession) => {
      queryClient.setQueryData(["model-session-current"], modelSession);
      queryClient.invalidateQueries({ queryKey: ["status"] });
      queryClient.invalidateQueries({ queryKey: ["runtime-status"] });
      queryClient.invalidateQueries({ queryKey: ["model-sessions"] });
    },
    onSettled: () => setActiveJob(null),
  });

  const createComparison = useMutation({
    mutationFn: ({ baseline, candidate }: {
      baseline: BenchmarkRun;
      candidate: BenchmarkRun;
    }) =>
      api<ComparisonRecord>("/api/comparisons", {
        method: "POST",
        body: JSON.stringify({
          baseline_run_id: baseline.id,
          candidate_run_id: candidate.id,
        }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["comparisons"] });
    },
  });

  const runMasked = useMutation({
    mutationFn: async () => {
      if (!baselineRun) throw new Error("Run a baseline cohort first");
      const submitted = await api<JobRecord>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          benchmark_id: baselineRun.benchmark_id,
          item_ids: baselineRun.items.map((item) => item.item_id),
        }),
      });
      const job = await waitForJob(submitted, setActiveJob);
      if (!job.result_id) throw new Error("Benchmark job has no result");
      return api<BenchmarkRun>(`/api/runs/${job.result_id}`);
    },
    onSuccess: (nextRun) => {
      setMaskedRun(nextRun);
      setRoutingVariant("masked");
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      if (
        baselineRun?.status === "completed" &&
        nextRun.status === "completed"
      ) {
        createComparison.mutate({ baseline: baselineRun, candidate: nextRun });
      }
    },
    onSettled: () => setActiveJob(null),
  });

  const model = modelsQuery.data?.[0];
  const benchmark = selectedBenchmark;
  const visibleItems = itemsQuery.data?.items ?? [];
  const selectedCount = selectedItems.size;
  const allVisibleSelected =
    visibleItems.length > 0 &&
    visibleItems.every((item) => selectedItems.has(item.id));
  const currentProfile = currentModelQuery.data?.profile ?? null;
  const profiles = profilesQuery.data ?? [];
  const savedProposal = proposal
    ? [createProfile.data, ...profiles].find(
        (saved) =>
          saved !== undefined &&
          JSON.stringify(saved.profile) === JSON.stringify(proposal.profile),
      )
    : undefined;
  const activeProfileId = currentModelQuery.data?.profile_id ?? (
    currentProfile
      ? profiles.find(
        (saved) =>
          JSON.stringify(saved.profile) === JSON.stringify(currentProfile),
        )?.id ?? null
      : null
  );
  const profileMatchesProposal = Boolean(
    proposal &&
      currentProfile &&
      JSON.stringify(proposal.profile) === JSON.stringify(currentProfile),
  );
  const error =
    statusQuery.error ||
    runtimeQuery.error ||
    currentModelQuery.error ||
    modelsQuery.error ||
    benchmarksQuery.error ||
    itemsQuery.error ||
    runsQuery.error ||
    modelSessionsQuery.error ||
    profilesQuery.error ||
    comparisonsQuery.error ||
    prepareBenchmark.error ||
    importCustomBenchmark.error ||
    cancelActiveJob.error ||
    loadModel.error ||
    runBaseline.error ||
    proposeProfile.error ||
    createProfile.error ||
    loadProfile.error ||
    runMasked.error ||
    createComparison.error;
  const selectedCategories = useMemo(() => {
    if (!itemsQuery.data) return [];
    return itemsQuery.data.categories;
  }, [itemsQuery.data]);
  const comparisonRows = useMemo(() => {
    if (!baselineRun || !maskedRun) return [];
    const maskedItems = new Map(
      maskedRun.items.map((item) => [item.item_id, item]),
    );
    return baselineRun.items.map((baseline) => ({
      baseline,
      masked: maskedItems.get(baseline.item_id),
    }));
  }, [baselineRun, maskedRun]);
  const pairedScoreDelta =
    baselineRun?.score != null && maskedRun?.score != null
      ? maskedRun.score - baselineRun.score
      : null;

  if (sessionQuery.isPending) {
    return <div className="app-loading">Preparing the workbench…</div>;
  }
  if (sessionQuery.error) {
    return <div className="app-loading error">{sessionQuery.error.message}</div>;
  }
  if (!sessionQuery.data?.authenticated) {
    return (
      <LoginScreen
        pending={login.isPending}
        error={login.error instanceof Error ? login.error.message : null}
        onLogin={(token) => login.mutate(token)}
      />
    );
  }

  function toggleItem(itemId: string) {
    setSelectedItems((current) => {
      const next = new Set(current);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  }

  function chooseBenchmark(benchmarkId: string) {
    initializedSelection.current = null;
    setSelectedBenchmarkId(benchmarkId);
    setSelectedItems(new Set());
    setBenchmarkSearch("");
    setItemOffset(0);
    setBaselineRun(null);
    setMaskedRun(null);
    setProposal(null);
    setRoutingVariant("baseline");
  }

  function toggleVisibleItems() {
    setSelectedItems((current) => {
      const next = new Set(current);
      for (const item of visibleItems) {
        if (allVisibleSelected) next.delete(item.id);
        else next.add(item.id);
      }
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
          <a className="nav-item" href="#compare" aria-label="Compare runs">
            ⇄<span>Compare</span>
          </a>
          <a className="nav-item" href="#archive" aria-label="Research archive">
            ◷<span>Archive</span>
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

        {activeJob && (
          <div className="job-banner" role="status" aria-live="polite">
            <div>
              <span className="section-label">
                {activeJob.kind === "model_load"
                  ? "Model job"
                  : activeJob.kind === "dataset_prepare"
                    ? "Dataset job"
                    : "Benchmark job"}
              </span>
              <strong>
                {activeJob.status === "queued"
                  ? "Queued"
                  : activeJob.status === "cancelled"
                    ? "Cancelled"
                    : "Working"}
              </strong>
            </div>
            <progress
              max={Math.max(activeJob.progress_total, 1)}
              value={activeJob.progress_current}
            />
            <span>
              {formatProgress(activeJob)}
            </span>
            {activeJob.kind !== "model_load" &&
              (activeJob.status === "queued" || activeJob.status === "running") && (
                <button
                  className="text-button"
                  disabled={cancelActiveJob.isPending}
                  onClick={() => cancelActiveJob.mutate(activeJob.id)}
                >
                  {cancelActiveJob.isPending ? "Cancelling…" : "Cancel"}
                </button>
              )}
          </div>
        )}

        {activeJob?.kind === "model_load" && runtimeQuery.data?.managed && (
          <details className="runtime-monitor" open>
            <summary>
              vLLM startup log
              <span>
                {runtimeQuery.data.pid
                  ? `PID ${runtimeQuery.data.pid}`
                  : "Waiting for process"}
              </span>
            </summary>
            <pre>
              {runtimeQuery.data.log_tail || "Waiting for launcher output…"}
            </pre>
          </details>
        )}

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
              disabled={loadModel.isPending || Boolean(activeJob)}
            >
              {loadModel.isPending
                ? "Loading…"
                : currentProfile
                  ? "Restore baseline model"
                  : statusQuery.data?.model_state === "ready"
                    ? "Reload baseline model"
                    : "Load model"}
            </button>
          </article>

          <article className="card benchmark-card" id="benchmark">
            <div className="card-heading">
              <div>
                <span className="section-label">02 · Baseline</span>
                <h2>{benchmark?.name ?? "Benchmark fixture"}</h2>
              </div>
              <span className="count-badge">
                {selectedCount}/{benchmark?.item_count ?? 0}
              </span>
            </div>
            <label className="benchmark-select">
              Benchmark dataset
              <select
                value={selectedBenchmarkId}
                onChange={(event) => chooseBenchmark(event.target.value)}
              >
                {benchmarksQuery.data?.map((candidate) => (
                  <option key={candidate.id} value={candidate.id}>
                    {candidate.name}{candidate.ready ? "" : " · prepare first"}
                  </option>
                ))}
              </select>
            </label>
            <p>{benchmark?.description}</p>
            {benchmark && (
              <div className="dataset-meta">
                <span>{benchmark.kind}</span>
                <span>{benchmark.split}</span>
                <span>{benchmark.license}</span>
                <span title={benchmark.revision}>
                  rev {benchmark.revision.slice(0, 10)}
                </span>
              </div>
            )}
            <div className="tag-row">
              {selectedCategories.map((category) => (
                <span className="tag" key={category}>{category}</span>
              ))}
            </div>
            <input
              ref={customBenchmarkInput}
              type="file"
              accept="application/jsonl,.jsonl"
              aria-label="Import custom benchmark JSONL"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) importCustomBenchmark.mutate(file);
              }}
              hidden
            />
            <div className="benchmark-actions">
              <button
                className="text-button"
                disabled={importCustomBenchmark.isPending || Boolean(activeJob)}
                onClick={() => customBenchmarkInput.current?.click()}
              >
                {importCustomBenchmark.isPending ? "Importing…" : "Import JSONL"}
              </button>
              {benchmark?.ready ? (
                <button
                  className="primary-button"
                  disabled={
                    statusQuery.data?.model_state !== "ready" ||
                    currentProfile !== null ||
                    currentModelQuery.isPending ||
                    selectedCount === 0 ||
                    runBaseline.isPending ||
                    Boolean(activeJob)
                  }
                  onClick={() => runBaseline.mutate()}
                >
                  {runBaseline.isPending ? "Running…" : "Run baseline"}
                </button>
              ) : (
                <button
                  className="primary-button"
                  disabled={prepareBenchmark.isPending || Boolean(activeJob)}
                  onClick={() => prepareBenchmark.mutate()}
                >
                  {prepareBenchmark.isPending ? "Preparing…" : "Prepare dataset"}
                </button>
              )}
            </div>
          </article>
        </section>

        {benchmark?.ready && (
          <details className="item-drawer">
            <summary>
              Choose benchmark items <span>{selectedCount} selected</span>
            </summary>
            <div className="item-toolbar">
              <input
                type="search"
                value={benchmarkSearch}
                placeholder="Search prompts, categories, or item IDs"
                aria-label="Search benchmark items"
                onChange={(event) => {
                  setBenchmarkSearch(event.target.value);
                  setItemOffset(0);
                }}
              />
              <button className="text-button" onClick={toggleVisibleItems}>
                {allVisibleSelected ? "Clear page" : "Select page"}
              </button>
              <span>
                {itemsQuery.data?.total
                  ? `${itemOffset + 1}–${Math.min(itemOffset + visibleItems.length, itemsQuery.data.total)} of ${itemsQuery.data.total}`
                  : "No matching items"}
              </span>
              <button
                className="text-button"
                disabled={itemOffset === 0}
                onClick={() => setItemOffset(Math.max(0, itemOffset - 100))}
              >
                Previous
              </button>
              <button
                className="text-button"
                disabled={
                  !itemsQuery.data ||
                  itemOffset + itemsQuery.data.limit >= itemsQuery.data.total
                }
                onClick={() => setItemOffset(itemOffset + 100)}
              >
                Next
              </button>
            </div>
            <div className="item-list">
              {visibleItems.map((item) => (
                <label className="item-row" key={item.id}>
                  <input
                    type="checkbox"
                    checked={selectedItems.has(item.id)}
                    onChange={() => toggleItem(item.id)}
                  />
                  <span className="item-copy">
                    <strong>{item.prompt}</strong>
                    <small>
                      {item.category} · {item.scoring}
                      {item.expected ? ` · expected ${item.expected}` : ""}
                    </small>
                  </span>
                </label>
              ))}
            </div>
          </details>
        )}

        {routingRun && (
          <section className="run-summary">
            <div>
              <span className="section-label">
                {routingVariant === "masked" ? "Masked run" : "Baseline run"}
              </span>
              <h2>{formatScore(routingRun.score)}</h2>
              <p>
                {routingRun.completed_items}/{routingRun.total_items} items captured
                {routingRun.status === "cancelled" ? " before cancellation" : ""}.
              </p>
              <div className="run-export-links">
                <a href={`/api/runs/${routingRun.id}/export?format=json`} download>
                  Export JSON
                </a>
                <a href={`/api/runs/${routingRun.id}/export?format=csv`} download>
                  Export CSV
                </a>
              </div>
            </div>
            <div
              className="score-ring"
              style={
                {
                  "--score": `${(routingRun.score ?? 0) * 360}deg`,
                } as React.CSSProperties
              }
            >
              <span>{routingRun.completed_items}/{routingRun.total_items}</span>
            </div>
            <div className="transition-list">
              {routingRun.items
                .filter((item) => item.passed === false)
                .map((item) => (
                  <div key={item.item_id}>
                    <span>{item.item_id}</span>
                    <strong>{item.error ?? item.output}</strong>
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
              <div className="analysis-controls">
                {maskedRun && (
                  <div className="segmented-control" aria-label="Run variant">
                    <button
                      className={routingVariant === "baseline" ? "selected" : ""}
                      onClick={() => setRoutingVariant("baseline")}
                    >Base</button>
                    <button
                      className={routingVariant === "masked" ? "selected" : ""}
                      onClick={() => setRoutingVariant("masked")}
                    >Masked</button>
                  </div>
                )}
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
            </div>
            <div className="heatmap-frame">
              <ExpertHeatmap summary={routingQuery.data} metric={metric} />
            </div>
            <div className="profile-workbench">
              <div>
                <span className="section-label">Profile proposal</span>
                <h3>Keep {keepPerLayer} experts per layer</h3>
                <p>
                  Rank experts from the baseline capture by observed {metric === "routing_mass" ? "routing mass" : "selection count"},
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
                <div className="profile-activation">
                  <span className={`validation-state ${proposal.validation.valid ? "valid" : "invalid"}`}>
                    {proposal.validation.valid
                      ? "Fork-compatible profile"
                      : proposal.validation.errors.join(", ")}
                  </span>
                  {!savedProposal ? (
                    <div className="profile-save-row">
                      <input
                        aria-label="Expert profile name"
                        value={profileName}
                        onChange={(event) => setProfileName(event.target.value)}
                      />
                      <button
                        className="primary-button"
                        disabled={
                          !proposal.validation.valid ||
                          !profileName.trim() ||
                          createProfile.isPending
                        }
                        onClick={() =>
                          createProfile.mutate({
                            name: profileName,
                            description: `Fixed-budget ${metric === "routing_mass" ? "routing mass" : "selection count"} proposal from ${baselineRun?.id ?? "baseline"}.`,
                            model_id: modelId,
                            profile: proposal.profile,
                            source: "proposal",
                            source_run_id: baselineRun?.id,
                            metric:
                              metric === "selection_counts"
                                ? "selection_count"
                                : metric,
                            observed_mass_retained:
                              proposal.observed_mass_retained,
                          })
                        }
                      >
                        {createProfile.isPending ? "Saving…" : "Save profile"}
                      </button>
                    </div>
                  ) : profileMatchesProposal ? (
                    <button
                      className="primary-button"
                      disabled={
                        baselineRun?.status !== "completed" ||
                        runMasked.isPending ||
                        Boolean(activeJob)
                      }
                      onClick={() => runMasked.mutate()}
                    >
                      {runMasked.isPending
                        ? "Running paired cohort…"
                        : maskedRun
                          ? "Rerun masked cohort"
                          : "Run masked cohort"}
                    </button>
                  ) : (
                    <button
                      className="primary-button"
                      disabled={
                        !proposal.validation.valid ||
                        loadProfile.isPending ||
                        Boolean(activeJob)
                      }
                      onClick={() => loadProfile.mutate(savedProposal)}
                    >
                      {loadProfile.isPending
                        ? "Restarting with profile…"
                        : "Load this profile"}
                    </button>
                  )}
                </div>
              </div>
            )}
          </section>
        )}

        {baselineRun && maskedRun && (
          <section className="comparison-section" id="compare">
            <div className="section-intro">
              <div>
                <span className="section-label">04 · Paired comparison</span>
                <h2>What changed under the mask</h2>
              </div>
              <span className="cohort-lock">
                Same {baselineRun.total_items}-item cohort
              </span>
            </div>
            <div className="comparison-metrics">
              <CompareMetric
                label="Baseline score"
                value={formatScore(baselineRun.score, false)}
              />
              <CompareMetric
                label="Masked score"
                value={formatScore(maskedRun.score, false)}
              />
              <CompareMetric
                label="Score delta"
                value={formatDelta(pairedScoreDelta)}
                tone={
                  pairedScoreDelta == null
                    ? "neutral"
                    : pairedScoreDelta < 0
                    ? "negative"
                    : pairedScoreDelta > 0
                      ? "positive"
                      : "neutral"
                }
              />
              <CompareMetric
                label="Regressions"
                value={String(
                  comparisonRows.filter(
                    ({ baseline, masked }) =>
                      baseline.passed === true && masked?.passed === false,
                  ).length,
                )}
                tone={
                  comparisonRows.some(
                    ({ baseline, masked }) =>
                      baseline.passed === true && masked?.passed === false,
                  )
                    ? "negative"
                    : "neutral"
                }
              />
            </div>
            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Item</th>
                    <th>Baseline</th>
                    <th>Masked</th>
                    <th>Transition</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonRows.map(({ baseline, masked }) => {
                    const transition = !masked
                      ? "missing"
                      : baseline.passed == null || masked.passed == null
                        ? "unscored"
                      : baseline.passed === masked.passed
                        ? baseline.passed
                          ? "retained"
                          : "unchanged"
                        : baseline.passed
                          ? "regression"
                          : "recovery";
                    return (
                      <tr key={baseline.item_id}>
                        <td>
                          <strong>{baseline.item_id}</strong>
                          <small>{baseline.prompt}</small>
                        </td>
                        <td
                          className={resultClass(baseline.passed)}
                          data-label="Baseline"
                        >
                          {baseline.output}
                        </td>
                        <td
                          className={resultClass(masked?.passed)}
                          data-label="Masked"
                        >
                          {masked?.output ?? "—"}
                        </td>
                        <td data-label="Transition">
                          <span className={`transition-badge ${transition}`}>
                            {transition}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {model && (
          <ResearchArchive
            runs={runsQuery.data ?? []}
            profiles={profiles}
            comparisons={comparisonsQuery.data ?? []}
            sessions={modelSessionsQuery.data ?? []}
            modelId={model.id}
            topology={model.topology}
            activeProfileId={activeProfileId}
            busy={
              Boolean(activeJob) ||
              loadProfile.isPending ||
              createProfile.isPending
            }
            onOpenRun={(run, pairedBaseline) => {
              setProposal(null);
              if (pairedBaseline) {
                setBaselineRun(pairedBaseline);
                setMaskedRun(run);
                setRoutingVariant("masked");
                return;
              }
              const session = modelSessionsQuery.data?.find(
                (candidate) => candidate.id === run.model_session_id,
              );
              if (session?.profile) {
                setBaselineRun(null);
                setMaskedRun(run);
                setRoutingVariant("masked");
              } else {
                setBaselineRun(run);
                setMaskedRun(null);
                setRoutingVariant("baseline");
              }
            }}
            onOpenComparison={(baseline, candidate) => {
              setBaselineRun(baseline);
              setMaskedRun(candidate);
              setRoutingVariant("masked");
              setProposal(null);
            }}
            onLoadProfile={(profile) => {
              setProposal(null);
              loadProfile.mutate(profile);
            }}
            onCreateProfile={(request) => createProfile.mutateAsync(request)}
          />
        )}

        <footer>
          <span>MoE Atelier · local research mode</span>
          <span>Raw routing, honest comparisons.</span>
        </footer>
      </main>
    </div>
  );
}

function LoginScreen({
  pending,
  error,
  onLogin,
}: {
  pending: boolean;
  error: string | null;
  onLogin: (token: string) => void;
}) {
  const [token, setToken] = useState("");
  return (
    <main className="login-shell">
      <section className="login-card">
        <span className="login-monogram">M</span>
        <p className="kicker">MoE Atelier</p>
        <h1>Enter the workbench.</h1>
        <p className="lede">
          This Runpod is private by design. Use the access token configured on
          the Pod template.
        </p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            onLogin(token);
          }}
        >
          <label htmlFor="access-token">Access token</label>
          <input
            autoComplete="current-password"
            id="access-token"
            name="access-token"
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            autoFocus
          />
          {error && <div className="login-error">{error}</div>}
          <button className="primary-button" disabled={!token || pending}>
            {pending ? "Checking…" : "Continue"}
          </button>
        </form>
        <small>The token stays in this request and is never saved by the browser.</small>
      </section>
    </main>
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

function formatScore(score: number | null, includeLabel = true) {
  if (score == null) return "Unscored";
  const value = `${Math.round(score * 100)}%`;
  return includeLabel ? `${value} passed` : value;
}

function formatDelta(delta: number | null) {
  if (delta == null) return "—";
  return `${delta >= 0 ? "+" : ""}${Math.round(delta * 100)} pp`;
}

function resultClass(value: boolean | null | undefined) {
  return value == null ? "unscored" : value ? "pass" : "fail";
}

function formatProgress(job: JobRecord) {
  if (job.kind !== "dataset_prepare") {
    return `${job.progress_current}/${job.progress_total}`;
  }
  return `${formatBytes(job.progress_current)} / ${formatBytes(job.progress_total)}`;
}

function formatBytes(value: number) {
  if (value < 1_000) return `${value} B`;
  return `${(value / 1_000).toFixed(value < 100_000 ? 1 : 0)} kB`;
}

function CompareMetric({
  value,
  label,
  tone = "neutral",
}: {
  value: string;
  label: string;
  tone?: "neutral" | "positive" | "negative";
}) {
  return (
    <div className={`compare-metric ${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
