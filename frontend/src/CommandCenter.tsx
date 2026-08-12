import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { AgenticWorkbench } from "./AgenticWorkbench";
import { ApiError, api, waitForJob } from "./api";
import { AppLink, AppShell, PageHeader } from "./AppShell";
import {
  evaluationContractPayload,
  executionPolicyPayload,
  generationConfigPayload,
  runItemKey,
  withGenerationConfig,
} from "./contracts";
import {
  defaultEvaluationContract,
  defaultExecutionPolicy,
  EvaluationControls,
} from "./EvaluationControls";
import { ExpertExplorer } from "./ExpertExplorer";
import {
  createFullProfile,
  normalizeEngagementMatrix,
  observedMassRetained,
} from "./expertSelection";
import { BenchmarkPerformancePanel } from "./PerformancePanel";
import { navigate, routeHref, type AppRoute } from "./router";
import { benchmarkSelectionInitialization } from "./selectionInitialization";
import type {
  AgentRun,
  AgentRunComparison,
  BenchmarkInfo,
  BenchmarkItem,
  BenchmarkItemPage,
  BenchmarkProblemDetail,
  BenchmarkRunPage,
  ComparisonRecord,
  CreateExpertProfileRequest,
  EvaluationContract,
  ExecutionPolicy,
  ExpertProfile,
  JobRecord,
  ModelRegistryEntry,
  ModelSession,
  RoutingExploreRequest,
  RoutingExploreResponse,
  RoutingSourceReference,
  RunDetail,
  RunItemResult,
  RunItemResultPage,
  SavedExpertProfile,
  SystemStatus,
  TrialRoutingSummary,
} from "./types";

interface CommandCenterProps {
  route: AppRoute;
  status: SystemStatus | null;
  currentModelSession: ModelSession | null;
  activeJob: JobRecord | null;
  onJobConflict: (job: JobRecord) => void;
}

export function CommandCenter({
  route,
  status,
  currentModelSession,
  activeJob,
  onJobConflict,
}: CommandCenterProps) {
  const benchmarks = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api<BenchmarkInfo[]>("/api/benchmarks"),
  });
  const models = useQuery({
    queryKey: ["models"],
    queryFn: () => api<ModelRegistryEntry[]>("/api/models"),
  });
  const model = models.data?.find(
    (candidate) => candidate.id === currentModelSession?.model_id,
  ) ?? models.data?.[0] ?? null;

  let content: React.ReactNode;
  if (route.name === "benchmarks") {
    content = <BenchmarkCatalogPage benchmarks={benchmarks.data ?? []} />;
  } else if (route.name === "benchmark") {
    content = (
      <BenchmarkDetailPage
        benchmarkId={route.benchmarkId}
        modelReady={status?.model_state === "ready"}
        busy={Boolean(activeJob)}
      />
    );
  } else if (route.name === "runs") {
    content = <RunArchivePage />;
  } else if (route.name === "run") {
    content = <BenchmarkRunPage runId={route.runId} model={model} />;
  } else if (route.name === "agentRuns" || route.name === "agentRun") {
    content = (
      <AgentCommandPage
        runId={route.name === "agentRun" ? route.runId : null}
        status={status}
        currentModelSession={currentModelSession}
        activeJob={activeJob}
        onJobConflict={onJobConflict}
      />
    );
  } else if (route.name === "agentComparison") {
    content = (
      <AgentComparisonDetailPage
        baselineRunId={route.baselineRunId}
        candidateRunId={route.candidateRunId}
        model={model}
      />
    );
  } else if (route.name === "profiles") {
    content = <ProfileArchivePage />;
  } else if (route.name === "profile") {
    content = <ProfileDetailPage profileId={route.profileId} model={model} />;
  } else if (route.name === "profileStudio") {
    content = <ProfileStudioPage route={route} model={model} />;
  } else if (route.name === "comparisons") {
    content = <ComparisonArchivePage />;
  } else if (route.name === "comparison") {
    content = <ComparisonDetailPage comparisonId={route.comparisonId} model={model} />;
  } else {
    content = <NotFoundPage />;
  }

  return (
    <AppShell route={route} status={status} activeJob={activeJob}>
      {content}
    </AppShell>
  );
}

function BenchmarkCatalogPage({ benchmarks }: { benchmarks: BenchmarkInfo[] }) {
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<"all" | BenchmarkInfo["kind"]>("all");
  const filtered = benchmarks.filter((benchmark) => {
    const text = `${benchmark.name} ${benchmark.description} ${benchmark.categories.join(" ")}`.toLowerCase();
    return text.includes(search.toLowerCase()) && (kind === "all" || benchmark.kind === kind);
  });
  return (
    <div className="command-page benchmark-catalog-page">
      <PageHeader
        eyebrow="Benchmark library"
        title="Choose the work that should shape the model."
        description="Inspect every problem and its evaluation contract before spending a token. Curated cohorts make iteration fast; full datasets remain available for formal runs."
        actions={<AppLink className="secondary-button" href="/runs">Run archive →</AppLink>}
        meta={<><span>{benchmarks.length} adapters</span><span>{benchmarks.filter((item) => item.ready).length} ready</span></>}
      />
      <div className="catalog-toolbar">
        <input
          type="search"
          placeholder="Search benchmarks, domains, or workload types"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <div className="segmented-control" aria-label="Benchmark kind">
          {(["all", "standard", "fixture", "custom"] as const).map((value) => (
            <button
              key={value}
              className={kind === value ? "selected" : ""}
              onClick={() => setKind(value)}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      <div className="benchmark-catalog-grid">
        {filtered.map((benchmark) => (
          <article className="benchmark-catalog-card" key={benchmark.id}>
            <div className="catalog-card-topline">
              <span className={`readiness-dot ${benchmark.ready ? "ready" : "pending"}`}>
                {benchmark.ready ? "Ready" : "Prepare"}
              </span>
              <span>{benchmark.item_count.toLocaleString()} problems</span>
            </div>
            <h2>{benchmark.name}</h2>
            <p>{benchmark.description}</p>
            <div className="tag-row">
              {benchmark.categories.slice(0, 4).map((category) => (
                <span className="tag" key={category}>{category}</span>
              ))}
            </div>
            <dl>
              <div><dt>Scorer</dt><dd>{humanize(benchmark.scoring)}</dd></div>
              <div><dt>Split</dt><dd>{benchmark.split}</dd></div>
              <div><dt>License</dt><dd>{benchmark.license}</dd></div>
            </dl>
            <AppLink className="primary-button" href={routeHref("benchmark", benchmark.id)}>
              Open benchmark
            </AppLink>
          </article>
        ))}
        {filtered.length === 0 && <CommandEmpty title="No benchmarks match" detail="Try a broader search or another workload type." />}
      </div>
    </div>
  );
}

function BenchmarkDetailPage({
  benchmarkId,
  modelReady,
  busy,
}: {
  benchmarkId: string;
  modelReady: boolean;
  busy: boolean;
}) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [focusedItemId, setFocusedItemId] = useState<string | null>(null);
  const [activePanel, setActivePanel] = useState<"problem" | "contract">("problem");
  const [contract, setContract] = useState<EvaluationContract>(() =>
    defaultEvaluationContract("Benchmark scorer", {
      kind: "exact",
      label: "Answer matches the reference",
      weight: 1,
      visibility: "public",
    }),
  );
  const [policy, setPolicy] = useState<ExecutionPolicy>(() => defaultExecutionPolicy(false));
  const benchmarkQuery = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api<BenchmarkInfo[]>("/api/benchmarks"),
  });
  const benchmark = benchmarkQuery.data?.find((item) => item.id === benchmarkId) ?? null;
  const itemsQuery = useQuery({
    queryKey: ["benchmark-items", benchmarkId, search, 0],
    queryFn: () => {
      const parameters = new URLSearchParams({ offset: "0", limit: "200" });
      if (search.trim()) parameters.set("search", search.trim());
      return api<BenchmarkItemPage>(
        `/api/benchmarks/${encodeURIComponent(benchmarkId)}/items?${parameters}`,
      );
    },
    enabled: benchmark?.ready === true,
  });
  const items = itemsQuery.data?.items ?? [];
  const focused = items.find((item) => item.id === focusedItemId) ?? items[0] ?? null;
  const problemQuery = useQuery({
    queryKey: ["benchmark-problem", benchmarkId, focused?.id],
    queryFn: () =>
      api<BenchmarkProblemDetail>(
        `/api/benchmarks/${encodeURIComponent(benchmarkId)}/items/${encodeURIComponent(focused!.id)}`,
      ),
    enabled: Boolean(focused),
  });
  const initialized = useRef(false);
  const authoritativeContractLoaded = useRef(false);
  const selectionInitializedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!benchmark || initialized.current) return;
    const nextContract = benchmark.evaluation_contract ??
      defaultContractForBenchmark(benchmark);
    setContract(nextContract);
    setPolicy(withGenerationConfig(
      benchmark.execution_policy ?? defaultExecutionPolicy(false),
      benchmark.default_generation,
    ));
    initialized.current = true;
  }, [benchmark]);

  useEffect(() => {
    if (authoritativeContractLoaded.current || !problemQuery.data) return;
    if (problemQuery.data.success_criteria.public_contract) {
      setContract(problemQuery.data.success_criteria.public_contract);
    }
    setPolicy((current) => ({
      ...problemQuery.data.default_execution_policy,
      temperature: current.temperature,
      seed: current.seed,
      enable_thinking: current.enable_thinking,
      reasoning_visibility: current.reasoning_visibility,
      generation_max_tokens: current.generation_max_tokens,
    }));
    authoritativeContractLoaded.current = true;
  }, [problemQuery.data]);

  useEffect(() => {
    const initialization = benchmarkSelectionInitialization(
      selectionInitializedFor.current,
      benchmarkId,
      itemsQuery.data?.items,
    );
    if (!initialization) return;
    selectionInitializedFor.current = initialization.benchmarkId;
    setSelected(new Set(initialization.selectedIds));
    setFocusedItemId(initialization.focusedItemId);
  }, [benchmarkId, itemsQuery.data]);

  const run = useMutation({
    mutationFn: async () => {
      const submitted = await api<JobRecord>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          benchmark_id: benchmarkId,
          item_ids: [...selected],
          generation: generationConfigPayload(
            policy,
            benchmark?.default_generation ?? {
              temperature: 0,
              max_tokens: 512,
              seed: 0,
              enable_thinking: false,
            },
          ),
          execution_policy: executionPolicyPayload(policy),
          evaluation_contract: evaluationContractPayload(contract),
        }),
      });
      const completed = await waitForJob(submitted);
      if (!completed.result_id) throw new Error("Benchmark completed without a run ID");
      return completed.result_id;
    },
    onSuccess: (runId) => {
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
      navigate(routeHref("run", runId));
    },
  });

  if (!benchmark && benchmarkQuery.isPending) return <CommandLoading label="Opening benchmark…" />;
  if (!benchmark) return <NotFoundPage noun="benchmark" />;

  return (
    <div className="command-page benchmark-detail-page">
      <PageHeader
        eyebrow={`${benchmark.kind} benchmark`}
        title={benchmark.name}
        description={benchmark.description}
        actions={
          <button
            className="primary-button"
            disabled={!benchmark.ready || !modelReady || busy || run.isPending || selected.size === 0}
            onClick={() => run.mutate()}
          >
            {run.isPending ? "Running cohort…" : `Run ${selected.size} selected`}
          </button>
        }
        meta={
          <>
            <span>{benchmark.item_count.toLocaleString()} problems</span>
            <span>{benchmark.split}</span>
            <span>{humanize(benchmark.scoring)}</span>
            <span>rev {benchmark.revision.slice(0, 10)}</span>
          </>
        }
      />
      {run.error && <CommandError error={run.error} />}
      {!benchmark.ready ? (
        <CommandEmpty title="Dataset preparation required" detail="Return to the dashboard to prepare this pinned dataset revision before running it." />
      ) : (
        <div className="benchmark-builder-grid">
          <aside className="problem-browser">
            <header>
              <span className="section-label">Problem cohort</span>
              <strong>{selected.size}/{itemsQuery.data?.total ?? 0}</strong>
            </header>
            <input
              type="search"
              value={search}
              placeholder="Search problems"
              onChange={(event) => setSearch(event.target.value)}
            />
            <div className="problem-browser-actions">
              <button className="text-button" onClick={() => setSelected(new Set(items.map((item) => item.id)))}>Select page</button>
              <button className="text-button" onClick={() => setSelected(new Set())}>Clear</button>
            </div>
            <div className="problem-browser-list">
              {items.map((item, index) => (
                <button
                  className={focused?.id === item.id ? "focused" : ""}
                  key={item.id}
                  onClick={() => setFocusedItemId(item.id)}
                >
                  <input
                    type="checkbox"
                    checked={selected.has(item.id)}
                    aria-label={`Include ${item.id}`}
                    onClick={(event) => event.stopPropagation()}
                    onChange={() => setSelected((current) => toggleSet(current, item.id))}
                  />
                  <span><strong>{index + 1}. {item.id}</strong><small>{item.category} · {humanize(item.scoring)}</small></span>
                </button>
              ))}
            </div>
          </aside>

          <div className="mobile-command-tabs global" role="tablist">
            <button className={activePanel === "problem" ? "selected" : ""} onClick={() => setActivePanel("problem")}>Problem</button>
            <button className={activePanel === "contract" ? "selected" : ""} onClick={() => setActivePanel("contract")}>Contract</button>
          </div>
          <section className={`problem-inspector ${activePanel === "problem" ? "mobile-panel-active" : "mobile-panel-hidden"}`}>
            <div>
              {focused ? (
                <ProblemDetail
                  item={focused}
                  contract={contract}
                  detail={problemQuery.data ?? null}
                />
              ) : <CommandEmpty title="Choose a problem" detail="Its prompt and success criteria will appear here." />}
            </div>
          </section>
          <aside className={`contract-inspector ${activePanel === "contract" ? "mobile-panel-active" : "mobile-panel-hidden"}`}>
            <EvaluationControls
              contract={contract}
              policy={policy}
              onContractChange={setContract}
              onPolicyChange={setPolicy}
            />
          </aside>
        </div>
      )}
    </div>
  );
}

function RunArchivePage() {
  const [offset, setOffset] = useState(0);
  const runs = useQuery({
    queryKey: ["runs", offset],
    queryFn: () => api<BenchmarkRunPage>(`/api/runs?offset=${offset}&limit=50`),
  });
  return (
    <div className="command-page">
      <PageHeader
        eyebrow="Benchmark runs"
        title="Every run remains attributable."
        description="Open a run to inspect its exact cohort, answers, scoring, performance, routing, model session, and export artifacts."
        actions={<AppLink className="primary-button" href="/benchmarks">New benchmark run</AppLink>}
      />
      <div className="command-record-list">
        {(runs.data?.items ?? []).map((run) => (
          <AppLink className="command-record-row" href={routeHref("run", run.id)} key={run.id}>
            <span className={`record-state ${run.status}`}>{run.status}</span>
            <div><strong>{run.benchmark_id}</strong><small>{shortId(run.id)} · {formatDate(run.created_at)}</small></div>
            <Metric label="Score" value={formatPercent(run.score)} />
            <Metric label="Items" value={`${run.completed_items}/${run.total_items}`} />
            <span className="record-arrow">→</span>
          </AppLink>
        ))}
        {runs.isPending && <CommandLoading label="Loading runs…" />}
        {!runs.isPending && runs.data?.total === 0 && <CommandEmpty title="No benchmark runs yet" detail="Choose a benchmark to create the first immutable cohort." />}
      </div>
      {(runs.data?.total ?? 0) > 50 && (
        <div className="artifact-actions command-pagination">
          <button className="secondary-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous</button>
          <span>{offset + 1}–{Math.min(offset + 50, runs.data?.total ?? 0)} of {runs.data?.total ?? 0}</span>
          <button className="secondary-button" disabled={offset + 50 >= (runs.data?.total ?? 0)} onClick={() => setOffset(offset + 50)}>Next</button>
        </div>
      )}
    </div>
  );
}

function BenchmarkRunPage({ runId, model }: { runId: string; model: ModelRegistryEntry | null }) {
  const [selectedItemKey, setSelectedItemKey] = useState<string | null>(null);
  const [itemOffset, setItemOffset] = useState(0);
  const [mobilePanel, setMobilePanel] = useState<"items" | "result" | "experts" | "contract">("result");
  const detailQuery = useQuery({
    queryKey: ["run-detail", runId],
    queryFn: () => api<RunDetail>(`/api/runs/${encodeURIComponent(runId)}/detail`),
  });
  const itemPageQuery = useQuery({
    queryKey: ["run-items", runId, itemOffset],
    queryFn: () => api<RunItemResultPage>(
      `/api/runs/${encodeURIComponent(runId)}/items?offset=${itemOffset}&limit=100`,
    ),
  });
  const run = detailQuery.data?.run ?? null;
  const visibleItems = itemPageQuery.data?.items ?? [];
  const selected = visibleItems.find((item) => runItemKey(item) === selectedItemKey) ?? visibleItems[0] ?? null;
  const explore = useRoutingExplore(
    run ? [{ kind: "benchmark_run", id: run.id }] : [],
    [],
    model,
  );
  const contract = detailQuery.data?.provenance?.evaluation_contract ??
    detailQuery.data?.cohort?.evaluation_contract ?? null;
  const recordedPolicy = detailQuery.data?.provenance?.execution_policy ??
    detailQuery.data?.cohort?.execution_policy ?? null;
  const recordedGeneration = detailQuery.data?.provenance?.generation ??
    detailQuery.data?.cohort?.generation ?? null;
  const policy = recordedPolicy && recordedGeneration
    ? withGenerationConfig(recordedPolicy, recordedGeneration)
    : recordedPolicy;

  useEffect(() => {
    if (visibleItems[0] && selectedItemKey == null) {
      setSelectedItemKey(runItemKey(visibleItems[0]));
    }
  }, [visibleItems, selectedItemKey]);

  if (detailQuery.isPending) return <CommandLoading label="Opening run command center…" />;
  if (!run) return <NotFoundPage noun="run" />;

  return (
    <div className="command-page run-command-page">
      <PageHeader
        eyebrow="Benchmark command center"
        title={`${run.benchmark_id} · ${formatPercent(run.score)}`}
        description={`Run ${shortId(run.id)} preserves ${run.total_items} problem decisions and their expert-routing trace.`}
        actions={<><a className="secondary-button" href={`/api/runs/${run.id}/export?format=json`} download>Export JSON</a><AppLink className="primary-button" href={`/profiles/new?run=${encodeURIComponent(run.id)}`}>Create expert profile</AppLink></>}
        meta={<><span className={`record-state ${run.status}`}>{run.status}</span><span>{run.completed_items}/{run.total_items} complete</span><span>{run.scored_items} scored</span><span>{detailQuery.data?.saved_profile ? `profile ${detailQuery.data.saved_profile.profile_fingerprint.slice(0, 10)}` : "baseline"}</span></>}
      />
      <MobileCommandTabs
        value={mobilePanel}
        options={["items", "result", "experts", "contract"]}
        onChange={setMobilePanel}
      />
      <div className="run-command-grid">
        <aside className={mobileClass(mobilePanel, "items", "run-item-rail")}>
          <header><span className="section-label">Cohort</span><strong>{run.completed_items}</strong></header>
          {visibleItems.map((item, index) => (
            <button
              className={selected && runItemKey(selected) === runItemKey(item) ? "selected" : ""}
              key={runItemKey(item)}
              onClick={() => { setSelectedItemKey(runItemKey(item)); setMobilePanel("result"); }}
            >
              <span className={`result-index ${resultTone(item.passed)}`}>{itemOffset + index + 1}</span>
              <span><strong>{item.item_id}</strong><small>Attempt {item.attempt} · {item.passed == null ? "unscored" : item.passed ? "passed" : "failed"} · {formatDuration(item.latency_ms)}</small></span>
            </button>
          ))}
          {itemPageQuery.isPending && <CommandLoading label="Loading results…" />}
          {(itemPageQuery.data?.total ?? 0) > 100 && (
            <div className="run-item-pagination">
              <button disabled={itemOffset === 0} onClick={() => { setItemOffset(Math.max(0, itemOffset - 100)); setSelectedItemKey(null); }}>Previous</button>
              <small>{itemOffset + 1}–{Math.min(itemOffset + 100, itemPageQuery.data?.total ?? 0)} of {itemPageQuery.data?.total ?? 0}</small>
              <button disabled={itemOffset + 100 >= (itemPageQuery.data?.total ?? 0)} onClick={() => { setItemOffset(itemOffset + 100); setSelectedItemKey(null); }}>Next</button>
            </div>
          )}
        </aside>
        <section className={mobileClass(mobilePanel, "result", "run-result-stage")}>
          {selected && <RunItemDetail item={selected} />}
          <BenchmarkPerformancePanel
            items={visibleItems}
            performance={run.performance}
            completedItems={run.completed_items}
          />
        </section>
        <aside className={mobilePanel === "experts" || mobilePanel === "contract" ? "run-inspector mobile-panel-active" : "run-inspector mobile-panel-hidden"}>
          <div className="inspector-tabs">
            <button className={mobilePanel === "experts" ? "selected" : ""} onClick={() => setMobilePanel("experts")}>Experts</button>
            <button className={mobilePanel === "contract" ? "selected" : ""} onClick={() => setMobilePanel("contract")}>Contract</button>
          </div>
          {mobilePanel !== "contract" ? (
            explore.data ? <ExpertExplorer data={explore.data} title="Run engagement" /> : <InspectorLoading error={explore.error} />
          ) : (
            contract && policy ? (
              <EvaluationControls contract={contract} policy={policy} onContractChange={() => undefined} onPolicyChange={() => undefined} readOnly />
            ) : (
              <CommandEmpty
                title="Legacy evaluation metadata unavailable"
                detail="This run predates persisted evaluation contracts. No current defaults are substituted for its unknown historical settings."
              />
            )
          )}
        </aside>
      </div>
    </div>
  );
}

function AgentCommandPage({
  runId,
  status,
  currentModelSession,
  activeJob,
  onJobConflict,
}: {
  runId: string | null;
  status: SystemStatus | null;
  currentModelSession: ModelSession | null;
  activeJob: JobRecord | null;
  onJobConflict: (job: JobRecord) => void;
}) {
  return (
    <div className="command-page agent-command-page">
      <PageHeader
        eyebrow="Agentic workflows"
        title={runId ? "Coding run command center" : "Give the model a real workspace."}
        description={runId
          ? "Reasoning, responses, commands, output, verification, performance, and expert routing stay visibly separate and attributable."
          : "Select a benchmark task pack, execution policy, agent scaffold, and isolated Daytona environment."}
        actions={runId ? <><AgentCompareLauncher candidateRunId={runId} /><AppLink className="secondary-button" href="/agent-runs">New agentic run</AppLink></> : undefined}
      />
      <AgenticWorkbench
        mode={status?.mode ?? "mock"}
        modelState={status?.model_state ?? "unloaded"}
        currentModelSession={currentModelSession}
        busy={Boolean(activeJob && activeJob.kind !== "agent_run")}
        requestedRunId={runId}
        onRequestedRunOpened={() => undefined}
        onActivityChange={() => undefined}
        onJobConflict={onJobConflict}
        commandCenter
      />
    </div>
  );
}

function AgentCompareLauncher({ candidateRunId }: { candidateRunId: string }) {
  const runs = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => api<AgentRun[]>("/api/agent-runs"),
  });
  const candidate = runs.data?.find((run) => run.id === candidateRunId) ?? null;
  const compatible = (runs.data ?? []).filter(
    (run) =>
      run.id !== candidateRunId &&
      run.status === "completed" &&
      run.compatibility_fingerprint === candidate?.compatibility_fingerprint,
  );
  const [baselineRunId, setBaselineRunId] = useState("");

  useEffect(() => {
    if (baselineRunId || compatible.length === 0) return;
    setBaselineRunId(
      compatible.find((run) => !run.profile_id)?.id ?? compatible[0].id,
    );
  }, [baselineRunId, compatible]);

  if (!candidate || candidate.status !== "completed") return null;
  return (
    <div className="agent-compare-launcher">
      <select
        aria-label="Agentic baseline run"
        value={baselineRunId}
        onChange={(event) => setBaselineRunId(event.target.value)}
      >
        {compatible.length === 0 && <option value="">No compatible baseline</option>}
        {compatible.map((run) => (
          <option value={run.id} key={run.id}>
            {run.profile_id ? "Profile" : "Baseline"} {shortId(run.id)} · {run.passed_trials}/{run.total_trials}
          </option>
        ))}
      </select>
      <button
        className="primary-button"
        disabled={!baselineRunId}
        onClick={() =>
          navigate(
            `/agent-comparisons/${encodeURIComponent(baselineRunId)}/${encodeURIComponent(candidateRunId)}`,
          )
        }
      >
        Compare run
      </button>
    </div>
  );
}

function AgentComparisonDetailPage({
  baselineRunId,
  candidateRunId,
  model,
}: {
  baselineRunId: string;
  candidateRunId: string;
  model: ModelRegistryEntry | null;
}) {
  const comparison = useQuery({
    queryKey: ["agent-comparison", baselineRunId, candidateRunId],
    queryFn: () =>
      api<AgentRunComparison>("/api/agent-comparisons", {
        method: "POST",
        body: JSON.stringify({
          baseline_run_id: baselineRunId,
          candidate_run_id: candidateRunId,
        }),
      }),
    retry: false,
  });
  const baseline = useQuery({
    queryKey: ["agent-run", baselineRunId],
    queryFn: () => api<AgentRun>(`/api/agent-runs/${encodeURIComponent(baselineRunId)}`),
  });
  const candidate = useQuery({
    queryKey: ["agent-run", candidateRunId],
    queryFn: () => api<AgentRun>(`/api/agent-runs/${encodeURIComponent(candidateRunId)}`),
  });
  const baselineSources: RoutingSourceReference[] = (baseline.data?.trials ?? [])
    .filter((trial) => trial.routed_inference_calls > 0)
    .map((trial) => ({ kind: "agent_trial", id: trial.id }));
  const candidateSources: RoutingSourceReference[] = (candidate.data?.trials ?? [])
    .filter((trial) => trial.routed_inference_calls > 0)
    .map((trial) => ({ kind: "agent_trial", id: trial.id }));
  const explore = useRoutingExplore(candidateSources, baselineSources, model);

  if (comparison.isPending) return <CommandLoading label="Pairing agentic trials…" />;
  if (!comparison.data) {
    return (
      <div className="command-page">
        <PageHeader
          eyebrow="Agentic comparison"
          title="These runs could not be paired."
          description="Agent comparisons require the same task pack, agent, provider, budgets, reasoning mode, and task attempts."
        />
        <CommandError error={comparison.error} />
      </div>
    );
  }
  const result = comparison.data;
  return (
    <div className="command-page agent-comparison-page">
      <PageHeader
        eyebrow="Paired agentic experiment"
        title={result.name}
        description="The same task attempts, execution contract, agent scaffold, and sandbox provider—measured before and after the expert profile."
        actions={<><AppLink className="secondary-button" href={routeHref("agentRun", baselineRunId)}>Open baseline</AppLink><AppLink className="secondary-button" href={routeHref("agentRun", candidateRunId)}>Open candidate</AppLink></>}
        meta={<><span>{result.trial_count} paired trials</span><span>{result.task_pack_id}</span><span>contract {result.compatibility_fingerprint.slice(0, 10)}</span></>}
      />
      <div className="comparison-hero-metrics agentic-comparison-metrics">
        <Metric label="Baseline pass" value={`${result.baseline_passed_trials}/${result.trial_count}`} />
        <Metric label="Profile pass" value={`${result.candidate_passed_trials}/${result.trial_count}`} />
        <Metric label="Reward delta" value={formatDelta(result.mean_reward_delta)} tone={(result.mean_reward_delta ?? 0) < 0 ? "negative" : "positive"} />
        <Metric label="Regressions" value={String(result.regressions)} tone={result.regressions ? "negative" : undefined} />
        <Metric label="Recoveries" value={String(result.recoveries)} tone={result.recoveries ? "positive" : undefined} />
      </div>
      <AgentComparisonPerformance result={result} />
      {candidateSources.length > 0 && baselineSources.length > 0 ? (
        explore.data ? <ExpertExplorer data={explore.data} defaultMode="delta" title="Agentic routing delta" /> : <InspectorLoading error={explore.error} />
      ) : (
        <CommandEmpty title="Routing delta unavailable" detail="At least one paired run did not retain routed inference artifacts." />
      )}
      <div className="comparison-command-table agent-pair-table">
        <header><span>Task attempt</span><span>Baseline</span><span>Profile</span><span>Transition</span></header>
        {result.pairs.map((pair) => (
          <div key={`${pair.task_id}:${pair.attempt}`}>
            <span><strong>{pair.task_id}</strong><small>Attempt {pair.attempt} · reward {formatMaybeNumber(pair.baseline_reward)} → {formatMaybeNumber(pair.candidate_reward)}</small></span>
            <span className={trialResultTone(pair.baseline_status)}>{humanize(pair.baseline_status)}</span>
            <span className={trialResultTone(pair.candidate_status)}>{humanize(pair.candidate_status)}</span>
            <span><em className={`transition-badge ${pair.transition}`}>{humanize(pair.transition)}</em></span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentComparisonPerformance({ result }: { result: AgentRunComparison }) {
  type ComparisonMetric = readonly [
    string,
    number | null | undefined,
    number | null | undefined,
    "duration" | "number" | "rate" | "cost",
  ];
  const metrics: ComparisonMetric[] = [
    ["Model time", result.baseline_performance.model_time_ms, result.candidate_performance.model_time_ms, "duration"],
    ["Sandbox time", result.baseline_performance.sandbox_time_ms, result.candidate_performance.sandbox_time_ms, "duration"],
    ["Total tokens", result.baseline_performance.total_tokens, result.candidate_performance.total_tokens, "number"],
    ["Reported TPS", result.baseline_performance.reported_mean_tps, result.candidate_performance.reported_mean_tps, "rate"],
    ["Estimated cost", result.baseline_performance.estimated_cost_usd, result.candidate_performance.estimated_cost_usd, "cost"],
  ];
  if (
    result.baseline_performance.inference_cost_usd != null ||
    result.candidate_performance.inference_cost_usd != null
  ) {
    metrics.push(["Inference cost", result.baseline_performance.inference_cost_usd, result.candidate_performance.inference_cost_usd, "cost"]);
  }
  if (
    result.baseline_performance.judge_cost_usd != null ||
    result.candidate_performance.judge_cost_usd != null
  ) {
    metrics.push(["Judge cost", result.baseline_performance.judge_cost_usd, result.candidate_performance.judge_cost_usd, "cost"]);
  }
  return (
    <section className="agent-comparison-performance">
      <header><span className="section-label">Paired performance</span><span>Baseline</span><span>Profile</span><span>Delta</span></header>
      {metrics.map(([label, base, candidate, format]) => (
        <div key={label}><strong>{label}</strong><span>{formatComparisonValue(base, format)}</span><span>{formatComparisonValue(candidate, format)}</span><span className={(candidate ?? 0) - (base ?? 0) > 0 ? "positive" : "negative"}>{formatSignedComparisonValue(base, candidate, format)}</span></div>
      ))}
    </section>
  );
}

function ProfileArchivePage() {
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: () => api<SavedExpertProfile[]>("/api/profiles") });
  return (
    <div className="command-page">
      <PageHeader
        eyebrow="Expert profiles"
        title="Reusable hypotheses, not anonymous masks."
        description="Each immutable profile records its source workload, routing metric, manual decisions, parent revision, topology validation, and fingerprint."
        actions={<AppLink className="primary-button" href="/profiles/new">New custom profile</AppLink>}
      />
      <div className="profile-command-grid">
        {(profiles.data ?? []).map((profile) => (
          <article className="profile-command-card" key={profile.id}>
            <div><span className="artifact-kind profile">{profile.source}</span><time>{formatDate(profile.created_at)}</time></div>
            <h2>{profile.name}</h2>
            <p>{profile.description || "No research note attached."}</p>
            <dl>
              <div><dt>Eligible</dt><dd>{formatPercent(profile.validation.retained_fraction)}</dd></div>
              <div><dt>Observed mass</dt><dd>{formatPercent(profile.observed_mass_retained)}</dd></div>
              <div><dt>Fingerprint</dt><dd>{profile.profile_fingerprint.slice(0, 10)}</dd></div>
            </dl>
            <AppLink className="primary-button" href={routeHref("profile", profile.id)}>Open profile</AppLink>
          </article>
        ))}
        {profiles.isPending && <CommandLoading label="Loading profiles…" />}
        {!profiles.isPending && profiles.data?.length === 0 && <CommandEmpty title="No profiles yet" detail="Open any benchmark or agentic run and choose Create expert profile." />}
      </div>
    </div>
  );
}

function ProfileDetailPage({ profileId, model }: { profileId: string; model: ModelRegistryEntry | null }) {
  const profile = useQuery({
    queryKey: ["profile", profileId],
    queryFn: () => api<SavedExpertProfile>(`/api/profiles/${encodeURIComponent(profileId)}`),
  });
  if (profile.isPending) return <CommandLoading label="Opening profile…" />;
  if (!profile.data) return <NotFoundPage noun="profile" />;
  const sourceQuery = profile.data.source_trial_id
    ? `trial=${encodeURIComponent(profile.data.source_trial_id)}`
    : profile.data.source_run_id
      ? `run=${encodeURIComponent(profile.data.source_run_id)}`
      : "";
  const reviseHref = `/profiles/new?profile=${encodeURIComponent(profile.data.id)}${sourceQuery ? `&${sourceQuery}` : ""}`;
  return (
    <div className="command-page profile-detail-page">
      <PageHeader
        eyebrow={`${profile.data.source} expert profile`}
        title={profile.data.name}
        description={profile.data.description || "No research note attached."}
        actions={<><a className="secondary-button" href={`/api/profiles/${profile.data.id}/export`} download>Export JSON</a><AppLink className="primary-button" href={reviseHref}>Open in Profile Studio</AppLink></>}
        meta={<><span>{formatPercent(profile.data.validation.retained_fraction)} eligible</span><span>{formatPercent(profile.data.observed_mass_retained)} observed mass</span><span>{profile.data.profile_fingerprint.slice(0, 14)}</span></>}
      />
      <section className="profile-provenance" aria-label="Profile provenance">
        <div><span className="section-label">Selection</span><strong>{humanize(profile.data.selection_strategy ?? "manual")}</strong><small>{humanize(profile.data.metric ?? "not recorded")}</small></div>
        <div><span className="section-label">Workload sources</span><strong>{profile.data.source_refs?.length ?? (profile.data.source_run_id || profile.data.source_trial_id ? 1 : 0)}</strong><small>{profile.data.source_refs?.map((source) => `${source.kind === "agent_trial" ? "trial" : "run"} ${shortId(source.id)} ×${source.weight}`).join(" · ") || "Legacy provenance"}</small></div>
        <div><span className="section-label">Source fingerprint</span><strong>{profile.data.source_fingerprint?.slice(0, 14) ?? "—"}</strong><small>{profile.data.parent_profile_id ? `Revision of ${shortId(profile.data.parent_profile_id)}` : "Original revision"}</small></div>
      </section>
      <ProfileShape profile={profile.data.profile} model={model} />
    </div>
  );
}

function ProfileStudioPage({
  route,
  model,
}: {
  route: Extract<AppRoute, { name: "profileStudio" }>;
  model: ModelRegistryEntry | null;
}) {
  const queryClient = useQueryClient();
  const [sources, setSources] = useState<RoutingSourceReference[]>(() => {
    if (route.trialIds.length) {
      return route.trialIds.map((id) => ({ kind: "agent_trial", id, weight: 1 }));
    }
    if (route.runId) return [{ kind: "benchmark_run", id: route.runId }];
    return [];
  });
  const [name, setName] = useState("Custom workload profile");
  const [description, setDescription] = useState("");
  const [draft, setDraft] = useState<ExpertProfile | null>(null);
  const [referenceProfile, setReferenceProfile] = useState<ExpertProfile | null>(null);
  const [selectionMetric, setSelectionMetric] = useState<"routing_mass" | "selection_count">("routing_mass");
  const [selectionStrategy, setSelectionStrategy] = useState("manual");
  const [selectionConfig, setSelectionConfig] = useState<Record<string, unknown>>({
    metric: "routing_mass",
  });
  const initializedFingerprint = useRef<string | null>(null);
  const parentSourcesInitialized = useRef(false);
  const runs = useQuery({
    queryKey: ["runs", "profile-studio"],
    queryFn: () => api<BenchmarkRunPage>("/api/runs?limit=100"),
  });
  const agentRuns = useQuery({ queryKey: ["agent-runs"], queryFn: () => api<AgentRun[]>("/api/agent-runs") });
  const parentProfile = useQuery({
    queryKey: ["profile", route.profileId],
    queryFn: () => api<SavedExpertProfile>(`/api/profiles/${route.profileId}`),
    enabled: Boolean(route.profileId),
  });
  const explore = useRoutingExplore(sources, [], model);

  useEffect(() => {
    if (!parentProfile.data || parentSourcesInitialized.current) return;
    parentSourcesInitialized.current = true;
    const restoredSources = parentProfile.data.source_refs?.length
      ? parentProfile.data.source_refs
      : parentProfile.data.source_trial_id
        ? [{ kind: "agent_trial" as const, id: parentProfile.data.source_trial_id, weight: 1 }]
        : parentProfile.data.source_run_id
          ? [{ kind: "benchmark_run" as const, id: parentProfile.data.source_run_id, weight: 1 }]
          : [];
    if (restoredSources.length) {
      setSources(restoredSources);
      initializedFingerprint.current = null;
    }
    if (parentProfile.data.metric) setSelectionMetric(parentProfile.data.metric);
    if (parentProfile.data.selection_strategy) {
      setSelectionStrategy(parentProfile.data.selection_strategy);
      setSelectionConfig(parentProfile.data.selection_config ?? {});
    }
  }, [parentProfile.data]);

  useEffect(() => {
    if (!explore.data || initializedFingerprint.current === explore.data.fingerprint) return;
    const initial = parentProfile.data?.profile ??
      createFullProfile(explore.data.layer_ids, explore.data.num_experts);
    setDraft(initial);
    setReferenceProfile(initial);
    if (parentProfile.data) {
      setName(`${parentProfile.data.name} · revision`);
      setDescription(parentProfile.data.description);
    } else {
      setName(`Custom profile · ${explore.data.sources.map((source) => source.label).join(" + ")}`);
      setDescription("Custom expert selection derived from recorded workload routing.");
    }
    initializedFingerprint.current = explore.data.fingerprint;
  }, [explore.data, parentProfile.data]);

  const save = useMutation({
    mutationFn: (request: CreateExpertProfileRequest) =>
      api<SavedExpertProfile>("/api/profiles", {
        method: "POST",
        body: JSON.stringify(request),
      }),
    onSuccess: (saved) => {
      void queryClient.invalidateQueries({ queryKey: ["profiles"] });
      navigate(routeHref("profile", saved.id));
    },
  });

  function toggleSource(source: RoutingSourceReference) {
    setSources((current) => {
      const exists = current.some((item) => item.kind === source.kind && item.id === source.id);
      return exists
        ? current.filter((item) => item.kind !== source.kind || item.id !== source.id)
        : [...current, { ...source, weight: source.weight ?? 1 }];
    });
    initializedFingerprint.current = null;
  }

  function updateSourceWeight(source: RoutingSourceReference, weight: number) {
    setSources((current) => current.map((item) =>
      item.kind === source.kind && item.id === source.id
        ? { ...item, weight: Math.min(100, Math.max(0.1, weight || 1)) }
        : item,
    ));
    initializedFingerprint.current = null;
  }

  function saveProfile() {
    if (!draft || !explore.data || !model) return;
    const primary = sources[0];
    save.mutate({
      name: name.trim(),
      description: description.trim(),
      model_id: explore.data.model_id || model.id,
      profile: draft,
      source:
        sources.length > 1
          ? "multi_source"
          : primary?.kind === "agent_trial"
            ? "agentic"
            : primary?.kind === "benchmark_run"
              ? "proposal"
              : "manual",
      source_run_id: primary?.kind === "benchmark_run" ? primary.id : null,
      source_trial_id: primary?.kind === "agent_trial" ? primary.id : null,
      parent_profile_id: parentProfile.data?.id ?? null,
      metric: selectionMetric,
      source_refs: sources.map((source) => ({ ...source, weight: source.weight ?? 1 })),
      source_fingerprint: explore.data.fingerprint,
      selection_strategy: selectionStrategy,
      selection_config: { ...selectionConfig, metric: selectionMetric },
      observed_mass_retained: observedMassRetained(
        draft,
        explore.data.layer_ids,
        explore.data.num_experts,
        explore.data.routing_mass,
      ),
    });
  }

  return (
    <div className="command-page profile-studio-page">
      <PageHeader
        eyebrow="Profile Studio"
        title="Paint the model you want to keep."
        description="Combine workload traces, rank experts by real routing mass, then make deliberate layer-level edits. Aggregate is the full layer × expert topology—not a misleading collapse of expert IDs."
        actions={<button className="primary-button" disabled={!draft || !name.trim() || save.isPending} onClick={saveProfile}>{save.isPending ? "Saving revision…" : "Save immutable profile"}</button>}
      />
      {save.error && <CommandError error={save.error} />}
      <div className="profile-studio-meta">
        <label>Profile name<input value={name} maxLength={120} onChange={(event) => setName(event.target.value)} /></label>
        <label>Research note<input value={description} maxLength={1000} onChange={(event) => setDescription(event.target.value)} /></label>
      </div>
      <details className="profile-source-picker" open={sources.length === 0}>
        <summary><span>Workload sources</span><strong>{sources.length} selected</strong></summary>
        <div className="profile-source-columns">
          <section>
            <span className="section-label">Benchmark runs</span>
            {(runs.data?.items ?? []).map((run) => {
              const selected = sources.find((source) =>
                source.kind === "benchmark_run" && source.id === run.id
              );
              return (
                <label key={run.id}>
                  <input type="checkbox" checked={Boolean(selected)} onChange={() => toggleSource({ kind: "benchmark_run", id: run.id })} />
                  <span><strong>{run.benchmark_id}</strong><small>{formatPercent(run.score)} · {shortId(run.id)}</small></span>
                  {selected && <SourceWeightInput source={selected} label={run.benchmark_id} onChange={updateSourceWeight} />}
                </label>
              );
            })}
          </section>
          <section>
            <span className="section-label">Agentic trials</span>
            {(agentRuns.data ?? []).flatMap((run) => run.trials.map((trial) => {
              const selected = sources.find((source) =>
                source.kind === "agent_trial" && source.id === trial.id
              );
              return (
                <label key={trial.id}>
                  <input type="checkbox" checked={Boolean(selected)} onChange={() => toggleSource({ kind: "agent_trial", id: trial.id })} />
                  <span><strong>{trial.title}</strong><small>{trial.status} · {shortId(trial.id)}</small></span>
                  {selected && <SourceWeightInput source={selected} label={trial.title} onChange={updateSourceWeight} />}
                </label>
              );
            }))}
          </section>
        </div>
      </details>
      {sources.length === 0 ? (
        <CommandEmpty title="Choose at least one workload trace" detail="Benchmark runs and agentic trials can both become custom profile sources." />
      ) : explore.data && draft ? (
        <ExpertExplorer
          data={explore.data}
          profile={draft}
          referenceProfile={referenceProfile}
          onProfileChange={setDraft}
          initialMetric={selectionMetric}
          onMetricChange={(metric) => {
            setSelectionMetric(metric);
            setSelectionConfig((current) => ({ ...current, metric }));
          }}
          onStrategyChange={(strategy, config) => {
            setSelectionStrategy(strategy);
            setSelectionConfig(config);
          }}
          defaultMode="selection"
          title="Custom expert selection"
        />
      ) : (
        <InspectorLoading error={explore.error} />
      )}
    </div>
  );
}

function SourceWeightInput({
  source,
  label,
  onChange,
}: {
  source: RoutingSourceReference;
  label: string;
  onChange: (source: RoutingSourceReference, weight: number) => void;
}) {
  return (
    <input
      className="source-weight-input"
      type="number"
      min={0.1}
      max={100}
      step={0.1}
      value={source.weight ?? 1}
      aria-label={`Weight for ${label}`}
      title="Relative workload weight"
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => onChange(source, Number(event.target.value))}
    />
  );
}

function ComparisonArchivePage() {
  const comparisons = useQuery({ queryKey: ["comparisons"], queryFn: () => api<ComparisonRecord[]>("/api/comparisons") });
  return (
    <div className="command-page">
      <PageHeader eyebrow="Paired comparisons" title="Measure what pruning actually changed." description="Every comparison locks the model, benchmark cohort, generation contract, and profile fingerprint before measuring regressions and recoveries." />
      <div className="command-record-list">
        {(comparisons.data ?? []).map((comparison) => (
          <AppLink className="command-record-row comparison-record" href={routeHref("comparison", comparison.id)} key={comparison.id}>
            <span className="artifact-kind comparison">Paired</span>
            <div><strong>{comparison.name}</strong><small>{comparison.benchmark_id} · {comparison.cohort_item_ids.length} items</small></div>
            <Metric label="Baseline" value={formatPercent(comparison.baseline_score)} />
            <Metric label="Profile" value={formatPercent(comparison.candidate_score)} />
            <Metric label="Delta" value={formatDelta(comparison.score_delta)} tone={(comparison.score_delta ?? 0) < 0 ? "negative" : "positive"} />
            <span className="record-arrow">→</span>
          </AppLink>
        ))}
        {!comparisons.isPending && comparisons.data?.length === 0 && <CommandEmpty title="No paired comparisons yet" detail="Run the same locked cohort against a baseline and a loaded expert profile." />}
      </div>
    </div>
  );
}

function ComparisonDetailPage({ comparisonId, model }: { comparisonId: string; model: ModelRegistryEntry | null }) {
  const [itemOffset, setItemOffset] = useState(0);
  const comparison = useQuery({ queryKey: ["comparison", comparisonId], queryFn: () => api<ComparisonRecord>(`/api/comparisons/${encodeURIComponent(comparisonId)}`) });
  const baselineItemsQuery = useQuery({
    queryKey: ["run-items", comparison.data?.baseline_run_id, itemOffset],
    queryFn: () => api<RunItemResultPage>(`/api/runs/${comparison.data?.baseline_run_id}/items?offset=${itemOffset}&limit=100`),
    enabled: Boolean(comparison.data),
  });
  const candidateItemsQuery = useQuery({
    queryKey: ["run-items", comparison.data?.candidate_run_id, itemOffset],
    queryFn: () => api<RunItemResultPage>(`/api/runs/${comparison.data?.candidate_run_id}/items?offset=${itemOffset}&limit=100`),
    enabled: Boolean(comparison.data),
  });
  const explore = useRoutingExplore(
    comparison.data ? [{ kind: "benchmark_run", id: comparison.data.candidate_run_id }] : [],
    comparison.data ? [{ kind: "benchmark_run", id: comparison.data.baseline_run_id }] : [],
    model,
  );
  if (comparison.isPending) return <CommandLoading label="Opening paired comparison…" />;
  if (!comparison.data) return <NotFoundPage noun="comparison" />;
  const baselineItems = new Map((baselineItemsQuery.data?.items ?? []).map((item) => [runItemKey(item), item]));
  return (
    <div className="command-page comparison-detail-page">
      <PageHeader eyebrow="Paired experiment" title={comparison.data.name} description="Candidate minus baseline. Positive routing deltas indicate experts receiving more probability mass under the profiled run." meta={<><span>{comparison.data.cohort_item_ids.length} locked items</span><span>profile {comparison.data.profile_fingerprint.slice(0, 10)}</span></>} />
      <div className="comparison-hero-metrics">
        <Metric label="Baseline" value={formatPercent(comparison.data.baseline_score)} />
        <Metric label="Profile" value={formatPercent(comparison.data.candidate_score)} />
        <Metric label="Score delta" value={formatDelta(comparison.data.score_delta)} tone={(comparison.data.score_delta ?? 0) < 0 ? "negative" : "positive"} />
        <Metric label="Regressions" value={String(comparison.data.regressions)} tone={comparison.data.regressions ? "negative" : undefined} />
        <Metric label="Recoveries" value={String(comparison.data.recoveries)} tone={comparison.data.recoveries ? "positive" : undefined} />
      </div>
      {explore.data ? <ExpertExplorer data={explore.data} defaultMode="delta" title="Routing delta" /> : <InspectorLoading error={explore.error} />}
      <div className="comparison-command-table">
        <header><span>Problem</span><span>Baseline</span><span>Profile</span><span>Transition</span></header>
        {(candidateItemsQuery.data?.items ?? []).map((item) => {
          const base = baselineItems.get(runItemKey(item));
          const transition = resultTransition(base?.passed, item.passed);
          return <div key={runItemKey(item)}><span><strong>{item.item_id}</strong><small>Attempt {item.attempt} · {item.prompt}</small></span><span className={resultTone(base?.passed)}>{resultLabel(base?.passed)}</span><span className={resultTone(item.passed)}>{resultLabel(item.passed)}</span><span><em className={`transition-badge ${transition}`}>{transition}</em></span></div>;
        })}
      </div>
      {(candidateItemsQuery.data?.total ?? 0) > 100 && (
        <div className="artifact-actions command-pagination">
          <button className="secondary-button" disabled={itemOffset === 0} onClick={() => setItemOffset(Math.max(0, itemOffset - 100))}>Previous</button>
          <span>{itemOffset + 1}–{Math.min(itemOffset + 100, candidateItemsQuery.data?.total ?? 0)} of {candidateItemsQuery.data?.total ?? 0}</span>
          <button className="secondary-button" disabled={itemOffset + 100 >= (candidateItemsQuery.data?.total ?? 0)} onClick={() => setItemOffset(itemOffset + 100)}>Next</button>
        </div>
      )}
    </div>
  );
}

function useRoutingExplore(
  sources: RoutingSourceReference[],
  comparisonSources: RoutingSourceReference[],
  model: ModelRegistryEntry | null,
) {
  return useQuery({
    queryKey: ["routing-explore", sources, comparisonSources],
    queryFn: () => fetchRoutingExplore({
      sources,
      comparison_sources: comparisonSources.length ? comparisonSources : undefined,
      metric: "routing_mass",
    }, model),
    enabled: sources.length > 0 && Boolean(model),
    retry: false,
  });
}

async function fetchRoutingExplore(
  request: RoutingExploreRequest,
  model: ModelRegistryEntry | null,
): Promise<RoutingExploreResponse> {
  try {
    return await api<RoutingExploreResponse>("/api/routing/explore", {
      method: "POST",
      body: JSON.stringify(request),
    });
  } catch (error) {
    if (!(error instanceof ApiError) || ![404, 405].includes(error.status)) throw error;
    if (
      !model?.topology ||
      request.sources.length !== 1 ||
      request.comparison_sources?.length
    ) throw error;
    const source = request.sources[0];
    const summary = source.kind === "benchmark_run"
      ? await api<TrialRoutingSummary>(`/api/runs/${encodeURIComponent(source.id)}/routing`)
      : await api<TrialRoutingSummary>(`/api/trials/${encodeURIComponent(source.id)}/routing`);
    return {
      fingerprint: `${source.kind}:${source.id}`,
      aggregation: "weighted_source_normalized",
      model_id: model.id,
      layer_ids: summary.layer_ids,
      num_experts: summary.selection_counts[0]?.length ?? model.topology.num_experts,
      top_k: model.topology.top_k,
      selection_counts: normalizeEngagementMatrix(summary.selection_counts),
      routing_mass: normalizeEngagementMatrix(summary.routing_mass),
      total_routed_slots: summary.total_routed_slots,
      captured_inference_calls: summary.captured_inference_calls,
      total_inference_calls: summary.inference_calls,
      served_tokens: summary.served_tokens,
      sources: [{ ...source, label: source.id.slice(0, 8) }],
      filter_capabilities: {
        item: source.kind === "benchmark_run",
        trial: source.kind === "agent_trial",
        step_type: false,
        outcome: false,
      },
    };
  }
}

function ProblemDetail({
  item,
  contract,
  detail,
}: {
  item: BenchmarkItem;
  contract: EvaluationContract;
  detail: BenchmarkProblemDetail | null;
}) {
  const criteria = detail?.success_criteria.public_criteria ??
    item.success_criteria ?? contract.criteria;
  const hiddenCount = detail?.success_criteria.hidden_criteria_count ??
    criteria.filter((criterion) => criterion.visibility === "hidden").length;
  const hiddenHash = detail?.success_criteria.hidden_criteria_hash ??
    item.verifier_hash;
  return (
    <article className="problem-detail">
      <header><span className="section-label">{item.category}</span><strong>{item.id}</strong></header>
      <section><span className="problem-section-label">Problem</span><pre>{detail?.rendered_prompt ?? item.prompt}</pre></section>
      <section className="success-criteria-preview"><span className="problem-section-label">Public success criteria</span>{criteria.filter((criterion) => criterion.visibility === "public").map((criterion, index) => <div key={criterion.id ?? index}><span>{index + 1}</span><p><strong>{criterion.label}</strong>{criterion.description && <small>{criterion.description}</small>}</p><em>{criterion.weight}×</em></div>)}{hiddenCount > 0 && <div className="hidden-criterion"><span>◇</span><p><strong>{hiddenCount} protected verifier {hiddenCount === 1 ? "check" : "checks"}</strong><small>Hidden assets remain withheld from the model. Fingerprint: {hiddenHash?.slice(0, 12) ?? "recorded at run time"}</small></p></div>}</section>
      {item.expected && <details><summary>View reference answer</summary><pre>{item.expected}</pre></details>}
      {Object.keys(item.metadata).length > 0 && <details><summary>Dataset metadata</summary><pre>{JSON.stringify(item.metadata, null, 2)}</pre></details>}
    </article>
  );
}

function RunItemDetail({ item }: { item: RunItemResult }) {
  return (
    <article className="run-item-detail">
      <header><div><span className="section-label">Selected result · attempt {item.attempt}</span><h2>{item.item_id}</h2></div><span className={`result-stamp ${resultTone(item.passed)}`}>{resultLabel(item.passed)}</span></header>
      <section><span>Problem</span><pre>{item.prompt}</pre></section>
      <div className="answer-comparison"><section><span>Model response</span><pre>{item.output || "No response captured."}</pre></section><section><span>Reference / success target</span><pre>{item.expected || "No public reference answer."}</pre></section></div>
      {item.error && <div className="run-item-error">{item.error}</div>}
      <footer><span>{formatDuration(item.latency_ms)} latency</span><span>{item.prompt_tokens.toLocaleString()} prompt tokens</span><span>{item.completion_tokens.toLocaleString()} completion tokens</span><span>{humanize(item.scoring)} scorer</span></footer>
    </article>
  );
}

function ProfileShape({ profile, model }: { profile: ExpertProfile; model: ModelRegistryEntry | null }) {
  const layerIds = model?.topology?.routed_layer_ids ??
    Object.keys(profile.layers).map(Number).sort((a, b) => a - b);
  const numExperts = model?.topology?.num_experts ??
    Math.max(...Object.values(profile.layers).flatMap((layer) => layer.keep), 0) + 1;
  return (
    <section className="profile-shape">
      <header><span className="section-label">Layer eligibility</span><strong>{layerIds.length} routed layers</strong></header>
      <div>{layerIds.map((layerId) => {
        const keep = profile.layers[String(layerId)]?.keep ??
          Array.from({ length: numExperts }, (_, index) => index);
        const keptPath = keep
          .filter((expertId) => expertId >= 0 && expertId < numExperts)
          .map((expertId) => `M${expertId} 0h1v1h-1z`)
          .join("");
        return (
          <article key={layerId}>
            <span>Layer {layerId}</span>
            <svg
              aria-label={`Layer ${layerId}: ${keep.length} of ${numExperts} experts eligible`}
              preserveAspectRatio="none"
              role="img"
              viewBox={`0 0 ${numExperts} 1`}
            >
              <rect className="masked" height="1" width={numExperts} />
              <path className="kept" d={keptPath} />
            </svg>
            <strong>{keep.length}/{numExperts}</strong>
          </article>
        );
      })}</div>
    </section>
  );
}

function MobileCommandTabs<T extends string>({ value, options, onChange }: { value: T; options: T[]; onChange: (value: T) => void }) {
  return <div className="mobile-command-tabs global" role="tablist">{options.map((option) => <button key={option} className={value === option ? "selected" : ""} onClick={() => onChange(option)}>{option}</button>)}</div>;
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "positive" | "negative" }) {
  return <div className={`command-metric ${tone ?? ""}`}><strong>{value}</strong><span>{label}</span></div>;
}

function InspectorLoading({ error }: { error: unknown }) {
  return error ? <CommandError error={error} /> : <CommandLoading label="Aggregating routing telemetry…" />;
}

function CommandLoading({ label }: { label: string }) {
  return <div className="command-loading"><span className="live-dot" />{label}</div>;
}

function CommandError({ error }: { error: unknown }) {
  return <div className="command-error" role="alert">{error instanceof Error ? error.message : "Something went wrong."}</div>;
}

function CommandEmpty({ title, detail }: { title: string; detail: string }) {
  return <div className="command-empty"><span>◇</span><strong>{title}</strong><p>{detail}</p></div>;
}

function NotFoundPage({ noun = "page" }: { noun?: string }) {
  return <div className="command-page"><CommandEmpty title={`${noun[0].toUpperCase()}${noun.slice(1)} not found`} detail="The artifact may have been removed or the address is incomplete." /><AppLink className="primary-button not-found-home" href="/">Return home</AppLink></div>;
}

function defaultContractForBenchmark(benchmark: BenchmarkInfo) {
  return defaultEvaluationContract(`${benchmark.name} scoring`, {
    kind: scoringKind(benchmark.scoring),
    label: scoringDescription(benchmark.scoring),
    weight: 1,
    visibility: "public",
  });
}

function scoringKind(scoring: BenchmarkInfo["scoring"] | undefined) {
  if (scoring === "gsm8k") return "numeric" as const;
  if (scoring === "ungraded") return "ungraded" as const;
  if (scoring === "exact" || scoring === "contains" || scoring === "regex") {
    return scoring;
  }
  if (scoring === "multiple_choice") return "benchmark_default" as const;
  if (scoring === "json_schema") return "json" as const;
  return "benchmark_default" as const;
}

function scoringDescription(scoring: BenchmarkInfo["scoring"] | undefined) {
  const labels: Record<string, string> = {
    exact: "Normalized output exactly matches the reference",
    gsm8k: "Extracted final numeric answer matches the reference",
    numeric: "Numeric answer falls within the configured tolerance",
    regex: "Response satisfies the required regular expression",
    contains: "Response contains the required answer fragment",
    ifeval: "All objective instruction-following constraints pass",
    livebench: "Pinned LiveBench task-specific deterministic evaluator passes",
    multiple_choice: "Pinned final-answer letter extractor matches the reference option",
    json_schema: "Response validates against the required JSON schema",
    executable: "Executable verifier passes",
    instruction_constraints: "All objective instruction constraints pass",
    llm_judge: "Frontier-model judge meets the rubric threshold",
    ungraded: "No deterministic scorer; judge or human review required",
  };
  return labels[scoring ?? "exact"] ?? humanize(scoring ?? "exact");
}

function toggleSet(current: Set<string>, value: string) {
  const next = new Set(current);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

function mobileClass(current: string, expected: string, base: string) {
  return `${base} ${current === expected ? "mobile-panel-active" : "mobile-panel-hidden"}`;
}

function resultTone(result: boolean | null | undefined) {
  return result == null ? "unscored" : result ? "pass" : "fail";
}

function resultLabel(result: boolean | null | undefined) {
  return result == null ? "Unscored" : result ? "Passed" : "Failed";
}

function resultTransition(baseline: boolean | null | undefined, candidate: boolean | null | undefined) {
  if (baseline == null || candidate == null) return "unscored";
  if (baseline && !candidate) return "regression";
  if (!baseline && candidate) return "recovery";
  return baseline ? "retained" : "retained failure";
}

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function shortId(value: string) {
  return value.slice(0, 8);
}

function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${Math.round(value * 1000) / 10}%`;
}

function formatDelta(value: number | null) {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${Math.round(value * 1000) / 10} pp`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value));
}

function formatDuration(milliseconds: number) {
  return milliseconds < 1_000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1_000).toFixed(1)} s`;
}

function trialResultTone(status: string) {
  if (status === "passed") return "pass";
  if (status === "failed" || status === "error") return "fail";
  return "unscored";
}

function formatMaybeNumber(value: number | null) {
  return value == null ? "—" : value.toFixed(2);
}

function formatComparisonValue(
  value: number | null | undefined,
  format: "duration" | "number" | "rate" | "cost",
) {
  if (value == null) return "—";
  if (format === "duration") return formatDuration(value);
  if (format === "rate") return `${value.toFixed(1)} t/s`;
  if (format === "cost") return `$${value.toFixed(value < 0.01 ? 4 : 2)}`;
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

function formatSignedComparisonValue(
  baseline: number | null | undefined,
  candidate: number | null | undefined,
  format: "duration" | "number" | "rate" | "cost",
) {
  if (baseline == null || candidate == null) return "—";
  const delta = candidate - baseline;
  const prefix = delta >= 0 ? "+" : "−";
  return `${prefix}${formatComparisonValue(Math.abs(delta), format)}`;
}
