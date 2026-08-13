import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { ApiError } from "../api";
import { AppLink } from "../AppShell";
import { navigate } from "../router";
import type { EvaluationContract, ModelSession } from "../types";
import {
  buildExperimentCreateRequest,
  cancelExperiment,
  createExperiment,
  getExperiment,
  listAgents,
  listExperimentEvents,
  listExperiments,
  listModels,
  listProfiles,
  listSandboxProviders,
  listWorkloads,
} from "./data";
import {
  codingTimeline,
  driftRoutingSummaries,
  liveProgress,
  mergeRunEvents,
  stateDifferences,
} from "./progress";
import {
  defaultWorkloadUnitSelection,
  MAX_RENDERED_WORKLOAD_UNITS,
  MAX_SELECTED_WORKLOAD_UNITS,
  orderedWorkloadUnitSelection,
  plannedExperimentUnits,
  selectAllWorkloadUnits,
  uniqueWorkloadUnitIds,
} from "./cohortSelection";
import { v2Href } from "./router";
import type {
  DriftCheckpoint,
  ExperimentCreateRequest,
  ExperimentLane,
  ExperimentRecord,
  ExperimentUnit,
  LaneRole,
  LaneUnitResult,
  RunEvent,
  WorkloadDescriptor,
  WorkloadKind,
} from "./types";
import {
  V2Empty,
  V2Error,
  V2Loading,
  V2Metric,
  V2PageHeader,
  V2Status,
} from "./V2Shell";
import { InlineProfileName } from "./Profiles";

const activeStatuses = new Set(["queued", "preparing", "running", "cancelling"]);

export function ExperimentsPage() {
  const [kind, setKind] = useState<"all" | WorkloadKind>("all");
  const [search, setSearch] = useState("");
  const experiments = useQuery({
    queryKey: ["v2-experiments"],
    queryFn: listExperiments,
    refetchInterval: (query) =>
      query.state.data?.items.some((item) => activeStatuses.has(item.status))
        ? 1_500
        : false,
    retry: false,
  });
  const items = experiments.data?.items ?? [];
  const filtered = items.filter((experiment) => {
    const matchesKind = kind === "all" || experiment.workload_kind === kind;
    const haystack = `${experiment.name} ${experiment.workload_name} ${experiment.model_name}`.toLowerCase();
    return matchesKind && haystack.includes(search.toLowerCase());
  });
  const unavailable = experiments.error instanceof ApiError && experiments.error.status === 404;

  return (
    <div className="v2-page v2-experiment-index">
      <V2PageHeader
        eyebrow="Experiments"
        title="One workbench. Every intervention."
        description="Answer, coding, and state-drift workloads use the same durable lane, progress, evaluation, performance, and routing contract."
        meta={
          <>
            <span>{items.length} durable experiments</span>
            <span>{items.filter((item) => activeStatuses.has(item.status)).length} active</span>
            <span>{items.filter((item) => item.lanes.length > 1).length} paired</span>
          </>
        }
        actions={<AppLink className="v2-primary-button" href="/experiments/new">Create experiment</AppLink>}
      />

      <div className="v2-toolbar">
        <input
          type="search"
          value={search}
          placeholder="Search experiments"
          aria-label="Search experiments"
          onChange={(event) => setSearch(event.target.value)}
        />
        <div className="v2-segments" aria-label="Workload type">
          {(["all", "answer", "coding", "state_drift"] as const).map((value) => (
            <button
              key={value}
              className={kind === value ? "selected" : ""}
              aria-pressed={kind === value}
              onClick={() => setKind(value)}
            >
              {value === "state_drift" ? "State drift" : humanize(value)}
            </button>
          ))}
        </div>
      </div>

      {experiments.isPending ? (
        <V2Loading label="Loading experiment archive…" />
      ) : unavailable ? (
        <V2Empty
          title="The V2 experiment service is not active yet."
          detail="The unified frontend is ready. Start a backend build exposing /api/experiments, or use a preserved V1 workflow in the meantime."
          action={<AppLink className="v2-secondary-button" href="/benchmarks">Open V1 benchmark library</AppLink>}
        />
      ) : experiments.error ? (
        <V2Error error={experiments.error} />
      ) : filtered.length === 0 ? (
        <V2Empty
          title={items.length ? "No experiments match." : "No experiments yet."}
          detail={items.length ? "Try another filter or search phrase." : "Choose a workload and create the first baseline or paired experiment."}
          action={<AppLink className="v2-primary-button" href="/experiments/new">Create experiment</AppLink>}
        />
      ) : (
        <div className="v2-experiment-grid">
          {filtered.map((experiment) => (
            <ExperimentCard experiment={experiment} key={experiment.id} />
          ))}
        </div>
      )}
    </div>
  );
}

function ExperimentCard({ experiment }: { experiment: ExperimentRecord }) {
  const progress = liveProgress(experiment, []);
  return (
    <AppLink className="v2-experiment-card" href={v2Href("experiment", experiment.id)}>
      <header>
        <span className={`v2-kind-mark ${experiment.workload_kind}`}>{kindLabel(experiment.workload_kind)}</span>
        <V2Status value={experiment.status} />
      </header>
      <div className="v2-experiment-card-copy">
        <h2>{experiment.name}</h2>
        <p>{experiment.workload_name}</p>
        <small>{experiment.model_name}</small>
      </div>
      <div className="v2-lane-preview">
        {experiment.lanes.map((lane) => (
          <div key={lane.id}>
            <span><strong>{lane.label}</strong><em>{lane.intervention.name}</em></span>
            <progress max={Math.max(lane.progress_total, 1)} value={lane.progress_current} />
            <small>{lane.progress_current}/{lane.progress_total || "—"}</small>
          </div>
        ))}
      </div>
      <footer>
        <span>{formatOutcomeCounts(progress.passed, progress.failed, progress.unscored, progress.completed)}</span>
        <time>{formatDate(experiment.created_at)}</time>
        <strong>Open →</strong>
      </footer>
    </AppLink>
  );
}

export function CreateExperimentPage({
  currentSession,
  workloadId,
}: {
  currentSession: ModelSession | null;
  workloadId: string | null;
}) {
  const [kind, setKind] = useState<WorkloadKind>("answer");
  const [selectedWorkloadId, setSelectedWorkloadId] = useState(workloadId ?? "");
  const [selectedWorkloadUnitIds, setSelectedWorkloadUnitIds] = useState(
    new Set<string>(),
  );
  const selectionWorkloadKey = useRef<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("");
  const modelTouched = useRef(false);
  const [candidateProfileId, setCandidateProfileId] = useState("");
  const [paired, setPaired] = useState(true);
  const [name, setName] = useState("");
  const nameTouched = useRef(false);
  const [horizon, setHorizon] = useState(8);
  const [conditions, setConditions] = useState(new Set(["oracle_reset", "chained", "state_anchored"]));
  const [seed, setSeed] = useState(0);
  const [evaluationMode, setEvaluationMode] = useState<"deterministic" | "record_only">("deterministic");
  const [generationMaxTokens, setGenerationMaxTokens] = useState(512);
  const [totalTokenBudget, setTotalTokenBudget] = useState(32768);
  const [perUnitTimeoutSeconds, setPerUnitTimeoutSeconds] = useState(300);
  const [maxTurns, setMaxTurns] = useState(8);
  const [maxCommands, setMaxCommands] = useState(8);
  const [dependencySpan, setDependencySpan] = useState(2);
  const [branchCount, setBranchCount] = useState(2);
  const [rollbackDepth, setRollbackDepth] = useState(2);
  const [distractorRatio, setDistractorRatio] = useState(0.25);
  const [toolErrorRate, setToolErrorRate] = useState(0.1);
  const [stateSize, setStateSize] = useState(12);
  const [agentId, setAgentId] = useState("");
  const [sandboxProviderId, setSandboxProviderId] = useState("");
  const workloads = useQuery({ queryKey: ["v2-workloads"], queryFn: listWorkloads });
  const models = useQuery({ queryKey: ["v2-models"], queryFn: listModels });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });
  const agents = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    enabled: kind === "coding",
    staleTime: 30_000,
  });
  const providers = useQuery({
    queryKey: ["sandbox-providers"],
    queryFn: listSandboxProviders,
    enabled: kind === "coding",
    staleTime: 30_000,
  });
  const candidates = (workloads.data ?? []).filter((item) => item.kind === kind);
  const selectedWorkload = candidates.find((item) => item.id === selectedWorkloadId) ?? null;
  const orderedSelectedUnitIds = selectedWorkload
    ? orderedWorkloadUnitSelection(
        selectedWorkload.unit_ids,
        selectedWorkloadUnitIds,
      )
    : [];
  const plannedUnits = plannedExperimentUnits({
    conditionCount: conditions.size,
    kind,
    paired,
    selectedUnitCount: orderedSelectedUnitIds.length,
  });

  useEffect(() => {
    if (!models.data?.length) return;
    const residentModel = models.data.find(
      (model) => model.id === currentSession?.model_id,
    );
    if (!modelTouched.current && residentModel && selectedModelId !== residentModel.id) {
      setSelectedModelId(residentModel.id);
      return;
    }
    if (selectedModelId) return;
    setSelectedModelId(
      residentModel?.id ?? models.data.find((model) => model.enabled)?.id ?? models.data[0].id,
    );
  }, [currentSession?.model_id, models.data, selectedModelId]);

  useEffect(() => {
    if (kind !== "coding" || !agents.data?.length) return;
    if (agents.data.some((agent) => agent.id === agentId && agent.available)) return;
    setAgentId(
      agents.data.find((agent) => agent.is_default && agent.available)?.id ??
      agents.data.find((agent) => agent.available)?.id ??
      "",
    );
  }, [agentId, agents.data, kind]);

  useEffect(() => {
    if (kind !== "coding" || !providers.data?.length) return;
    if (providers.data.some((provider) =>
      provider.id === sandboxProviderId && provider.configured && provider.status !== "error"
    )) return;
    setSandboxProviderId(
      providers.data.find((provider) => provider.status === "ready")?.id ??
      providers.data.find((provider) => provider.configured && provider.status !== "error")?.id ??
      "",
    );
  }, [kind, providers.data, sandboxProviderId]);

  useEffect(() => {
    if (workloadId && workloads.data) {
      const requested = workloads.data.find((item) => item.id === workloadId);
      if (requested) {
        setKind(requested.kind);
        setSelectedWorkloadId(requested.id);
        return;
      }
    }
    if (candidates.some((item) => item.id === selectedWorkloadId)) return;
    setSelectedWorkloadId(candidates.find((item) => item.readiness === "ready")?.id ?? candidates[0]?.id ?? "");
  }, [candidates, selectedWorkloadId, workloadId, workloads.data]);

  useEffect(() => {
    const unitIds = selectedWorkload?.unit_ids ?? [];
    const selectionKey = selectedWorkload
      ? [
          selectedWorkload.id,
          unitIds.length,
          unitIds[0] ?? "",
          unitIds.at(-1) ?? "",
        ].join(":")
      : null;
    if (selectionWorkloadKey.current === selectionKey) return;
    selectionWorkloadKey.current = selectionKey;
    setSelectedWorkloadUnitIds(
      new Set(defaultWorkloadUnitSelection(unitIds)),
    );
  }, [selectedWorkload]);

  useEffect(() => {
    if (nameTouched.current || !selectedWorkload) return;
    setName(`${selectedWorkload.title} · ${paired ? "baseline vs profile" : "baseline"}`);
  }, [paired, selectedWorkload]);

  const create = useMutation({
    mutationFn: (request: ExperimentCreateRequest) => createExperiment(request),
    onSuccess: (experiment) => navigate(v2Href("experiment", experiment.id)),
  });

  function submit() {
    if (!selectedWorkload || !selectedModelId) return;
    const criterionKind = evaluationMode === "record_only" ? "ungraded" : "benchmark_default";
    const evaluationContract = {
      version: 1 as const,
      name: evaluationMode === "record_only" ? "Record-only evaluation" : kind === "state_drift" ? "Exact canonical state" : kind === "coding" ? "Trusted sandbox verifier" : "Pinned dataset scorer",
      description: evaluationMode === "record_only" ? "Persist outputs, performance, and routing without assigning pass/fail." : "Use the workload adapter's pinned deterministic evaluator.",
      criteria: [{
        id: "workload-correctness",
        kind: criterionKind,
        label: evaluationMode === "record_only" ? "Recorded outcome" : "Workload correctness",
        visibility: "public" as const,
        required: evaluationMode !== "record_only",
        weight: 1,
        case_sensitive: true,
        strip_whitespace: true,
      }],
      aggregation: "all_required" as const,
      pass_threshold: 1,
      judge: null,
      judge_weight: 0,
      judge_can_override_deterministic_failure: false,
    } satisfies EvaluationContract;
    create.mutate(buildExperimentCreateRequest({
      name: name.trim(),
      modelId: selectedModelId,
      workloadId: selectedWorkload.id,
      workloadUnitIds: orderedSelectedUnitIds,
      kind,
      candidateProfileId: paired ? candidateProfileId || null : null,
      agentId,
      sandboxProviderId,
      conditions: [...conditions],
      driftParameters: {
        dependency_span: dependencySpan,
        branch_count: branchCount,
        rollback_depth: rollbackDepth,
        distractor_ratio: distractorRatio,
        tool_error_rate: toolErrorRate,
        state_size: stateSize,
      },
      horizon,
      seed,
      generation: {
        temperature: 0,
        max_tokens: generationMaxTokens,
        seed,
        enable_thinking: false,
      },
      evaluationContract,
      executionPolicy: {
        attempts: 1,
        concurrency: 1,
        timeout_seconds: Math.min(
          86_400,
          Math.max(perUnitTimeoutSeconds, perUnitTimeoutSeconds * plannedUnits),
        ),
        per_item_timeout_seconds: perUnitTimeoutSeconds,
        max_turns: kind === "coding" || kind === "state_drift" ? Math.max(maxTurns, kind === "state_drift" ? horizon : 1) : null,
        max_commands: kind === "coding" ? maxCommands : null,
        max_tokens: totalTokenBudget,
        max_cost_usd: null,
        fail_fast: false,
      },
    }));
  }

  const residentModelReady = currentSession?.state === "ready" &&
    currentSession.model_id === selectedModelId;
  const codingExecutorReady = kind !== "coding" || Boolean(
    agents.data?.some((agent) => agent.id === agentId && agent.available) &&
    providers.data?.some((provider) =>
      provider.id === sandboxProviderId && provider.configured && provider.status !== "error"
    ),
  );
  const ready = Boolean(
    name.trim() &&
    selectedModelId &&
    residentModelReady &&
    selectedWorkload?.readiness === "ready" &&
    orderedSelectedUnitIds.length > 0 &&
    orderedSelectedUnitIds.length <= MAX_SELECTED_WORKLOAD_UNITS &&
    (!paired || candidateProfileId) &&
    (kind !== "state_drift" || conditions.size > 0) &&
    codingExecutorReady &&
    generationMaxTokens >= 1 &&
    totalTokenBudget >= generationMaxTokens &&
    perUnitTimeoutSeconds >= 1 &&
    maxTurns >= 1 &&
    maxCommands >= 1 &&
    (kind !== "state_drift" || (
      dependencySpan < stateSize &&
      branchCount <= stateSize &&
      rollbackDepth + 2 <= horizon &&
      distractorRatio >= 0 && distractorRatio <= .9 &&
      toolErrorRate >= 0 && toolErrorRate <= .9
    )) &&
    Number.isInteger(seed),
  );

  return (
    <div className="v2-page v2-create-page">
      <V2PageHeader
        eyebrow="Create experiment"
        title="Define the comparison once."
        description="The workload, cohort, model, contexts, evaluation contract, and execution budget become one durable experiment."
        actions={
          <button className="v2-primary-button" disabled={!ready || create.isPending} onClick={submit}>
            {create.isPending ? "Creating…" : "Create & run"}
          </button>
        }
      />
      {create.error && <V2Error error={create.error} />}
      <div className="v2-create-grid">
        <section className="v2-create-main">
          <fieldset className="v2-form-section">
            <legend><span>01</span> Model</legend>
            <label>
              Warm engine
              <select value={selectedModelId} onChange={(event) => { modelTouched.current = true; setSelectedModelId(event.target.value); }}>
                {(models.data ?? []).map((model) => (
                  <option key={model.id} value={model.id} disabled={!model.enabled}>{model.display_name}</option>
                ))}
              </select>
            </label>
            <p>Loading different weights is expensive. Baseline and candidate profiles below are fast intervention contexts on this engine.</p>
            {!residentModelReady && selectedModelId && (
              <p className="v2-form-warning" role="status">
                This experiment requires the selected model to be resident and ready. <AppLink href="/models">Load it in Models →</AppLink>
              </p>
            )}
          </fieldset>

          <fieldset className="v2-form-section">
            <legend><span>02</span> Workload</legend>
            <div className="v2-kind-picker">
              {(["answer", "coding", "state_drift"] as const).map((value) => (
                <button
                  type="button"
                  key={value}
                  className={kind === value ? "selected" : ""}
                  onClick={() => {
                    selectionWorkloadKey.current = null;
                    setKind(value);
                    setSelectedWorkloadId("");
                    setSelectedWorkloadUnitIds(new Set());
                  }}
                >
                  <strong>{kindLabel(value)}</strong>
                  <small>{kindDescription(value)}</small>
                </button>
              ))}
            </div>
            <label>
              Task or cohort
              <select
                value={selectedWorkloadId}
                onChange={(event) => {
                  selectionWorkloadKey.current = null;
                  setSelectedWorkloadId(event.target.value);
                  setSelectedWorkloadUnitIds(new Set());
                }}
              >
                {candidates.map((workload) => (
                  <option key={workload.id} value={workload.id} disabled={workload.readiness !== "ready"}>
                    {workload.title}{workload.readiness === "ready" ? "" : ` · ${humanize(workload.readiness)}`}
                  </option>
                ))}
              </select>
            </label>
            {selectedWorkload && <WorkloadSelectionSummary workload={selectedWorkload} />}
            {selectedWorkload && (
              <WorkloadUnitSelection
                onChange={setSelectedWorkloadUnitIds}
                selected={selectedWorkloadUnitIds}
                workload={selectedWorkload}
              />
            )}
            {kind === "state_drift" && (
              <div className="v2-drift-config">
                <label>Horizon<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>{[2, 4, 8, 12, 16, 24].map((value) => <option key={value}>{value}</option>)}</select></label>
                <div>
                  <span>Conditions</span>
                  {["oracle_reset", "chained", "state_anchored"].map((condition) => (
                    <label key={condition}><input type="checkbox" checked={conditions.has(condition)} onChange={() => setConditions(toggleSet(conditions, condition))} />{humanize(condition)}</label>
                  ))}
                </div>
                <div className="v2-drift-dimensions">
                  <span>Scenario dimensions</span>
                  <label>State size<input type="number" min="3" max="128" value={stateSize} onChange={(event) => setStateSize(Number(event.target.value))} /></label>
                  <label>Dependency span<input type="number" min="1" max="32" value={dependencySpan} onChange={(event) => setDependencySpan(Number(event.target.value))} /></label>
                  <label>Branches<input type="number" min="1" max="16" value={branchCount} onChange={(event) => setBranchCount(Number(event.target.value))} /></label>
                  <label>Rollback depth<input type="number" min="0" max="16" value={rollbackDepth} onChange={(event) => setRollbackDepth(Number(event.target.value))} /></label>
                  <label>Distractor ratio<input type="number" min="0" max="0.9" step="0.05" value={distractorRatio} onChange={(event) => setDistractorRatio(Number(event.target.value))} /></label>
                  <label>Tool-error rate<input type="number" min="0" max="0.9" step="0.05" value={toolErrorRate} onChange={(event) => setToolErrorRate(Number(event.target.value))} /></label>
                </div>
              </div>
            )}
            {kind === "coding" && (
              <div className="v2-coding-config">
                <label>
                  Agent
                  <select value={agentId} onChange={(event) => setAgentId(event.target.value)}>
                    <option value="">{agents.isPending ? "Loading agents…" : "Choose an available agent"}</option>
                    {(agents.data ?? []).map((agent) => (
                      <option key={agent.id} value={agent.id} disabled={!agent.available}>
                        {agent.label}{agent.available ? "" : " · unavailable"}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Sandbox provider
                  <select value={sandboxProviderId} onChange={(event) => setSandboxProviderId(event.target.value)}>
                    <option value="">{providers.isPending ? "Loading providers…" : "Choose a configured provider"}</option>
                    {(providers.data ?? []).map((provider) => (
                      <option key={provider.id} value={provider.id} disabled={!provider.configured || provider.status === "error"}>
                        {provider.label} · {provider.configured ? humanize(provider.status) : "Not configured"}
                      </option>
                    ))}
                  </select>
                </label>
                {(agents.error || providers.error) && (
                  <p className="v2-form-warning" role="alert">Coding execution options could not be loaded. Retry this page before submitting.</p>
                )}
              </div>
            )}
          </fieldset>

          <fieldset className="v2-form-section">
            <legend><span>03</span> Intervention contexts</legend>
            <div className="v2-context-builder">
              <article><span>Baseline lane</span><strong>All experts eligible</strong><small>Resident warm engine · no weight reload</small></article>
              <label className="v2-pair-toggle"><input type="checkbox" checked={paired} onChange={(event) => setPaired(event.target.checked)} /><span>Run a paired candidate lane</span></label>
              {paired && (
                <label>
                  Candidate profile
                  <select value={candidateProfileId} onChange={(event) => setCandidateProfileId(event.target.value)}>
                    <option value="">Choose a named profile</option>
                    {(profiles.data ?? []).filter((profile) => profile.model_id === selectedModelId).map((profile) => (
                      <option key={profile.id} value={profile.id}>{profile.name} · {profile.profile_fingerprint.slice(0, 8)}</option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          </fieldset>
        </section>

        <aside className="v2-create-contract">
          <section>
            <span className="v2-section-label">Experiment identity</span>
            <label>Name<input autoFocus value={name} maxLength={120} onChange={(event) => { nameTouched.current = true; setName(event.target.value); }} onFocus={(event) => event.currentTarget.select()} /></label>
            <small>The human name is editable later. Fingerprints remain authoritative.</small>
          </section>
          <section>
            <span className="v2-section-label">Evaluation contract</span>
            <label>
              Verdict policy
              <select value={evaluationMode} onChange={(event) => setEvaluationMode(event.target.value as "deterministic" | "record_only")}>
                <option value="deterministic">Pinned deterministic evaluator</option>
                <option value="record_only">Record only · no pass/fail</option>
              </select>
            </label>
            <strong>{evaluationMode === "record_only" ? "Outputs remain unscored" : kind === "state_drift" ? "Exact canonical state" : kind === "coding" ? "Trusted sandbox verifier" : "Dataset scorer"}</strong>
            <p>{selectedWorkload?.public_success_criteria[0] ?? "The workload adapter owns observable success criteria."}</p>
            <small>Evaluation is workload-owned and persisted with the experiment result.</small>
          </section>
          <section>
            <span className="v2-section-label">Deterministic dimensions</span>
            <div className="v2-compact-fields">
              <label>Seed<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
              <label>Response tokens<input type="number" min="1" max="4096" value={generationMaxTokens} onChange={(event) => setGenerationMaxTokens(Number(event.target.value))} /></label>
            </div>
            <p>Temperature stays at 0 so baseline and candidate lanes remain aligned.</p>
          </section>
          <section>
            <span className="v2-section-label">Execution budget</span>
            <div className="v2-compact-fields">
              <label>Tokens / lane<input type="number" min="1" max="10000000" value={totalTokenBudget} onChange={(event) => setTotalTokenBudget(Number(event.target.value))} /></label>
              <label>Seconds / unit<input type="number" min="1" max="7200" value={perUnitTimeoutSeconds} onChange={(event) => setPerUnitTimeoutSeconds(Number(event.target.value))} /></label>
              {(kind === "coding" || kind === "state_drift") && <label>Max turns<input type="number" min="1" max="1024" value={maxTurns} onChange={(event) => setMaxTurns(Number(event.target.value))} /></label>}
              {kind === "coding" && <label>Max commands<input type="number" min="1" max="4096" value={maxCommands} onChange={(event) => setMaxCommands(Number(event.target.value))} /></label>}
            </div>
            <dl className="v2-create-budget-summary">
              <div><dt>Runtime</dt><dd>{selectedWorkload?.runtime?.label ?? "Warm model engine"}</dd></div>
              <div><dt>Units</dt><dd>{kind === "state_drift" ? `${orderedSelectedUnitIds.length} selected scenarios × ${conditions.size} conditions × 1 horizon` : `${orderedSelectedUnitIds.length} selected`}</dd></div>
              <div><dt>Lane units</dt><dd>{plannedUnits || "—"}</dd></div>
              <div><dt>Cancellation</dt><dd>Safe unit boundaries</dd></div>
              <div><dt>Lanes</dt><dd>{paired ? "Baseline + candidate" : "Baseline"}</dd></div>
            </dl>
            <p>The adapter enforces its trusted timeout and token limits; budgets do not decide success.</p>
          </section>
        </aside>
      </div>
    </div>
  );
}

function WorkloadSelectionSummary({ workload }: { workload: WorkloadDescriptor }) {
  return (
    <div className="v2-workload-selection-summary">
      <V2Status value={workload.readiness} />
      <div><strong>{workload.title}</strong><p>{workload.public_problem_statement}</p></div>
      <dl><div><dt>Source</dt><dd>{workload.source}</dd></div><div><dt>Units</dt><dd>{workload.unit_count}</dd></div><div><dt>Horizon</dt><dd>{workload.expected_horizon ?? "—"}</dd></div></dl>
      {workload.blocked_reasons.length > 0 && <small>{workload.blocked_reasons.join(" ")}</small>}
    </div>
  );
}

function WorkloadUnitSelection({
  onChange,
  selected,
  workload,
}: {
  onChange: (selected: Set<string>) => void;
  selected: ReadonlySet<string>;
  workload: WorkloadDescriptor;
}) {
  const [search, setSearch] = useState("");
  const unitIds = uniqueWorkloadUnitIds(workload.unit_ids);
  const selectedIds = orderedWorkloadUnitSelection(unitIds, selected);
  const query = search.trim().toLowerCase();
  const matchingIds = query
    ? unitIds.filter((unitId) => unitId.toLowerCase().includes(query))
    : unitIds;
  const visibleIds = matchingIds.slice(0, MAX_RENDERED_WORKLOAD_UNITS);
  const atLimit = selectedIds.length >= MAX_SELECTED_WORKLOAD_UNITS;

  useEffect(() => setSearch(""), [workload.id]);

  function toggle(unitId: string) {
    const next = new Set(selectedIds);
    if (next.has(unitId)) {
      next.delete(unitId);
    } else if (!atLimit) {
      next.add(unitId);
    }
    onChange(next);
  }

  return (
    <section className="v2-unit-picker" aria-labelledby="v2-unit-picker-title">
      <header>
        <div>
          <span className="v2-section-label" id="v2-unit-picker-title">
            {workload.kind === "state_drift" ? "Scenario subset" : "Task subset"}
          </span>
          <strong>{selectedIds.length} of {unitIds.length} selected</strong>
        </div>
        <div>
          <button
            disabled={unitIds.length === 0}
            onClick={() => onChange(new Set(selectAllWorkloadUnits(unitIds)))}
            type="button"
          >
            {unitIds.length > MAX_SELECTED_WORKLOAD_UNITS
              ? `Select first ${MAX_SELECTED_WORKLOAD_UNITS.toLocaleString()}`
              : "Select all"}
          </button>
          <button
            disabled={selectedIds.length === 0}
            onClick={() => onChange(new Set())}
            type="button"
          >
            Clear
          </button>
        </div>
      </header>
      {unitIds.length > 12 && (
        <input
          aria-label="Filter workload units"
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Filter task or scenario IDs"
          type="search"
          value={search}
        />
      )}
      <div className="v2-unit-picker-list">
        {visibleIds.map((unitId) => (
          <label key={unitId}>
            <input
              checked={selected.has(unitId)}
              disabled={atLimit && !selected.has(unitId)}
              onChange={() => toggle(unitId)}
              type="checkbox"
            />
            <code>{unitId}</code>
          </label>
        ))}
      </div>
      {unitIds.length === 0 && (
        <small className="v2-form-warning" role="alert">
          This backend did not publish stable unit IDs, so the workload cannot be
          submitted through the V2 experiment contract.
        </small>
      )}
      {unitIds.length > MAX_SELECTED_WORKLOAD_UNITS && (
        <small className="v2-form-warning" role="status">
          The request contract allows at most {MAX_SELECTED_WORKLOAD_UNITS.toLocaleString()} units.
          Large cohorts are intentionally bounded and never submitted in full.
        </small>
      )}
      {matchingIds.length > visibleIds.length && (
        <small>
          Showing the first {visibleIds.length.toLocaleString()} matching IDs. Use
          search to reach another unit; selection still covers all checked IDs.
        </small>
      )}
      {selectedIds.length === 0 && unitIds.length > 0 && (
        <small className="v2-form-warning" role="status">
          Select at least one {workload.kind === "state_drift" ? "scenario" : "task"}.
        </small>
      )}
    </section>
  );
}

export function ExperimentDetailPage({ experimentId }: { experimentId: string }) {
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [cursor, setCursor] = useState(0);
  const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
  const [mobilePanel, setMobilePanel] = useState<"units" | "work" | "inspect">("work");
  const [laneRole, setLaneRole] = useState<LaneRole>("baseline");
  const [inspector, setInspector] = useState<"performance" | "evaluation" | "experts" | "contract" | "events">("performance");
  const experiment = useQuery({
    queryKey: ["v2-experiment", experimentId],
    queryFn: () => getExperiment(experimentId),
    refetchInterval: (query) => activeStatuses.has(query.state.data?.status ?? "") ? 1_000 : false,
    retry: false,
  });
  const profiles = useQuery({
    queryKey: ["profiles"],
    queryFn: listProfiles,
  });
  const eventPage = useQuery({
    queryKey: ["v2-experiment-events", experimentId, cursor],
    queryFn: () => listExperimentEvents(experimentId, cursor),
    enabled: Boolean(experiment.data),
    refetchInterval: activeStatuses.has(experiment.data?.status ?? "") ? 1_000 : false,
    retry: false,
  });

  useEffect(() => {
    if (!eventPage.data) return;
    setEvents((current) => mergeRunEvents(current, eventPage.data.items));
    setCursor((current) => Math.max(current, eventPage.data!.next_sequence));
  }, [eventPage.data]);

  useEffect(() => {
    if (!experiment.data?.units.length) return;
    if (selectedUnitId && experiment.data.units.some((unit) => unit.id === selectedUnitId)) return;
    setSelectedUnitId(experiment.data.active_unit_id ?? experiment.data.units[0].id);
  }, [experiment.data, selectedUnitId]);

  const cancel = useMutation({
    mutationFn: () => cancelExperiment(experimentId),
    onSuccess: () => queryClient.invalidateQueries({
      queryKey: ["v2-experiment", experimentId],
    }),
  });

  if (experiment.isPending) return <div className="v2-page"><V2Loading label="Opening experiment command center…" /></div>;
  if (experiment.error || !experiment.data) return <div className="v2-page"><V2Error error={experiment.error ?? new Error("Experiment not found")} /></div>;
  const profileById = new Map(
    (profiles.data ?? []).map((profile) => [profile.id, profile]),
  );
  const record = {
    ...experiment.data,
    lanes: experiment.data.lanes.map((lane) => {
      const currentProfile = lane.intervention.profile_id
        ? profileById.get(lane.intervention.profile_id)
        : null;
      if (!currentProfile) return lane;
      return {
        ...lane,
        label: lane.role === "candidate" ? currentProfile.name : lane.label,
        intervention: {
          ...lane.intervention,
          name: currentProfile.name,
        },
      };
    }),
  };
  const selected = record.units.find((unit) => unit.id === selectedUnitId) ?? record.units[0] ?? null;
  const progress = liveProgress(record, events);
  const latestEvent = events.at(-1);
  const paired = record.lanes.some((lane) => lane.role === "candidate");

  return (
    <div className="v2-page v2-experiment-detail">
      <V2PageHeader
        eyebrow={`${kindLabel(record.workload_kind)} experiment`}
        title={record.name}
        description={`${record.workload_name} · ${record.model_name}`}
        meta={<><V2Status value={record.status} /><span>{record.lanes.length} lane{record.lanes.length === 1 ? "" : "s"}</span><span>{record.cohort_fingerprint ? `cohort ${record.cohort_fingerprint.slice(0, 10)}` : "cohort pending"}</span></>}
        actions={activeStatuses.has(record.status) ? <button className="v2-danger-button" disabled={cancel.isPending || record.status === "cancelling"} onClick={() => cancel.mutate()}>{cancel.isPending || record.status === "cancelling" ? "Cancelling safely…" : "Cancel experiment"}</button> : <AppLink className="v2-secondary-button" href={`/experiments/new?workload=${encodeURIComponent(record.workload_id)}`}>Run again</AppLink>}
      />
      {cancel.error && <V2Error error={cancel.error} />}

      <section className="v2-live-summary" aria-live="polite">
        <div className="v2-live-phase"><span className={activeStatuses.has(record.status) ? "v2-live-dot" : ""} /><strong>{humanize(progress.phase || record.status)}</strong><small>{latestEvent?.message ?? (record.error || "Durable experiment state is synchronized.")}</small></div>
        <div className="v2-live-progress"><span>{progress.completed}/{progress.total || "—"} lane units</span><progress max={Math.max(progress.total, 1)} value={progress.completed} /><small>{formatOutcomeCounts(progress.passed, progress.failed, progress.unscored)}</small></div>
        <V2Metric label="Elapsed" value={formatDuration(progress.elapsedMs)} detail={progress.etaMs == null ? "ETA measuring" : `${formatDuration(progress.etaMs)} remaining`} />
        <V2Metric label="Tokens" value={formatCompact(progress.promptTokens + progress.completionTokens)} detail={progress.currentTps == null ? `${formatRate(progress.meanTps)} completed mean` : `${formatRate(progress.currentTps)} current`} />
      </section>

      <div className="v2-mobile-command-tabs" role="tablist" aria-label="Experiment panels">
        {(["units", "work", "inspect"] as const).map((panel) => (
          <button key={panel} role="tab" aria-selected={mobilePanel === panel} aria-controls={`v2-${panel}-panel`} className={mobilePanel === panel ? "selected" : ""} onClick={() => setMobilePanel(panel)}>{humanize(panel)}</button>
        ))}
      </div>

      <div className="v2-command-grid">
        <aside id="v2-units-panel" className={`v2-unit-rail ${mobilePanel === "units" ? "mobile-active" : ""}`}>
          <header><span className="v2-section-label">Cohort</span><strong>{record.units.length}</strong></header>
          <div className="v2-unit-list">
            {record.units.map((unit) => (
              <button key={unit.id} className={selected?.id === unit.id ? "selected" : ""} onClick={() => { setSelectedUnitId(unit.id); setMobilePanel("work"); }}>
                <span className="v2-unit-index">{unit.index}</span>
                <span><strong>{unit.title}</strong><small>{unit.subtitle || humanize(unit.status)}</small></span>
                <em className={`v2-pair-dots ${paired ? "paired" : ""}`} aria-label={unitStatusLabel(unit)}><i className={resultTone(unit.baseline)} /><i className={resultTone(unit.candidate)} /></em>
              </button>
            ))}
            {record.units.length === 0 && <V2Empty title="Preparing cohort" detail="Run units will appear here before their first model request." />}
          </div>
        </aside>

        <section id="v2-work-panel" className={`v2-work-stage ${mobilePanel === "work" ? "mobile-active" : ""}`}>
          <div className="v2-stage-toolbar">
            <div><span className="v2-section-label">Selected unit</span><strong>{selected?.title ?? "Waiting for first unit"}</strong></div>
            {paired && (
              <div className="v2-segments v2-lane-tabs" aria-label="Focused lane">
                {record.lanes.map((lane) => <button key={lane.id} className={laneRole === lane.role ? "selected" : ""} aria-pressed={laneRole === lane.role} onClick={() => setLaneRole(lane.role)}>{lane.label}</button>)}
              </div>
            )}
          </div>
          {selected ? <WorkloadStage experiment={record} unit={selected} laneRole={laneRole} events={events} /> : <V2Empty title="The experiment is preparing." detail="The first durable run unit will appear without requiring a refresh." />}
        </section>

        <aside id="v2-inspect-panel" className={`v2-inspector ${mobilePanel === "inspect" ? "mobile-active" : ""}`}>
          <div className="v2-inspector-tabs" role="tablist" aria-label="Experiment inspector">
            {(["performance", "evaluation", "experts", "contract", "events"] as const).map((tab) => <button key={tab} role="tab" aria-selected={inspector === tab} className={inspector === tab ? "selected" : ""} onClick={() => setInspector(tab)}>{humanize(tab)}</button>)}
          </div>
          <ExperimentInspector experiment={record} unit={selected} events={events} tab={inspector} />
        </aside>
      </div>
    </div>
  );
}

function WorkloadStage({ experiment, unit, laneRole, events }: { experiment: ExperimentRecord; unit: ExperimentUnit; laneRole: LaneRole; events: RunEvent[] }) {
  if (experiment.workload_kind === "state_drift") return <DriftStage experiment={experiment} unit={unit} />;
  if (experiment.workload_kind === "coding") return <CodingStage experiment={experiment} unit={unit} laneRole={laneRole} events={events} />;
  return <AnswerStage experiment={experiment} unit={unit} />;
}

function AnswerStage({ experiment, unit }: { experiment: ExperimentRecord; unit: ExperimentUnit }) {
  const lanes = experiment.lanes.map((lane) => ({ lane, result: lane.role === "baseline" ? unit.baseline : unit.candidate }));
  return (
    <div className="v2-answer-stage">
      <article className="v2-prompt-card"><header><span>Prompt</span><small>{unit.id}</small></header><pre>{unit.prompt ?? "Prompt metadata pending."}</pre></article>
      <div className={`v2-lane-results ${lanes.length > 1 ? "paired" : ""}`}>
        {lanes.map(({ lane, result }) => (
          <article key={lane.id} className="v2-lane-result">
            <header><span><strong>{lane.label}</strong><small>{lane.intervention.name}</small></span><ResultBadge result={result} /></header>
            {result?.reasoning && <details><summary>Model reasoning</summary><pre>{result.reasoning}</pre></details>}
            <div className="v2-response-copy">{result?.output ?? <em>Waiting for response…</em>}</div>
            {result?.error && <p className="v2-inline-error">{result.error}</p>}
            <div className="v2-criterion-list">{result?.criteria.map((criterion) => <div key={criterion.label}><span className={criterion.passed == null ? "neutral" : criterion.passed ? "positive" : "negative"}>{criterion.passed == null ? "—" : criterion.passed ? "✓" : "×"}</span><p><strong>{criterion.label}</strong>{criterion.detail && <small>{criterion.detail}</small>}</p></div>)}</div>
          </article>
        ))}
      </div>
    </div>
  );
}

function CodingStage({ experiment, unit, laneRole, events }: { experiment: ExperimentRecord; unit: ExperimentUnit; laneRole: LaneRole; events: RunEvent[] }) {
  const result = laneRole === "baseline" ? unit.baseline : unit.candidate;
  const lane = experiment.lanes.find((candidate) => candidate.role === laneRole);
  const timeline = codingTimeline(
    result,
    events,
    unit.run_unit_ids[laneRole],
    lane?.id ?? null,
  );
  return (
    <div className="v2-coding-stage">
      <header className="v2-coding-heading"><div><span className="v2-section-label">{lane?.label ?? humanize(laneRole)} trajectory</span><h2>{unit.title}</h2><p>{unit.prompt}</p></div><ResultBadge result={result} /></header>
      <div className="v2-trajectory-feed" aria-live={activeStatuses.has(experiment.status) ? "polite" : "off"}>
        {timeline.map((raw, index) => {
          const item = raw as Record<string, unknown>;
          const type = String(item.type ?? "update");
          const title = String(item.title ?? humanize(type));
          const content = String(item.content ?? item.output ?? item.command ?? "");
          return <article className={`v2-trajectory-event ${type}`} key={String(item.id ?? item.sequence ?? index)}><span>{eventGlyph(type)}</span><div><header><strong>{title}</strong><small>{type}</small></header>{content && <pre>{content}</pre>}</div></article>;
        })}
        {timeline.length === 0 && <V2Empty title="Waiting for the first turn" detail="Reasoning, commands, observations, verification, and cleanup events will remain distinct." />}
      </div>
    </div>
  );
}

function DriftStage({ experiment, unit }: { experiment: ExperimentRecord; unit: ExperimentUnit }) {
  const baseline = unit.baseline?.checkpoints ?? [];
  const candidate = unit.candidate?.checkpoints ?? [];
  const indices = [...new Set([...baseline, ...candidate].map((checkpoint) => checkpoint.index))].sort((left, right) => left - right);
  const [selectedIndex, setSelectedIndex] = useState(indices[0] ?? 1);
  useEffect(() => {
    if (indices.includes(selectedIndex)) return;
    setSelectedIndex(indices[0] ?? 1);
  }, [indices, selectedIndex]);
  const baselineCheckpoint = baseline.find((item) => item.index === selectedIndex) ?? null;
  const candidateCheckpoint = candidate.find((item) => item.index === selectedIndex) ?? null;
  return (
    <div className="v2-drift-stage">
      <DriftMetricsStrip experiment={experiment} />
      <DriftResearchMetrics experiment={experiment} />
      <section className="v2-drift-timeline">
        <header><span className="v2-section-label">Checkpoint fidelity</span><small>Exact state is authoritative</small></header>
        <div className="v2-fidelity-chart" aria-label="State fidelity by checkpoint">
          {indices.map((index) => {
            const base = baseline.find((item) => item.index === index);
            const profile = candidate.find((item) => item.index === index);
            return <button key={index} className={selectedIndex === index ? "selected" : ""} onClick={() => setSelectedIndex(index)} aria-label={`Checkpoint ${index}, baseline ${formatPercent(base?.state_fidelity)}, candidate ${formatPercent(profile?.state_fidelity)}`}><span style={{ "--fidelity": `${Math.max(0.08, base?.state_fidelity ?? 0) * 100}%` } as React.CSSProperties} /><i style={{ "--fidelity": `${Math.max(0.08, profile?.state_fidelity ?? 0) * 100}%` } as React.CSSProperties} /><em>{index}</em></button>;
          })}
          {indices.length === 0 && <p>Checkpoints will appear as the scenario advances.</p>}
        </div>
        <footer><span><i className="baseline" />Baseline</span><span><i className="candidate" />Candidate</span></footer>
      </section>
      <div className="v2-checkpoint-comparison">
        <CheckpointCard label="Baseline" checkpoint={baselineCheckpoint} />
        <CheckpointCard label="Candidate profile" checkpoint={candidateCheckpoint} />
      </div>
      <StateDiff checkpoint={candidateCheckpoint ?? baselineCheckpoint} />
    </div>
  );
}

function DriftMetricsStrip({ experiment }: { experiment: ExperimentRecord }) {
  const baseline = experiment.drift_metrics.baseline;
  const candidate = experiment.drift_metrics.candidate;
  return <div className="v2-drift-metrics"><V2Metric label="Baseline AUC" value={formatPercent(baseline?.area_under_fidelity_curve)} /><V2Metric label="Candidate AUC" value={formatPercent(candidate?.area_under_fidelity_curve)} /><V2Metric label="First divergence" value={candidate?.first_divergence_checkpoint == null ? "—" : `Step ${candidate.first_divergence_checkpoint}`} /><V2Metric label="Excess mask drift" value={formatSigned(candidate?.excess_mask_drift)} tone={(candidate?.excess_mask_drift ?? 0) > 0 ? "negative" : "neutral"} /></div>;
}

function DriftResearchMetrics({ experiment }: { experiment: ExperimentRecord }) {
  const baseline = experiment.drift_metrics.baseline;
  const candidate = experiment.drift_metrics.candidate;
  const parameters = experiment.drift_parameters;
  return (
    <section className="v2-drift-research">
      <header><span className="v2-section-label">Drift decomposition</span><small>Exact-state metrics · descriptive</small></header>
      <div className="v2-drift-research-grid">
        <V2Metric label="Final success" value={formatPercent(candidate?.final_success)} detail={`baseline ${formatPercent(baseline?.final_success)}`} />
        <V2Metric label="Transition accuracy" value={formatPercent(candidate?.transition_accuracy)} detail={`baseline ${formatPercent(baseline?.transition_accuracy)}`} />
        <V2Metric label="Local competence" value={formatPercent(candidate?.local_competence)} detail={`Δ ${formatSigned(candidate?.local_capability_delta)}`} />
        <V2Metric label="Chained fidelity" value={formatPercent(candidate?.chained_fidelity)} />
        <V2Metric label="Compounding penalty" value={formatSigned(candidate?.compounding_penalty)} />
        <V2Metric label="Paired drift Δ" value={formatSigned(candidate?.paired_drift_delta)} />
        <V2Metric label="Recovery rate" value={formatPercent(candidate?.recovery_rate)} detail={`baseline ${formatPercent(baseline?.recovery_rate)}`} />
        <V2Metric label="Rollback exact" value={formatPercent(candidate?.rollback_correctness)} />
        <V2Metric label="Growth slope" value={formatDecimal(candidate?.divergence_growth_slope)} />
        <V2Metric label="Reliable horizon" value={candidate?.horizon_at_80 == null ? "—" : String(candidate.horizon_at_80)} detail={`50%: ${candidate?.horizon_at_50 ?? "—"}`} />
        <V2Metric label="Invariant violations" value={formatOptionalCompact(candidate?.invariant_violations)} />
        <V2Metric label="Invalid calls" value={formatOptionalCompact(candidate?.invalid_tool_calls)} />
        <V2Metric label="Collateral mutations" value={formatOptionalCompact(candidate?.collateral_mutations)} />
        <V2Metric label="Survival at horizon" value={formatPercent(candidate?.survival_curve.at(-1))} />
      </div>
      {parameters && <footer>{Object.entries(parameters).map(([key, value]) => <span key={key}>{humanize(key)} <strong>{typeof value === "number" && value < 1 ? `${Math.round(value * 100)}%` : value}</strong></span>)}</footer>}
    </section>
  );
}

function CheckpointCard({ label, checkpoint }: { label: string; checkpoint: DriftCheckpoint | null }) {
  return <article><header><span><strong>{label}</strong><small>{checkpoint?.label ?? "Checkpoint pending"}</small></span>{checkpoint && <V2Status value={checkpoint.status} />}</header><dl><div><dt>Fidelity</dt><dd>{formatPercent(checkpoint?.state_fidelity)}</dd></div><div><dt>Distance</dt><dd>{checkpoint?.state_distance ?? "—"}</dd></div><div><dt>Invalid tools</dt><dd>{checkpoint?.invalid_tool_calls ?? "—"}</dd></div><div><dt>Tool error</dt><dd>{checkpoint?.tool_error_injected ? "Injected · retry expected" : "None"}</dd></div><div><dt>Routing delta</dt><dd>{formatSigned(checkpoint?.routing_delta)}</dd></div></dl>{checkpoint?.invariant_violations.length ? <div className="v2-invariant-list"><span>Invariant violations</span>{checkpoint.invariant_violations.map((item) => <p key={item}>{item}</p>)}</div> : <small>No invariant violation recorded.</small>}</article>;
}

function StateDiff({ checkpoint }: { checkpoint: DriftCheckpoint | null }) {
  if (!checkpoint) return <V2Empty title="State diff pending" detail="Expected and actual canonical state will align here at every checkpoint." />;
  const differences = stateDifferences(checkpoint.expected_state, checkpoint.actual_state);
  return <section className="v2-state-diff"><header><span className="v2-section-label">Expected ↔ actual state</span><strong>{differences.length} changed path{differences.length === 1 ? "" : "s"}</strong></header>{differences.length ? <div>{differences.map((difference) => <article key={difference.path}><code>{difference.path}</code><pre>{formatJson(difference.expected)}</pre><span>→</span><pre>{formatJson(difference.actual)}</pre></article>)}</div> : <p className="v2-state-exact">✓ State is exactly canonical at this checkpoint.</p>}</section>;
}

function ExperimentInspector({ experiment, unit, events, tab }: { experiment: ExperimentRecord; unit: ExperimentUnit | null; events: RunEvent[]; tab: "performance" | "evaluation" | "experts" | "contract" | "events" }) {
  if (tab === "events") return <EventInspector events={events} />;
  if (tab === "contract") return <ContractInspector experiment={experiment} />;
  if (tab === "experts") return <ExpertInspector experiment={experiment} unit={unit} />;
  if (tab === "evaluation") return <div className="v2-inspector-body"><span className="v2-section-label">Evaluation</span><h3>Observable outcomes</h3>{experiment.lanes.map((lane) => { const result = lane.role === "baseline" ? unit?.baseline : unit?.candidate; return <article className="v2-evaluation-summary" key={lane.id}><header><strong>{lane.label}</strong><ResultBadge result={result ?? null} /></header><p>{result?.status === "unscored" ? "Completed without a scored verdict" : result?.criteria.length ? `${result.criteria.filter((item) => item.passed).length}/${result.criteria.length} criteria passed` : "Evaluation pending"}</p></article>; })}</div>;
  return <div className="v2-inspector-body"><span className="v2-section-label">Performance</span><h3>Lane budgets and speed</h3>{experiment.lanes.map((lane) => <LanePerformanceCard key={lane.id} lane={lane} />)}</div>;
}

function ContractInspector({ experiment }: { experiment: ExperimentRecord }) {
  if (experiment.contract_provenance === "legacy_unknown") {
    return (
      <div className="v2-inspector-body v2-legacy-contract">
        <span className="v2-section-label">Legacy contract provenance</span>
        <h3>Evaluation and execution contract unknown</h3>
        <p>
          This archived run predates immutable contract persistence. Missing
          values are unknown; they are not V2 defaults and must not be treated
          as reproducible evidence.
        </p>
        <details>
          <summary>Observed legacy executor metadata</summary>
          <pre className="v2-json-panel">{JSON.stringify(experiment.execution_config, null, 2)}</pre>
        </details>
      </div>
    );
  }
  return (
    <div className="v2-inspector-body">
      <span className="v2-section-label">Immutable contract</span>
      <h3>Execution and evaluation</h3>
      <pre className="v2-json-panel">{JSON.stringify({
        evaluation: experiment.evaluation_contract,
        policy: experiment.execution_policy,
        generation: experiment.generation,
        drift_parameters: experiment.drift_parameters,
        executor: experiment.execution_config,
      }, null, 2)}</pre>
    </div>
  );
}

function ExpertInspector({
  experiment,
  unit,
}: {
  experiment: ExperimentRecord;
  unit: ExperimentUnit | null;
}) {
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });
  const routing = unit ? driftRoutingSummaries(unit) : [];
  const highlighted = routing[0] ?? null;
  return (
    <div className="v2-inspector-body">
      <span className="v2-section-label">Routing context</span>
      <h3>Active expert provenance</h3>
      {experiment.lanes.map((lane) => {
        const profile = lane.intervention.profile_id
          ? profiles.data?.find((item) => item.id === lane.intervention.profile_id)
          : null;
        return (
          <article className="v2-context-inspector" key={lane.id}>
            <header><strong>{lane.label}</strong><V2Status value={lane.intervention.active ? "ready" : lane.status} /></header>
            <div className="v2-run-profile-name">
              {profile
                ? <InlineProfileName profile={profile} />
                : <strong>{lane.intervention.name}</strong>}
              <small>
                {profile
                  ? "Editable display name · fingerprint remains immutable"
                  : lane.role === "baseline"
                    ? "Built-in baseline context"
                    : "Archived profile name · current profile record unavailable"}
              </small>
            </div>
            <dl><div><dt>Profile fingerprint</dt><dd>{lane.intervention.profile_fingerprint?.slice(0, 12) ?? "Baseline"}</dd></div><div><dt>Context</dt><dd>{lane.intervention.context_fingerprint?.slice(0, 12) ?? "Pending"}</dd></div></dl>
          </article>
        );
      })}
      {highlighted && (
        <section className="v2-routing-comparison">
          <header>
            <div><span className="v2-section-label">Paired routing shift</span><strong>Checkpoint {highlighted.checkpoint}</strong></div>
            {highlighted.firstDivergence && <V2Status value="first divergence" />}
          </header>
          <div className="v2-routing-metrics">
            <V2Metric label="Selection overlap" value={formatPercent(highlighted.selectionOverlap)} />
            <V2Metric label="Routing mass JSD" value={formatDecimal(highlighted.routingMassJsDivergence)} />
            <V2Metric label="Selection slots" value={formatOptionalCompact(highlighted.candidateSelectionCount)} detail={`baseline ${formatOptionalCompact(highlighted.baselineSelectionCount)} · Δ ${formatSigned(highlighted.selectionCountDelta)}`} />
            <V2Metric label="Routing mass" value={formatDecimal(highlighted.candidateRoutingMass)} detail={`baseline ${formatDecimal(highlighted.baselineRoutingMass)} · Δ ${formatSigned(highlighted.routingMassDelta)}`} />
          </div>
          <div className="v2-routing-shifts">
            <span className="v2-section-label">Largest selection-share shifts</span>
            {highlighted.shifts.slice(0, 5).map((shift) => (
              <article key={`${shift.layer}:${shift.expert}`}>
                <code>L{shift.layer} · E{shift.expert}</code>
                <span>{formatPercent(shift.baselineShare)} → {formatPercent(shift.candidateShare)}</span>
                <strong className={(shift.delta ?? 0) >= 0 ? "positive" : "negative"}>{formatPercentagePoint(shift.delta)}</strong>
              </article>
            ))}
            {highlighted.shifts.length === 0 && <small>No per-expert shift evidence was recorded.</small>}
          </div>
          {routing.length > 1 && (
            <details>
              <summary>{routing.length} aligned checkpoint comparisons</summary>
              <div className="v2-routing-checkpoint-list">
                {routing.map((item) => <span key={item.checkpoint}>#{item.checkpoint} · overlap {formatPercent(item.selectionOverlap)} · JSD {formatDecimal(item.routingMassJsDivergence)}</span>)}
              </div>
            </details>
          )}
        </section>
      )}
      {unit && !highlighted && (
        <pre className="v2-json-panel">{JSON.stringify({ baseline: unit.baseline?.routing, candidate: unit.candidate?.routing }, null, 2)}</pre>
      )}
    </div>
  );
}

function LanePerformanceCard({ lane }: { lane: ExperimentLane }) {
  return <article className="v2-performance-card"><header><strong>{lane.label}</strong><small>{lane.intervention.name}</small></header><div><V2Metric label="Elapsed" value={formatDuration(lane.performance.elapsed_ms)} /><V2Metric label="Total tokens" value={formatCompact(lane.performance.total_tokens)} /><V2Metric label="Mean TPS" value={formatRate(lane.performance.mean_tps)} /><V2Metric label="Estimated cost" value={lane.performance.estimated_cost_usd == null ? "Unknown" : `$${lane.performance.estimated_cost_usd.toFixed(4)}`} /></div></article>;
}

function EventInspector({ events }: { events: RunEvent[] }) {
  return <div className="v2-inspector-body v2-event-inspector"><span className="v2-section-label">Durable event stream</span><h3>{events.length} persisted updates</h3>{[...events].reverse().map((event) => <article key={event.sequence}><span>{event.sequence}</span><div><strong>{event.message}</strong><small>{[event.lane_role, event.unit_id, event.phase].filter(Boolean).join(" · ")}</small></div><time>{formatTime(event.created_at)}</time></article>)}{events.length === 0 && <p>Waiting for the first persisted event.</p>}</div>;
}

function ResultBadge({ result }: { result: LaneUnitResult | null }) {
  if (!result) return <V2Status value="queued" />;
  return <V2Status value={result.passed == null ? result.status : result.passed ? "passed" : "failed"} />;
}

function resultTone(result: LaneUnitResult | null) {
  if (!result) return "pending";
  if (result.status === "unscored") return "unscored";
  if (result.passed == null) return "pending";
  return result.passed ? "pass" : "fail";
}

function unitStatusLabel(unit: ExperimentUnit) {
  return `Baseline ${resultTone(unit.baseline)}${unit.candidate ? `, candidate ${resultTone(unit.candidate)}` : ""}`;
}

function toggleSet(current: Set<string>, value: string) {
  const next = new Set(current);
  if (next.has(value)) next.delete(value); else next.add(value);
  return next;
}

function kindLabel(kind: WorkloadKind) {
  if (kind === "state_drift") return "State drift";
  return kind === "coding" ? "Coding" : "Answer";
}

function kindDescription(kind: WorkloadKind) {
  if (kind === "state_drift") return "State transitions across longer horizons";
  if (kind === "coding") return "Tool-using work in an isolated repository";
  return "One-request questions and deterministic scoring";
}

function eventGlyph(type: string) {
  if (type.includes("command") || type.includes("tool")) return "$";
  if (type.includes("reason")) return "◇";
  if (type.includes("verif") || type.includes("evaluat")) return "✓";
  if (type.includes("error") || type.includes("stderr")) return "!";
  return "·";
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown date" : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(date);
}

function formatTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(value: number | null | undefined) {
  if (value == null) return "—";
  if (value < 1_000) return `${Math.round(value)} ms`;
  const seconds = value / 1_000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function formatCompact(value: number) {
  return new Intl.NumberFormat(undefined, { notation: value >= 10_000 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
}

function formatOptionalCompact(value: number | null | undefined) {
  return value == null ? "—" : formatCompact(value);
}

function formatOutcomeCounts(
  passed: number,
  failed: number,
  unscored: number,
  completed?: number,
) {
  if (completed != null && passed + failed + unscored < completed) {
    return `${completed} completed · ${passed} passed`;
  }
  return [
    `${passed} passed`,
    `${failed} failed`,
    unscored ? `${unscored} unscored` : null,
  ].filter(Boolean).join(" · ");
}

function formatRate(value: number | null | undefined) {
  return value == null ? "Unknown" : `${value.toFixed(1)} tok/s`;
}

function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
}

function formatSigned(value: number | null | undefined) {
  return value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatDecimal(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(4);
}

function formatPercentagePoint(value: number | null | undefined) {
  return value == null
    ? "—"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
}

function formatJson(value: unknown) {
  return value === undefined ? "<missing>" : JSON.stringify(value, null, 2);
}
