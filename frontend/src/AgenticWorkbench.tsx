import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { activeJobFromConflict, api } from "./api";
import { AppLink } from "./AppShell";
import {
  evaluationContractPayload,
  executionPolicyPayload,
  generationConfigPayload,
} from "./contracts";
import {
  defaultEvaluationContract,
  defaultExecutionPolicy,
  EvaluationControls,
} from "./EvaluationControls";
import { ExpertExplorer } from "./ExpertExplorer";
import { normalizeEngagementMatrix } from "./expertSelection";
import { TrajectoryViewer } from "./TrajectoryViewer";
import type {
  AgentDefinition,
  AgentRun,
  AgentRunStatus,
  AgentTask,
  AgentTaskPack,
  AgentTrajectory,
  AgentTrialArtifacts,
  AgentTrialSummary,
  CreateExpertProfileRequest,
  CreateAgentRunRequest,
  JobRecord,
  ModelSession,
  ModelState,
  ProfileProposal,
  ProviderPreflight,
  SandboxProvider,
  SavedExpertProfile,
  SystemStatus,
  TrialRoutingSummary,
  RoutingExploreResponse,
  EvaluationContract,
  ExecutionPolicy,
} from "./types";

const MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8";

interface AgenticWorkbenchProps {
  mode: SystemStatus["mode"];
  modelState: ModelState;
  currentModelSession: ModelSession | null;
  busy: boolean;
  requestedRunId: string | null;
  onRequestedRunOpened: () => void;
  onActivityChange: (active: boolean) => void;
  onJobConflict: (job: JobRecord) => void;
  commandCenter?: boolean;
}

export function AgenticWorkbench({
  mode,
  modelState,
  currentModelSession,
  busy,
  requestedRunId,
  onRequestedRunOpened,
  onActivityChange,
  onJobConflict,
  commandCenter = false,
}: AgenticWorkbenchProps) {
  const queryClient = useQueryClient();
  const [selectedPackId, setSelectedPackId] = useState("");
  const [selectedTaskIds, setSelectedTaskIds] = useState<Set<string>>(
    new Set(),
  );
  const [selectedProviderId, setSelectedProviderId] = useState("");
  const [selectedAgentId, setSelectedAgentId] = useState("");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<ProviderPreflight | null>(null);
  const [runNotice, setRunNotice] = useState<string | null>(null);
  const openedRequestedRun = useRef<string | null>(null);
  const [executionPolicy, setExecutionPolicy] = useState<ExecutionPolicy>(() =>
    defaultExecutionPolicy(true),
  );
  const [evaluationContract, setEvaluationContract] =
    useState<EvaluationContract>(() =>
      defaultEvaluationContract("Trusted coding verifier", {
        kind: "benchmark_default",
        label: "Protected test suite passes",
        description:
          "The trusted verifier runs after the agent stops, including after a budget limit.",
        weight: 1,
        visibility: "public",
      }),
    );

  const agentsQuery = useQuery({
    queryKey: ["agents"],
    queryFn: () => api<AgentDefinition[]>("/api/agents"),
    staleTime: 30_000,
  });
  const providersQuery = useQuery({
    queryKey: ["sandbox-providers"],
    queryFn: () => api<SandboxProvider[]>("/api/sandbox-providers"),
    staleTime: 30_000,
  });
  const packsQuery = useQuery({
    queryKey: ["agent-task-packs"],
    queryFn: () => api<AgentTaskPack[]>("/api/agent-task-packs"),
    staleTime: 30_000,
  });
  const tasksQuery = useQuery({
    queryKey: ["agent-task-pack-tasks", selectedPackId],
    queryFn: () =>
      api<AgentTask[]>(
        `/api/agent-task-packs/${encodeURIComponent(selectedPackId)}/tasks`,
      ),
    enabled: Boolean(selectedPackId),
    staleTime: 30_000,
  });
  const runsQuery = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => api<AgentRun[]>("/api/agent-runs"),
  });
  const runQuery = useQuery({
    queryKey: ["agent-run", activeRunId],
    queryFn: () => api<AgentRun>(`/api/agent-runs/${activeRunId}`),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) =>
      isActiveRun((query.state.data as AgentRun | undefined)?.status)
        ? 1_000
        : false,
  });
  const activeRun = runQuery.data ??
    runsQuery.data?.find((run) => run.id === activeRunId) ?? null;
  const selectedTrial = activeRun?.trials.find(
    (trial) => trial.id === selectedTrialId,
  ) ?? null;
  const trialLive = isActiveTrial(selectedTrial?.status);
  const trajectoryQuery = useQuery({
    queryKey: ["agent-trajectory", selectedTrialId],
    queryFn: () =>
      api<AgentTrajectory>(`/api/trials/${selectedTrialId}/trajectory`),
    enabled: Boolean(selectedTrialId),
    refetchInterval: trialLive ? 750 : false,
    retry: false,
  });
  const artifactsQuery = useQuery({
    queryKey: ["agent-artifacts", selectedTrialId],
    queryFn: () =>
      api<AgentTrialArtifacts>(`/api/trials/${selectedTrialId}/artifacts`),
    enabled: Boolean(selectedTrialId),
    refetchInterval: trialLive ? 1_000 : false,
    retry: false,
  });
  const routingQuery = useQuery({
    queryKey: ["agent-routing", selectedTrialId],
    queryFn: () =>
      api<TrialRoutingSummary>(`/api/trials/${selectedTrialId}/routing`),
    enabled: Boolean(selectedTrialId && selectedTrial?.routed_inference_calls),
    refetchInterval: trialLive ? 1_500 : false,
    retry: false,
  });

  const providers = providersQuery.data ?? [];
  const agents = agentsQuery.data ?? [];
  const packs = packsQuery.data ?? [];
  const tasks = tasksQuery.data ?? [];
  const selectedPack = packs.find((pack) => pack.id === selectedPackId) ?? null;
  const selectedProvider =
    providers.find((provider) => provider.id === selectedProviderId) ?? null;
  const selectedAgent =
    agents.find((agent) => agent.id === selectedAgentId) ?? null;
  const modelReady =
    modelState === "ready" && currentModelSession?.state === "ready";
  const providerReady = preflight
    ? preflight.status === "ready"
    : selectedProvider?.status === "ready";
  const canStart = Boolean(
    modelReady &&
      selectedPack?.ready &&
      providerReady &&
      selectedAgent?.available &&
      selectedTaskIds.size > 0 &&
      !busy &&
      !isActiveRun(activeRun?.status),
  );

  useEffect(() => {
    if (selectedPackId || packs.length === 0) return;
    setSelectedPackId(packs.find((pack) => pack.ready)?.id ?? packs[0].id);
  }, [packs, selectedPackId]);

  useEffect(() => {
    const taskIds = new Set(tasks.map((task) => task.id));
    if (activeRun?.task_pack_id === selectedPackId) {
      setSelectedTaskIds(
        new Set(activeRun.task_ids.filter((taskId) => taskIds.has(taskId))),
      );
      return;
    }
    setSelectedTaskIds(taskIds);
  }, [activeRun?.id, selectedPackId, tasks]);

  useEffect(() => {
    if (selectedAgentId || agents.length === 0) return;
    setSelectedAgentId(
      agents.find((agent) => agent.is_default && agent.available)?.id ??
        agents.find((agent) => agent.available)?.id ??
        agents[0].id,
    );
  }, [agents, selectedAgentId]);

  useEffect(() => {
    if (!selectedAgent) return;
    const budgets = selectedAgent.default_budgets;
    setExecutionPolicy((current) => ({
      ...current,
      max_turns: budgets.max_turns,
      max_tokens: budgets.max_tokens,
      max_commands: budgets.max_commands,
      timeout_seconds: budgets.timeout_seconds,
    }));
  }, [selectedAgent?.id]);

  useEffect(() => {
    if (selectedProviderId || providers.length === 0) return;
    const preferred =
      mode === "mock"
        ? providers.find((provider) => provider.id === "fake")
        : providers.find((provider) => provider.id === "daytona");
    setSelectedProviderId(
      preferred?.id ??
        providers.find((provider) => provider.status === "ready")?.id ??
        providers[0].id,
    );
  }, [mode, providers, selectedProviderId]);

  useEffect(() => {
    setPreflight(null);
  }, [selectedProviderId]);

  useEffect(() => {
    if (!requestedRunId || openedRequestedRun.current === requestedRunId) return;
    openedRequestedRun.current = requestedRunId;
    setActiveRunId(requestedRunId);
    setSelectedTrialId(null);
    onRequestedRunOpened();
    window.requestAnimationFrame(() => {
      document.getElementById("agentic")?.scrollIntoView({ block: "start" });
    });
  }, [onRequestedRunOpened, requestedRunId]);

  useEffect(() => {
    if (activeRunId || !runsQuery.data) return;
    const interrupted = runsQuery.data.find((run) => isActiveRun(run.status));
    if (interrupted) setActiveRunId(interrupted.id);
  }, [activeRunId, runsQuery.data]);

  useEffect(() => {
    if (!activeRun) return;
    const selectedStillExists = activeRun.trials.some(
      (trial) => trial.id === selectedTrialId,
    );
    if (activeRun.active_trial_id && isActiveRun(activeRun.status)) {
      setSelectedTrialId(activeRun.active_trial_id);
    } else if (!selectedStillExists) {
      setSelectedTrialId(
        activeRun.trials.find((trial) => isActiveTrial(trial.status))?.id ??
          activeRun.trials[0]?.id ??
          null,
      );
    }
  }, [activeRun, selectedTrialId]);

  useEffect(() => {
    if (!activeRun) return;
    setSelectedPackId(activeRun.task_pack_id);
    setSelectedProviderId(activeRun.sandbox_provider_id);
    setSelectedAgentId(activeRun.agent_id);
  }, [activeRun?.id]);

  useEffect(() => {
    onActivityChange(isActiveRun(activeRun?.status));
  }, [activeRun?.status, onActivityChange]);

  useEffect(() => {
    if (!activeRun || isActiveRun(activeRun.status)) return;
    void queryClient.invalidateQueries({ queryKey: ["agent-runs"] });
  }, [activeRun, queryClient]);

  const preflightProvider = useMutation({
    mutationFn: (providerId: string) =>
      api<ProviderPreflight>(
        `/api/sandbox-providers/${encodeURIComponent(providerId)}/preflight`,
        { method: "POST" },
      ),
    onSuccess: (result) => {
      setPreflight(result);
      void queryClient.invalidateQueries({ queryKey: ["sandbox-providers"] });
    },
  });

  const startRun = useMutation({
    mutationFn: async (request: CreateAgentRunRequest) => {
      let job: JobRecord;
      try {
        job = await api<JobRecord>("/api/agent-runs", {
          method: "POST",
          body: JSON.stringify(request),
        });
      } catch (error) {
        const serverJob = activeJobFromConflict(error);
        if (!serverJob) throw error;
        if (serverJob.kind === "agent_run" && serverJob.result_id) {
          setActiveRunId(serverJob.result_id);
          setSelectedTrialId(null);
          setRunNotice("Rejoined the coding run already active on the server.");
          void queryClient.invalidateQueries({ queryKey: ["agent-runs"] });
        } else {
          onJobConflict(serverJob);
        }
        return null;
      }
      if (!job.result_id) throw new Error("Agent job has no run ID");
      return job;
    },
    onMutate: () => setRunNotice(null),
    onSuccess: (job) => {
      if (!job) return;
      setActiveRunId(job.result_id);
      setSelectedTrialId(null);
      void queryClient.invalidateQueries({ queryKey: ["agent-runs"] });
    },
    onError: () => {
      void queryClient.invalidateQueries({ queryKey: ["active-job"] });
      void queryClient.invalidateQueries({ queryKey: ["agent-runs"] });
    },
  });

  useEffect(() => {
    if (activeRunId && startRun.isError) startRun.reset();
  }, [activeRunId, startRun.isError]);

  const cancelRun = useMutation({
    mutationFn: (runId: string) =>
      api<AgentRun>(`/api/agent-runs/${runId}/cancel`, { method: "POST" }),
    onSuccess: (run) => {
      queryClient.setQueryData(["agent-run", run.id], run);
      void queryClient.invalidateQueries({ queryKey: ["agent-runs"] });
    },
  });

  const setupError =
    agentsQuery.error ||
    providersQuery.error ||
    packsQuery.error ||
    tasksQuery.error ||
    runsQuery.error ||
    runQuery.error ||
    preflightProvider.error ||
    startRun.error ||
    cancelRun.error;
  const selectedTaskList = useMemo(
    () => tasks.filter((task) => selectedTaskIds.has(task.id)),
    [selectedTaskIds, tasks],
  );

  function toggleTask(taskId: string) {
    setSelectedTaskIds((current) => {
      const next = new Set(current);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  }

  function submitRun() {
    if (!currentModelSession || !selectedAgent || !selectedProvider) return;
    startRun.mutate({
      task_pack_id: selectedPackId,
      task_ids: selectedTaskList.map((task) => task.id),
      agent_id: selectedAgent.id,
      sandbox_provider_id: selectedProvider.id,
      model_session_id: currentModelSession.id,
      budgets: {
        max_turns:
          executionPolicy.max_turns ?? selectedAgent.default_budgets.max_turns,
        max_tokens:
          executionPolicy.max_tokens ?? selectedAgent.default_budgets.max_tokens,
        max_commands:
          executionPolicy.max_commands ??
          selectedAgent.default_budgets.max_commands,
        timeout_seconds: executionPolicy.timeout_seconds,
      },
      attempts: executionPolicy.attempts ?? 1,
      seed: executionPolicy.seed ?? 0,
      generation: generationConfigPayload(executionPolicy, {
        temperature: 0,
        max_tokens: 2048,
        seed: 0,
        enable_thinking: true,
      }),
      reasoning_mode: executionPolicy.reasoning_visibility ?? "compact",
      execution_policy: executionPolicyPayload(executionPolicy),
      evaluation_contract: evaluationContractPayload(evaluationContract),
    });
  }

  return (
    <section
      className={`agentic-section ${commandCenter ? "agentic-command-workbench" : ""} ${activeRun ? "has-active-run" : ""}`}
      id="agentic"
    >
      <div className="section-intro agentic-intro">
        <div>
          <span className="section-label">03 · Agentic coding</span>
          <h2>Watch the model work, turn by turn.</h2>
          <p>
            Run code in an isolated sandbox while every model inference, command,
            verifier result, and expert-routing capture remains linked.
          </p>
        </div>
        <span className="sandbox-boundary">Code runs outside this Pod</span>
      </div>

      {setupError && (
        <div className="agentic-error" role="alert">
          {(setupError as Error).message}
        </div>
      )}
      {runNotice && (
        <div className="job-recovery-note" role="status">
          <span>{runNotice}</span>
        </div>
      )}

      <div className="agent-setup-grid">
        <article className="agent-setup-card task-pack-card">
          <div className="agent-card-heading">
            <div>
              <span className="section-label">Task pack</span>
              <h3>{selectedPack?.name ?? "Loading task packs…"}</h3>
            </div>
            <span className="count-badge">
              {selectedTaskIds.size}/{tasks.length}
            </span>
          </div>
          <label className="agent-select-field">
            Coding task pack
            <select
              value={selectedPackId}
              disabled={isActiveRun(activeRun?.status)}
              onChange={(event) => {
                setSelectedPackId(event.target.value);
                setSelectedTaskIds(new Set());
              }}
            >
              {packs.map((pack) => (
                <option key={pack.id} value={pack.id} disabled={!pack.ready}>
                  {pack.name}{pack.ready ? "" : " · unavailable"}
                </option>
              ))}
            </select>
          </label>
          <p>{selectedPack?.description}</p>
          <div className="validation-stamps">
            <span className={selectedPack?.oracle_passed ? "valid" : "pending"}>
              {selectedPack?.oracle_passed ? "✓ Oracle passes" : "Oracle pending"}
            </span>
            <span className={selectedPack?.noop_failed ? "valid" : "pending"}>
              {selectedPack?.noop_failed ? "✓ No-op fails" : "No-op pending"}
            </span>
          </div>
          <div className="agent-task-list" aria-label="Tasks in selected pack">
            {tasks.map((task) => (
              <label className="agent-task-row" key={task.id}>
                <input
                  type="checkbox"
                  checked={selectedTaskIds.has(task.id)}
                  disabled={isActiveRun(activeRun?.status)}
                  onChange={() => toggleTask(task.id)}
                />
                <span>
                  <strong>{task.title}</strong>
                  <small>
                    {[task.language, ...task.tags].filter(Boolean).join(" · ")}
                  </small>
                  <em>{task.instruction}</em>
                </span>
              </label>
            ))}
            {tasksQuery.isPending && <p className="agent-empty-copy">Loading tasks…</p>}
          </div>
        </article>

        <article className="agent-setup-card readiness-card">
          <span className="section-label">Readiness</span>
          <h3>Everything in its place.</h3>
          <div className="readiness-list">
            <ReadinessRow
              label="Model"
              detail={
                modelReady
                  ? currentModelSession?.profile_id
                    ? `Profile ${currentModelSession.profile_id.slice(0, 8)}`
                    : "Baseline ready"
                  : "Load the model first"
              }
              ready={modelReady}
            />
            <ReadinessRow
              label="Task pack"
              detail={selectedPack?.ready ? "Oracle/no-op verified" : "Not ready"}
              ready={selectedPack?.ready === true}
            />
            <ReadinessRow
              label="Sandbox"
              detail={providerStatusLabel(selectedProvider, preflight)}
              ready={providerReady}
            />
          </div>
          <label className="agent-select-field">
            Sandbox provider
            <select
              value={selectedProviderId}
              disabled={isActiveRun(activeRun?.status)}
              onChange={(event) => setSelectedProviderId(event.target.value)}
            >
              {providers.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.label}
                </option>
              ))}
            </select>
          </label>
          {selectedProvider && !selectedProvider.configured && (
            <p className="provider-guidance">
              Add {selectedProvider.required_env.join(" or ")} to the Pod, then
              restart the app.
            </p>
          )}
          {preflight?.error && (
            <p className="provider-guidance error">{preflight.error.message}</p>
          )}
          {selectedProvider && selectedProvider.id !== "fake" && (
            <button
              className="text-button provider-test-button"
              disabled={
                !selectedProvider.configured || preflightProvider.isPending
              }
              onClick={() => preflightProvider.mutate(selectedProvider.id)}
            >
              {preflightProvider.isPending ? "Testing connection…" : "Test connection"}
            </button>
          )}
          <small className="no-spend-note">
            Connection checks are authenticated and read-only. They do not create a
            sandbox. Task files go to the selected provider; credentials never do.
          </small>
        </article>

        <article className="agent-setup-card run-plan-card">
          <span className="section-label">Run plan</span>
          <h3>One careful attempt.</h3>
          <label className="agent-select-field">
            Agent scaffold
            <select
              value={selectedAgentId}
              disabled={isActiveRun(activeRun?.status)}
              onChange={(event) => setSelectedAgentId(event.target.value)}
            >
              {agents.map((agent) => (
                <option key={agent.id} value={agent.id} disabled={!agent.available}>
                  {agent.label}{agent.available ? "" : " · unavailable"}
                </option>
              ))}
            </select>
          </label>
          <p>{selectedAgent?.description}</p>
          {selectedAgent && (
            <dl className="agent-budget-list">
              <div><dt>Turns</dt><dd>{executionPolicy.max_turns ?? "—"}</dd></div>
              <div><dt>Total tokens</dt><dd>{executionPolicy.max_tokens == null ? "—" : formatCompactNumber(executionPolicy.max_tokens)}</dd></div>
              <div><dt>Commands</dt><dd>{executionPolicy.max_commands ?? "—"}</dd></div>
              <div><dt>Time</dt><dd>{formatMinutes(executionPolicy.timeout_seconds)}</dd></div>
            </dl>
          )}
          <button
            className="primary-button agent-start-button"
            disabled={!canStart || startRun.isPending}
            onClick={submitRun}
          >
            {startRun.isPending
              ? "Starting trial…"
              : `Start ${selectedTaskIds.size}-task run`}
          </button>
          {!canStart && !isActiveRun(activeRun?.status) && (
            <small className="start-guidance">
              Model, task pack, provider, and at least one task must be ready.
            </small>
          )}
        </article>
      </div>

      <details className="agent-contract-drawer" open={commandCenter && !activeRun}>
        <summary>
          <span>Evaluation & execution contract</span>
          <small>Success is independent from turns, tokens, commands, and time</small>
        </summary>
        <EvaluationControls
          contract={evaluationContract}
          policy={executionPolicy}
          onContractChange={setEvaluationContract}
          onPolicyChange={setExecutionPolicy}
          agentic
          readOnly={isActiveRun(activeRun?.status)}
        />
      </details>

      {activeRun && (
        <AgentRunMonitor
          run={activeRun}
          selectedTrial={selectedTrial}
          trajectory={trajectoryQuery.data ?? null}
          artifacts={artifactsQuery.data ?? null}
          routing={routingQuery.data ?? null}
          cancelling={cancelRun.isPending}
          onCancel={() => cancelRun.mutate(activeRun.id)}
          onSelectTrial={setSelectedTrialId}
          onRunAgain={submitRun}
          canRunAgain={canStart}
        />
      )}
    </section>
  );
}

function AgentRunMonitor({
  run,
  selectedTrial,
  trajectory,
  artifacts,
  routing,
  cancelling,
  onCancel,
  onSelectTrial,
  onRunAgain,
  canRunAgain,
}: {
  run: AgentRun;
  selectedTrial: AgentTrialSummary | null;
  trajectory: AgentTrajectory | null;
  artifacts: AgentTrialArtifacts | null;
  routing: TrialRoutingSummary | null;
  cancelling: boolean;
  onCancel: () => void;
  onSelectTrial: (trialId: string) => void;
  onRunAgain: () => void;
  canRunAgain: boolean;
}) {
  const live = isActiveRun(run.status);
  const progressTotal = Math.max(run.total_trials, 1);
  const [mobilePanel, setMobilePanel] = useState<
    "tasks" | "trajectory" | "experts"
  >("trajectory");
  const routedTrialIds = run.trials
    .filter((trial) => trial.routed_inference_calls > 0)
    .map((trial) => trial.id);
  const wholeRunProfileHref = routedTrialIds.length
    ? `/profiles/new?${routedTrialIds.map((trialId) =>
        `trial=${encodeURIComponent(trialId)}`
      ).join("&")}`
    : null;
  return (
    <div className="agent-run-monitor">
      <header className="agent-run-header" aria-live="polite">
        <div>
          <span className="section-label">
            {live ? "Live coding run" : "Coding run"}
          </span>
          <h3>{live ? "The agent is at work." : runSummary(run)}</h3>
          <p>
            {run.task_pack_name} · {profileLabel(run)} · {run.sandbox_provider_id}
          </p>
        </div>
        <div className="agent-run-progress">
          <strong>{run.completed_trials}/{run.total_trials}</strong>
          <span>tasks complete</span>
          <progress max={progressTotal} value={run.completed_trials} />
        </div>
        <div className="agent-run-actions">
          <span className={`run-state ${run.status}`}>{run.status}</span>
          <a
            className="agent-run-export"
            href={`/api/agent-runs/${run.id}/export`}
            download
          >
            Export coding run
          </a>
          {wholeRunProfileHref && (
            <AppLink className="agent-run-export" href={wholeRunProfileHref}>
              Profile whole run
            </AppLink>
          )}
          {live ? (
            <button
              className="danger-text-button"
              disabled={cancelling || run.status === "cancelling"}
              onClick={onCancel}
            >
              {cancelling || run.status === "cancelling"
                ? "Cleaning up…"
                : "Cancel & clean up"}
            </button>
          ) : (
            <button
              className="primary-button"
              disabled={!canRunAgain}
              onClick={onRunAgain}
            >
              Run same pack again
            </button>
          )}
        </div>
      </header>

      {run.error && <div className="agentic-error">{run.error}</div>}

      <div className="agent-mobile-tabs" role="tablist">
        {(["tasks", "trajectory", "experts"] as const).map((panel) => (
          <button
            key={panel}
            className={mobilePanel === panel ? "selected" : ""}
            onClick={() => setMobilePanel(panel)}
          >
            {panel}
          </button>
        ))}
      </div>

      <div className="agent-live-grid">
        <aside
          className={`trial-rail agent-mobile-panel ${mobilePanel === "tasks" ? "active" : ""}`}
          aria-label="Coding task trials"
        >
          {run.trials.map((trial, index) => (
            <button
              className={trial.id === selectedTrial?.id ? "selected" : ""}
              key={trial.id}
              onClick={() => {
                onSelectTrial(trial.id);
                setMobilePanel("trajectory");
              }}
            >
              <span className={`trial-index ${trial.status}`}>{index + 1}</span>
              <span>
                <strong>{trial.title}</strong>
                <small>
                  {trialStatusLabel(trial)}
                  {trial.reward != null
                    ? ` · ${Math.round(trial.reward * 100)}% reward`
                    : ""}
                </small>
              </span>
            </button>
          ))}
          {run.trials.length === 0 && (
            <div className="trial-rail-empty">
              <span>Queued</span>
              <p>Waiting for the first task record.</p>
            </div>
          )}
        </aside>
        <div className={`agent-trajectory-pane agent-mobile-panel ${mobilePanel === "trajectory" ? "active" : ""}`}>
          {selectedTrial ? (
            <TrajectoryViewer
              trial={selectedTrial}
              trajectory={trajectory}
              artifacts={artifacts}
              live={isActiveTrial(selectedTrial.status)}
            />
          ) : (
            <div className="trajectory-placeholder">
              <span>◇</span>
              <p>The selected task trajectory will open here.</p>
            </div>
          )}
        </div>
        <aside className={`agent-expert-pane agent-mobile-panel ${mobilePanel === "experts" ? "active" : ""}`}>
          {selectedTrial ? (
            <TrialRoutingCard
              key={selectedTrial.id}
              trial={selectedTrial}
              routing={routing}
              profileId={run.profile_id}
              profileFingerprint={run.profile_fingerprint}
            />
          ) : (
            <div className="trajectory-placeholder">
              <span>⌁</span>
              <p>Select a trial to inspect expert routing.</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function TrialRoutingCard({
  trial,
  routing,
  profileId,
  profileFingerprint,
}: {
  trial: AgentTrialSummary;
  routing: TrialRoutingSummary | null;
  profileId: string | null;
  profileFingerprint: string | null;
}) {
  const queryClient = useQueryClient();
  const [proposal, setProposal] = useState<ProfileProposal | null>(null);
  const [profileName, setProfileName] = useState(
    `${trial.title} · 64 experts/layer`,
  );
  const [savedProfile, setSavedProfile] = useState<SavedExpertProfile | null>(
    null,
  );

  const proposeProfile = useMutation({
    mutationFn: () =>
      api<ProfileProposal>(
        `/api/trials/${encodeURIComponent(trial.id)}/profile-proposal?keep_per_layer=64&metric=routing_mass`,
        { method: "POST" },
      ),
    onMutate: () => setSavedProfile(null),
    onSuccess: (nextProposal) => setProposal(nextProposal),
  });

  const saveProfile = useMutation({
    mutationFn: (request: CreateExpertProfileRequest) =>
      api<SavedExpertProfile>("/api/profiles", {
        method: "POST",
        body: JSON.stringify(request),
      }),
    onSuccess: (nextProfile) => {
      setSavedProfile(nextProfile);
      void queryClient.invalidateQueries({ queryKey: ["profiles"] });
    },
  });

  function saveProposal() {
    if (!proposal || !profileName.trim()) return;
    saveProfile.mutate({
      name: profileName.trim(),
      description: `Routing-mass profile learned from coding trial ${trial.id}.`,
      model_id: MODEL_ID,
      profile: proposal.profile,
      source: "agentic",
      source_trial_id: trial.id,
      metric: "routing_mass",
      observed_mass_retained: proposal.observed_mass_retained,
    });
  }

  const profileError = proposeProfile.error || saveProfile.error;
  const explorerData: RoutingExploreResponse | null = routing
    ? {
        fingerprint: `agent_trial:${trial.id}`,
        aggregation: "weighted_source_normalized",
        model_id: MODEL_ID,
        profile_id: routing.profile_id,
        profile_fingerprint: routing.profile_fingerprint,
        layer_ids: routing.layer_ids,
        num_experts: routing.selection_counts[0]?.length ?? 256,
        top_k: 8,
        selection_counts: normalizeEngagementMatrix(routing.selection_counts),
        routing_mass: normalizeEngagementMatrix(routing.routing_mass),
        total_routed_slots: routing.total_routed_slots,
        captured_inference_calls: routing.captured_inference_calls,
        total_inference_calls: routing.inference_calls,
        served_tokens: routing.served_tokens,
        sources: [
          {
            kind: "agent_trial",
            id: trial.id,
            label: trial.title,
            status: trial.status,
            profile_id: routing.profile_id,
          },
        ],
        filter_capabilities: {
          item: false,
          trial: true,
          step_type: false,
          outcome: true,
        },
      }
    : null;

  return (
    <section className="agent-routing-panel">
      <header>
        <div>
          <span className="section-label">Expert routing</span>
          <h3>Every model turn stays attributable.</h3>
          <p>
            {profileId
              ? `Profile ${profileFingerprint?.slice(0, 10) ?? profileId.slice(0, 8)}`
              : "Baseline model"}
          </p>
        </div>
        <div className="routing-capture-summary">
          <span>
            <strong>{routing?.captured_inference_calls ?? trial.routed_inference_calls}</strong>
            routed calls
          </span>
          <span>
            <strong>{routing?.inference_calls ?? trial.inference_calls}</strong>
            total calls
          </span>
          <span>
            <strong>{routing ? formatCompactNumber(routing.served_tokens) : "—"}</strong>
            served tokens
          </span>
        </div>
      </header>
      <div className="agent-profile-loop">
        <div className="agent-profile-loop-copy">
          <span className="section-label">Trace → expert profile</span>
          <strong>Turn this task’s routing into a custom reusable mask.</strong>
          <p>
            Open the full Profile Studio for layer-by-layer manual selection,
            global budgets, retained-mass targets, and multi-workload sources.
          </p>
          <AppLink
            className="primary-button agent-custom-profile-link"
            href={`/profiles/new?trial=${encodeURIComponent(trial.id)}`}
          >
            Open custom Profile Studio
          </AppLink>
        </div>
        {proposal ? (
          <>
            <dl className="agent-profile-stats">
              <div>
                <dt>Observed mass</dt>
                <dd>{formatPercent(proposal.observed_mass_retained)}</dd>
              </div>
              <div>
                <dt>Experts retained</dt>
                <dd>{formatPercent(proposal.validation.retained_fraction)}</dd>
              </div>
              <div>
                <dt>Validation</dt>
                <dd className={proposal.validation.valid ? "valid" : "invalid"}>
                  {proposal.validation.valid ? "Ready" : "Needs review"}
                </dd>
              </div>
            </dl>
            <div className="agent-profile-save">
              <label>
                Profile name
                <input
                  value={profileName}
                  maxLength={120}
                  onChange={(event) => {
                    setProfileName(event.target.value);
                    setSavedProfile(null);
                  }}
                />
              </label>
              <button
                className="primary-button"
                disabled={
                  !proposal.validation.valid ||
                  !profileName.trim() ||
                  saveProfile.isPending ||
                  Boolean(savedProfile)
                }
                onClick={saveProposal}
              >
                {savedProfile
                  ? "Profile saved"
                  : saveProfile.isPending
                    ? "Saving profile…"
                    : "Save profile"}
              </button>
            </div>
          </>
        ) : (
          <button
            className="text-button agent-profile-propose"
            disabled={!routing || proposeProfile.isPending}
            onClick={() => proposeProfile.mutate()}
          >
            {proposeProfile.isPending
              ? "Building profile…"
              : "Quick 64/layer proposal"}
          </button>
        )}
        {savedProfile && (
          <p className="agent-profile-saved" role="status">
            ✓ Saved “{savedProfile.name}” to Profiles
          </p>
        )}
        {profileError && (
          <p className="agent-profile-error" role="alert">
            {(profileError as Error).message}
          </p>
        )}
      </div>
      {explorerData ? (
        <ExpertExplorer data={explorerData} title="Trial engagement" />
      ) : (
        <p className="routing-waiting-copy">
          {trial.routed_inference_calls
            ? "Routing is being aggregated…"
            : "No routing artifacts have been captured for this task yet."}
        </p>
      )}
    </section>
  );
}

function ReadinessRow({
  label,
  detail,
  ready,
}: {
  label: string;
  detail: string;
  ready: boolean;
}) {
  return (
    <div className={ready ? "ready" : "waiting"}>
      <span aria-hidden="true">{ready ? "✓" : "○"}</span>
      <span>
        <strong>{label}</strong>
        <small>{detail}</small>
      </span>
    </div>
  );
}

function isActiveRun(status: AgentRunStatus | undefined) {
  return status === "queued" || status === "running" || status === "cancelling";
}

function isActiveTrial(status: AgentTrialSummary["status"] | undefined) {
  return (
    status === "queued" ||
    status === "provisioning" ||
    status === "running" ||
    status === "verifying" ||
    status === "cleaning"
  );
}

function providerStatusLabel(
  provider: SandboxProvider | null,
  preflight: ProviderPreflight | null,
) {
  if (!provider) return "Choose a provider";
  if (!provider.configured) return "Credential not configured";
  if (preflight?.status === "ready" || provider.status === "ready") {
    const latency = preflight?.latency_ms;
    return latency == null ? "Connection ready" : `Ready · ${Math.round(latency)} ms`;
  }
  if (preflight?.error) return preflight.error.message;
  if (provider.status === "error") return provider.error?.message ?? "Connection failed";
  return "Connection unchecked";
}

function trialStatusLabel(trial: AgentTrialSummary) {
  const labels: Partial<Record<AgentTrialSummary["status"], string>> = {
    queued: "Waiting",
    provisioning: "Creating sandbox",
    running: "Working",
    verifying: "Verifying",
    cleaning: "Cleaning up",
    passed: "Passed",
    failed: "Failed",
    error: "Error",
    cancelled: "Cancelled",
  };
  return labels[trial.status] ?? trial.status;
}

function profileLabel(run: AgentRun) {
  return run.profile_id
    ? `profile ${run.profile_fingerprint?.slice(0, 10) ?? run.profile_id.slice(0, 8)}`
    : "baseline model";
}

function runSummary(run: AgentRun) {
  if (run.status === "completed") {
    return `${run.passed_trials} / ${run.total_trials} tasks passed`;
  }
  if (run.status === "cancelled") return "Run cancelled and cleaned up.";
  if (run.status === "failed") return "The coding run failed.";
  return "The coding run is settling.";
}

function formatCompactNumber(value: number) {
  if (value < 1_000) return String(value);
  return `${(value / 1_000).toFixed(value < 10_000 ? 1 : 0)}k`;
}

function formatMinutes(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.round(seconds / 60)} min`;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value >= 0.995 ? 1 : 0)}%`;
}
