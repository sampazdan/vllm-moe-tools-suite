import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { AppLink } from "../AppShell";
import type {
  JobPhaseRecord,
  JobRecord,
  ModelSession,
  RuntimeStatus,
  SystemStatus,
} from "../types";
import {
  activateExpertContext,
  listModelLoadJobs,
  listModelSessions,
  listModels,
  listProfiles,
} from "./data";
import {
  canRetryModelLoad,
  currentModelLoadPhase,
  failedModelLoadPhase,
  failedModelLoadStage,
  formatDuration,
  modelLoadElapsedMs,
  modelLoadPhaseElapsedMs,
  phaseCounters,
} from "./modelLoad";
import type { ModelDescriptor } from "./types";
import {
  V2Error,
  V2Loading,
  V2Metric,
  V2PageHeader,
  V2Status,
} from "./V2Shell";

export function ModelsPage({
  activeJob,
  activeProfileId,
  connectionIssue,
  currentSession,
  onCancelJob,
  onLoadModel,
  onRetryModelLoad,
  runtime,
  status,
}: {
  activeJob: JobRecord | null;
  activeProfileId: string | null;
  connectionIssue: string | null;
  currentSession: ModelSession | null;
  onCancelJob: (jobId: string) => Promise<void>;
  onLoadModel: (modelId: string) => Promise<void>;
  onRetryModelLoad: (jobId: string) => Promise<void>;
  runtime: RuntimeStatus | null;
  status: SystemStatus | null;
}) {
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState(currentSession?.model_id ?? "");
  const [selectedLoadId, setSelectedLoadId] = useState<string | null>(null);
  const models = useQuery({ queryKey: ["v2-models"], queryFn: listModels });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });
  const sessions = useQuery({
    queryKey: ["model-sessions"],
    queryFn: listModelSessions,
    refetchInterval: activeJob?.kind === "model_load" ? 1_000 : false,
  });
  const loads = useQuery({
    queryKey: ["v2-model-loads"],
    queryFn: listModelLoadJobs,
    refetchInterval: activeJob?.kind === "model_load" ? 1_000 : false,
  });
  const modelItems = models.data ?? [];
  const selected = modelItems.find((model) => model.id === selectedModelId)
    ?? modelItems.find((model) => model.id === currentSession?.model_id)
    ?? modelItems[0]
    ?? null;
  const activeLoad = activeJob?.kind === "model_load" ? activeJob : null;
  const loadHistory = mergeActiveLoad(activeLoad, loads.data ?? []);
  const selectedLoad = loadHistory.find((job) => job.id === selectedLoadId)
    ?? activeLoad
    ?? loadHistory[0]
    ?? null;
  const selectedLoadSession = (sessions.data ?? []).find(
    (session) => session.id === selectedLoad?.result_id,
  ) ?? null;
  const conflictingJob = activeJob && activeJob.kind !== "model_load" ? activeJob : null;
  const loadingWeights = activeLoad !== null;
  const load = useMutation({
    mutationFn: (modelId: string) => onLoadModel(modelId),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ["v2-model-loads"] }),
  });
  const cancel = useMutation({
    mutationFn: (jobId: string) => onCancelJob(jobId),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ["v2-model-loads"] }),
  });
  const retry = useMutation({
    mutationFn: (jobId: string) => onRetryModelLoad(jobId),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ["v2-model-loads"] }),
  });
  const context = useMutation({
    mutationFn: (profileId: string | null) => activateExpertContext(profileId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["model-session-current"] });
      void queryClient.invalidateQueries({ queryKey: ["v2-active-context"] });
    },
  });
  const currentProfile = profiles.data?.find((profile) => profile.id === activeProfileId) ?? null;
  const matchingProfiles = (profiles.data ?? []).filter((profile) => profile.model_id === currentSession?.model_id);
  const residentModelReady = currentSession?.state === "ready";
  const locationHash = window.location.hash;

  useEffect(() => {
    if (!selectedModelId && currentSession?.model_id) {
      setSelectedModelId(currentSession.model_id);
    }
  }, [currentSession?.model_id, selectedModelId]);

  useEffect(() => {
    if (activeLoad) setSelectedLoadId(activeLoad.id);
  }, [activeLoad?.id]);

  useEffect(() => {
    if (locationHash !== "#active-model-load" || !selectedLoad) return;
    const frame = window.requestAnimationFrame(() => {
      document.getElementById("active-model-load")?.scrollIntoView({ block: "start" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [locationHash, selectedLoad?.id]);

  return (
    <div className="v2-page v2-models-page">
      <V2PageHeader
        eyebrow="Models"
        title="Warm weights once. Switch expert context quickly."
        description="Model loading and expert-mask activation are separate operations with different costs, compatibility rules, and provenance."
        meta={
          <>
            <V2Status value={status?.model_state ?? "unloaded"} />
            <span>{currentSession?.model_id ?? "No resident engine"}</span>
            <span>{currentProfile?.name ?? (currentSession?.profile_id ? "Custom expert context" : "Baseline context")}</span>
          </>
        }
      />

      <section className="v2-lifecycle-map" aria-label="Runtime lifecycle">
        <article className="expensive">
          <span>Minutes · expensive</span>
          <strong>01 Load model weights</strong>
          <p>Starts or replaces the vLLM process, allocates accelerators, and warms the selected model.</p>
          <em>POST /api/model-sessions</em>
        </article>
        <span aria-hidden="true">→</span>
        <article className="fast">
          <span>Milliseconds · hot</span>
          <strong>02 Activate expert context</strong>
          <p>Changes expert eligibility for the same resident weights. The process ID should remain unchanged.</p>
          <em>POST /api/expert-contexts/activate</em>
        </article>
        <span aria-hidden="true">→</span>
        <article>
          <span>Durable evidence</span>
          <strong>03 Run paired lanes</strong>
          <p>Experiments persist both profile and context fingerprints so each result remains attributable.</p>
          <AppLink href="/experiments/new">Create experiment →</AppLink>
        </article>
      </section>

      {conflictingJob && (
        <section className="v2-active-job-recovery" role="status">
          <div>
            <span className="v2-section-label">Active work adopted</span>
            <strong>{jobKindLabel(conflictingJob)} is {humanize(conflictingJob.status)}</strong>
            <p>A second model load was not started. Open or safely cancel the durable job, then retry.</p>
          </div>
          <AppLink className="v2-secondary-button" href={activeWorkHref(conflictingJob)}>Open active job</AppLink>
          <button
            className="v2-secondary-button"
            disabled={cancel.isPending || conflictingJob.status === "cancelling"}
            onClick={() => cancel.mutate(conflictingJob.id)}
          >
            {conflictingJob.status === "cancelling" ? "Cancelling…" : "Cancel active job"}
          </button>
        </section>
      )}

      {(models.isPending || profiles.isPending) ? (
        <V2Loading label="Reading the model and context registry…" />
      ) : models.error || profiles.error ? (
        <V2Error error={models.error ?? profiles.error} />
      ) : (
        <div className="v2-model-console">
          <section className="v2-model-registry">
            <header><div><span className="v2-section-label">Model registry</span><h2>Qualified engines</h2></div><strong>{modelItems.length}</strong></header>
            <div>
              {modelItems.map((model) => (
                <button
                  className={selected?.id === model.id ? "selected" : ""}
                  key={model.id}
                  onClick={() => setSelectedModelId(model.id)}
                >
                  <span className={currentSession?.model_id === model.id ? "running" : ""} />
                  <div><strong>{model.display_name}</strong><small>{model.id}</small></div>
                  <V2Status value={model.qualification_status ?? (model.enabled ? "ready" : "unsupported")} />
                </button>
              ))}
            </div>
          </section>

          {selected && (
            <section className="v2-model-detail">
              <header>
                <div><span className="v2-section-label">Selected weights</span><h2>{selected.display_name}</h2><code>{selected.id}</code></div>
                <button
                  className="v2-primary-button"
                  disabled={!selected.enabled || activeJob !== null || load.isPending || (residentModelReady && currentSession?.model_id === selected.id)}
                  onClick={() => load.mutate(selected.id)}
                >
                  {loadingWeights ? "Loading weights…" : activeJob ? "Background work active" : residentModelReady && currentSession?.model_id === selected.id ? "Resident engine" : "Load model"}
                </button>
              </header>
              {load.error && <V2Error error={load.error} />}
              <div className="v2-model-facts">
                <V2Metric label="Routed layers" value={selected.topology ? String(selected.topology.routed_layer_ids.length) : "Unverified"} detail={selected.topology ? `${selected.topology.num_layers} total layers` : "Loaded topology not discovered"} />
                <V2Metric label="Experts / layer" value={selected.topology ? String(selected.topology.num_experts) : "Unverified"} detail={selected.topology ? `${selected.topology.top_k} active per token` : "Fail closed until qualified"} />
                <V2Metric label="Precision" value={selected.dtype ?? "Registry default"} detail={selected.quantization ?? "No quantization reported"} />
                <V2Metric label="Minimum memory" value={selected.minimum_memory_gib == null ? "Unreported" : `${selected.minimum_memory_gib} GiB`} detail={selected.recommended_hardware} />
              </div>
              <p>{selected.notes || "No registry notes supplied."}</p>
              <div className="v2-capability-grid">
                <Capability label="Expert masking" capability={selected.masking} />
                <Capability label="Routing telemetry" capability={selected.routing_telemetry} />
                <Capability label="Hot context switch" capability={selected.hot_switch} />
              </div>
              {selected.failure_reason && <div className="v2-context-warning"><strong>Qualification issue</strong><span>{selected.failure_reason}</span></div>}
            </section>
          )}

          <section className="v2-context-console">
            <header><div><span className="v2-section-label">Expert context</span><h2>Resident eligibility</h2></div><V2Status value={currentSession?.state === "ready" ? "ready" : "blocked"} /></header>
            <button
              className={`v2-context-option ${!activeProfileId ? "selected" : ""}`}
              disabled={currentSession?.state !== "ready" || context.isPending || !activeProfileId}
              onClick={() => context.mutate(null)}
            >
              <span>∞</span><div><strong>Baseline</strong><small>All experts eligible</small></div><em>{!activeProfileId ? "Active" : "Activate"}</em>
            </button>
            {matchingProfiles.map((profile) => (
              <button
                className={`v2-context-option ${activeProfileId === profile.id ? "selected" : ""}`}
                disabled={currentSession?.state !== "ready" || context.isPending || activeProfileId === profile.id}
                key={profile.id}
                onClick={() => context.mutate(profile.id)}
              >
                <span>⌁</span><div><strong>{profile.name}</strong><small>{formatPercent(profile.validation.retained_fraction)} retained · {profile.profile_fingerprint.slice(0, 10)}</small></div><em>{activeProfileId === profile.id ? "Active" : "Activate"}</em>
              </button>
            ))}
            {currentSession?.state !== "ready" && <p>Load model weights before activating an expert context.</p>}
            {currentSession?.state === "ready" && matchingProfiles.length === 0 && <p>No saved profiles target this model. Build one from routing evidence.</p>}
            {context.error && <V2Error error={context.error} />}
            <footer><span>Context changes must report <code>weights_reloaded: false</code>.</span><AppLink href="/profiles">Manage profiles →</AppLink></footer>
          </section>
        </div>
      )}

      {loads.isPending || sessions.isPending ? (
        <V2Loading label="Recovering durable model-load history…" />
      ) : loads.error || sessions.error ? (
        <V2Error error={loads.error ?? sessions.error} />
      ) : selectedLoad ? (
        <ModelLoadLifecycle
          cancelling={cancel.isPending}
          connectionIssue={connectionIssue}
          job={selectedLoad}
          jobs={loadHistory}
          onCancel={() => cancel.mutate(selectedLoad.id)}
          onRetry={() => retry.mutate(selectedLoad.id)}
          onSelect={setSelectedLoadId}
          retryError={retry.error}
          retrying={retry.isPending}
          runtime={runtime}
          session={selectedLoadSession}
        />
      ) : (
        <section className="v2-model-load-empty">
          <span className="v2-section-label">Durable load history</span>
          <h2>No model-load attempts yet.</h2>
          <p>Select a qualified model and start a load. Its lifecycle will remain available after refresh.</p>
        </section>
      )}
    </div>
  );
}

function ModelLoadLifecycle({
  cancelling,
  connectionIssue,
  job,
  jobs,
  onCancel,
  onRetry,
  onSelect,
  retryError,
  retrying,
  runtime,
  session,
}: {
  cancelling: boolean;
  connectionIssue: string | null;
  job: JobRecord;
  jobs: JobRecord[];
  onCancel: () => void;
  onRetry: () => void;
  onSelect: (jobId: string) => void;
  retryError: unknown;
  retrying: boolean;
  runtime: RuntimeStatus | null;
  session: ModelSession | null;
}) {
  const running = ["queued", "running", "cancelling"].includes(job.status);
  const now = useClock(running);
  const currentPhase = currentModelLoadPhase(job);
  const failedPhase = failedModelLoadPhase(job);
  const failedStage = failedModelLoadStage(job);
  const diagnostics = [
    failedPhase?.diagnostics,
    runtime?.session_id === job.result_id ? runtime.log_tail : null,
  ].filter((value, index, values): value is string => (
    Boolean(value) && values.indexOf(value) === index
  )).join("\n\n--- Current managed runtime log tail ---\n");

  return (
    <section className="v2-model-load-lifecycle" id="active-model-load">
      <header>
        <div>
          <span className="v2-section-label">Durable model-load lifecycle</span>
          <h2>{session?.model_id ?? "Model-load attempt"}</h2>
          <code>{job.id}</code>
        </div>
        <div className="v2-model-load-summary">
          <V2Status value={job.status} />
          <strong>{humanize(currentPhase?.phase ?? job.status)}</strong>
          <span>{formatDuration(modelLoadElapsedMs(job, now))} elapsed</span>
        </div>
      </header>

      <nav className="v2-model-load-history" aria-label="Model-load attempts">
        {jobs.slice(0, 8).map((attempt) => (
          <button
            className={attempt.id === job.id ? "selected" : ""}
            key={attempt.id}
            onClick={() => onSelect(attempt.id)}
          >
            <strong>{formatTime(attempt.created_at)}</strong>
            <span>{humanize(attempt.status)}</span>
            <code>{attempt.id.slice(0, 8)}</code>
          </button>
        ))}
      </nav>

      {running && (
        <div className="v2-model-load-progress" role="status" aria-live="polite">
          <span><strong>{humanize(currentPhase?.phase ?? job.status)}</strong><small>Durable job adopted · safe to refresh this page</small></span>
          <progress max={Math.max(job.progress_total, 1)} value={job.progress_current} />
          <em>{job.progress_current}/{job.progress_total || "—"}</em>
          <button
            className="v2-secondary-button"
            disabled={cancelling || job.status === "cancelling"}
            onClick={onCancel}
          >
            {job.status === "cancelling" ? "Cleaning up…" : cancelling ? "Requesting cancel…" : "Cancel safely"}
          </button>
        </div>
      )}
      {connectionIssue && <div className="v2-context-warning"><strong>Connection interrupted</strong><span>{connectionIssue}. The durable backend load continues.</span></div>}
      {failedPhase && (
        <div className="v2-model-load-failure" role="alert">
          <div><span>{failedStage ? `Failed during ${humanize(failedStage.phase)}` : humanize(failedPhase.failure_code ?? "model_load_failed")}</span><strong>{failedPhase.detail ?? job.error ?? "Model loading failed."}</strong><p>{failedPhase.recovery_action ?? "Inspect diagnostics before retrying."}</p></div>
          {canRetryModelLoad(job) ? (
            <button className="v2-primary-button" disabled={retrying} onClick={onRetry}>{retrying ? "Retrying…" : "Retry same pinned model"}</button>
          ) : (
            <em>Automatic retry is disabled until cleanup safety is restored.</em>
          )}
        </div>
      )}
      {job.status === "cancelled" && (
        <div className="v2-model-load-cancelled">
          <div><strong>Load cancelled safely.</strong><span>The managed process cleanup completed; no ready engine was adopted.</span></div>
          <button className="v2-primary-button" disabled={retrying} onClick={onRetry}>{retrying ? "Retrying…" : "Retry same pinned model"}</button>
        </div>
      )}
      {retryError != null && <V2Error error={retryError} />}

      <ol className="v2-model-load-timeline">
        {(job.phase_history ?? []).map((record, index) => (
          <ModelLoadPhaseRow key={`${record.phase}-${record.started_at}-${index}`} now={now} record={record} />
        ))}
      </ol>

      <details className="v2-model-load-diagnostics">
        <summary>Diagnostics and launcher logs</summary>
        <p>Raw failure detail and the current managed-runtime tail stay collapsed so the recovery action remains readable.</p>
        <pre>{diagnostics || "No diagnostic output is available for this load."}</pre>
      </details>
    </section>
  );
}

function ModelLoadPhaseRow({
  now,
  record,
}: {
  now: number;
  record: JobPhaseRecord;
}) {
  const counters = phaseCounters(record);
  return (
    <li className={`${record.status} ${record.observability}`}>
      <span aria-hidden="true" />
      <div>
        <strong>{humanize(record.phase)}</strong>
        <p>{record.detail ?? "No phase detail was reported."}</p>
        {counters && <small>{counters}</small>}
      </div>
      <aside>
        <V2Status value={record.status} />
        <time>{formatDuration(modelLoadPhaseElapsedMs(record, now))}</time>
        <small>{record.source === "managed_runtime" ? "Managed runtime" : record.source === "application" ? "Application" : "Legacy source unknown"}</small>
      </aside>
    </li>
  );
}

function Capability({ label, capability }: { label: string; capability: ModelDescriptor["masking"] }) {
  const status = capability?.status ?? "manifest_only";
  return (
    <article>
      <header><strong>{label}</strong><V2Status value={status} /></header>
      <p>{capability?.reason ?? "This backend registry has not reported capability evidence."}</p>
    </article>
  );
}

function mergeActiveLoad(active: JobRecord | null, history: JobRecord[]) {
  if (!active) return history;
  return [active, ...history.filter((job) => job.id !== active.id)];
}

function activeWorkHref(job: JobRecord) {
  if (job.result_id && job.kind === "experiment_run") {
    return `/experiments/${encodeURIComponent(job.result_id)}`;
  }
  if (job.result_id && job.kind === "agent_run") {
    return `/agent-runs/${encodeURIComponent(job.result_id)}`;
  }
  if (job.result_id && job.kind === "benchmark_run") {
    return `/runs/${encodeURIComponent(job.result_id)}`;
  }
  return "/experiments";
}

function jobKindLabel(job: JobRecord) {
  if (job.kind === "experiment_run") return "Experiment";
  if (job.kind === "dataset_prepare") return "Dataset preparation";
  if (job.kind === "agent_run") return "Coding run";
  return "Benchmark run";
}

function useClock(active: boolean) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) {
      setNow(Date.now());
      return;
    }
    const interval = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(interval);
  }, [active]);
  return now;
}

function formatTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unknown time"
    : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
