import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { AppLink } from "../AppShell";
import type { JobRecord, ModelSession, SystemStatus } from "../types";
import {
  activateExpertContext,
  listModels,
  listProfiles,
} from "./data";
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
  currentSession,
  onLoadModel,
  status,
}: {
  activeJob: JobRecord | null;
  activeProfileId: string | null;
  currentSession: ModelSession | null;
  onLoadModel: (modelId: string) => Promise<void>;
  status: SystemStatus | null;
}) {
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState(currentSession?.model_id ?? "");
  const models = useQuery({ queryKey: ["v2-models"], queryFn: listModels });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });
  const modelItems = models.data ?? [];
  const selected = modelItems.find((model) => model.id === selectedModelId)
    ?? modelItems.find((model) => model.id === currentSession?.model_id)
    ?? modelItems[0]
    ?? null;
  const load = useMutation({ mutationFn: (modelId: string) => onLoadModel(modelId) });
  const context = useMutation({
    mutationFn: (profileId: string | null) => activateExpertContext(profileId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["model-session-current"] });
      void queryClient.invalidateQueries({ queryKey: ["v2-active-context"] });
    },
  });
  const currentProfile = profiles.data?.find((profile) => profile.id === activeProfileId) ?? null;
  const matchingProfiles = (profiles.data ?? []).filter((profile) => profile.model_id === currentSession?.model_id);
  const loadingWeights = activeJob?.kind === "model_load";

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
                  disabled={!selected.enabled || loadingWeights || load.isPending || currentSession?.model_id === selected.id}
                  onClick={() => load.mutate(selected.id)}
                >
                  {loadingWeights ? "Loading weights…" : currentSession?.model_id === selected.id ? "Resident engine" : "Load model"}
                </button>
              </header>
              {load.error && <V2Error error={load.error} />}
              {loadingWeights && activeJob && (
                <div className="v2-model-load-progress" role="status" aria-live="polite">
                  <span><strong>{humanize(activeJob.status)}</strong><small>Weight lifecycle job {activeJob.id.slice(0, 8)}</small></span>
                  <progress max={Math.max(activeJob.progress_total, 1)} value={activeJob.progress_current} />
                  <em>{activeJob.progress_current}/{activeJob.progress_total || "—"}</em>
                </div>
              )}
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
            <header><div><span className="v2-section-label">Expert context</span><h2>Resident eligibility</h2></div><V2Status value={currentSession ? "ready" : "blocked"} /></header>
            <button
              className={`v2-context-option ${!activeProfileId ? "selected" : ""}`}
              disabled={!currentSession || context.isPending || !activeProfileId}
              onClick={() => context.mutate(null)}
            >
              <span>∞</span><div><strong>Baseline</strong><small>All experts eligible</small></div><em>{!activeProfileId ? "Active" : "Activate"}</em>
            </button>
            {matchingProfiles.map((profile) => (
              <button
                className={`v2-context-option ${activeProfileId === profile.id ? "selected" : ""}`}
                disabled={!currentSession || context.isPending || activeProfileId === profile.id}
                key={profile.id}
                onClick={() => context.mutate(profile.id)}
              >
                <span>⌁</span><div><strong>{profile.name}</strong><small>{formatPercent(profile.validation.retained_fraction)} retained · {profile.profile_fingerprint.slice(0, 10)}</small></div><em>{activeProfileId === profile.id ? "Active" : "Activate"}</em>
              </button>
            ))}
            {!currentSession && <p>Load model weights before activating an expert context.</p>}
            {currentSession && matchingProfiles.length === 0 && <p>No saved profiles target this model. Build one from routing evidence.</p>}
            {context.error && <V2Error error={context.error} />}
            <footer><span>Context changes must report <code>weights_reloaded: false</code>.</span><AppLink href="/profiles">Manage profiles →</AppLink></footer>
          </section>
        </div>
      )}
    </div>
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

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
