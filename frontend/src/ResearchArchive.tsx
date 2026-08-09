import { useMemo, useRef, useState } from "react";

import type {
  BenchmarkRun,
  ComparisonRecord,
  CreateExpertProfileRequest,
  ExpertProfile,
  ModelSession,
  ModelTopology,
  SavedExpertProfile,
} from "./types";

type ArchiveTab = "runs" | "profiles" | "comparisons";

interface ResearchArchiveProps {
  runs: BenchmarkRun[];
  profiles: SavedExpertProfile[];
  comparisons: ComparisonRecord[];
  sessions: ModelSession[];
  modelId: string;
  topology: ModelTopology;
  activeProfileId: string | null;
  busy: boolean;
  onOpenRun: (run: BenchmarkRun, baseline?: BenchmarkRun) => void;
  onOpenComparison: (baseline: BenchmarkRun, candidate: BenchmarkRun) => void;
  onLoadProfile: (profile: SavedExpertProfile) => void;
  onCreateProfile: (
    request: CreateExpertProfileRequest,
  ) => Promise<SavedExpertProfile>;
}

export function ResearchArchive({
  runs,
  profiles,
  comparisons,
  sessions,
  modelId,
  topology,
  activeProfileId,
  busy,
  onOpenRun,
  onOpenComparison,
  onLoadProfile,
  onCreateProfile,
}: ResearchArchiveProps) {
  const [tab, setTab] = useState<ArchiveTab>("runs");
  const [editingProfile, setEditingProfile] =
    useState<SavedExpertProfile | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const sessionById = useMemo(
    () => new Map(sessions.map((session) => [session.id, session])),
    [sessions],
  );
  const runById = useMemo(
    () => new Map(runs.map((run) => [run.id, run])),
    [runs],
  );

  function compatibleBaseline(candidate: BenchmarkRun) {
    const candidateIds = candidate.items.map((item) => item.item_id).join("\0");
    return runs.find((run) => {
      const session = sessionById.get(run.model_session_id);
      return (
        !session?.profile &&
        run.benchmark_id === candidate.benchmark_id &&
        run.items.map((item) => item.item_id).join("\0") === candidateIds
      );
    });
  }

  async function importProfile(file: File | undefined) {
    if (!file) return;
    setImportError(null);
    try {
      const profile = JSON.parse(await file.text()) as ExpertProfile;
      const imported = await onCreateProfile({
        name: file.name.replace(/\.json$/i, "") || "Imported profile",
        description: `Imported from ${file.name}`,
        model_id: modelId,
        profile,
        source: "import",
      });
      setEditingProfile(imported);
      setTab("profiles");
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Import failed");
    } finally {
      if (importInput.current) importInput.current.value = "";
    }
  }

  return (
    <section className="archive-section" id="archive">
      <div className="section-intro archive-intro">
        <div>
          <span className="section-label">05 · Research archive</span>
          <h2>Return to every decision.</h2>
          <p>
            Runs, profiles, and paired comparisons are stored on the mounted
            volume and remain available after the Pod or application restarts.
          </p>
        </div>
        <div className="archive-totals" aria-label="Archive totals">
          <span><strong>{runs.length}</strong> runs</span>
          <span><strong>{profiles.length}</strong> profiles</span>
          <span><strong>{comparisons.length}</strong> comparisons</span>
        </div>
      </div>

      <div className="archive-tabs" role="tablist" aria-label="Research archive">
        {(["runs", "profiles", "comparisons"] as const).map((value) => (
          <button
            key={value}
            role="tab"
            aria-selected={tab === value}
            className={tab === value ? "selected" : ""}
            onClick={() => setTab(value)}
          >
            {value}
          </button>
        ))}
        {tab === "profiles" && (
          <div className="archive-tab-actions">
            <input
              ref={importInput}
              type="file"
              accept="application/json,.json"
              aria-label="Import expert profile JSON"
              onChange={(event) => void importProfile(event.target.files?.[0])}
              hidden
            />
            <button className="text-button" onClick={() => importInput.current?.click()}>
              Import profile
            </button>
          </div>
        )}
      </div>

      {importError && <div className="archive-error">{importError}</div>}

      {tab === "runs" && (
        <div className="archive-grid">
          {runs.map((run) => {
            const session = sessionById.get(run.model_session_id);
            const profiled = Boolean(session?.profile);
            const baseline = profiled ? compatibleBaseline(run) : undefined;
            return (
              <article className="archive-card" key={run.id}>
                <div className="archive-card-topline">
                  <span className={`artifact-kind ${profiled ? "masked" : "base"}`}>
                    {profiled ? "Profiled" : "Baseline"}
                  </span>
                  <time dateTime={run.created_at}>{formatDate(run.created_at)}</time>
                </div>
                <strong className="archive-score">{Math.round(run.score * 100)}%</strong>
                <h3>{run.benchmark_id}</h3>
                <p>
                  {run.total_items} items · {shortId(run.id)}
                  {profiled && !baseline ? " · no matching baseline" : ""}
                </p>
                <button
                  className="text-button"
                  onClick={() => onOpenRun(run, baseline)}
                >
                  {profiled && baseline ? "Open paired run" : "Open run"} →
                </button>
              </article>
            );
          })}
          {runs.length === 0 && <EmptyArtifact noun="runs" />}
        </div>
      )}

      {tab === "profiles" && !editingProfile && (
        <div className="archive-grid profile-grid">
          {profiles.map((profile) => (
            <article className="archive-card profile-card" key={profile.id}>
              <div className="archive-card-topline">
                <span className="artifact-kind profile">{profile.source}</span>
                <time dateTime={profile.created_at}>{formatDate(profile.created_at)}</time>
              </div>
              <h3>{profile.name}</h3>
              <p>{profile.description || "No research note attached."}</p>
              <dl>
                <div>
                  <dt>Eligible</dt>
                  <dd>{Math.round(profile.validation.retained_fraction * 100)}%</dd>
                </div>
                <div>
                  <dt>Fingerprint</dt>
                  <dd>{profile.profile_fingerprint.slice(0, 10)}</dd>
                </div>
              </dl>
              {activeProfileId === profile.id && (
                <span className="active-artifact">Loaded in runtime</span>
              )}
              <div className="artifact-actions">
                <button
                  className="primary-button"
                  disabled={busy || activeProfileId === profile.id}
                  onClick={() => onLoadProfile(profile)}
                >
                  {activeProfileId === profile.id ? "Loaded" : "Load profile"}
                </button>
                <button className="text-button" onClick={() => setEditingProfile(profile)}>
                  Revise
                </button>
                <a href={`/api/profiles/${profile.id}/export`} download>
                  Export JSON
                </a>
              </div>
            </article>
          ))}
          {profiles.length === 0 && <EmptyArtifact noun="profiles" />}
        </div>
      )}

      {tab === "profiles" && editingProfile && (
        <ProfileEditor
          key={editingProfile.id}
          profile={editingProfile}
          topology={topology}
          busy={busy}
          onCancel={() => setEditingProfile(null)}
          onSave={async (request) => {
            const saved = await onCreateProfile(request);
            setEditingProfile(saved);
          }}
        />
      )}

      {tab === "comparisons" && (
        <div className="comparison-archive-list">
          {comparisons.map((comparison) => {
            const baseline = runById.get(comparison.baseline_run_id);
            const candidate = runById.get(comparison.candidate_run_id);
            return (
              <article className="comparison-archive-row" key={comparison.id}>
                <div>
                  <span className="artifact-kind comparison">paired</span>
                  <h3>{comparison.name}</h3>
                  <p>
                    {comparison.cohort_item_ids.length} items · profile {comparison.profile_fingerprint.slice(0, 10)}
                  </p>
                </div>
                <ArchiveMetric label="Base" value={`${Math.round(comparison.baseline_score * 100)}%`} />
                <ArchiveMetric label="Masked" value={`${Math.round(comparison.candidate_score * 100)}%`} />
                <ArchiveMetric
                  label="Delta"
                  value={`${comparison.score_delta >= 0 ? "+" : ""}${Math.round(comparison.score_delta * 100)} pp`}
                  tone={comparison.score_delta < 0 ? "negative" : "positive"}
                />
                <ArchiveMetric label="Regressions" value={String(comparison.regressions)} tone={comparison.regressions ? "negative" : "neutral"} />
                <button
                  className="text-button"
                  disabled={!baseline || !candidate}
                  onClick={() => baseline && candidate && onOpenComparison(baseline, candidate)}
                >
                  Open comparison →
                </button>
              </article>
            );
          })}
          {comparisons.length === 0 && <EmptyArtifact noun="comparisons" />}
        </div>
      )}
    </section>
  );
}

function ProfileEditor({
  profile,
  topology,
  busy,
  onCancel,
  onSave,
}: {
  profile: SavedExpertProfile;
  topology: ModelTopology;
  busy: boolean;
  onCancel: () => void;
  onSave: (request: CreateExpertProfileRequest) => Promise<void>;
}) {
  const [draft, setDraft] = useState<ExpertProfile>(() =>
    structuredClone(profile.profile),
  );
  const [selectedLayer, setSelectedLayer] = useState(topology.routed_layer_ids[0]);
  const [name, setName] = useState(`${profile.name} · revision`);
  const [description, setDescription] = useState(profile.description);
  const [error, setError] = useState<string | null>(null);
  const keep = draft.layers[String(selectedLayer)]?.keep ?? [];
  const keepSet = new Set(keep);
  const layerValid = keep.length >= topology.top_k;
  const allLayersValid = topology.routed_layer_ids.every(
    (layerId) => (draft.layers[String(layerId)]?.keep.length ?? 0) >= topology.top_k,
  );

  function replaceLayer(nextKeep: number[]) {
    setDraft((current) => ({
      ...current,
      layers: {
        ...current.layers,
        [String(selectedLayer)]: { keep: [...nextKeep].sort((a, b) => a - b) },
      },
    }));
  }

  function toggleExpert(expertId: number) {
    replaceLayer(
      keepSet.has(expertId)
        ? keep.filter((value) => value !== expertId)
        : [...keep, expertId],
    );
  }

  async function saveRevision() {
    setError(null);
    try {
      await onSave({
        name,
        description,
        model_id: profile.model_id,
        profile: draft,
        source: "manual",
        parent_profile_id: profile.id,
      });
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Save failed");
    }
  }

  return (
    <div className="profile-editor">
      <header>
        <div>
          <span className="section-label">Immutable revision</span>
          <h3>Edit expert eligibility</h3>
          <p>
            Toggle experts for one routed layer at a time. Saving creates a new
            profile linked to <em>{profile.name}</em>; prior runs remain unchanged.
          </p>
        </div>
        <button className="text-button" onClick={onCancel}>Close editor</button>
      </header>

      <div className="editor-fields">
        <label>
          Revision name
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label>
          Research note
          <input value={description} onChange={(event) => setDescription(event.target.value)} />
        </label>
        <label>
          Routed layer
          <select
            value={selectedLayer}
            onChange={(event) => setSelectedLayer(Number(event.target.value))}
          >
            {topology.routed_layer_ids.map((layerId) => (
              <option key={layerId} value={layerId}>Layer {layerId}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="editor-toolbar">
        <span>
          <strong>{keep.length}</strong> / {topology.num_experts} eligible
        </span>
        <button className="text-button" onClick={() => replaceLayer([...Array(topology.num_experts).keys()])}>
          Select all
        </button>
        <button
          className="text-button"
          onClick={() => replaceLayer(profile.profile.layers[String(selectedLayer)].keep)}
        >
          Revert layer
        </button>
        <button
          className="text-button"
          disabled={selectedLayer === topology.routed_layer_ids[0]}
          onClick={() => {
            const position = topology.routed_layer_ids.indexOf(selectedLayer);
            const previous = topology.routed_layer_ids[position - 1];
            replaceLayer(draft.layers[String(previous)].keep);
          }}
        >
          Copy previous layer
        </button>
      </div>

      <div className="expert-picker" aria-label={`Experts for layer ${selectedLayer}`}>
        {[...Array(topology.num_experts).keys()].map((expertId) => (
          <button
            key={expertId}
            aria-pressed={keepSet.has(expertId)}
            className={keepSet.has(expertId) ? "kept" : "masked"}
            onClick={() => toggleExpert(expertId)}
          >
            {expertId}
          </button>
        ))}
      </div>

      <footer className="editor-footer">
        <span className={layerValid && allLayersValid ? "valid" : "invalid"}>
          {layerValid && allLayersValid
            ? `Valid top-${topology.top_k} floor across all layers`
            : `Each layer must keep at least ${topology.top_k} experts`}
        </span>
        {error && <span className="archive-error">{error}</span>}
        <button
          className="primary-button"
          disabled={busy || !name.trim() || !allLayersValid}
          onClick={() => void saveRevision()}
        >
          Save new revision
        </button>
      </footer>
    </div>
  );
}

function EmptyArtifact({ noun }: { noun: string }) {
  return (
    <div className="empty-artifact">
      <span>◇</span>
      <p>No {noun} yet. The first one will appear here automatically.</p>
    </div>
  );
}

function ArchiveMetric({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "positive" | "negative";
}) {
  return (
    <div className={`archive-metric ${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function shortId(id: string) {
  return id.slice(0, 8);
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
