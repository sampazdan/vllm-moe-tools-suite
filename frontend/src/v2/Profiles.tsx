import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { AppLink } from "../AppShell";
import type { ModelSession, SavedExpertProfile } from "../types";
import {
  activateExpertContext,
  getProfile,
  listProfiles,
  updateProfileMetadata,
} from "./data";
import { v2Href } from "./router";
import {
  V2Empty,
  V2Error,
  V2Loading,
  V2Metric,
  V2PageHeader,
  V2Status,
} from "./V2Shell";

export function ProfilesPage({
  activeProfileId,
  currentSession,
}: {
  activeProfileId: string | null;
  currentSession: ModelSession | null;
}) {
  const [search, setSearch] = useState("");
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });
  const filtered = (profiles.data ?? []).filter((profile) =>
    `${profile.name} ${profile.description} ${profile.model_id} ${profile.source}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );

  return (
    <div className="v2-page v2-profiles-page">
      <V2PageHeader
        eyebrow="Profiles"
        title="Name the intervention, preserve the evidence."
        description="Names and descriptions are mutable metadata. Expert eligibility, model identity, lineage, and fingerprints remain authoritative."
        meta={
          <>
            <span>{profiles.data?.length ?? 0} saved profiles</span>
            <span>{activeProfileId ? "1 profile context active" : "Baseline context active"}</span>
          </>
        }
        actions={
          <AppLink className="v2-primary-button" href="/profiles/new">
            Build profile
          </AppLink>
        }
      />

      <div className="v2-context-explainer">
        <span>Fast operation</span>
        <div><strong>Activate expert context</strong><p>Change eligible experts on the resident model process. No weight reload.</p></div>
        <AppLink href="/models">Manage warm engine →</AppLink>
      </div>

      <div className="v2-toolbar">
        <input
          aria-label="Search profiles"
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search names, model, provenance…"
          type="search"
          value={search}
        />
      </div>

      {profiles.isPending ? (
        <V2Loading label="Loading named expert contexts…" />
      ) : profiles.error ? (
        <V2Error error={profiles.error} />
      ) : filtered.length === 0 ? (
        <V2Empty
          title={profiles.data?.length ? "No profiles match." : "No profiles yet."}
          detail={profiles.data?.length ? "Try another search phrase." : "Use routing evidence to build a named, validated expert-eligibility profile."}
          action={<AppLink className="v2-primary-button" href="/profiles/new">Open Profile Studio</AppLink>}
        />
      ) : (
        <div className="v2-profile-grid">
          {filtered.map((profile) => (
            <ProfileCard
              active={activeProfileId === profile.id}
              currentSession={currentSession}
              key={profile.id}
              profile={profile}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ProfileCard({
  active,
  currentSession,
  profile,
}: {
  active: boolean;
  currentSession: ModelSession | null;
  profile: SavedExpertProfile;
}) {
  const queryClient = useQueryClient();
  const activation = useMutation({
    mutationFn: () => activateExpertContext(profile.id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["model-session-current"] });
      void queryClient.invalidateQueries({ queryKey: ["v2-active-context"] });
    },
  });
  const compatible = currentSession?.model_id === profile.model_id;
  return (
    <article className={`v2-profile-card ${active ? "active" : ""}`}>
      <header>
        <span className="v2-profile-glyph" aria-hidden="true">⌁</span>
        <div>
          <InlineProfileName profile={profile} />
          <small>{profile.model_id}</small>
        </div>
        <V2Status value={active ? "active" : profile.validation.valid ? "ready" : "failed"} />
      </header>
      <p>{profile.description || "No description supplied."}</p>
      <div className="v2-profile-retention">
        <span style={{ "--retained": `${profile.validation.retained_fraction * 100}%` } as React.CSSProperties} />
        <small>{formatPercent(profile.validation.retained_fraction)} experts retained</small>
      </div>
      <dl>
        <div><dt>Eligible</dt><dd>{profile.validation.eligible_experts}/{profile.validation.total_experts}</dd></div>
        <div><dt>Source</dt><dd>{humanize(profile.source)}</dd></div>
        <div><dt>Metric</dt><dd>{humanize(profile.metric ?? "manual")}</dd></div>
        <div><dt>Fingerprint</dt><dd><code>{profile.profile_fingerprint.slice(0, 12)}</code></dd></div>
      </dl>
      {activation.error && <V2Error error={activation.error} />}
      <footer>
        <AppLink className="v2-secondary-button" href={v2Href("profile", profile.id)}>Inspect</AppLink>
        <button
          className="v2-primary-button"
          disabled={active || !compatible || activation.isPending}
          onClick={() => activation.mutate()}
          title={!currentSession ? "Load this profile's model first" : !compatible ? "This profile belongs to a different model" : undefined}
        >
          {activation.isPending ? "Activating…" : active ? "Context active" : "Activate context"}
        </button>
      </footer>
    </article>
  );
}

export function ProfileDetailPage({
  activeProfileId,
  currentSession,
  profileId,
}: {
  activeProfileId: string | null;
  currentSession: ModelSession | null;
  profileId: string;
}) {
  const queryClient = useQueryClient();
  const profile = useQuery({
    queryKey: ["profiles", profileId],
    queryFn: () => getProfile(profileId),
  });
  const activation = useMutation({
    mutationFn: (id: string | null) => activateExpertContext(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["model-session-current"] });
      void queryClient.invalidateQueries({ queryKey: ["v2-active-context"] });
    },
  });
  if (profile.isPending) return <div className="v2-page"><V2Loading label="Opening profile evidence…" /></div>;
  if (profile.error || !profile.data) return <div className="v2-page"><V2Error error={profile.error ?? new Error("Profile not found")} /></div>;
  const item = profile.data;
  const active = activeProfileId === item.id;
  const compatible = currentSession?.model_id === item.model_id;
  const layerRows = Object.entries(item.profile.layers)
    .map(([layer, selection]) => ({ layer: Number(layer), keep: selection.keep.length }))
    .sort((left, right) => left.layer - right.layer);
  const maxKeep = Math.max(1, ...layerRows.map((row) => row.keep));

  return (
    <div className="v2-page v2-profile-detail">
      <V2PageHeader
        eyebrow="Expert profile"
        title={item.name}
        description={item.description || "Validated expert eligibility context."}
        meta={<><V2Status value={active ? "active" : item.validation.valid ? "ready" : "failed"} /><span>{item.model_id}</span><span>created {formatDate(item.created_at)}</span></>}
        actions={
          <>
            <a className="v2-secondary-button" download href={`/api/profiles/${encodeURIComponent(item.id)}/export`}>Export JSON</a>
            <AppLink className="v2-secondary-button" href={`/profiles/new?profile=${encodeURIComponent(item.id)}`}>Revise in Studio</AppLink>
            <button
              className={active ? "v2-danger-button" : "v2-primary-button"}
              disabled={!compatible || activation.isPending}
              onClick={() => activation.mutate(active ? null : item.id)}
            >
              {activation.isPending ? "Switching…" : active ? "Return to baseline" : "Activate context"}
            </button>
          </>
        }
      />
      {activation.error && <V2Error error={activation.error} />}
      {!compatible && (
        <div className="v2-context-warning">
          <strong>Load the matching model first.</strong>
          <span>This profile targets {item.model_id}; expert contexts cannot cross model weights.</span>
          <AppLink href="/models">Open Models</AppLink>
        </div>
      )}
      <div className="v2-profile-detail-grid">
        <section className="v2-profile-identity-card">
          <span className="v2-section-label">Editable identity</span>
          <InlineProfileName profile={item} large />
          <p>{item.description || "No description supplied."}</p>
          <small>Renaming calls PATCH /api/profiles/{item.id}. It never changes the profile fingerprint.</small>
        </section>
        <section className="v2-profile-proof-card">
          <span className="v2-section-label">Immutable provenance</span>
          <dl>
            <div><dt>Fingerprint</dt><dd><code>{item.profile_fingerprint}</code></dd></div>
            <div><dt>Source</dt><dd>{humanize(item.source)}</dd></div>
            <div><dt>Parent</dt><dd>{item.parent_profile_id ?? "None"}</dd></div>
            <div><dt>Evidence source</dt><dd>{item.source_fingerprint ?? item.source_run_id ?? "Manual selection"}</dd></div>
          </dl>
        </section>
      </div>
      <section className="v2-profile-overview">
        <V2Metric label="Eligible experts" value={String(item.validation.eligible_experts)} detail={`of ${item.validation.total_experts}`} />
        <V2Metric label="Retained" value={formatPercent(item.validation.retained_fraction)} />
        <V2Metric label="Observed mass" value={formatPercent(item.observed_mass_retained)} detail="when evidence is available" />
        <V2Metric label="Routed layers" value={String(layerRows.length)} />
      </section>
      <section className="v2-layer-profile">
        <header><div><span className="v2-section-label">Layer eligibility</span><h2>Retained experts by routed layer</h2></div><span>Count, not routing frequency</span></header>
        <div>
          {layerRows.map((row) => (
            <article key={row.layer}>
              <span>L{row.layer}</span>
              <div><i style={{ width: `${row.keep / maxKeep * 100}%` }} /></div>
              <strong>{row.keep}</strong>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function InlineProfileName({ profile, large = false }: { profile: SavedExpertProfile; large?: boolean }) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(profile.name);
  const duplicate = Boolean(
    queryClient.getQueryData<SavedExpertProfile[]>(["profiles"])?.some(
      (item) => item.id !== profile.id && item.name.trim().toLowerCase() === draft.trim().toLowerCase(),
    ),
  );
  useEffect(() => setDraft(profile.name), [profile.name]);
  const rename = useMutation({
    mutationFn: (name: string) => updateProfileMetadata(profile.id, { name }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["profiles", profile.id], updated);
      queryClient.setQueryData<SavedExpertProfile[]>(["profiles"], (current) =>
        current?.map((item) => item.id === updated.id ? updated : item),
      );
      setEditing(false);
    },
  });
  function submit() {
    const name = draft.trim();
    if (!name || name === profile.name) {
      setDraft(profile.name);
      setEditing(false);
      return;
    }
    rename.mutate(name);
  }
  if (!editing) {
    return (
      <button className={`v2-inline-name ${large ? "large" : ""}`} onClick={() => setEditing(true)} title="Rename profile">
        <strong>{profile.name}</strong><span aria-hidden="true">✎</span>
      </button>
    );
  }
  return (
    <div className={`v2-inline-name-editor ${large ? "large" : ""}`}>
      <label className="v2-visually-hidden" htmlFor={`profile-name-${profile.id}`}>Profile name</label>
      <input
        autoFocus
        id={`profile-name-${profile.id}`}
        maxLength={120}
        onChange={(event) => setDraft(event.target.value)}
        onFocus={(event) => event.currentTarget.select()}
        onKeyDown={(event) => {
          if (event.key === "Enter") submit();
          if (event.key === "Escape") { setDraft(profile.name); setEditing(false); }
        }}
        value={draft}
      />
      <button aria-label="Save profile name" disabled={rename.isPending || !draft.trim()} onClick={submit}>✓</button>
      <button aria-label="Cancel rename" disabled={rename.isPending} onClick={() => { setDraft(profile.name); setEditing(false); }}>×</button>
      {duplicate && <small className="v2-duplicate-warning">Another profile uses this name; fingerprints will still distinguish them.</small>}
      {rename.error && <small role="alert">{rename.error instanceof Error ? rename.error.message : "Rename failed"}</small>}
    </div>
  );
}

function formatPercent(value: number | null) {
  return value == null ? "—" : `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "unknown" : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
