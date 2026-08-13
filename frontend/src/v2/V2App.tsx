import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import App from "../App";
import { activeJobFromConflict, api, setCsrfToken } from "../api";
import { AppLink } from "../AppShell";
import { ProfileStudioPage } from "../CommandCenter";
import { navigate } from "../router";
import type {
  JobRecord,
  ModelSession,
  RuntimeStatus,
  SessionStatus,
  SystemStatus,
} from "../types";
import { useDurableJob } from "../useDurableJob";
import {
  cancelJob,
  getActiveExpertContext,
  listExperiments,
  listModels,
  listProfiles,
  loadModel,
  retryModelLoad,
} from "./data";
import {
  CreateExperimentPage,
  ExperimentDetailPage,
  ExperimentsPage,
} from "./Experiments";
import { ModelsPage } from "./Models";
import { ProfileDetailPage, ProfilesPage } from "./Profiles";
import { type V2Route, useV2Route } from "./router";
import { SettingsPage } from "./Settings";
import { V2Empty, V2Shell } from "./V2Shell";
import { WorkloadLibraryPage } from "./Workloads";

const activeExperimentStatuses = new Set(["queued", "preparing", "running", "cancelling"]);

export default function V2App() {
  const route = useV2Route();
  if (route.name === "legacy") {
    return route.path === "/legacy" ? <LegacyRedirect /> : <App />;
  }
  return <V2Workspace route={route} />;
}

function LegacyRedirect() {
  useEffect(() => navigate("/benchmarks", { replace: true }), []);
  return <div className="app-loading">Opening preserved workflows…</div>;
}

function V2Workspace({ route }: { route: Exclude<V2Route, { name: "legacy" }> }) {
  const queryClient = useQueryClient();
  const session = useQuery({
    queryKey: ["session"],
    queryFn: () => api<SessionStatus>("/api/session"),
    staleTime: 30_000,
  });
  const authenticated = session.data?.authenticated === true;
  const status = useQuery({
    queryKey: ["status"],
    queryFn: () => api<SystemStatus>("/api/system/status"),
    enabled: authenticated,
  });
  const runtime = useQuery({
    queryKey: ["runtime-status"],
    queryFn: () => api<RuntimeStatus>("/api/runtime/status"),
    enabled: authenticated,
    refetchInterval: (query) => query.state.data?.managed ? 2_000 : false,
  });
  const currentSession = useQuery({
    queryKey: ["model-session-current"],
    queryFn: () => api<ModelSession>("/api/model-sessions/current"),
    enabled:
      authenticated &&
      status.data?.model_state !== undefined &&
      status.data.model_state !== "unloaded",
    retry: false,
  });
  const activeContext = useQuery({
    queryKey: ["v2-active-context"],
    queryFn: getActiveExpertContext,
    enabled: authenticated && currentSession.data?.state === "ready",
    retry: false,
    refetchInterval: 2_000,
  });
  const profiles = useQuery({
    queryKey: ["profiles"],
    queryFn: listProfiles,
    enabled: authenticated,
  });
  const studioModels = useQuery({
    queryKey: ["v2-models"],
    queryFn: listModels,
    enabled: authenticated && route.name === "profileStudio",
  });
  const experiments = useQuery({
    queryKey: ["v2-experiments"],
    queryFn: listExperiments,
    enabled: authenticated,
    retry: false,
    refetchInterval: (query) => {
      if (query.state.error) return false;
      return query.state.data?.items.some((item) => activeExperimentStatuses.has(item.status))
        ? 1_500
        : 10_000;
    },
  });
  const durableJob = useDurableJob({
    enabled: authenticated,
    onRecovered: refreshAfterJob,
  });
  const login = useMutation({
    mutationFn: (token: string) => api<SessionStatus>("/api/session/login", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
    onSuccess: (nextSession) => {
      setCsrfToken(nextSession.csrf_token);
      queryClient.setQueryData(["session"], nextSession);
    },
  });

  useEffect(() => setCsrfToken(session.data?.csrf_token ?? null), [session.data]);

  async function refreshAfterJob(job: JobRecord) {
    if (job.kind === "model_load") {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["status"] }),
        queryClient.invalidateQueries({ queryKey: ["runtime-status"] }),
        queryClient.invalidateQueries({ queryKey: ["model-session-current"] }),
        queryClient.invalidateQueries({ queryKey: ["v2-active-context"] }),
        queryClient.invalidateQueries({ queryKey: ["v2-models"] }),
        queryClient.invalidateQueries({ queryKey: ["v2-model-loads"] }),
        queryClient.invalidateQueries({ queryKey: ["model-sessions"] }),
      ]);
      return;
    }
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["v2-experiments"] }),
      queryClient.invalidateQueries({ queryKey: ["v2-workloads"] }),
    ]);
  }

  async function loadAndWatchModel(modelId: string) {
    let submitted: JobRecord;
    try {
      submitted = await loadModel(modelId);
    } catch (error) {
      const existing = activeJobFromConflict(error);
      if (!existing) throw error;
      durableJob.adoptJob(existing);
      return;
    }
    queryClient.setQueryData(["active-job"], submitted);
    await durableJob.watchJob(submitted);
    await refreshAfterJob(submitted);
  }

  async function cancelActiveJob(jobId: string) {
    const job = await cancelJob(jobId);
    await queryClient.invalidateQueries({ queryKey: ["v2-model-loads"] });
    if (job.status === "cancelled") durableJob.retryRecovery();
  }

  async function retryAndWatchModel(jobId: string) {
    let submitted: JobRecord;
    try {
      submitted = await retryModelLoad(jobId);
    } catch (error) {
      const existing = activeJobFromConflict(error);
      if (!existing) throw error;
      durableJob.adoptJob(existing);
      return;
    }
    queryClient.setQueryData(["active-job"], submitted);
    await durableJob.watchJob(submitted);
    await refreshAfterJob(submitted);
  }

  if (session.isPending) return <div className="v2-boot"><span />Preparing the experiment workspace…</div>;
  if (session.error) return <div className="v2-boot error">{session.error.message}</div>;
  if (!session.data?.authenticated) {
    return (
      <V2Login
        error={login.error instanceof Error ? login.error.message : null}
        onLogin={(token) => login.mutate(token)}
        pending={login.isPending}
      />
    );
  }

  const activeExperiment = experiments.data?.items.find((item) => activeExperimentStatuses.has(item.status)) ?? null;
  const activeProfileId = activeContext.data?.profile_id ?? null;
  const activeProfile = profiles.data?.find((profile) => profile.id === activeProfileId) ?? null;
  return (
    <V2Shell
      activeExperiment={activeExperiment}
      activeJob={durableJob.activeJob}
      activeProfileName={activeProfile?.name ?? null}
      currentSession={currentSession.data ?? null}
      route={route}
      status={status.data ?? null}
    >
      {(durableJob.discoveryError || durableJob.recoveryError) && (
        <div className="v2-global-warning" role="alert">
          <span>{durableJob.recoveryError ?? durableJob.discoveryError}</span>
          <button onClick={durableJob.retryRecovery}>Check background work again</button>
        </div>
      )}
      {route.name === "experiments" && <ExperimentsPage />}
      {route.name === "experimentNew" && (
        <CreateExperimentPage
          currentSession={currentSession.data ?? null}
          workloadId={route.workloadId}
        />
      )}
      {route.name === "experiment" && <ExperimentDetailPage experimentId={route.experimentId} />}
      {route.name === "workloads" && <WorkloadLibraryPage />}
      {route.name === "profiles" && <ProfilesPage activeProfileId={activeProfileId} currentSession={currentSession.data ?? null} />}
      {route.name === "profileStudio" && (
        <ProfileStudioPage
          model={
            studioModels.data?.find(
              (model) => model.id === currentSession.data?.model_id,
            ) ?? studioModels.data?.[0] ?? null
          }
          route={route}
        />
      )}
      {route.name === "profile" && <ProfileDetailPage activeProfileId={activeProfileId} currentSession={currentSession.data ?? null} profileId={route.profileId} />}
      {route.name === "models" && (
        <ModelsPage
          activeJob={durableJob.activeJob}
          activeProfileId={activeProfileId}
          connectionIssue={durableJob.connectionIssue}
          currentSession={currentSession.data ?? null}
          onCancelJob={cancelActiveJob}
          onLoadModel={loadAndWatchModel}
          onRetryModelLoad={retryAndWatchModel}
          runtime={runtime.data ?? null}
          status={status.data ?? null}
        />
      )}
      {route.name === "settings" && (
        <SettingsPage
          runtime={runtime.data ?? null}
          session={session.data}
          status={status.data ?? null}
        />
      )}
      {route.name === "notFound" && (
        <div className="v2-page">
          <V2Empty
            title="This workspace route does not exist."
            detail="Open the experiment archive or return to a preserved V1 workflow."
            action={<AppLink className="v2-primary-button" href="/experiments">Open experiments</AppLink>}
          />
        </div>
      )}
    </V2Shell>
  );
}

function V2Login({
  error,
  onLogin,
  pending,
}: {
  error: string | null;
  onLogin: (token: string) => void;
  pending: boolean;
}) {
  const [token, setToken] = useState("");
  return (
    <main className="v2-login">
      <section>
        <span className="v2-login-mark">M</span>
        <div><span className="v2-section-label">MoE Atelier</span><h1>Open the experiment workspace.</h1><p>This runtime is private. Enter the access token configured for this deployment.</p></div>
        <form onSubmit={(event) => { event.preventDefault(); onLogin(token); }}>
          <label htmlFor="v2-access-token">Access token</label>
          <input autoComplete="current-password" autoFocus id="v2-access-token" name="access-token" onChange={(event) => setToken(event.target.value)} type="password" value={token} />
          {error && <small role="alert">{error}</small>}
          <button className="v2-primary-button" disabled={!token || pending}>{pending ? "Opening…" : "Open workspace"}</button>
        </form>
        <small>The token is sent to this deployment and is not saved by the browser.</small>
      </section>
    </main>
  );
}
