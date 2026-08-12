import type { ReactNode } from "react";

import { AppLink } from "../AppShell";
import type { JobRecord, ModelSession, SystemStatus } from "../types";
import type { V2Route } from "./router";
import type { ExperimentRecord } from "./types";

const navigation = [
  {
    href: "/experiments",
    label: "Experiments",
    shortLabel: "Runs",
    glyph: "◫",
    routes: ["experiments", "experiment", "experimentNew"],
  },
  {
    href: "/workloads",
    label: "Workload Library",
    shortLabel: "Library",
    glyph: "⌘",
    routes: ["workloads"],
  },
  {
    href: "/profiles",
    label: "Profiles",
    shortLabel: "Profiles",
    glyph: "⌁",
    routes: ["profiles", "profile"],
  },
  {
    href: "/models",
    label: "Models",
    shortLabel: "Models",
    glyph: "◇",
    routes: ["models"],
  },
  {
    href: "/settings",
    label: "Settings",
    shortLabel: "Settings",
    glyph: "⚙",
    routes: ["settings"],
  },
] as const;

export function V2Shell({
  route,
  status,
  currentSession,
  activeJob,
  activeExperiment,
  activeProfileName,
  children,
}: {
  route: V2Route;
  status: SystemStatus | null;
  currentSession: ModelSession | null;
  activeJob: JobRecord | null;
  activeExperiment: ExperimentRecord | null;
  activeProfileName: string | null;
  children: ReactNode;
}) {
  const ready = status?.model_state === "ready";
  return (
    <div className="v2-shell">
      <header className="v2-topbar">
        <AppLink className="v2-brand" href="/experiments" aria-label="MoE Atelier experiments">
          <span>M</span>
          <strong>MoE Atelier</strong>
        </AppLink>
        <nav className="v2-primary-nav" aria-label="Primary navigation">
          {navigation.map((item) => (
            <AppLink
              href={item.href}
              key={item.href}
              className={item.routes.includes(route.name as never) ? "active" : ""}
              aria-current={item.routes.includes(route.name as never) ? "page" : undefined}
            >
              <span aria-hidden="true">{item.glyph}</span>
              <em>{item.label}</em>
            </AppLink>
          ))}
        </nav>
        <div className="v2-runtime-strip">
          {activeExperiment && (
            <AppLink
              className="v2-active-work"
              href={`/experiments/${encodeURIComponent(activeExperiment.id)}`}
              title={`${activeExperiment.name} · ${experimentProgress(activeExperiment)}`}
            >
              <span className="v2-live-dot" />
              <strong>{activeExperiment.name}</strong>
              <em>{experimentProgress(activeExperiment)}</em>
            </AppLink>
          )}
          {activeJob && (!activeExperiment || activeJob.kind !== "experiment_run") && (
            <AppLink
              className="v2-active-work"
              href={activeWorkHref(activeJob)}
              title={`${jobLabel(activeJob)} ${activeJob.progress_current}/${activeJob.progress_total || "—"}`}
            >
              <span className="v2-live-dot" />
              <strong>{jobLabel(activeJob)}</strong>
              <em>{activeJob.progress_current}/{activeJob.progress_total || "—"}</em>
            </AppLink>
          )}
          <AppLink className={`v2-runtime-state ${ready ? "ready" : "idle"}`} href="/models">
            <span />
            <strong>{ready ? activeProfileName ?? "Baseline" : humanize(status?.model_state ?? "unloaded")}</strong>
            <em>{currentSession?.model_id ? compactModel(currentSession.model_id) : "No engine"}</em>
          </AppLink>
        </div>
      </header>

      <main className="v2-main">{children}</main>

      <nav className="v2-mobile-nav" aria-label="Primary mobile navigation">
        {navigation.map((item) => (
          <AppLink
            href={item.href}
            key={item.href}
            className={item.routes.includes(route.name as never) ? "active" : ""}
            aria-current={item.routes.includes(route.name as never) ? "page" : undefined}
            aria-label={item.label}
          >
            <span aria-hidden="true">{item.glyph}</span>
            <em>{item.shortLabel}</em>
          </AppLink>
        ))}
      </nav>
    </div>
  );
}

function experimentProgress(experiment: ExperimentRecord) {
  const laneUnits = experiment.lanes.reduce((total, lane) => total + lane.progress_total, 0);
  const completed = laneUnits
    ? experiment.lanes.reduce((total, lane) => total + lane.progress_current, 0)
    : experiment.progress_current;
  const units = laneUnits || experiment.progress_total;
  return `${completed}/${units || "—"}`;
}

export function V2PageHeader({
  eyebrow,
  title,
  description,
  meta,
  actions,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  meta?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="v2-page-header">
      <div>
        <span>{eyebrow}</span>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
        {meta && <div className="v2-page-meta">{meta}</div>}
      </div>
      {actions && <div className="v2-page-actions">{actions}</div>}
    </header>
  );
}

export function V2Status({ value }: { value: string }) {
  return <span className={`v2-status ${statusTone(value)}`}>{humanize(value)}</span>;
}

export function V2Metric({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "positive" | "negative" | "neutral";
}) {
  return (
    <div className={`v2-metric ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

export function V2Empty({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="v2-empty">
      <span aria-hidden="true">◇</span>
      <h2>{title}</h2>
      <p>{detail}</p>
      {action}
    </div>
  );
}

export function V2Error({ error }: { error: unknown }) {
  return (
    <div className="v2-error" role="alert">
      <strong>Could not complete this request.</strong>
      <span>{error instanceof Error ? error.message : "Unknown error"}</span>
    </div>
  );
}

export function V2Loading({ label = "Opening workspace…" }: { label?: string }) {
  return <div className="v2-loading" role="status"><span />{label}</div>;
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
  return job.kind === "model_load" ? "/models" : "/experiments";
}

function jobLabel(job: JobRecord) {
  if (job.kind === "experiment_run") return "Experiment";
  if (job.kind === "model_load") return "Loading model";
  if (job.kind === "dataset_prepare") return "Preparing data";
  if (job.kind === "agent_run") return "Coding run";
  return "Benchmark run";
}

function statusTone(value: string) {
  if (["ready", "completed", "passed", "exact", "recovered"].includes(value)) return "positive";
  if (["failed", "error", "blocked", "unsupported", "diverged"].includes(value)) return "negative";
  if (["running", "preparing", "queued", "cancelling"].includes(value)) return "live";
  return "neutral";
}

function compactModel(modelId: string) {
  return modelId.split("/").at(-1)?.replaceAll("-", " ") ?? modelId;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
