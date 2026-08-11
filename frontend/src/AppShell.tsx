import type { MouseEvent, ReactNode } from "react";

import { navigate, type AppRoute } from "./router";
import type { JobRecord, ModelState, SystemStatus } from "./types";

interface AppShellProps {
  route: AppRoute;
  status: SystemStatus | null;
  activeJob?: JobRecord | null;
  children: ReactNode;
}

const navigation = [
  { href: "/", names: ["dashboard"], glyph: "◇", label: "Home" },
  {
    href: "/benchmarks",
    names: ["benchmarks", "benchmark", "runs", "run"],
    glyph: "◫",
    label: "Bench",
  },
  {
    href: "/agent-runs",
    names: ["agentRuns", "agentRun", "agentComparison"],
    glyph: "⌘",
    label: "Agents",
  },
  {
    href: "/profiles",
    names: ["profiles", "profile", "profileStudio"],
    glyph: "⌁",
    label: "Profiles",
  },
  {
    href: "/comparisons",
    names: ["comparisons", "comparison"],
    glyph: "⇄",
    label: "Compare",
  },
] as const;

export function AppShell({ route, status, activeJob, children }: AppShellProps) {
  return (
    <div className="shell command-shell">
      <aside className="rail command-rail">
        <AppLink className="brand-mark" href="/" aria-label="MoE Atelier home">
          M
        </AppLink>
        <nav aria-label="Primary navigation">
          {navigation.map((item) => (
            <AppLink
              className={`nav-item ${item.names.includes(route.name as never) ? "active" : ""}`}
              href={item.href}
              key={item.href}
              aria-label={item.label}
            >
              {item.glyph}<span>{item.label}</span>
            </AppLink>
          ))}
        </nav>
        <span className="version">v{__APP_VERSION__}</span>
      </aside>

      <main className="command-main">
        <header className="topbar command-topbar">
          <div>
            <AppLink className="eyebrow brand-link" href="/">
              MoE Atelier
            </AppLink>
            <span className="topbar-note">Research command center</span>
          </div>
          <div className="command-status-group">
            {activeJob && (
              <AppLink className="active-job-chip" href={jobHref(activeJob)}>
                <span className="live-dot" />
                {jobLabel(activeJob)} · {activeJob.progress_current}/
                {activeJob.progress_total || "—"}
              </AppLink>
            )}
            <RuntimeStatus state={status?.model_state ?? "unloaded"} mode={status?.mode} />
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}

export function AppLink({
  href,
  children,
  onClick,
  ...props
}: React.AnchorHTMLAttributes<HTMLAnchorElement>) {
  function follow(event: MouseEvent<HTMLAnchorElement>) {
    onClick?.(event);
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      props.target === "_blank" ||
      !href?.startsWith("/")
    ) {
      return;
    }
    event.preventDefault();
    navigate(href);
  }

  return (
    <a {...props} href={href} onClick={follow}>
      {children}
    </a>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  meta,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  actions?: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="command-page-header">
      <div className="command-page-title">
        <span className="section-label">{eyebrow}</span>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
        {meta && <div className="page-meta">{meta}</div>}
      </div>
      {actions && <div className="command-page-actions">{actions}</div>}
    </header>
  );
}

function RuntimeStatus({
  state,
  mode,
}: {
  state: ModelState;
  mode?: SystemStatus["mode"];
}) {
  return (
    <div className={`status-pill ${state}`} title={`Model ${state}`}>
      <span />
      {state === "ready"
        ? mode === "mock"
          ? "Mock ready"
          : "Qwen ready"
        : state.replaceAll("_", " ")}
    </div>
  );
}

function jobHref(job: JobRecord) {
  if (job.kind === "agent_run" && job.result_id) {
    return `/agent-runs/${encodeURIComponent(job.result_id)}`;
  }
  if (job.kind === "benchmark_run" && job.result_id) {
    return `/runs/${encodeURIComponent(job.result_id)}`;
  }
  return "/";
}

function jobLabel(job: JobRecord) {
  if (job.kind === "agent_run") return "Agent run";
  if (job.kind === "benchmark_run") return "Benchmark";
  if (job.kind === "model_load") return "Model load";
  return "Dataset";
}
