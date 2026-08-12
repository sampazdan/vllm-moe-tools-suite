import { AppLink } from "../AppShell";
import type { RuntimeStatus, SessionStatus, SystemStatus } from "../types";
import { V2Metric, V2PageHeader, V2Status } from "./V2Shell";

export function SettingsPage({
  runtime,
  session,
  status,
}: {
  runtime: RuntimeStatus | null;
  session: SessionStatus;
  status: SystemStatus | null;
}) {
  return (
    <div className="v2-page v2-settings-page">
      <V2PageHeader
        eyebrow="Settings"
        title="Runtime and compatibility."
        description="This view keeps operational truth, authentication posture, and preserved V1 workflows close to the experiment workspace."
        meta={<><span>Frontend v{__APP_VERSION__}</span><V2Status value={status?.mode ?? "unknown"} /></>}
      />

      <section className="v2-settings-overview">
        <V2Metric label="Backend version" value={status?.version ?? "Unavailable"} />
        <V2Metric label="Runtime mode" value={(status?.mode ?? "unknown").toUpperCase()} />
        <V2Metric label="Model state" value={humanize(status?.model_state ?? "unloaded")} />
        <V2Metric label="Authentication" value={session.auth_required ? "Token required" : "Local access"} detail={session.expires_at ? `expires ${formatDate(session.expires_at)}` : undefined} />
      </section>

      <div className="v2-settings-grid">
        <section>
          <header><span className="v2-section-label">Managed runtime</span><h2>Process evidence</h2></header>
          <dl>
            <div><dt>Managed by app</dt><dd>{runtime?.managed ? "Yes" : "No"}</dd></div>
            <div><dt>Model</dt><dd>{runtime?.model_id ?? "None"}</dd></div>
            <div><dt>Process</dt><dd>{runtime?.pid ?? "Not running"}</dd></div>
            <div><dt>Session</dt><dd>{runtime?.session_id ?? "None"}</dd></div>
            <div><dt>Started</dt><dd>{runtime?.started_at ? formatDate(runtime.started_at) : "—"}</dd></div>
            <div><dt>Profile artifact</dt><dd>{runtime?.profile_path ?? "Baseline"}</dd></div>
          </dl>
          {runtime?.log_tail && <details><summary>Recent launcher log</summary><pre>{runtime.log_tail}</pre></details>}
        </section>

        <section>
          <header><span className="v2-section-label">V2 API contract</span><h2>Integration readiness</h2></header>
          <div className="v2-api-contract-list">
            <ApiContract method="GET · POST" path="/api/experiments" purpose="Durable creation and archive" />
            <ApiContract method="GET" path="/api/experiments/{id}/events" purpose="Cursor-based persisted progress" />
            <ApiContract method="GET" path="/api/workloads" purpose="Task-level metadata and readiness" />
            <ApiContract method="PATCH" path="/api/profiles/{id}" purpose="Mutable name and description only" />
            <ApiContract method="POST" path="/api/expert-contexts/activate" purpose="Hot expert context switch" />
          </div>
        </section>

        <section>
          <header><span className="v2-section-label">Preserved workflows</span><h2>V1 research tools</h2></header>
          <p>Existing benchmark, run, comparison, and Profile Studio routes remain available while their artifacts are adopted by the experiment contract.</p>
          <nav className="v2-legacy-links">
            <AppLink href="/benchmarks">Benchmark library <span>→</span></AppLink>
            <AppLink href="/runs">Benchmark runs <span>→</span></AppLink>
            <AppLink href="/agent-runs">Coding runs <span>→</span></AppLink>
            <AppLink href="/comparisons">Comparisons <span>→</span></AppLink>
            <AppLink href="/profiles/new">Profile Studio <span>→</span></AppLink>
          </nav>
        </section>

        <section>
          <header><span className="v2-section-label">Data location</span><h2>Durability boundary</h2></header>
          <p>Run metadata, event journals, logs, and runtime artifacts are owned by the backend data directory. Reloading this browser must not erase experiment progress.</p>
          <code className="v2-path-code">{status?.data_dir ?? "Backend data directory unavailable"}</code>
          <small>The V2 interface never treats in-memory React state as the authoritative run record.</small>
        </section>
      </div>
    </div>
  );
}

function ApiContract({ method, path, purpose }: { method: string; path: string; purpose: string }) {
  return <article><span>{method}</span><code>{path}</code><small>{purpose}</small></article>;
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
