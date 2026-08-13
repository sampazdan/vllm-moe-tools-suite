import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { AppLink } from "../AppShell";
import { listWorkloads } from "./data";
import { v2Href } from "./router";
import type { WorkloadDescriptor, WorkloadKind } from "./types";
import {
  V2Empty,
  V2Error,
  V2Loading,
  V2PageHeader,
  V2Status,
} from "./V2Shell";

type KindFilter = "all" | WorkloadKind;
type ReadinessFilter = "all" | "launchable" | "attention";

export function WorkloadLibraryPage() {
  const [kind, setKind] = useState<KindFilter>("all");
  const [readiness, setReadiness] = useState<ReadinessFilter>("all");
  const [search, setSearch] = useState("");
  const workloads = useQuery({
    queryKey: ["v2-workloads"],
    queryFn: listWorkloads,
  });
  const items = workloads.data ?? [];
  const filtered = useMemo(() => {
    const phrase = search.trim().toLowerCase();
    return items.filter((workload) => {
      if (kind !== "all" && workload.kind !== kind) return false;
      if (readiness === "launchable" && workload.readiness !== "ready") return false;
      if (readiness === "attention" && workload.readiness === "ready") return false;
      if (!phrase) return true;
      return [
        workload.title,
        workload.description,
        workload.family,
        workload.language,
        ...workload.tags,
      ].some((value) => value?.toLowerCase().includes(phrase));
    });
  }, [items, kind, readiness, search]);

  return (
    <div className="v2-page v2-library-page">
      <V2PageHeader
        eyebrow="Workload Library"
        title="Choose work at the task boundary."
        description="Launchability, source provenance, runtime constraints, and public success criteria are visible before an experiment exists."
        meta={
          <>
            <span>{items.length} tasks and cohorts</span>
            <span>{items.filter((item) => item.readiness === "ready").length} launchable</span>
            <span>{items.filter((item) => item.kind === "state_drift").length} state-drift scenarios</span>
          </>
        }
        actions={
          <AppLink className="v2-primary-button" href="/experiments/new">
            Create experiment
          </AppLink>
        }
      />

      <div className="v2-toolbar v2-library-toolbar">
        <input
          aria-label="Search workload library"
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search tasks, languages, families…"
          type="search"
          value={search}
        />
        <div className="v2-segments" aria-label="Workload type">
          {(["all", "answer", "coding", "state_drift"] as const).map((value) => (
            <button
              aria-pressed={kind === value}
              className={kind === value ? "selected" : ""}
              key={value}
              onClick={() => setKind(value)}
            >
              {value === "state_drift" ? "State drift" : humanize(value)}
            </button>
          ))}
        </div>
        <select
          aria-label="Readiness"
          onChange={(event) => setReadiness(event.target.value as ReadinessFilter)}
          value={readiness}
        >
          <option value="all">All readiness</option>
          <option value="launchable">Launchable now</option>
          <option value="attention">Needs attention</option>
        </select>
      </div>

      {workloads.isPending ? (
        <V2Loading label="Reading workload manifests…" />
      ) : workloads.error ? (
        <V2Error error={workloads.error} />
      ) : filtered.length === 0 ? (
        <V2Empty
          title="No workloads match."
          detail="Change the task type, readiness filter, or search phrase."
        />
      ) : (
        <div className="v2-workload-grid">
          {filtered.map((workload) => (
            <WorkloadCard key={workload.id} workload={workload} />
          ))}
        </div>
      )}
    </div>
  );
}

function WorkloadCard({ workload }: { workload: WorkloadDescriptor }) {
  const runnable = workload.readiness === "ready";
  return (
    <article className={`v2-workload-card ${runnable ? "ready" : "attention"}`}>
      <header>
        <span className={`v2-kind-mark ${workload.kind}`}>{kindLabel(workload.kind)}</span>
        <V2Status value={workload.readiness} />
      </header>
      <div className="v2-workload-copy">
        <span>{workload.family}</span>
        <h2>{workload.title}</h2>
        <p>{workload.public_problem_statement || workload.description}</p>
      </div>
      <dl className="v2-workload-facts">
        <div><dt>Unit</dt><dd>{workload.unit_count === 1 ? "1 task" : `${workload.unit_count} item cohort`}</dd></div>
        <div><dt>Language</dt><dd>{workload.language ?? "Model response"}</dd></div>
        <div><dt>Horizon</dt><dd>{workload.expected_horizon ?? (workload.kind === "coding" ? "Agent turns" : "Single request")}</dd></div>
        <div><dt>Runtime</dt><dd>{workload.runtime?.label ?? "Warm engine"}</dd></div>
      </dl>
      <section className="v2-readiness-evidence">
        <span className="v2-section-label">Launch evidence</span>
        <p><strong>{workload.source}</strong>{workload.source_revision ? ` · ${workload.source_revision}` : " · revision unavailable"}</p>
        {workload.runtime?.image_digest && <code>{compactDigest(workload.runtime.image_digest)}</code>}
        {workload.blocked_reasons.map((reason) => <small key={reason}>{reason}</small>)}
      </section>
      <section className="v2-public-contract">
        <span className="v2-section-label">Public success criteria</span>
        {workload.public_success_criteria.length ? (
          <ul>
            {workload.public_success_criteria.slice(0, 3).map((criterion) => <li key={criterion}>{criterion}</li>)}
          </ul>
        ) : (
          <p>Evaluator details are supplied by the workload manifest.</p>
        )}
      </section>
      <footer>
        <div className="v2-tag-list">
          {workload.tools.slice(0, 2).map((tool) => <span key={tool}>{tool}</span>)}
          {workload.tags.slice(0, 2).map((tag) => <span key={tag}>{tag}</span>)}
        </div>
        {runnable ? (
          <AppLink className="v2-primary-button" href={v2Href("newExperiment", workload.id)}>
            Use workload
          </AppLink>
        ) : (
          <button className="v2-secondary-button" disabled title={workload.blocked_reasons.join(" ")}>
            Not launchable
          </button>
        )}
      </footer>
    </article>
  );
}

function kindLabel(kind: WorkloadKind) {
  if (kind === "state_drift") return "State drift";
  return kind === "coding" ? "Coding" : "Answer";
}

function compactDigest(value: string) {
  const [algorithm, digest] = value.split(":", 2);
  return digest ? `${algorithm}:${digest.slice(0, 14)}…` : value.slice(0, 18);
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
