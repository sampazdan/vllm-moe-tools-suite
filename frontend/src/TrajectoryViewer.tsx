import { useEffect, useRef, useState } from "react";

import { AgentPerformancePanel } from "./PerformancePanel";
import type {
  AgentTrajectory,
  AgentTrialArtifacts,
  AgentTrialSummary,
  TrajectoryStep,
  TrajectoryStepType,
} from "./types";

interface TrajectoryViewerProps {
  trial: AgentTrialSummary;
  trajectory: AgentTrajectory | null;
  artifacts: AgentTrialArtifacts | null;
  live: boolean;
}

const eventLabels: Record<TrajectoryStepType, string> = {
  system: "System",
  user: "Task",
  assistant: "Model response",
  tool: "Tool call",
  observation: "Observation",
  verifier: "Verifier",
  task: "Task",
  reasoning: "Explicit reasoning",
  model: "Model response",
  tool_call: "Tool call",
  command: "Command",
  stdout: "Standard output",
  stderr: "Standard error",
  routing: "Expert routing",
};

export function TrajectoryViewer({
  trial,
  trajectory,
  artifacts,
  live,
}: TrajectoryViewerProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [reasoningView, setReasoningView] = useState<"off" | "compact" | "full">(
    "compact",
  );
  const feedRef = useRef<HTMLDivElement>(null);
  const steps = trajectory?.steps ?? [];

  useEffect(() => {
    setExpandedSteps(new Set());
  }, [trial.id]);

  useEffect(() => {
    if (live && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [live, steps.length]);

  function toggleStep(stepId: string) {
    setExpandedSteps((current) => {
      const next = new Set(current);
      if (next.has(stepId)) next.delete(stepId);
      else next.add(stepId);
      return next;
    });
  }

  return (
    <div className="trajectory-viewer">
      <header className="trajectory-heading">
        <div>
          <span className="section-label">Live trajectory</span>
          <h3>{trial.title}</h3>
          <p>
            {trial.task_id} · {trial.turns} turns · {trial.commands} commands
          </p>
        </div>
        <div className="trajectory-heading-actions">
          <label className="reasoning-view-control">
            Reasoning
            <select
              value={reasoningView}
              onChange={(event) =>
                setReasoningView(event.target.value as typeof reasoningView)
              }
            >
              <option value="off">Off</option>
              <option value="compact">Compact</option>
              <option value="full">Full</option>
            </select>
          </label>
          <span className={`trial-state ${trial.status}`}>
            {formatTrialStatus(trial.status)}
          </span>
          <a
            className="text-button trajectory-export"
            href={`/api/trials/${trial.id}/export/atif`}
            download
          >
            Export ATIF
          </a>
        </div>
      </header>

      <div className="trajectory-budget" aria-label="Trial usage">
        <Usage value={trial.turns} label="turns" />
        <Usage value={trial.commands} label="commands" />
        <Usage
          value={formatCompactNumber(
            trial.prompt_tokens + trial.completion_tokens,
          )}
          label="tokens"
        />
        <Usage
          value={`${trial.routed_inference_calls}/${trial.inference_calls}`}
          label="routed calls"
        />
      </div>

      <AgentPerformancePanel trial={trial} trajectory={trajectory} compact />

      <div
        className="trajectory-feed"
        aria-live={live ? "polite" : "off"}
        ref={feedRef}
      >
        {steps
          .filter((step) => step.type !== "reasoning" || reasoningView !== "off")
          .map((step) => (
          <div
            className={`trajectory-step ${step.type} ${step.phase ? `phase-${step.phase}` : ""}`}
            key={step.id}
          >
            <span className="trajectory-marker" aria-hidden="true">
              {stepMarker(step.type)}
            </span>
            <article>
              <header>
                <div>
                  <span className="event-kind">{eventLabels[step.type]}</span>
                  <strong>{step.title}</strong>
                </div>
                <div className="event-time">
                  {step.turn != null && <span>Turn {step.turn}</span>}
                  <time dateTime={step.timestamp}>{formatStepTime(step.timestamp)}</time>
                </div>
              </header>
              {step.command && <pre className="command-line">$ {step.command}</pre>}
              {step.content && (
                <StepContent
                  step={step}
                  expanded={
                    expandedSteps.has(step.id) ||
                    (step.type === "reasoning" && reasoningView === "full")
                  }
                  compactReasoning={
                    step.type === "reasoning" && reasoningView === "compact"
                  }
                  onToggle={() => toggleStep(step.id)}
                />
              )}
              <div className="event-meta">
                {step.exit_code != null && (
                  <span className={step.exit_code === 0 ? "ok" : "bad"}>
                    Exit {step.exit_code}
                  </span>
                )}
                {step.duration_ms != null && (
                  <span>{formatDuration(step.duration_ms)}</span>
                )}
                {step.inference && (
                  <>
                    <span>
                      {step.inference.total_tokens ??
                        step.inference.prompt_tokens +
                          step.inference.completion_tokens} tokens
                    </span>
                    {step.inference.reasoning_tokens != null && (
                      <span>{step.inference.reasoning_tokens} reasoning</span>
                    )}
                    {step.inference.ttft_ms != null && (
                      <span>{formatDuration(step.inference.ttft_ms)} TTFT</span>
                    )}
                    {step.inference.tokens_per_second != null && (
                      <span>{step.inference.tokens_per_second.toFixed(1)} t/s</span>
                    )}
                    <span>
                      {step.inference.routing_artifact_id
                        ? "Routing captured"
                        : "Routing unavailable"}
                    </span>
                  </>
                )}
              </div>
            </article>
          </div>
        ))}
        {steps.length === 0 && (
          <div className="trajectory-empty">
            <span>{live ? "Working" : "No events"}</span>
            <p>
              {live
                ? "The first model or sandbox event will appear here."
                : "This trial did not retain a trajectory."}
            </p>
          </div>
        )}
      </div>

      {artifacts?.verifier && (
        <section
          className={`verifier-result ${artifacts.verifier.status}`}
          aria-label="Verifier result"
        >
          <div className="verifier-result-heading">
            <div>
              <span className="section-label">Verifier result</span>
              <h4>{artifacts.verifier.summary}</h4>
            </div>
            <strong>
              {artifacts.verifier.reward == null
                ? artifacts.verifier.status
                : `${Math.round(artifacts.verifier.reward * 100)}% reward`}
            </strong>
          </div>
          {artifacts.verifier.output && (
            <details>
              <summary>Show verifier output</summary>
              <pre>{artifacts.verifier.output}</pre>
            </details>
          )}
        </section>
      )}

      {artifacts?.patch && (
        <details className="patch-artifact">
          <summary>
            <span>Show patch</span>
            <small>
              {artifacts.files_changed} files · +{artifacts.additions} / −
              {artifacts.deletions}
            </small>
          </summary>
          <pre>{artifacts.patch}</pre>
        </details>
      )}

      {artifacts?.exports && artifacts.exports.length > 0 && (
        <div className="trajectory-artifact-links">
          {artifacts.exports.map((artifact) => (
            <a href={artifact.download_url} key={artifact.name} download>
              {artifact.name}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function StepContent({
  step,
  expanded,
  compactReasoning,
  onToggle,
}: {
  step: TrajectoryStep;
  expanded: boolean;
  compactReasoning: boolean;
  onToggle: () => void;
}) {
  const expandable = step.truncated || step.content.length > 900;
  return (
    <div
      className={`step-content ${expanded ? "expanded" : ""} ${compactReasoning ? "compact-reasoning" : ""}`}
    >
      <pre>{step.content}</pre>
      {expandable && (
        <button className="text-button" onClick={onToggle}>
          {expanded ? "Show less" : "Show full output"}
        </button>
      )}
    </div>
  );
}

function Usage({ value, label }: { value: string | number; label: string }) {
  return (
    <span>
      <strong>{value}</strong>
      {label}
    </span>
  );
}

function stepMarker(type: TrajectoryStepType) {
  if (type === "assistant" || type === "model") return "M";
  if (type === "user" || type === "task") return "→";
  if (type === "reasoning") return "R";
  if (type === "tool" || type === "tool_call") return "◆";
  if (type === "command") return "$";
  if (type === "observation" || type === "stdout") return "↳";
  if (type === "stderr") return "!";
  if (type === "verifier") return "✓";
  if (type === "routing") return "⌁";
  return "·";
}

function formatTrialStatus(status: AgentTrialSummary["status"]) {
  return status.replaceAll("_", " ");
}

function formatStepTime(value: string) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatDuration(milliseconds: number) {
  return milliseconds < 1_000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1_000).toFixed(1)} s`;
}

function formatCompactNumber(value: number) {
  if (value < 1_000) return String(value);
  return `${(value / 1_000).toFixed(value < 10_000 ? 1 : 0)}k`;
}
