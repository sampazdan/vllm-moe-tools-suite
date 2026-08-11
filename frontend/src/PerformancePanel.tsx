import type {
  AgentPerformanceSummary,
  AgentTrajectory,
  AgentTrialSummary,
  RunItemResult,
  RunPerformance,
} from "./types";

export function AgentPerformancePanel({
  trial,
  trajectory,
  compact = false,
}: {
  trial: AgentTrialSummary;
  trajectory: AgentTrajectory | null;
  compact?: boolean;
}) {
  const calls = (trajectory?.steps ?? [])
    .map((step) => step.inference)
    .filter((call) => call != null);
  const reported = trial.performance ?? {};
  const totalTokens = reported.total_tokens ??
    ((reported.prompt_tokens != null || reported.completion_tokens != null)
      ? (reported.prompt_tokens ?? trial.prompt_tokens) +
        (reported.reasoning_tokens ?? 0) +
        (reported.completion_tokens ?? trial.completion_tokens)
      : trial.prompt_tokens + trial.completion_tokens);
  const callTps = calls
    .map((call) => call.tokens_per_second)
    .filter((value): value is number => value != null && Number.isFinite(value));
  const callLatency = calls.reduce((time, call) => time + call.latency_ms, 0);
  const fallbackTps = calls.length && callLatency > 0
    ? calls.reduce((tokens, call) => tokens + call.completion_tokens, 0) /
      (callLatency / 1_000)
    : null;
  const meanTps = reported.mean_tps ?? average(callTps) ?? fallbackTps;
  const ttft = average(
    calls
      .map((call) => call.ttft_ms)
      .filter((value): value is number => value != null),
  );
  const cost = reported.estimated_cost_usd ?? sumNullable(
    calls.map((call) => call.estimated_cost_usd),
  );
  const costDetail = [
    reported.inference_cost_usd != null
      ? `${formatCost(reported.inference_cost_usd)} inference`
      : null,
    reported.judge_cost_usd != null
      ? `${formatCost(reported.judge_cost_usd)} judge`
      : null,
  ].filter((value): value is string => value != null).join(" · ") || undefined;
  const timing = reported as AgentPerformanceSummary;

  return (
    <section className={`performance-panel ${compact ? "compact" : ""}`}>
      <header>
        <span className="section-label">Performance</span>
        <strong>{calls.length} model calls</strong>
      </header>
      <div className="performance-metrics">
        <PerformanceMetric
          label="Decode speed"
          value={formatRate(meanTps)}
          detail={reported.mean_tps != null || callTps.length ? "reported" : "observed e2e"}
        />
        <PerformanceMetric label="Mean TTFT" value={formatDuration(ttft)} />
        <PerformanceMetric label="Total tokens" value={formatCompact(totalTokens)} />
        <PerformanceMetric
          label="Known cost"
          value={formatCost(cost)}
          detail={costDetail}
        />
        {reported.judge_cost_uncertain && (
          <PerformanceMetric
            label="Conservative judge debit"
            value={formatCost(reported.judge_cost_debit_usd)}
            detail="provider usage unavailable; actual spend is unknown"
          />
        )}
        {!compact && (
          <>
            <PerformanceMetric
              label="Model time"
              value={formatDuration(timing.model_time_ms)}
            />
            <PerformanceMetric
              label="Sandbox time"
              value={formatDuration(timing.sandbox_time_ms)}
            />
            <PerformanceMetric
              label="Verifier time"
              value={formatDuration(timing.verifier_time_ms)}
            />
            <PerformanceMetric
              label="Wall time"
              value={formatDuration(timing.wall_time_ms)}
            />
          </>
        )}
      </div>
      {!compact && (
        <div className="performance-token-mix">
          <TokenSegment label="Prompt" value={reported.prompt_tokens ?? trial.prompt_tokens} />
          <TokenSegment label="Reasoning" value={reported.reasoning_tokens} />
          <TokenSegment
            label="Completion"
            value={reported.completion_tokens ?? trial.completion_tokens}
          />
        </div>
      )}
    </section>
  );
}

export function BenchmarkPerformancePanel({
  items,
  performance,
  completedItems,
}: {
  items: RunItemResult[];
  performance?: RunPerformance | null;
  completedItems?: number;
}) {
  const completed = items.filter((item) => item.error == null);
  const pageLatency = completed.reduce((total, item) => total + item.latency_ms, 0);
  const pagePromptTokens = completed.reduce((total, item) => total + item.prompt_tokens, 0);
  const pageCompletionTokens = completed.reduce(
    (total, item) => total + item.completion_tokens,
    0,
  );
  const count = completedItems ?? completed.length;
  const latency = performance?.model_time_ms ?? pageLatency;
  const promptTokens = performance?.prompt_tokens ?? pagePromptTokens;
  const completionTokens = performance?.completion_tokens ?? pageCompletionTokens;
  const e2eTps = performance?.mean_tokens_per_second ?? (
    latency > 0 ? completionTokens / (latency / 1_000) : null
  );
  return (
    <section className="performance-panel benchmark-performance">
      <header>
        <span className="section-label">Performance</span>
        <strong>{count} completed items</strong>
      </header>
      <div className="performance-metrics">
        <PerformanceMetric label="E2E output speed" value={formatRate(e2eTps)} />
        <PerformanceMetric
          label="Mean latency"
          value={formatDuration(count ? latency / count : null)}
        />
        <PerformanceMetric label="Prompt tokens" value={formatCompact(promptTokens)} />
        <PerformanceMetric label="Completion tokens" value={formatCompact(completionTokens)} />
        {performance?.judge_cost_usd != null && (
          <PerformanceMetric
            label="Judge spend this run"
            value={formatCost(performance.judge_cost_usd)}
          />
        )}
        {performance?.judge_equivalent_cost_usd != null && (
          <PerformanceMetric
            label="Judge equivalent value"
            value={formatCost(performance.judge_equivalent_cost_usd)}
            detail="includes cached verdict usage"
          />
        )}
        {performance?.judge_cost_uncertain && (
          <PerformanceMetric
            label="Conservative judge debit"
            value={formatCost(performance.judge_budget_debit_usd)}
            detail="provider usage unavailable; actual spend is unknown"
          />
        )}
      </div>
      <p className="performance-note">
        Time-to-first-token appears only when the serving runtime reports a
        streaming timestamp; it is never inferred from total latency.
      </p>
    </section>
  );
}

function PerformanceMetric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div>
      <strong>{value}</strong>
      <span>{label}</span>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function TokenSegment({ label, value }: { label: string; value: number | null | undefined }) {
  return (
    <span className={value == null ? "unavailable" : ""}>
      <strong>{formatCompact(value)}</strong> {label}
    </span>
  );
}

function average(values: number[]) {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : null;
}

function sumNullable(values: Array<number | null | undefined>) {
  const available = values.filter((value): value is number => value != null);
  return available.length
    ? available.reduce((total, value) => total + value, 0)
    : null;
}

function formatRate(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)} t/s`;
}

function formatDuration(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value < 1_000) return `${Math.round(value)} ms`;
  if (value < 60_000) return `${(value / 1_000).toFixed(1)} s`;
  return `${(value / 60_000).toFixed(1)} min`;
}

function formatCompact(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

function formatCost(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}
