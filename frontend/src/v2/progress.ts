import type {
  ExperimentRecord,
  ExperimentUnit,
  LaneRole,
  LaneUnitResult,
  RunEvent,
} from "./types";

export interface LiveProgress {
  phase: string;
  currentUnit: string | null;
  completed: number;
  total: number;
  passed: number;
  failed: number;
  unscored: number;
  promptTokens: number;
  completionTokens: number;
  currentTps: number | null;
  meanTps: number | null;
  elapsedMs: number | null;
  etaMs: number | null;
}

export interface StateDifference {
  path: string;
  expected: unknown;
  actual: unknown;
}

export interface DriftRoutingSummary {
  checkpoint: number;
  selectionOverlap: number | null;
  routingMassJsDivergence: number | null;
  firstDivergence: boolean;
  shifts: Array<{
    layer: number;
    expert: number;
    baselineShare: number | null;
    candidateShare: number | null;
    delta: number | null;
  }>;
}

export function mergeRunEvents(
  current: RunEvent[],
  incoming: RunEvent[],
): RunEvent[] {
  const merged = new Map(current.map((event) => [event.sequence, event]));
  for (const event of incoming) merged.set(event.sequence, event);
  return [...merged.values()].sort((left, right) => left.sequence - right.sequence);
}

export function codingTimeline(
  result: LaneUnitResult | null,
  events: RunEvent[],
  runUnitId: string | null,
  laneId: string | null,
): Array<Record<string, unknown>> {
  const durable = runUnitId && laneId
    ? events
      .filter((event) =>
        event.type === "trajectory_step" &&
        event.unit_id === runUnitId &&
        event.lane_id === laneId
      )
      .map((event): Record<string, unknown> | null => {
        const step = isRecord(event.payload.trajectory_step)
          ? event.payload.trajectory_step
          : null;
        return step
          ? {
              ...step,
              event_sequence: event.sequence,
              event_created_at: event.created_at,
            }
          : null;
      })
      .filter((step): step is Record<string, unknown> => step !== null)
    : [];
  const merged = new Map<string, Record<string, unknown>>();
  [...durable, ...(result?.trajectory ?? [])].forEach((step, index) => {
    const key = trajectoryKey(step, index);
    merged.set(key, { ...merged.get(key), ...step });
  });
  return [...merged.values()].sort((left, right) => {
    const leftSequence = numericValue(left.sequence ?? left.event_sequence);
    const rightSequence = numericValue(right.sequence ?? right.event_sequence);
    return leftSequence - rightSequence;
  });
}

export function driftRoutingSummaries(unit: ExperimentUnit): DriftRoutingSummary[] {
  const checkpoints = unit.candidate?.checkpoints.length
    ? unit.candidate.checkpoints
    : unit.baseline?.checkpoints ?? [];
  return checkpoints
    .filter((checkpoint) => checkpoint.routing_comparison !== null)
    .map((checkpoint) => {
      const comparison = checkpoint.routing_comparison!;
      return {
        checkpoint: checkpoint.index,
        selectionOverlap: comparison.selection_overlap,
        routingMassJsDivergence: comparison.routing_mass_js_divergence,
        firstDivergence: comparison.at_candidate_first_divergence,
        shifts: comparison.largest_selection_shifts.map((shift) => ({
          layer: shift.layer,
          expert: shift.expert,
          baselineShare: shift.baseline_share,
          candidateShare: shift.candidate_share,
          delta: shift.delta,
        })),
      };
    })
    .sort((left, right) =>
      Number(right.firstDivergence) - Number(left.firstDivergence) ||
      left.checkpoint - right.checkpoint
    );
}

export function liveProgress(
  experiment: ExperimentRecord,
  events: RunEvent[],
  laneRole?: LaneRole,
): LiveProgress {
  const lanes = laneRole
    ? experiment.lanes.filter((lane) => lane.role === laneRole)
    : experiment.lanes;
  const latest = [...events]
    .reverse()
    .find((event) => !laneRole || event.lane_role === laneRole);
  const laneCompleted = lanes.reduce((total, lane) => total + lane.progress_current, 0);
  const laneTotal = lanes.reduce((sum, lane) => sum + lane.progress_total, 0);
  const completed = !laneRole && laneTotal === 0
    ? experiment.progress_current
    : laneCompleted;
  const total = !laneRole && laneTotal === 0
    ? experiment.progress_total
    : laneTotal;
  const performance = lanes[0]?.performance;
  return {
    phase: latest?.phase ?? experiment.phase,
    currentUnit: latest?.unit_id ?? experiment.active_unit_id,
    completed,
    total,
    passed: !laneRole && laneTotal === 0
      ? experiment.passed
      : lanes.reduce((sum, lane) => sum + lane.passed, 0),
    failed: !laneRole && laneTotal === 0
      ? experiment.failed
      : lanes.reduce((sum, lane) => sum + lane.failed, 0),
    unscored: !laneRole && laneTotal === 0
      ? experiment.unscored
      : lanes.reduce((sum, lane) => sum + lane.unscored, 0),
    promptTokens: lanes.reduce(
      (sum, lane) => sum + lane.performance.prompt_tokens,
      0,
    ),
    completionTokens: lanes.reduce(
      (sum, lane) => sum + lane.performance.completion_tokens,
      0,
    ),
    currentTps: performance?.current_tps ?? null,
    meanTps: performance?.mean_tps ?? null,
    elapsedMs: performance?.elapsed_ms ?? null,
    etaMs: performance?.eta_ms ?? null,
  };
}

export function stateDifferences(expected: unknown, actual: unknown) {
  const differences: StateDifference[] = [];
  walkState("$", expected, actual, differences);
  return differences;
}

function walkState(
  path: string,
  expected: unknown,
  actual: unknown,
  differences: StateDifference[],
) {
  if (Object.is(expected, actual)) return;
  if (isRecord(expected) && isRecord(actual)) {
    const keys = new Set([...Object.keys(expected), ...Object.keys(actual)]);
    for (const key of [...keys].sort()) {
      walkState(`${path}.${key}`, expected[key], actual[key], differences);
    }
    return;
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    const length = Math.max(expected.length, actual.length);
    for (let index = 0; index < length; index += 1) {
      walkState(`${path}[${index}]`, expected[index], actual[index], differences);
    }
    return;
  }
  differences.push({ path, expected, actual });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function trajectoryKey(step: Record<string, unknown>, fallback: number) {
  if (typeof step.id === "string" && step.id) return `id:${step.id}`;
  const sequence = numericValue(step.sequence);
  if (Number.isFinite(sequence)) return `sequence:${sequence}`;
  const eventSequence = numericValue(step.event_sequence);
  if (Number.isFinite(eventSequence)) return `event:${eventSequence}`;
  return `fallback:${fallback}`;
}

function numericValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : Number.POSITIVE_INFINITY;
}
