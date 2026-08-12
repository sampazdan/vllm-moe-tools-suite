import { ApiError, api } from "../api.ts";
import type {
  AgentDefinition,
  AgentTaskPack,
  BenchmarkInfo,
  EvaluationContract,
  ExecutionPolicy,
  GenerationConfig,
  ModelRegistryEntry,
  SandboxProvider,
  SavedExpertProfile,
} from "../types";
import type {
  ContextActivationResult,
  DriftCheckpoint,
  DriftMetrics,
  ExperimentCreateRequest,
  ExperimentEventPage,
  ExperimentLane,
  ExperimentPage,
  ExperimentRecord,
  ExperimentStatus,
  ExperimentUnit,
  LanePerformance,
  LaneRole,
  LaneUnitResult,
  ModelDescriptor,
  ProfileMetadataPatch,
  RoutingCheckpointComparison,
  RunEvent,
  WorkloadDescriptor,
  WorkloadKind,
  WorkloadReadiness,
} from "./types";

const emptyPerformance: LanePerformance = {
  elapsed_ms: null,
  eta_ms: null,
  prompt_tokens: 0,
  reasoning_tokens: null,
  completion_tokens: 0,
  total_tokens: 0,
  current_tps: null,
  mean_tps: null,
  estimated_cost_usd: null,
};

export async function listExperiments(): Promise<ExperimentPage> {
  const response = await api<unknown>("/api/experiments?limit=100");
  const record = asRecord(response);
  const rawItems = Array.isArray(response)
    ? response
    : arrayValue(record?.items ?? record?.experiments);
  return {
    items: rawItems.map(normalizeExperiment),
    total: numberValue(record?.total, rawItems.length),
    offset: numberValue(record?.offset, 0),
    limit: numberValue(record?.limit, rawItems.length || 100),
  };
}

export async function getExperiment(id: string): Promise<ExperimentRecord> {
  const response = await api<unknown>(`/api/experiments/${encodeURIComponent(id)}`);
  return normalizeExperiment(response);
}

export async function createExperiment(request: ExperimentCreateRequest) {
  const response = await api<unknown>("/api/experiments", {
    method: "POST",
    body: JSON.stringify(request),
  });
  const record = asRecord(response);
  const experiment = record?.experiment ?? response;
  return normalizeExperiment(experiment);
}

export function buildExperimentCreateRequest({
  agentId,
  candidateProfileId,
  conditions,
  horizon,
  kind,
  modelId,
  name,
  sandboxProviderId,
  seed,
  workloadId,
  workloadUnitIds,
}: {
  agentId: string;
  candidateProfileId: string | null;
  conditions: string[];
  horizon: number;
  kind: WorkloadKind;
  modelId: string;
  name: string;
  sandboxProviderId: string;
  seed: number;
  workloadId: string;
  workloadUnitIds: string[];
}): ExperimentCreateRequest {
  const request: ExperimentCreateRequest = {
    name,
    model_id: modelId,
    workload_id: workloadId,
    candidate_profile_id: candidateProfileId,
    seeds: [seed],
  };
  if (kind === "coding") {
    return {
      ...request,
      agent_id: agentId,
      sandbox_provider_id: sandboxProviderId,
    };
  }
  if (kind === "state_drift") {
    return {
      ...request,
      scenario_ids: workloadUnitIds,
      conditions,
      horizons: [horizon],
    };
  }
  return request;
}

export async function cancelExperiment(id: string) {
  const response = await api<unknown>(
    `/api/experiments/${encodeURIComponent(id)}/cancel`,
    { method: "POST" },
  );
  return normalizeExperiment(asRecord(response)?.experiment ?? response);
}

export async function listExperimentEvents(
  id: string,
  after: number,
): Promise<ExperimentEventPage> {
  const response = await api<unknown>(
    `/api/experiments/${encodeURIComponent(id)}/events?after=${after}&limit=500`,
  );
  const record = asRecord(response);
  const rawItems = Array.isArray(response)
    ? response
    : arrayValue(record?.items ?? record?.events);
  const items = rawItems.map((item, index) => normalizeEvent(item, id, after + index + 1));
  return {
    items,
    next_sequence: numberValue(
      record?.next_sequence,
      items.at(-1)?.sequence ?? after,
    ),
  };
}

export async function listWorkloads(): Promise<WorkloadDescriptor[]> {
  try {
    const response = await api<unknown>("/api/workloads?include_unavailable=1");
    const record = asRecord(response);
    const rawItems = Array.isArray(response)
      ? response
      : arrayValue(record?.items ?? record?.workloads);
    return rawItems.map(normalizeWorkload);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error;
    return legacyWorkloads();
  }
}

export async function listModels(): Promise<ModelDescriptor[]> {
  const models = await api<ModelRegistryEntry[]>("/api/models");
  return models.map((model) => normalizeModel(model));
}

export function listAgents() {
  return api<AgentDefinition[]>("/api/agents");
}

export function listSandboxProviders() {
  return api<SandboxProvider[]>("/api/sandbox-providers");
}

export function listProfiles() {
  return api<SavedExpertProfile[]>("/api/profiles");
}

export function getProfile(id: string) {
  return api<SavedExpertProfile>(`/api/profiles/${encodeURIComponent(id)}`);
}

export function updateProfileMetadata(
  id: string,
  patch: ProfileMetadataPatch,
) {
  return api<SavedExpertProfile>(`/api/profiles/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function activateExpertContext(profileId: string | null) {
  return api<ContextActivationResult>("/api/expert-contexts/activate", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId }),
  });
}

export async function getActiveExpertContext() {
  const response = await api<unknown>("/api/expert-contexts/current");
  const record = asRecord(response);
  if (!record) return null;
  const kind = record.kind === "expert_mask" ? "expert_mask" : "baseline";
  return {
    id: stringValue(record.context_id ?? record.id, `${kind}-context`),
    kind,
    name: kind === "baseline" ? "Baseline" : "Expert mask",
    profile_id: nullableString(record.profile_id),
    profile_fingerprint: nullableString(record.profile_fingerprint),
    context_fingerprint: nullableString(record.context_fingerprint),
    topology_fingerprint: nullableString(record.topology_fingerprint),
    active: true,
  } satisfies import("./types").InterventionContextRef;
}

export function loadModel(modelId: string) {
  return api<import("../types").JobRecord>("/api/model-sessions", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

export function normalizeExperiment(value: unknown): ExperimentRecord {
  const envelope = asRecord(value) ?? {};
  const record = asRecord(envelope.experiment) ?? envelope;
  const workload = asRecord(record.workload);
  const model = asRecord(record.model);
  const rawLanes = arrayValue(record.lanes);
  const rawRuns = arrayValue(envelope.workload_runs ?? record.workload_runs);
  const rawUnits = arrayValue(envelope.units ?? record.units ?? record.run_units);
  const kind = workloadKind(record.workload_kind ?? workload?.kind ?? record.kind);
  const lanes = rawLanes.map((lane, index) => {
    const laneRecord = asRecord(lane);
    const laneId = stringValue(laneRecord?.id, laneRole(laneRecord?.role, index));
    const run = rawRuns.find((candidate) => asRecord(candidate)?.lane_id === laneId);
    const laneUnits = rawUnits.filter((candidate) => asRecord(candidate)?.lane_id === laneId);
    return normalizeLane(lane, index, run, laneUnits);
  });
  const status = experimentStatus(record.status);
  const progressCurrent = numberValue(record.completed_units, 0);
  const progressTotal = numberValue(record.total_units, 0);
  const passed = numberValue(record.passed_units, 0);
  const failed = numberValue(
    record.failed_units,
    rawUnits.filter((unit) => ["failed", "error"].includes(
      stringValue(asRecord(unit)?.status, "queued"),
    )).length,
  );
  const unscored = numberValue(
    record.unscored_units,
    rawUnits.filter((unit) => asRecord(unit)?.status === "unscored").length,
  );
  const activeUnit = rawUnits.find((unit) => asRecord(unit)?.status === "running");
  const activeUnitRecord = asRecord(activeUnit);
  const comparison = asRecord(record.comparison);
  const laneMetrics = Object.fromEntries(lanes.map((lane) => {
    const run = rawRuns.find((candidate) => asRecord(candidate)?.lane_id === lane.id);
    return [lane.role, normalizeDriftMetrics(
      asRecord(asRecord(run)?.aggregate_metrics),
      comparison,
      lane.role,
    )];
  })) as Partial<Record<LaneRole, DriftMetrics | null>>;
  const modelId = stringValue(record.model_id ?? model?.id, "unknown-model");
  return {
    id: stringValue(record.id, "unknown-experiment"),
    name: stringValue(record.name, `${humanize(kind)} experiment`),
    status,
    workload_kind: kind,
    workload_id: stringValue(record.workload_id ?? workload?.id, "unknown-workload"),
    workload_name: stringValue(
      record.workload_name ?? workload?.title ?? workload?.name,
      humanize(kind),
    ),
    model_id: modelId,
    model_name: stringValue(record.model_name ?? model?.display_name, compactModel(modelId)),
    execution_config: asRecord(record.execution_config) ?? {},
    cohort_fingerprint: nullableString(
      record.cohort_fingerprint ?? workload?.content_fingerprint ?? workload?.fingerprint,
    ),
    created_at: stringValue(record.created_at, new Date(0).toISOString()),
    started_at: nullableString(record.started_at),
    completed_at: nullableString(record.completed_at),
    active_unit_id: activeUnitRecord
      ? unitAlignment(activeUnitRecord)
      : nullableString(record.active_unit_id),
    active_lane_id: nullableString(record.active_lane_id ?? activeUnitRecord?.lane_id),
    phase: stringValue(record.phase, status),
    progress_current: progressCurrent,
    progress_total: progressTotal,
    passed,
    failed,
    unscored,
    lanes,
    units: normalizeExperimentUnits(rawUnits, lanes, kind, workload, comparison),
    evaluation_contract: nullableTyped<EvaluationContract>(record.evaluation_contract),
    execution_policy: nullableTyped<ExecutionPolicy>(record.execution_policy),
    generation: nullableTyped<GenerationConfig>(record.generation),
    drift_metrics: {
      baseline: laneMetrics.baseline ?? normalizeDriftMetrics(
        asRecord(record.drift_metrics)?.baseline ?? record.baseline_drift_metrics,
        comparison,
        "baseline",
      ),
      candidate: laneMetrics.candidate ?? normalizeDriftMetrics(
        asRecord(record.drift_metrics)?.candidate ?? record.candidate_drift_metrics,
        comparison,
        "candidate",
      ),
    },
    comparison,
    error: nullableString(record.error),
    legacy_source: normalizeLegacySource(record.legacy_source),
  };
}

export function normalizeWorkload(value: unknown): WorkloadDescriptor {
  const record = asRecord(value) ?? {};
  const runtime = asRecord(record.runtime);
  const metadata = asRecord(record.metadata) ?? {};
  const readinessRecord = asRecord(record.readiness);
  const readiness = workloadReadiness(
    readinessRecord?.status ?? record.readiness ?? record.preparation_status ?? record.preparation_state,
    Boolean(record.ready ?? readinessRecord?.launchable),
  );
  const criteria = arrayValue(
    record.public_success_criteria ?? record.success_criteria,
  ).map((item) => stringValue(item, "")).filter(Boolean);
  const units = arrayValue(record.units ?? record.tasks);
  return {
    id: stringValue(record.id, "unknown-workload"),
    kind: workloadKind(record.kind ?? record.workload_kind),
    title: stringValue(record.title ?? record.name, "Untitled workload"),
    description: stringValue(record.description, ""),
    family: stringValue(record.family ?? record.pack_id ?? record.parent_id ?? metadata.family, "first-party"),
    pack_id: nullableString(record.pack_id ?? record.parent_id),
    source: stringValue(record.source, "first-party"),
    source_revision: nullableString(record.source_revision ?? record.revision),
    language: nullableString(record.language),
    difficulty: nullableString(record.difficulty),
    expected_horizon: nullableNumber(record.expected_horizon),
    tools: stringArray(record.tools ?? record.tool_names),
    tags: stringArray(record.tags ?? metadata.tags),
    public_problem_statement: stringValue(
      record.public_problem_statement ?? record.instruction,
      stringValue(record.description, ""),
    ),
    public_success_criteria: criteria,
    runtime: runtime
      ? {
          label: stringValue(runtime.label ?? runtime.name, "Isolated runtime"),
          image: nullableString(runtime.image ?? runtime.image_ref),
          image_digest: nullableString(runtime.image_digest),
          timeout_seconds: nullableNumber(runtime.timeout_seconds),
        }
      : typeof record.runtime === "string" && record.runtime.trim()
        ? {
            label: record.runtime,
            image: null,
            image_digest: runtimeDigest(record.runtime),
            timeout_seconds: runtimeTimeout(record.runtime),
          }
        : null,
    readiness,
    blocked_reasons: blockedReasons(record, readinessRecord),
    unit_count: numberValue(
      record.unit_count ?? record.task_count,
      stringArray(record.unit_ids).length || units.length || 1,
    ),
    unit_ids: stringArray(record.unit_ids).length
      ? stringArray(record.unit_ids)
      : units.map((unit) => stringValue(asRecord(unit)?.id, "")).filter(Boolean),
    metadata,
  };
}

export function normalizeModel(value: ModelRegistryEntry): ModelDescriptor {
  const record = value as ModelRegistryEntry & Record<string, unknown>;
  return {
    ...value,
    architecture: nullableString(record.architecture) ?? undefined,
    dtype: nullableString(record.dtype) ?? undefined,
    quantization: nullableString(record.quantization),
    tensor_parallel_size: numberValue(record.tensor_parallel_size, 1),
    recommended_hardware: nullableString(record.recommended_hardware) ?? undefined,
    minimum_memory_gib: nullableNumber(record.minimum_memory_gib),
    qualification_status: modelCapabilityStatus(record.qualification_status),
    masking: capability(record.masking_status, record.failure_reason),
    routing_telemetry: capability(record.routing_telemetry_status, record.failure_reason),
    hot_switch: capability(record.hot_switch_status, record.failure_reason),
    last_live_evidence: nullableString(record.last_live_evidence),
    failure_reason: nullableString(record.failure_reason),
  };
}

function normalizeLane(
  value: unknown,
  index: number,
  runValue?: unknown,
  unitValues: unknown[] = [],
): ExperimentLane {
  const record = asRecord(value) ?? {};
  const run = asRecord(runValue) ?? {};
  const intervention = asRecord(record.intervention ?? record.context) ?? {};
  const role = laneRole(record.role, index);
  const completed = numberValue(
    run.completed_units ?? record.progress_current ?? record.completed_units,
    unitValues.filter((unit) => terminalUnitStatus(asRecord(unit)?.status)).length,
  );
  const passed = numberValue(
    run.passed_units ?? record.passed ?? record.passed_units,
    unitValues.filter((unit) => asRecord(unit)?.status === "passed").length,
  );
  const failed = numberValue(
    record.failed ?? record.failed_units,
    unitValues.filter((unit) => ["failed", "error"].includes(
      stringValue(asRecord(unit)?.status, "queued"),
    )).length,
  );
  const unscored = numberValue(
    record.unscored ?? record.unscored_units,
    unitValues.filter((unit) => asRecord(unit)?.status === "unscored").length,
  );
  const aggregate = asRecord(run.aggregate_metrics);
  return {
    id: stringValue(record.id, role),
    role,
    label: stringValue(record.label, role === "baseline" ? "Baseline" : "Candidate"),
    status: experimentStatus(record.status),
    intervention: {
      id: stringValue(
        intervention.id ?? intervention.context_id ?? record.context_id,
        `${role}-context`,
      ),
      kind:
        stringValue(intervention.kind, role === "baseline" ? "baseline" : "expert_mask") ===
        "baseline"
          ? "baseline"
          : "expert_mask",
      name: stringValue(
        intervention.name ?? record.profile_name ?? record.label,
        role === "baseline" ? "Baseline" : "Candidate profile",
      ),
      profile_id: nullableString(intervention.profile_id ?? record.profile_id),
      profile_fingerprint: nullableString(
        intervention.profile_fingerprint ?? record.profile_fingerprint,
      ),
      context_fingerprint: nullableString(
        intervention.context_fingerprint ?? record.context_fingerprint,
      ),
      topology_fingerprint: nullableString(intervention.topology_fingerprint),
      active: Boolean(intervention.active ?? record.active),
    },
    progress_current: completed,
    progress_total: numberValue(
      run.total_units ?? record.progress_total ?? record.total_units,
      unitValues.length,
    ),
    passed,
    failed,
    unscored,
    score: nullableNumber(
      record.score ?? record.mean_reward ?? aggregate?.mean_state_fidelity_auc,
    ),
    performance: aggregateLanePerformance(unitValues),
  };
}

function normalizeUnit(value: unknown, index: number): ExperimentUnit {
  const record = asRecord(value) ?? {};
  const results = asRecord(record.results);
  return {
    id: stringValue(record.id ?? record.unit_id, `unit-${index + 1}`),
    run_unit_ids: {
      baseline: nullableString(record.id ?? record.unit_id),
      candidate: null,
    },
    index: numberValue(record.index ?? record.sequence, index + 1),
    title: stringValue(record.title ?? record.item_id ?? record.task_id, `Unit ${index + 1}`),
    subtitle: stringValue(record.subtitle ?? record.category, ""),
    prompt: nullableString(record.prompt ?? record.problem_statement),
    status: stringValue(record.status, "queued"),
    baseline: normalizeUnitResult(record.baseline ?? results?.baseline),
    candidate: normalizeUnitResult(record.candidate ?? results?.candidate),
  };
}

function normalizeExperimentUnits(
  values: unknown[],
  lanes: ExperimentLane[],
  kind: WorkloadKind,
  workload: Record<string, unknown> | null,
  comparison: Record<string, unknown> | null,
): ExperimentUnit[] {
  const detailed = values.some((value) => {
    const record = asRecord(value);
    return Boolean(record?.lane_id && record.unit_key);
  });
  if (!detailed) return values.map((value, index) => normalizeUnit(value, index));
  const roles = new Map(lanes.map((lane) => [lane.id, lane.role]));
  const grouped = new Map<string, ExperimentUnit>();
  for (const value of values) {
    const record = asRecord(value) ?? {};
    const alignment = unitAlignment(record);
    const role = roles.get(stringValue(record.lane_id, "")) ?? "baseline";
    const current = grouped.get(alignment);
    const result = normalizeDetailedUnitResult(record, kind, comparison);
    const status = stringValue(record.status, "queued");
    const runUnitId = nullableString(record.id);
    const prompt = nullableString(record.prompt ?? record.problem_statement) ??
      resultPrompt(record.result, kind) ??
      firstTransitionInstruction(record.result) ??
      nullableString(workload?.public_problem_statement);
    if (current) {
      current[role] = result;
      current.run_unit_ids[role] = runUnitId;
      current.prompt ??= prompt;
      current.status = combinedUnitStatus(current.status, status);
      continue;
    }
    grouped.set(alignment, {
      id: alignment,
      run_unit_ids: {
        baseline: role === "baseline" ? runUnitId : null,
        candidate: role === "candidate" ? runUnitId : null,
      },
      index: grouped.size + 1,
      title: stringValue(
        record.scenario_id ?? workload?.name ?? workload?.title,
        humanize(alignment.split(":")[0] ?? alignment),
      ),
      subtitle: [
        nullableString(record.condition)?.replaceAll("_", " "),
        record.horizon == null ? null : `horizon ${String(record.horizon)}`,
        record.seed == null ? null : `seed ${String(record.seed)}`,
      ].filter(Boolean).join(" · "),
      prompt,
      status,
      baseline: role === "baseline" ? result : null,
      candidate: role === "candidate" ? result : null,
    });
  }
  return [...grouped.values()];
}

function unitAlignment(record: Record<string, unknown>) {
  return [
    record.scenario_id,
    record.condition,
    record.horizon == null ? null : `h${String(record.horizon)}`,
    record.seed == null ? null : `s${String(record.seed)}`,
  ].filter((part) => part != null && part !== "").join(":") ||
    stringValue(record.unit_key, stringValue(record.id, "unit"))
      .replace(/:(baseline|candidate)$/, "");
}

function normalizeDetailedUnitResult(
  record: Record<string, unknown>,
  workloadKindValue: WorkloadKind,
  comparison: Record<string, unknown> | null,
): LaneUnitResult {
  const result = asRecord(record.result) ?? {};
  const item = asRecord(result.item) ?? {};
  const evaluation = asRecord(record.evaluation) ?? {};
  const deterministic = asRecord(evaluation.deterministic) ?? {};
  const itemEvaluation = asRecord(item.evaluation) ?? {};
  const metrics = asRecord(evaluation.metrics) ?? asRecord(result.metrics) ?? {};
  const kind = explicitWorkloadKind(result.kind) ?? workloadKindValue;
  const passed = nullableBoolean(
    evaluation.passed ?? deterministic.passed ?? itemEvaluation.passed ??
    item.passed ?? result.passed,
  ) ?? (
    record.status === "passed" ? true : record.status === "failed" ? false : null
  );
  const score = nullableNumber(
    evaluation.score ?? deterministic.combined_score ?? itemEvaluation.combined_score ??
    result.score ?? result.reward ?? metrics.area_under_state_fidelity_curve,
  );
  const criteria = normalizeResultCriteria({
    kind,
    record,
    result,
    item,
    evaluation,
    deterministic,
    itemEvaluation,
    passed,
  });
  const routing = normalizeRoutingEvidence(result.routing);
  const routingRecords = Array.isArray(routing) ? routing : [];
  if (kind === "state_drift" && score != null) {
    criteria.push({
      label: "State fidelity across checkpoints",
      passed: score === 1,
      detail: `${(score * 100).toFixed(1)}% area under the fidelity curve`,
    });
  }
  return {
    kind,
    status: stringValue(record.status, "queued"),
    passed,
    score,
    output: normalizeResultOutput(result, item, kind),
    reasoning: nullableString(result.reasoning ?? item.reasoning),
    error: nullableString(record.error ?? result.error ?? item.error),
    criteria,
    checkpoints: arrayValue(result.transitions ?? result.checkpoints).map(
      (checkpoint, index) => normalizeCheckpoint(
        checkpoint,
        index,
        routingRecords,
        routingComparisonFor(record, checkpoint, index, comparison),
      ),
    ),
    performance: record.performance ? normalizePerformance(record.performance) : null,
    routing,
    trajectory: trajectoryItems(result.trajectory),
  };
}

function normalizeUnitResult(value: unknown): LaneUnitResult | null {
  const record = asRecord(value);
  if (!record) return null;
  return {
    kind: explicitWorkloadKind(record.kind),
    status: stringValue(record.status, "unknown"),
    passed: nullableBoolean(record.passed),
    score: nullableNumber(record.score ?? record.reward),
    output: nullableString(record.output ?? record.response),
    reasoning: nullableString(record.reasoning),
    error: nullableString(record.error),
    criteria: arrayValue(record.criteria).map((criterion) => {
      const item = asRecord(criterion) ?? {};
      return {
        label: stringValue(item.label ?? item.name, "Criterion"),
        passed: nullableBoolean(item.passed),
        detail: nullableString(item.detail ?? item.message),
      };
    }),
    checkpoints: arrayValue(record.checkpoints).map(
      (checkpoint, index) => normalizeCheckpoint(checkpoint, index),
    ),
    performance: record.performance ? normalizePerformance(record.performance) : null,
    routing: normalizeRoutingEvidence(record.routing),
    trajectory: trajectoryItems(record.trajectory),
  };
}

function normalizeCheckpoint(
  value: unknown,
  index: number,
  routingRecords: Array<Record<string, unknown>> = [],
  routingComparison: RoutingCheckpointComparison | null = null,
): DriftCheckpoint {
  const record = asRecord(value) ?? {};
  const checkpoint = numberValue(record.index ?? record.checkpoint, index + 1);
  const rawStatus = typeof record.exact === "boolean"
    ? record.exact ? "exact" : "diverged"
    : stringValue(record.status, "pending");
  const status = ["pending", "exact", "diverged", "recovered", "failed"].includes(rawStatus)
    ? rawStatus as DriftCheckpoint["status"]
    : "pending";
  return {
    index: checkpoint,
    label: stringValue(record.label ?? record.instruction, `Checkpoint ${index + 1}`),
    status,
    expected_state: record.expected_state ?? null,
    actual_state: record.actual_state ?? null,
    state_fidelity: nullableNumber(record.state_fidelity),
    state_distance: nullableNumber(record.state_distance),
    invariant_violations: stringArray(record.invariant_violations),
    invalid_tool_calls: numberValue(
      record.invalid_tool_calls,
      record.valid_tool_call === false ? 1 : 0,
    ),
    routing_delta: nullableNumber(record.routing_delta),
    routing_evidence: routingRecords.find(
      (candidate) => nullableNumber(candidate.checkpoint) === checkpoint,
    ) ?? null,
    routing_comparison: routingComparison,
  };
}

function normalizePerformance(value: unknown): LanePerformance {
  const record = asRecord(value) ?? {};
  const prompt = numberValue(record.prompt_tokens, 0);
  const completion = numberValue(record.completion_tokens, 0);
  const reasoning = nullableNumber(record.reasoning_tokens);
  return {
    elapsed_ms: nullableNumber(record.elapsed_ms ?? record.wall_time_ms ?? record.latency_ms),
    eta_ms: nullableNumber(record.eta_ms),
    prompt_tokens: prompt,
    reasoning_tokens: reasoning,
    completion_tokens: completion,
    total_tokens: numberValue(
      record.total_tokens,
      prompt + completion + (reasoning ?? 0),
    ),
    current_tps: nullableNumber(record.current_tps ?? record.tokens_per_second),
    mean_tps: nullableNumber(record.mean_tps ?? record.mean_tokens_per_second),
    estimated_cost_usd: nullableNumber(record.estimated_cost_usd),
  };
}

function normalizeDriftMetrics(
  value: unknown,
  comparison?: Record<string, unknown> | null,
  role?: LaneRole,
): DriftMetrics | null {
  const record = asRecord(value);
  if (!record) return null;
  return {
    final_success: nullableNumber(record.final_success ?? record.final_success_rate),
    transition_accuracy: nullableNumber(
      record.transition_accuracy ?? record.mean_transition_accuracy,
    ),
    area_under_fidelity_curve: nullableNumber(
      record.area_under_fidelity_curve ?? record.state_fidelity_auc ?? record.mean_state_fidelity_auc,
    ),
    first_divergence_checkpoint: nullableNumber(
      record.first_divergence_checkpoint ??
      (role === "baseline"
        ? comparison?.baseline_median_first_divergence
        : comparison?.candidate_median_first_divergence),
    ),
    recovery_rate: nullableNumber(record.recovery_rate),
    horizon_at_80: nullableNumber(
      record.horizon_at_80 ?? record.horizon_at_80_percent_reliability,
    ),
    horizon_at_50: nullableNumber(
      record.horizon_at_50 ?? record.horizon_at_50_percent_reliability,
    ),
    compounding_penalty: nullableNumber(
      record.compounding_penalty ??
      (role === "baseline"
        ? comparison?.baseline_compounding_penalty
        : comparison?.candidate_compounding_penalty),
    ),
    excess_mask_drift: nullableNumber(
      record.excess_mask_drift ?? comparison?.excess_compounding_penalty ??
      comparison?.baseline_to_mask_paired_drift_delta,
    ),
  };
}

export function normalizeEvent(
  value: unknown,
  experimentId: string,
  fallbackSequence: number,
): RunEvent {
  const record = asRecord(value) ?? {};
  return {
    experiment_id: stringValue(record.experiment_id, experimentId),
    sequence: numberValue(record.sequence, fallbackSequence),
    type: stringValue(record.type ?? record.event_type ?? record.kind, "update"),
    phase: nullableString(record.phase),
    message: stringValue(
      record.message,
      humanize(stringValue(record.type ?? record.kind, "update")),
    ),
    lane_id: nullableString(record.lane_id),
    lane_role: nullableLaneRole(record.lane_role ?? record.role),
    unit_id: nullableString(record.unit_id ?? record.run_unit_id),
    checkpoint: nullableNumber(record.checkpoint),
    created_at: stringValue(record.created_at ?? record.timestamp, new Date(0).toISOString()),
    payload: asRecord(record.payload ?? record.data) ?? {},
  };
}

function aggregateLanePerformance(values: unknown[]): LanePerformance {
  const snapshots = values
    .map((value) => asRecord(asRecord(value)?.performance))
    .filter((value): value is Record<string, unknown> => value !== null);
  if (!snapshots.length) return { ...emptyPerformance };
  const prompt = snapshots.reduce((total, item) => total + numberValue(item.prompt_tokens, 0), 0);
  const completion = snapshots.reduce((total, item) => total + numberValue(item.completion_tokens, 0), 0);
  const reasoningValues = snapshots.map((item) => nullableNumber(item.reasoning_tokens));
  const elapsed = snapshots.reduce(
    (total, item) => total + numberValue(item.elapsed_ms ?? item.latency_ms, 0),
    0,
  );
  const tpsValues = snapshots
    .map((item) => nullableNumber(item.tokens_per_second))
    .filter((value): value is number => value !== null);
  return {
    elapsed_ms: elapsed || null,
    eta_ms: null,
    prompt_tokens: prompt,
    reasoning_tokens: reasoningValues.some((value) => value !== null)
      ? reasoningValues.reduce<number>((total, value) => total + (value ?? 0), 0)
      : null,
    completion_tokens: completion,
    total_tokens: snapshots.reduce(
      (total, item) => total + numberValue(item.total_tokens, 0),
      0,
    ),
    current_tps: tpsValues.at(-1) ?? null,
    mean_tps: tpsValues.length
      ? tpsValues.reduce((total, value) => total + value, 0) / tpsValues.length
      : null,
    estimated_cost_usd: null,
  };
}

function firstTransitionInstruction(value: unknown) {
  const result = asRecord(value);
  const first = asRecord(arrayValue(result?.transitions)[0]);
  return nullableString(first?.instruction);
}

function resultPrompt(value: unknown, kind: WorkloadKind) {
  const result = asRecord(value);
  const item = asRecord(result?.item);
  const nested = asRecord(result?.answer ?? result?.response);
  if (kind === "coding") {
    const task = asRecord(result?.task);
    return nullableString(result?.prompt ?? task?.prompt ?? task?.instruction);
  }
  return nullableString(
    result?.prompt ?? item?.prompt ?? nested?.prompt ?? nested?.input,
  );
}

function normalizeResultOutput(
  result: Record<string, unknown>,
  item: Record<string, unknown>,
  kind: WorkloadKind,
) {
  const nested = asRecord(result.answer ?? result.response);
  const value = kind === "answer"
    ? result.output ?? item.output ?? nested?.output ?? nested?.content ??
      result.response ?? item.response
    : result.output ?? item.output ?? result.response ?? item.response;
  return displayText(value);
}

function normalizeResultCriteria({
  kind,
  record,
  result,
  item,
  evaluation,
  deterministic,
  itemEvaluation,
  passed,
}: {
  kind: WorkloadKind;
  record: Record<string, unknown>;
  result: Record<string, unknown>;
  item: Record<string, unknown>;
  evaluation: Record<string, unknown>;
  deterministic: Record<string, unknown>;
  itemEvaluation: Record<string, unknown>;
  passed: boolean | null;
}): LaneUnitResult["criteria"] {
  const sources = [
    deterministic.criteria,
    evaluation.criteria,
    itemEvaluation.criteria,
    item.criteria,
    result.criteria,
  ];
  const source = sources.find((candidate) => arrayValue(candidate).length > 0);
  const criteria = arrayValue(source).map((criterion, index) => {
    const candidate = asRecord(criterion);
    if (!candidate) {
      return {
        label: stringValue(criterion, `Criterion ${index + 1}`),
        passed: null,
        detail: null,
      };
    }
    const criterionPassed = nullableBoolean(
      candidate.passed ?? candidate.satisfied ?? candidate.success,
    );
    const score = nullableNumber(candidate.score);
    return {
      label: stringValue(
        candidate.label ?? candidate.name,
        humanize(stringValue(candidate.criterion_id ?? candidate.id, `Criterion ${index + 1}`)),
      ),
      passed: criterionPassed,
      detail: nullableString(
        candidate.explanation ?? candidate.detail ?? candidate.message ?? candidate.error,
      ) ?? (score == null ? null : `${(score * 100).toFixed(1)}% score`),
    };
  });
  if (criteria.length) return criteria;

  const status = stringValue(record.status, "queued");
  const label = kind === "state_drift"
    ? "Final canonical state"
    : kind === "coding"
      ? "Trusted verifier"
      : "Evaluation contract";
  return [{
    label,
    passed,
    detail: passed == null
      ? terminalUnitStatus(status)
        ? "Completed without a scored verdict"
        : "Evaluation has not completed"
      : passed
        ? kind === "state_drift"
          ? "Final state is exact"
          : kind === "coding"
            ? "Trusted verifier accepted the workspace"
            : "Response satisfied the task scorer"
        : kind === "state_drift"
          ? "Final state diverged"
          : kind === "coding"
            ? "Trusted verifier rejected the workspace"
            : "Response did not satisfy the task scorer",
  }];
}

function normalizeRoutingEvidence(value: unknown): LaneUnitResult["routing"] {
  const record = asRecord(value);
  if (record) return record;
  const records = arrayValue(value)
    .map((item) => asRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null);
  return records.length ? records : null;
}

function routingComparisonFor(
  unit: Record<string, unknown>,
  checkpointValue: unknown,
  checkpointIndex: number,
  comparison: Record<string, unknown> | null,
): RoutingCheckpointComparison | null {
  const checkpoint = numberValue(
    asRecord(checkpointValue)?.checkpoint ?? asRecord(checkpointValue)?.index,
    checkpointIndex + 1,
  );
  const pair = arrayValue(comparison?.routing_checkpoint_pairs)
    .map((value) => asRecord(value))
    .find((candidate) =>
      candidate !== null &&
      candidate.scenario_id === unit.scenario_id &&
      candidate.condition === unit.condition &&
      candidate.horizon === unit.horizon &&
      candidate.seed === unit.seed &&
      candidate.checkpoint === checkpoint
    );
  if (!pair) return null;
  return {
    selection_overlap: nullableNumber(pair.selection_overlap),
    routing_mass_js_divergence: nullableNumber(pair.routing_mass_js_divergence),
    at_candidate_first_divergence: pair.at_candidate_first_divergence === true,
    largest_selection_shifts: arrayValue(pair.largest_selection_shifts)
      .map((value) => asRecord(value))
      .filter((value): value is Record<string, unknown> => value !== null)
      .map((shift) => ({
        layer: numberValue(shift.layer, 0),
        expert: numberValue(shift.expert, 0),
        baseline_share: nullableNumber(shift.baseline_share),
        candidate_share: nullableNumber(shift.candidate_share),
        delta: nullableNumber(shift.delta),
      })),
  };
}

function trajectoryItems(value: unknown) {
  const record = asRecord(value);
  return arrayValue(record?.steps ?? value).map((item) => asRecord(item) ?? {});
}

function explicitWorkloadKind(value: unknown): WorkloadKind | null {
  return value === "answer" || value === "coding" || value === "state_drift"
    ? value
    : null;
}

function displayText(value: unknown): string | null {
  if (typeof value === "string") return value.trim() ? value : null;
  if (value == null) return null;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function combinedUnitStatus(current: string, incoming: string) {
  if (current === "running" || incoming === "running") return "running";
  if (current === "failed" || incoming === "failed") return "failed";
  if (current === "error" || incoming === "error") return "error";
  if (current === "passed" && incoming === "passed") return "passed";
  if (terminalUnitStatus(current) && terminalUnitStatus(incoming)) return "completed";
  return current === "queued" ? incoming : current;
}

function terminalUnitStatus(value: unknown) {
  return ["passed", "failed", "unscored", "error", "cancelled", "completed"].includes(
    stringValue(value, "queued"),
  );
}

function blockedReasons(
  record: Record<string, unknown>,
  readiness: Record<string, unknown> | null,
) {
  const reasons = stringArray(
    readiness?.blocked_reasons ?? record.blocked_reasons ?? record.limitations,
  );
  const singular = nullableString(record.blocked_reason);
  return singular && !reasons.includes(singular) ? [singular, ...reasons] : reasons;
}

function runtimeDigest(value: string) {
  return value.match(/(?:sha256:)?[0-9a-f]{64}/i)?.[0] ?? null;
}

function runtimeTimeout(value: string) {
  const seconds = value.match(/([0-9]+(?:\.[0-9]+)?)s timeout/i)?.[1];
  return seconds ? Number(seconds) : null;
}

function modelCapabilityStatus(
  value: unknown,
): NonNullable<ModelDescriptor["qualification_status"]> {
  return ["qualified", "experimental", "manifest_only", "unsupported"].includes(
    stringValue(value, "unsupported"),
  )
    ? stringValue(value, "unsupported") as NonNullable<ModelDescriptor["qualification_status"]>
    : "unsupported";
}

function capability(
  value: unknown,
  reason: unknown,
): NonNullable<ModelDescriptor["masking"]> {
  const status = modelCapabilityStatus(value);
  return {
    status,
    supported: status === "qualified" || status === "experimental",
    reason: nullableString(reason),
  };
}

function compactModel(value: string) {
  return value.split("/").at(-1)?.replaceAll("-", " ") ?? value;
}

async function legacyWorkloads(): Promise<WorkloadDescriptor[]> {
  const [benchmarks, packs] = await Promise.all([
    api<BenchmarkInfo[]>("/api/benchmarks"),
    api<AgentTaskPack[]>("/api/agent-task-packs"),
  ]);
  const answer = benchmarks.map((benchmark) => normalizeWorkload({
    id: benchmark.id,
    kind: "answer",
    title: benchmark.name,
    description: benchmark.description,
    family: benchmark.kind,
    source: benchmark.source,
    revision: benchmark.revision,
    tags: benchmark.categories,
    public_problem_statement: benchmark.description,
    public_success_criteria: [humanize(benchmark.scoring)],
    ready: benchmark.ready,
    readiness: benchmark.ready ? "ready" : "preparing",
    unit_count: benchmark.item_count,
    unit_ids: [],
    runtime: { label: "Warm model request", timeout_seconds: null },
  }));
  const taskGroups = await Promise.all(packs.map(async (pack) => {
    const tasks = await api<Array<Record<string, unknown>>>(
      `/api/agent-task-packs/${encodeURIComponent(pack.id)}/tasks`,
    );
    return tasks.map((task) => normalizeWorkload({
      ...task,
      id: `${pack.id}:${String(task.id)}`,
      kind: "coding",
      pack_id: pack.id,
      family: pack.id,
      source: pack.source,
      revision: pack.revision,
      ready: pack.ready,
      readiness: pack.ready ? "ready" : "blocked",
      unit_count: 1,
      unit_ids: [String(task.id)],
      runtime: {
        label: task.language ? `${String(task.language)} isolated sandbox` : "Isolated sandbox",
        timeout_seconds: task.timeout_seconds,
      },
    }));
  }));
  return [...answer, ...taskGroups.flat(), ...fallbackDriftWorkloads()];
}

function fallbackDriftWorkloads(): WorkloadDescriptor[] {
  return [
    ["ledger-reconciliation-v1", "Ledger and account reconciliation", 16],
    ["order-lifecycle-v1", "Order lifecycle and inventory correction", 12],
    ["dependency-release-v1", "Dependency graph and release planning", 16],
  ].map(([id, title, horizon]) => normalizeWorkload({
    id,
    kind: "state_drift",
    title,
    description: "Deterministic state-machine scenario with oracle-reset, chained, and state-anchored conditions.",
    family: "state-drift-first-party",
    source: "first-party manifest",
    expected_horizon: horizon,
    tools: ["typed state tools", "snapshot", "rollback"],
    tags: ["state", "horizon", "deterministic"],
    success_criteria: [
      "Preserve exact canonical state through every checkpoint.",
      "Avoid invariant violations and collateral mutations.",
      "Recover correctly after seeded divergence or rollback.",
    ],
    readiness: "manifest_only",
    blocked_reasons: ["The V2 state-drift adapter is not exposed by this backend build."],
    unit_count: 1,
    unit_ids: [id],
    runtime: { label: "Deterministic state machine", timeout_seconds: 900 },
  }));
}

function normalizeLegacySource(value: unknown) {
  const record = asRecord(value);
  if (!record) return null;
  return {
    kind: stringValue(record.kind, "legacy"),
    id: stringValue(record.id, "unknown"),
  };
}

function workloadKind(value: unknown): WorkloadKind {
  return value === "coding" || value === "state_drift" ? value : "answer";
}

function workloadReadiness(value: unknown, ready: boolean): WorkloadReadiness {
  if (ready) return "ready";
  const candidate = stringValue(value, "blocked");
  return [
    "ready",
    "preparing",
    "source_pinned",
    "manifest_only",
    "reference_only",
    "blocked",
    "unsupported",
  ].includes(candidate)
    ? candidate as WorkloadReadiness
    : "blocked";
}

function experimentStatus(value: unknown): ExperimentStatus {
  const candidate = stringValue(value, "queued");
  return [
    "queued",
    "preparing",
    "running",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
  ].includes(candidate)
    ? candidate as ExperimentStatus
    : "queued";
}

function laneRole(value: unknown, index: number): LaneRole {
  return value === "candidate" || (value == null && index > 0)
    ? "candidate"
    : "baseline";
}

function nullableLaneRole(value: unknown): LaneRole | null {
  return value === "baseline" || value === "candidate" ? value : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringArray(value: unknown): string[] {
  return arrayValue(value).map((item) => stringValue(item, "")).filter(Boolean);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nullableBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function nullableTyped<T>(value: unknown): T | null {
  return value && typeof value === "object" ? value as T : null;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export { emptyPerformance };
