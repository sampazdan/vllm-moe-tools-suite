import type {
  EvaluationContract,
  ExecutionPolicy,
  ExpertProfile,
  GenerationConfig,
  JobRecord,
  ModelRegistryEntry,
  SavedExpertProfile,
} from "../types";

export type WorkloadKind = "answer" | "coding" | "state_drift";

export type WorkloadReadiness =
  | "ready"
  | "preparing"
  | "source_pinned"
  | "manifest_only"
  | "reference_only"
  | "blocked"
  | "unsupported";

export type ExperimentStatus =
  | "queued"
  | "preparing"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export type LaneRole = "baseline" | "candidate";

export interface WorkloadDescriptor {
  id: string;
  kind: WorkloadKind;
  title: string;
  description: string;
  family: string;
  pack_id: string | null;
  source: string;
  source_revision: string | null;
  language: string | null;
  difficulty: string | null;
  expected_horizon: number | null;
  tools: string[];
  tags: string[];
  public_problem_statement: string;
  public_success_criteria: string[];
  runtime: {
    label: string;
    image: string | null;
    image_digest: string | null;
    timeout_seconds: number | null;
  } | null;
  readiness: WorkloadReadiness;
  blocked_reasons: string[];
  unit_count: number;
  unit_ids: string[];
  metadata: Record<string, unknown>;
}

export interface ModelCapability {
  status: "qualified" | "experimental" | "manifest_only" | "unsupported";
  supported: boolean;
  reason: string | null;
}

export interface ModelDescriptor extends ModelRegistryEntry {
  architecture?: string;
  dtype?: string;
  quantization?: string | null;
  tensor_parallel_size?: number;
  recommended_hardware?: string;
  minimum_memory_gib?: number | null;
  qualification_status?: ModelCapability["status"];
  masking?: ModelCapability;
  routing_telemetry?: ModelCapability;
  hot_switch?: ModelCapability;
  last_live_evidence?: string | null;
  failure_reason?: string | null;
}

export interface InterventionContextRef {
  id: string;
  kind: "baseline" | "expert_mask";
  name: string;
  profile_id: string | null;
  profile_fingerprint: string | null;
  context_fingerprint: string | null;
  topology_fingerprint?: string | null;
  active: boolean;
}

export interface LanePerformance {
  elapsed_ms: number | null;
  eta_ms: number | null;
  prompt_tokens: number;
  reasoning_tokens: number | null;
  completion_tokens: number;
  total_tokens: number;
  current_tps: number | null;
  mean_tps: number | null;
  estimated_cost_usd: number | null;
}

export interface DriftCheckpoint {
  index: number;
  label: string;
  status: "pending" | "exact" | "diverged" | "recovered" | "failed";
  expected_state: unknown;
  actual_state: unknown;
  state_fidelity: number | null;
  state_distance: number | null;
  invariant_violations: string[];
  invalid_tool_calls: number;
  routing_delta: number | null;
  routing_evidence: Record<string, unknown> | null;
  routing_comparison: RoutingCheckpointComparison | null;
}

export interface RoutingSelectionShift {
  layer: number;
  expert: number;
  baseline_share: number | null;
  candidate_share: number | null;
  delta: number | null;
}

export interface RoutingCheckpointComparison {
  selection_overlap: number | null;
  routing_mass_js_divergence: number | null;
  at_candidate_first_divergence: boolean;
  largest_selection_shifts: RoutingSelectionShift[];
}

export interface LaneUnitResult {
  kind: WorkloadKind | null;
  status: string;
  passed: boolean | null;
  score: number | null;
  output: string | null;
  reasoning: string | null;
  error: string | null;
  criteria: Array<{
    label: string;
    passed: boolean | null;
    detail?: string | null;
  }>;
  checkpoints: DriftCheckpoint[];
  performance: LanePerformance | null;
  routing: Record<string, unknown> | Array<Record<string, unknown>> | null;
  trajectory: Array<Record<string, unknown>>;
}

export interface ExperimentUnit {
  id: string;
  run_unit_ids: Record<LaneRole, string | null>;
  index: number;
  title: string;
  subtitle: string;
  prompt: string | null;
  status: string;
  baseline: LaneUnitResult | null;
  candidate: LaneUnitResult | null;
}

export interface ExperimentLane {
  id: string;
  role: LaneRole;
  label: string;
  status: ExperimentStatus;
  intervention: InterventionContextRef;
  progress_current: number;
  progress_total: number;
  passed: number;
  failed: number;
  unscored: number;
  score: number | null;
  performance: LanePerformance;
}

export interface DriftMetrics {
  final_success: number | null;
  transition_accuracy: number | null;
  area_under_fidelity_curve: number | null;
  first_divergence_checkpoint: number | null;
  recovery_rate: number | null;
  horizon_at_80: number | null;
  horizon_at_50: number | null;
  compounding_penalty: number | null;
  excess_mask_drift: number | null;
}

export interface ExperimentRecord {
  id: string;
  name: string;
  status: ExperimentStatus;
  workload_kind: WorkloadKind;
  workload_id: string;
  workload_name: string;
  model_id: string;
  model_name: string;
  execution_config: Record<string, unknown>;
  cohort_fingerprint: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  active_unit_id: string | null;
  active_lane_id: string | null;
  phase: string;
  progress_current: number;
  progress_total: number;
  passed: number;
  failed: number;
  unscored: number;
  lanes: ExperimentLane[];
  units: ExperimentUnit[];
  evaluation_contract: EvaluationContract | null;
  execution_policy: ExecutionPolicy | null;
  generation: GenerationConfig | null;
  drift_metrics: Record<LaneRole, DriftMetrics | null>;
  comparison: Record<string, unknown> | null;
  error: string | null;
  legacy_source: { kind: string; id: string } | null;
}

export interface RunEvent {
  experiment_id: string;
  sequence: number;
  type: string;
  phase: string | null;
  message: string;
  lane_id: string | null;
  lane_role: LaneRole | null;
  unit_id: string | null;
  checkpoint: number | null;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface ExperimentCreateRequest {
  name: string;
  model_id: string;
  workload_id: string;
  candidate_profile_id: string | null;
  agent_id?: string;
  sandbox_provider_id?: string;
  scenario_ids?: string[];
  conditions?: string[];
  horizons?: number[];
  seeds: number[];
}

export interface ExperimentPage {
  items: ExperimentRecord[];
  total: number;
  offset: number;
  limit: number;
}

export interface ExperimentEventPage {
  items: RunEvent[];
  next_sequence: number;
}

export interface ProfileMetadataPatch {
  name?: string;
  description?: string;
}

export interface ContextActivationResult {
  old_context_id: string | null;
  old_context_fingerprint: string | null;
  new_context_id: string;
  new_context_fingerprint: string;
  topology_fingerprint: string;
  duration_ms: number;
  process_id: number | null;
  weights_reloaded: false;
  activated_at: string;
}

export type {
  ExpertProfile,
  JobRecord,
  SavedExpertProfile,
};
