export type ModelState =
  | "unloaded"
  | "starting"
  | "ready"
  | "stopping"
  | "stopped"
  | "failed";

export interface SystemStatus {
  mode: "mock" | "vllm";
  version: string;
  model_state: ModelState;
  data_dir: string;
}

export interface RuntimeStatus {
  managed: boolean;
  model_id: string;
  pid: number | null;
  session_id: string | null;
  started_at: string | null;
  log_path: string | null;
  profile_path: string | null;
  log_tail: string;
}

export interface SessionStatus {
  auth_required: boolean;
  authenticated: boolean;
  csrf_token: string | null;
  expires_at: string | null;
}

export type JobKind =
  | "model_load"
  | "benchmark_run"
  | "dataset_prepare"
  | "agent_run"
  | "experiment_run";
export type JobStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobRecord {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress_current: number;
  progress_total: number;
  result_id: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface ModelTopology {
  num_layers: number;
  num_experts: number;
  top_k: number;
  routed_layer_ids: number[];
}

export interface ModelRegistryEntry {
  id: string;
  display_name: string;
  enabled: boolean;
  revision: string | null;
  topology: ModelTopology | null;
  notes: string;
}

export interface ModelSession {
  id: string;
  model_id: string;
  state: ModelState;
  mode: "mock" | "vllm";
  profile: ExpertProfile | null;
  profile_id: string | null;
  created_at: string;
}

export type BenchmarkKind = "fixture" | "standard" | "custom";
export type ScoringMode =
  | "exact"
  | "gsm8k"
  | "regex"
  | "contains"
  | "ifeval"
  | "livebench"
  | "numeric"
  | "multiple_choice"
  | "json_schema"
  | "executable"
  | "instruction_constraints"
  | "llm_judge"
  | "ungraded";

export type EvaluationCriterionKind =
  | "benchmark_default"
  | "exact"
  | "contains"
  | "regex"
  | "numeric"
  | "multiple_choice"
  | "json"
  | "ungraded"
  | "verifier";

export interface EvaluationCriterion {
  id: string;
  kind: EvaluationCriterionKind;
  label: string;
  description?: string;
  visibility: "public" | "hidden";
  required: boolean;
  weight: number;
  case_sensitive: boolean;
  strip_whitespace: boolean;
  expected?: string | null;
  pattern?: string | null;
  numeric_tolerance?: number;
  json_schema?: Record<string, unknown> | null;
  verifier_command?: string | null;
  verifier_timeout_seconds?: number;
}

export interface EvaluationContract {
  version?: 1;
  name: string;
  description?: string;
  criteria: EvaluationCriterion[];
  aggregation: "all_required" | "weighted_threshold";
  pass_threshold: number;
  judge: LLMJudgeConfig | null;
  judge_weight: number;
  judge_can_override_deterministic_failure: boolean;
  fingerprint?: string;
}

export interface LLMJudgeConfig {
  provider: "anthropic" | "openai" | "fake";
  model: string;
  mode: "single" | "reference" | "pairwise";
  rubric: string;
  pass_threshold: number;
  repetitions: number;
  temperature?: number | null;
  max_output_tokens: number;
  input_cost_per_million_usd?: number | null;
  output_cost_per_million_usd?: number | null;
  rubric_hash?: string;
  fingerprint?: string;
}

export interface ExecutionPolicy {
  version?: 1;
  concurrency?: number;
  max_turns?: number | null;
  max_tokens: number | null;
  max_commands?: number | null;
  timeout_seconds: number;
  per_item_timeout_seconds?: number;
  max_cost_usd?: number | null;
  fail_fast?: boolean;
  attempts?: number;
  fingerprint?: string;
  // Generation presentation fields are stripped before policy submission.
  temperature?: number;
  seed?: number | null;
  enable_thinking?: boolean;
  reasoning_visibility?: "off" | "compact" | "full";
  generation_max_tokens?: number;
}

export interface GenerationConfig {
  temperature: number;
  max_tokens: number;
  seed: number | null;
  enable_thinking: boolean;
}

export interface BenchmarkInfo {
  id: string;
  name: string;
  description: string;
  item_count: number;
  categories: string[];
  kind: BenchmarkKind;
  source: string;
  revision: string;
  split: string;
  license: string;
  ready: boolean;
  scoring: ScoringMode;
  prompt_template_version: string;
  default_generation: GenerationConfig;
  evaluation_contract?: EvaluationContract | null;
  execution_policy?: ExecutionPolicy | null;
}

export interface BenchmarkItem {
  id: string;
  prompt: string;
  expected: string;
  category: string;
  scoring: ScoringMode;
  metadata: Record<string, string | number | boolean | null>;
  success_criteria?: EvaluationCriterion[];
  verifier_hash?: string | null;
}

export interface BenchmarkItemPage {
  benchmark: BenchmarkInfo;
  items: BenchmarkItem[];
  total: number;
  offset: number;
  limit: number;
  categories: string[];
}

export interface BenchmarkProblemDetail {
  benchmark: BenchmarkInfo;
  item: BenchmarkItem;
  rendered_prompt: string;
  success_criteria: {
    summary: string;
    public_criteria: EvaluationCriterion[];
    hidden_criteria_count: number;
    hidden_criteria_hash: string | null;
    evaluation_contract_fingerprint: string;
    public_contract: EvaluationContract | null;
  };
  default_execution_policy: ExecutionPolicy;
}

export interface BenchmarkDatasetRecord {
  id: string;
  info: BenchmarkInfo;
  content_hash: string;
  created_at: string;
}

export interface RunItemResult {
  item_id: string;
  attempt: number;
  prompt: string;
  expected: string;
  output: string;
  passed: boolean | null;
  scoring: ScoringMode;
  error: string | null;
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  judge_budget_debit_usd?: number;
  judge_cost_uncertain?: boolean;
}

export interface BenchmarkRunSummary {
  id: string;
  benchmark_id: string;
  model_session_id: string;
  status: string;
  score: number | null;
  scored_items: number;
  completed_items: number;
  total_items: number;
  cohort_id: string | null;
  created_at: string;
  evaluation_contract_fingerprint?: string | null;
  execution_policy_fingerprint?: string | null;
  performance?: RunPerformance | null;
}

export interface BenchmarkRun extends BenchmarkRunSummary {
  items: RunItemResult[];
}

export interface BenchmarkRunPage {
  items: BenchmarkRunSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface RunItemResultPage {
  run_id: string;
  items: RunItemResult[];
  total: number;
  offset: number;
  limit: number;
}

export interface RunPerformance {
  wall_time_ms: number;
  model_time_ms: number;
  evaluation_time_ms: number;
  prompt_tokens: number;
  reasoning_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  mean_tokens_per_second: number | null;
  p50_tokens_per_second: number | null;
  p95_tokens_per_second: number | null;
  inference_cost_usd: number | null;
  judge_cost_usd: number | null;
  judge_equivalent_cost_usd: number | null;
  judge_budget_debit_usd: number;
  judge_cost_uncertain: boolean;
  estimated_cost_usd: number | null;
}

export interface RunDetail {
  run: BenchmarkRunSummary;
  model_session: ModelSession;
  saved_profile: SavedExpertProfile | null;
  cohort: {
    id: string;
    item_count: number;
    generation?: GenerationConfig;
    evaluation_contract: EvaluationContract | null;
    evaluation_contract_fingerprint?: string | null;
    execution_policy: ExecutionPolicy | null;
    execution_policy_fingerprint?: string | null;
  } | null;
  provenance: {
    generation?: GenerationConfig;
    evaluation_contract: EvaluationContract | null;
    evaluation_contract_fingerprint: string | null;
    hidden_criteria_count: number;
    hidden_criteria_hash: string | null;
    execution_policy: ExecutionPolicy | null;
    execution_policy_fingerprint: string | null;
    profile_id: string | null;
    profile_fingerprint: string | null;
  } | null;
}

export interface RoutingSummary {
  run_id: string;
  layer_ids: number[];
  selection_counts: number[][];
  routing_mass: number[][];
  total_routed_slots: number;
}

export type RoutingSourceKind = "benchmark_run" | "agent_trial";

export interface RoutingSourceReference {
  kind: RoutingSourceKind;
  id: string;
  weight?: number;
}

export interface RoutingExploreSource extends RoutingSourceReference {
  label: string;
  status?: string | null;
  profile_id?: string | null;
}

export interface RoutingExploreRequest {
  sources: RoutingSourceReference[];
  comparison_sources?: RoutingSourceReference[];
  metric: "routing_mass" | "selection_count";
  filters?: {
    item_ids?: string[];
    trial_ids?: string[];
    step_types?: string[];
    passed?: boolean;
  };
}

export interface RoutingExploreResponse {
  fingerprint: string;
  aggregation: "weighted_source_normalized";
  model_id: string;
  profile_id?: string | null;
  profile_fingerprint?: string | null;
  layer_ids: number[];
  num_experts: number;
  top_k: number;
  selection_counts: number[][];
  routing_mass: number[][];
  comparison_selection_counts?: number[][] | null;
  comparison_routing_mass?: number[][] | null;
  total_routed_slots: number;
  captured_inference_calls?: number | null;
  total_inference_calls?: number | null;
  served_tokens?: number | null;
  sources: RoutingExploreSource[];
  comparison_sources?: RoutingExploreSource[];
  filter_capabilities: {
    item: boolean;
    trial: boolean;
    step_type: boolean;
    outcome: boolean;
  };
}

export interface ExpertProfile {
  version: 1;
  layers: Record<string, { keep: number[] }>;
}

export interface ProfileValidation {
  valid: boolean;
  errors: string[];
  eligible_experts: number;
  total_experts: number;
  retained_fraction: number;
}

export interface ProfileProposal {
  profile: ExpertProfile;
  validation: ProfileValidation;
  observed_mass_retained: number;
}

export type ProfileSource =
  | "proposal"
  | "agentic"
  | "manual"
  | "import"
  | "multi_source";

export interface CreateExpertProfileRequest {
  name: string;
  description?: string;
  model_id: string;
  profile: ExpertProfile;
  source?: ProfileSource;
  source_run_id?: string | null;
  source_trial_id?: string | null;
  parent_profile_id?: string | null;
  metric?: "routing_mass" | "selection_count" | null;
  observed_mass_retained?: number | null;
  source_refs?: Array<RoutingSourceReference & { weight: number }>;
  source_fingerprint?: string | null;
  selection_strategy?: string | null;
  selection_config?: Record<string, unknown>;
}

export interface ProfileSelectionStrategy {
  kind:
    | "manual"
    | "fixed_per_layer"
    | "global_budget"
    | "cumulative_mass"
    | "import";
  metric: "routing_mass" | "selection_count";
  parameters: Record<string, number | string | boolean | null>;
}

export interface SavedExpertProfile {
  id: string;
  name: string;
  description: string;
  model_id: string;
  profile: ExpertProfile;
  profile_fingerprint: string;
  source: ProfileSource;
  source_run_id: string | null;
  source_trial_id: string | null;
  source_refs?: Array<RoutingSourceReference & { weight: number }>;
  source_fingerprint?: string | null;
  parent_profile_id: string | null;
  metric: "routing_mass" | "selection_count" | null;
  selection_strategy?: string | null;
  selection_config?: Record<string, unknown>;
  validation: ProfileValidation;
  observed_mass_retained: number | null;
  created_at: string;
}

export interface ComparisonRecord {
  id: string;
  name: string;
  baseline_run_id: string;
  candidate_run_id: string;
  profile_id: string | null;
  profile_fingerprint: string;
  benchmark_id: string;
  cohort_item_ids: string[];
  baseline_score: number | null;
  candidate_score: number | null;
  score_delta: number | null;
  regressions: number;
  recoveries: number;
  retained_passes: number;
  retained_failures: number;
  unscored_items: number;
  created_at: string;
}

export interface AgentBudgets {
  max_turns: number;
  max_tokens: number;
  max_commands: number;
  timeout_seconds: number;
}

export interface AgentDefinition {
  id: string;
  label: string;
  description: string;
  revision: string;
  available: boolean;
  is_default: boolean;
  tool_names: string[];
  default_budgets: AgentBudgets;
}

export interface ProviderError {
  code: string;
  message: string;
  retryable: boolean;
}

export interface SandboxProviderCapabilities {
  create: boolean;
  exec: boolean;
  upload: boolean;
  download: boolean;
  delete: boolean;
}

export type SandboxProviderStatus =
  | "not_configured"
  | "unchecked"
  | "ready"
  | "error";

export interface SandboxProvider {
  id: string;
  label: string;
  configured: boolean;
  credential_mode: "api_key" | "jwt" | null;
  required_env: string[];
  optional_env: string[];
  region: string | null;
  capabilities: SandboxProviderCapabilities;
  status: SandboxProviderStatus;
  last_checked_at: string | null;
  error: ProviderError | null;
}

export interface ProviderPreflight {
  configured: boolean;
  reachable: boolean;
  authenticated: boolean;
  status: Exclude<SandboxProviderStatus, "unchecked">;
  latency_ms: number | null;
  checked_at: string;
  api_url_host: string | null;
  region: string | null;
  error: ProviderError | null;
}

export interface AgentTaskPack {
  id: string;
  name: string;
  description: string;
  source: string;
  revision: string;
  fingerprint: string;
  task_count: number;
  ready: boolean;
  oracle_passed: boolean;
  noop_failed: boolean;
  tags: string[];
}

export interface AgentTask {
  id: string;
  title: string;
  instruction: string;
  language: string | null;
  tags: string[];
  timeout_seconds: number;
}

export type AgentRunStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelling"
  | "cancelled";

export type AgentTrialStatus =
  | "queued"
  | "provisioning"
  | "running"
  | "verifying"
  | "cleaning"
  | "passed"
  | "failed"
  | "error"
  | "cancelled";

export interface AgentTrialSummary {
  id: string;
  agent_run_id: string;
  task_id: string;
  title: string;
  attempt: number;
  seed: number;
  status: AgentTrialStatus;
  reward: number | null;
  turns: number;
  commands: number;
  prompt_tokens: number;
  completion_tokens: number;
  inference_calls: number;
  routed_inference_calls: number;
  termination_reason: string | null;
  sandbox_status: string;
  updated_at: string;
  performance?: AgentPerformanceSummary | null;
}

export interface AgentPerformanceSummary {
  wall_time_ms?: number | null;
  queue_time_ms?: number | null;
  provisioning_time_ms?: number | null;
  model_time_ms?: number | null;
  sandbox_time_ms?: number | null;
  verifier_time_ms?: number | null;
  prompt_tokens?: number | null;
  reasoning_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  mean_tps?: number | null;
  p50_tps?: number | null;
  p95_tps?: number | null;
  inference_cost_usd?: number | null;
  judge_cost_usd?: number | null;
  judge_cost_debit_usd?: number | null;
  judge_cost_uncertain?: boolean;
  estimated_cost_usd?: number | null;
}

export interface AgentRunPerformanceSummary {
  wall_time_ms: number;
  model_time_ms: number;
  sandbox_time_ms: number;
  verifier_time_ms: number;
  prompt_tokens: number;
  reasoning_tokens: number | null;
  completion_tokens: number;
  total_tokens: number;
  reported_mean_tps: number | null;
  reported_tps_trials: number;
  inference_cost_usd?: number | null;
  judge_cost_usd?: number | null;
  estimated_cost_usd: number | null;
}

export interface AgentTrialPair {
  task_id: string;
  attempt: number;
  baseline_trial_id: string;
  candidate_trial_id: string;
  baseline_status: AgentTrialStatus;
  candidate_status: AgentTrialStatus;
  baseline_reward: number | null;
  candidate_reward: number | null;
  reward_delta: number | null;
  transition:
    | "regression"
    | "recovery"
    | "retained_pass"
    | "retained_failure"
    | "unscored";
  baseline_performance: AgentPerformanceSummary | null;
  candidate_performance: AgentPerformanceSummary | null;
}

export interface AgentRunComparison {
  id: string;
  name: string;
  baseline_run_id: string;
  candidate_run_id: string;
  task_pack_id: string;
  compatibility_fingerprint: string;
  baseline_profile_id: string | null;
  candidate_profile_id: string | null;
  baseline_profile_fingerprint: string | null;
  candidate_profile_fingerprint: string | null;
  trial_count: number;
  baseline_passed_trials: number;
  candidate_passed_trials: number;
  baseline_mean_reward: number | null;
  candidate_mean_reward: number | null;
  mean_reward_delta: number | null;
  regressions: number;
  recoveries: number;
  retained_passes: number;
  retained_failures: number;
  unscored: number;
  baseline_performance: AgentRunPerformanceSummary;
  candidate_performance: AgentRunPerformanceSummary;
  pairs: AgentTrialPair[];
}

export interface AgentRun {
  id: string;
  job_id: string;
  task_pack_id: string;
  task_pack_name: string;
  model_session_id: string;
  profile_id: string | null;
  profile_fingerprint: string | null;
  agent_id: string;
  sandbox_provider_id: string;
  compatibility_fingerprint: string;
  status: AgentRunStatus;
  task_ids: string[];
  total_trials: number;
  completed_trials: number;
  passed_trials: number;
  mean_reward: number | null;
  reasoning_mode?: "off" | "compact" | "full";
  active_trial_id: string | null;
  trials: AgentTrialSummary[];
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  budgets?: AgentBudgets | null;
  execution_policy?: ExecutionPolicy | null;
  evaluation_contract?: EvaluationContract | null;
}

export interface CreateAgentRunRequest {
  task_pack_id: string;
  task_ids: string[];
  agent_id: string;
  sandbox_provider_id: string;
  model_session_id: string;
  budgets?: AgentBudgets;
  attempts?: number;
  seed?: number;
  generation?: GenerationConfig;
  reasoning_mode?: "off" | "compact" | "full";
  execution_policy?: ExecutionPolicy;
  evaluation_contract?: EvaluationContract;
}

export interface InferenceCallSummary {
  id: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  routing_artifact_id: string | null;
  routed_layers: number;
  total_routed_slots: number;
  reasoning_tokens?: number | null;
  total_tokens?: number | null;
  ttft_ms?: number | null;
  prefill_ms?: number | null;
  decode_ms?: number | null;
  tokens_per_second?: number | null;
  estimated_cost_usd?: number | null;
  model_id?: string | null;
  finish_reason?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export type TrajectoryStepType =
  | "system"
  | "user"
  | "assistant"
  | "tool"
  | "observation"
  | "verifier"
  | "task"
  | "reasoning"
  | "model"
  | "tool_call"
  | "command"
  | "stdout"
  | "stderr"
  | "routing";

export interface TrajectoryStep {
  id: string;
  sequence: number;
  timestamp: string;
  type: TrajectoryStepType;
  title: string;
  content: string;
  tool_name: string | null;
  command: string | null;
  exit_code: number | null;
  duration_ms: number | null;
  truncated: boolean;
  inference: InferenceCallSummary | null;
  phase?: "setup" | "reasoning" | "action" | "tool" | "verification" | null;
  turn?: number | null;
  stream?: "stdout" | "stderr" | null;
  reasoning_visibility?: "explicit" | "none" | null;
  metadata?: Record<string, unknown>;
}

export interface AgentTrajectory {
  trial_id: string;
  format: "ATIF";
  schema_version: string;
  steps: TrajectoryStep[];
  reasoning_mode?: "off" | "compact" | "full";
  updated_at: string;
}

export interface VerifierResult {
  status: "pending" | "passed" | "failed" | "error";
  reward: number | null;
  summary: string;
  output: string;
  exit_code: number | null;
  duration_ms: number | null;
}

export interface AgentArtifactLink {
  name: string;
  media_type: string;
  download_url: string;
}

export interface AgentTrialArtifacts {
  trial_id: string;
  patch: string | null;
  patch_sha256: string | null;
  files_changed: number;
  additions: number;
  deletions: number;
  verifier: VerifierResult | null;
  exports?: AgentArtifactLink[];
}

export interface TrialRoutingSummary extends RoutingSummary {
  trial_id: string;
  inference_count: number;
  inference_calls: number;
  captured_inference_calls: number;
  served_tokens: number;
  profile_id: string | null;
  profile_fingerprint: string | null;
}
