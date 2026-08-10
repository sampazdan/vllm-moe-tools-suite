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
  | "agent_run";
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
  topology: ModelTopology;
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
  | "ungraded";

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
}

export interface BenchmarkItem {
  id: string;
  prompt: string;
  expected: string;
  category: string;
  scoring: ScoringMode;
  metadata: Record<string, string | number | boolean | null>;
}

export interface BenchmarkItemPage {
  benchmark: BenchmarkInfo;
  items: BenchmarkItem[];
  total: number;
  offset: number;
  limit: number;
  categories: string[];
}

export interface BenchmarkDatasetRecord {
  id: string;
  info: BenchmarkInfo;
  content_hash: string;
  created_at: string;
}

export interface RunItemResult {
  item_id: string;
  prompt: string;
  expected: string;
  output: string;
  passed: boolean | null;
  scoring: ScoringMode;
  error: string | null;
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface BenchmarkRun {
  id: string;
  benchmark_id: string;
  model_session_id: string;
  status: string;
  score: number | null;
  scored_items: number;
  completed_items: number;
  total_items: number;
  items: RunItemResult[];
  cohort_id: string | null;
  created_at: string;
}

export interface RoutingSummary {
  run_id: string;
  layer_ids: number[];
  selection_counts: number[][];
  routing_mass: number[][];
  total_routed_slots: number;
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

export type ProfileSource = "proposal" | "agentic" | "manual" | "import";

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
  parent_profile_id: string | null;
  metric: "routing_mass" | "selection_count" | null;
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
  active_trial_id: string | null;
  trials: AgentTrialSummary[];
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface CreateAgentRunRequest {
  task_pack_id: string;
  task_ids: string[];
  agent_id: string;
  sandbox_provider_id: string;
  model_session_id: string;
  budgets?: AgentBudgets;
}

export interface InferenceCallSummary {
  id: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  routing_artifact_id: string | null;
  routed_layers: number;
  total_routed_slots: number;
}

export type TrajectoryStepType =
  | "system"
  | "user"
  | "assistant"
  | "tool"
  | "observation"
  | "verifier";

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
}

export interface AgentTrajectory {
  trial_id: string;
  format: "ATIF";
  schema_version: string;
  steps: TrajectoryStep[];
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
