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

export type JobKind = "model_load" | "benchmark_run" | "dataset_prepare";
export type JobStatus =
  | "queued"
  | "running"
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

export type ProfileSource = "proposal" | "manual" | "import";

export interface CreateExpertProfileRequest {
  name: string;
  description?: string;
  model_id: string;
  profile: ExpertProfile;
  source?: ProfileSource;
  source_run_id?: string | null;
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
