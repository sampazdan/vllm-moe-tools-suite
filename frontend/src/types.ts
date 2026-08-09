export type ModelState =
  | "unloaded"
  | "starting"
  | "ready"
  | "stopping"
  | "failed";

export interface SystemStatus {
  mode: "mock" | "vllm";
  version: string;
  model_state: ModelState;
  data_dir: string;
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

export interface BenchmarkInfo {
  id: string;
  name: string;
  description: string;
  item_count: number;
  categories: string[];
}

export interface BenchmarkItem {
  id: string;
  prompt: string;
  expected: string;
  category: string;
}

export interface RunItemResult {
  item_id: string;
  prompt: string;
  expected: string;
  output: string;
  passed: boolean;
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface BenchmarkRun {
  id: string;
  benchmark_id: string;
  model_session_id: string;
  status: string;
  score: number;
  completed_items: number;
  total_items: number;
  items: RunItemResult[];
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
