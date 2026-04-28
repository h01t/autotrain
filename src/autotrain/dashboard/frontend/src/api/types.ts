export interface Run {
  id: string
  repo_path: string
  metric_name: string
  metric_target: number
  metric_direction: string
  status: string | null
  best_metric_value: number | null
  best_iteration: number | null
  total_iterations: number
  total_api_cost: number
  git_branch: string | null
  config_snapshot: string | null
  created_at: string | null
  updated_at: string | null
}

export interface Iteration {
  id: number | null
  run_id: string
  iteration_num: number
  state: string
  outcome: string | null
  metric_value: number | null
  commit_hash: string | null
  agent_reasoning: string | null
  agent_hypothesis: string | null
  changes_summary: string | null
  duration_seconds: number | null
  api_cost: number | null
  error_message: string | null
  checkpoint_path: string | null
  resumed_from_checkpoint: boolean
  created_at: string | null
}

export interface MetricSnapshot {
  id: number | null
  run_id: string
  iteration_num: number
  metric_name: string
  value: number
  timestamp: string | null
}

export interface EpochMetric {
  id: number | null
  run_id: string
  iteration_num: number
  epoch: number
  metrics: Record<string, number>
  timestamp: string | null
}

export interface GpuSnapshot {
  id: number | null
  run_id: string
  gpu_index: number
  utilization_pct: number | null
  memory_used_mb: number | null
  memory_total_mb: number | null
  temperature_c: number | null
  timestamp: string | null
}

export type OutcomeType =
  | 'improved'
  | 'regressed'
  | 'crashed'
  | 'sandbox_rejected'
  | 'no_change'
  | 'timeout'

export const OUTCOME_COLORS: Record<string, string> = {
  improved: '#22c55e',
  regressed: '#ef4444',
  crashed: '#f97316',
  sandbox_rejected: '#eab308',
  no_change: '#94a3b8',
  timeout: '#f97316',
}

export const OUTCOME_ICONS: Record<string, string> = {
  improved: '+',
  regressed: '-',
  crashed: '!',
  sandbox_rejected: 'x',
  no_change: '=',
  timeout: 'T',
}

// -- Dashboard Control Types --

export interface ConfigValidationError {
  field: string
  message: string
}

export interface ValidateConfigResponse {
  valid: boolean
  errors: ConfigValidationError[]
  warnings: string[]
}

export interface PreflightGpuInfo {
  index: number
  name: string | null
  memory_total_mb: number | null
  memory_free_mb: number | null
  utilization_pct: number | null
}

export interface PreflightResult {
  check: string
  passed: boolean
  message: string
  detail?: string | null
  suggestion?: string | null
}

export interface PreflightResponse {
  passed: boolean
  checks: PreflightResult[]
  gpus: PreflightGpuInfo[]
  duration_seconds: number
}

export interface CreateRunResponse {
  run_id: string
  status: string
  message: string
  config_errors: ConfigValidationError[]
}

export interface RunActionResponse {
  run_id: string
  action: string
  success: boolean
  message: string
  previous_status: string | null
  new_status: string | null
}

export interface RunStatusResponse {
  run_id: string
  status: string
  is_active: boolean
  pid: number | null
  uptime_seconds: number | null
}
