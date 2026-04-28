import type {
  Run,
  Iteration,
  MetricSnapshot,
  EpochMetric,
  GpuSnapshot,
  ValidateConfigResponse,
  PreflightResponse,
  CreateRunResponse,
  RunActionResponse,
  RunStatusResponse,
} from './types'

const BASE = '/api/v1'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

// -- Runs --
export const fetchRuns = () => fetchJson<Run[]>(`${BASE}/runs`)

export const fetchRun = (runId: string) =>
  fetchJson<Run>(`${BASE}/runs/${runId}`)

// -- Run Creation & Control --
export const createRun = (configYaml: string, startImmediately = true) =>
  fetchJson<CreateRunResponse>(`${BASE}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      config_yaml: configYaml,
      start_immediately: startImmediately,
    }),
  })

export const startRun = (runId: string) =>
  fetchJson<RunActionResponse>(`${BASE}/runs/${runId}/start`, {
    method: 'POST',
  })

export const stopRun = (runId: string) =>
  fetchJson<RunActionResponse>(`${BASE}/runs/${runId}/stop`, {
    method: 'POST',
  })

export const restartRun = (runId: string) =>
  fetchJson<RunActionResponse>(`${BASE}/runs/${runId}/restart`, {
    method: 'POST',
  })

export const fetchRunStatus = (runId: string) =>
  fetchJson<RunStatusResponse>(`${BASE}/runs/${runId}/status`)

// -- Preflight & Validation --
export const validateConfig = (configYaml: string) =>
  fetchJson<ValidateConfigResponse>(`${BASE}/runs/validate-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_yaml: configYaml }),
  })

export const runPreflight = (body: {
  repo_path: string
  mode?: string
  ssh_host?: string
  ssh_port?: number
  gpu_device?: string
  venv_activate?: string
  train_command?: string
}) =>
  fetchJson<PreflightResponse>(`${BASE}/runs/preflight`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

// -- Iterations --
export const fetchIterations = (runId: string, limit = 100) =>
  fetchJson<Iteration[]>(`${BASE}/runs/${runId}/iterations?limit=${limit}`)

export const fetchBestIterations = (
  runId: string,
  direction = 'maximize',
  limit = 5,
) =>
  fetchJson<Iteration[]>(
    `${BASE}/runs/${runId}/iterations/best?direction=${direction}&limit=${limit}`,
  )

// -- Metrics --
export const fetchMetrics = (runId: string) =>
  fetchJson<MetricSnapshot[]>(`${BASE}/runs/${runId}/metrics`)

// -- Epochs --
export const fetchEpochMetrics = (runId: string, iterationNum?: number) => {
  const params = iterationNum != null ? `?iteration_num=${iterationNum}` : ''
  return fetchJson<EpochMetric[]>(`${BASE}/runs/${runId}/epochs${params}`)
}

// -- GPU --
export const fetchGpuSnapshots = (runId: string, limit = 500) =>
  fetchJson<GpuSnapshot[]>(`${BASE}/runs/${runId}/gpu?limit=${limit}`)

export const fetchGpuLatest = (runId: string) =>
  fetchJson<GpuSnapshot>(`${BASE}/runs/${runId}/gpu/latest`)

// -- New API functions (Milestone #2 revisions) --

export const resumeRun = (runId: string, startImmediately = true) =>
  fetchJson<ResumeRunResponse>(`${BASE}/runs/${runId}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, start_immediately: startImmediately }),
  })

export const fetchDefaults = () =>
  fetchJson<DefaultsResponse>(`${BASE}/defaults`)

export const saveConfig = (repoPath: string, configYaml: string) =>
  fetchJson<SaveConfigResponse>(`${BASE}/save-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_path: repoPath, config_yaml: configYaml }),
  })

// Types imported from types.ts above, re-exported via local types
import type {
  ResumeRunResponse,
  RunLogsResponse,
  ArtifactsListResponse,
  DefaultsResponse,
  SaveConfigResponse,
} from './types'
