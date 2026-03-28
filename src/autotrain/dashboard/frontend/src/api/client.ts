import type {
  Run,
  Iteration,
  MetricSnapshot,
  EpochMetric,
  GpuSnapshot,
} from './types'

const BASE = '/api/v1'

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

// -- Runs --
export const fetchRuns = () => fetchJson<Run[]>(`${BASE}/runs`)

export const fetchRun = (runId: string) =>
  fetchJson<Run>(`${BASE}/runs/${runId}`)

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
