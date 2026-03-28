import { useQuery } from '@tanstack/react-query'
import {
  fetchRun,
  fetchIterations,
  fetchMetrics,
  fetchEpochMetrics,
  fetchGpuSnapshots,
  fetchGpuLatest,
} from '../api/client'

export function useRun(runId: string) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: () => fetchRun(runId),
    enabled: !!runId,
  })
}

export function useIterations(runId: string) {
  return useQuery({
    queryKey: ['iterations', runId],
    queryFn: () => fetchIterations(runId, 100),
    enabled: !!runId,
  })
}

export function useMetrics(runId: string) {
  return useQuery({
    queryKey: ['metrics', runId],
    queryFn: () => fetchMetrics(runId),
    enabled: !!runId,
  })
}

export function useEpochMetrics(runId: string, iterationNum?: number) {
  return useQuery({
    queryKey: ['epochs', runId, iterationNum],
    queryFn: () => fetchEpochMetrics(runId, iterationNum),
    enabled: !!runId,
  })
}

export function useGpuSnapshots(runId: string) {
  return useQuery({
    queryKey: ['gpu', runId],
    queryFn: () => fetchGpuSnapshots(runId),
    enabled: !!runId,
    staleTime: 30000,         // WS pushes updates; only refetch if stale 30s+
    refetchInterval: 30000,   // Fallback polling — WebSocket is primary
  })
}

export function useGpuLatest(runId: string) {
  return useQuery({
    queryKey: ['gpuLatest', runId],
    queryFn: () => fetchGpuLatest(runId),
    enabled: !!runId,
    retry: false,
    staleTime: 30000,
    refetchInterval: 30000,
  })
}
