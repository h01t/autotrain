import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { GpuSnapshot, Run } from '../api/types'

export function useWebSocket(runId: string | undefined) {
  const queryClient = useQueryClient()
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!runId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/ws/${runId}`

    function connect() {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          switch (msg.type) {
            case 'run_updated':
            case 'run_completed':
              // Push run data directly into cache
              if (msg.data && msg.data.id) {
                queryClient.setQueryData(['run', runId], msg.data as Run)
                // Update runs list in-place
                queryClient.setQueryData<Run[]>(['runs'], (old) => {
                  if (!old) return old
                  return old.map((r) => (r.id === msg.data.id ? msg.data : r))
                })
              }
              break

            case 'iteration_added':
              // Invalidate iterations/metrics — these are lists that need full refresh
              queryClient.invalidateQueries({ queryKey: ['iterations', runId] })
              queryClient.invalidateQueries({ queryKey: ['metrics', runId] })
              queryClient.invalidateQueries({ queryKey: ['run', runId] })
              queryClient.invalidateQueries({ queryKey: ['runs'] })
              break

            case 'metric_added':
              if (msg.data && msg.data.id) {
                queryClient.setQueryData(['run', runId], msg.data as Run)
                queryClient.setQueryData<Run[]>(['runs'], (old) => {
                  if (!old) return old
                  return old.map((r) => (r.id === msg.data.id ? msg.data : r))
                })
              }
              queryClient.invalidateQueries({ queryKey: ['metrics', runId] })
              break

            case 'gpu_snapshot':
              // Push GPU data directly — no HTTP round-trip needed
              if (msg.data) {
                const snapshot = msg.data as GpuSnapshot
                queryClient.setQueryData(['gpuLatest', runId], snapshot)
                queryClient.setQueryData<GpuSnapshot[]>(['gpu', runId], (old) => {
                  if (!old) return [snapshot]
                  const next = [...old, snapshot]
                  // Keep last 300 points — enough for chart, prevents unbounded growth
                  return next.length > 300 ? next.slice(-300) : next
                })
              }
              break
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onclose = () => {
        setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
    }
  }, [runId, queryClient])
}
