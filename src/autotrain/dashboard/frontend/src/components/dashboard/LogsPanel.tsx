import { useState, useEffect } from 'react'
import type { RunLogsResponse } from '../../api/types'

interface LogsPanelProps {
  runId: string
}

export function LogsPanel({ runId }: LogsPanelProps) {
  const [data, setData] = useState<RunLogsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [tail, setTail] = useState(200)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch(`/api/v1/runs/${runId}/logs?tail=${tail}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled) {
          setData(d)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [runId, tail])

  if (loading || !data) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 animate-pulse">
        <div className="h-4 bg-gray-800 rounded w-24 mb-3" />
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-3 bg-gray-800 rounded" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">
          Training Logs
          <span className="ml-2 text-xs text-gray-600">
            ({data.total_lines} lines{data.truncated ? ', truncated' : ''})
          </span>
        </h3>
        <select
          value={tail}
          onChange={(e) => setTail(Number(e.target.value))}
          className="bg-gray-800 text-gray-300 text-xs rounded px-2 py-1 border border-gray-700"
        >
          <option value={50}>Last 50</option>
          <option value={200}>Last 200</option>
          <option value={500}>Last 500</option>
          <option value={1000}>Last 1000</option>
        </select>
      </div>
      {data.lines.length === 0 ? (
        <p className="text-gray-600 text-xs italic">No log output yet.</p>
      ) : (
        <pre className="text-xs font-mono text-gray-400 bg-gray-950 rounded p-3 overflow-auto max-h-96 whitespace-pre-wrap">
          {data.lines.join('\n')}
        </pre>
      )}
    </div>
  )
}
