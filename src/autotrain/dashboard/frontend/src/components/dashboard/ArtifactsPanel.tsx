import { useState, useEffect } from 'react'
import type { ArtifactsListResponse } from '../../api/types'

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}

interface ArtifactsPanelProps {
  runId: string
}

export function ArtifactsPanel({ runId }: ArtifactsPanelProps) {
  const [data, setData] = useState<ArtifactsListResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetch(`/api/v1/runs/${runId}/artifacts`)
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
  }, [runId])

  if (loading || !data) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 animate-pulse">
        <div className="h-4 bg-gray-800 rounded w-20 mb-3" />
        <div className="space-y-2">
          <div className="h-3 bg-gray-800 rounded w-full" />
          <div className="h-3 bg-gray-800 rounded w-3/4" />
        </div>
      </div>
    )
  }

  const dlUrl = (path: string) =>
    `/api/v1/runs/${runId}/artifacts/${encodeURIComponent(path)}`

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Artifacts
        <span className="ml-2 text-xs text-gray-600">
          ({data.artifacts.length} files, {fmtBytes(data.total_bytes)})
        </span>
      </h3>
      {data.artifacts.length === 0 ? (
        <p className="text-gray-600 text-xs italic">
          No artifacts yet. Checkpoints and model files appear here after
          training runs.
        </p>
      ) : (
        <ul className="space-y-1">
          {data.artifacts.map((a) => (
            <li
              key={a.path}
              className="flex items-center justify-between text-xs py-1"
            >
              <span className="text-gray-400 font-mono truncate mr-2">
                {a.name}
              </span>
              <span className="text-gray-600 flex-shrink-0">
                {fmtBytes(a.size_bytes)}
              </span>
              <a
                href={dlUrl(a.path)}
                download
                className="ml-3 text-blue-400 hover:text-blue-300 flex-shrink-0"
              >
                ↓
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
