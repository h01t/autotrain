import { useState, useEffect } from 'react'
import type { RunConfigResponse } from '../../api/types'

interface ConfigDrawerProps {
  runId: string
}

export function ConfigDrawer({ runId }: ConfigDrawerProps) {
  const [data, setData] = useState<RunConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open && data) return // already loaded
    let cancelled = false
    setLoading(true)
    fetch(`/api/v1/runs/${runId}/config`)
      .then((r) => {
        if (!r.ok) throw new Error('not found')
        return r.json()
      })
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
  }, [runId, open])

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className="text-xs text-gray-500 hover:text-gray-300 transition-colors underline"
      >
        {open ? 'Hide Config' : 'View Config'}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setOpen(false)}
          />
          <div className="relative w-full max-w-lg bg-gray-900 border-l border-gray-800 overflow-y-auto p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-300">
                Run Configuration
              </h3>
              <button
                onClick={() => setOpen(false)}
                className="text-gray-500 hover:text-gray-300 text-lg"
              >
                ✕
              </button>
            </div>
            {loading ? (
              <div className="animate-pulse space-y-2">
                <div className="h-3 bg-gray-800 rounded w-3/4" />
                <div className="h-3 bg-gray-800 rounded w-1/2" />
                <div className="h-3 bg-gray-800 rounded w-5/6" />
              </div>
            ) : data?.config_yaml ? (
              <pre className="text-xs font-mono text-gray-400 bg-gray-950 rounded p-3 overflow-auto max-h-[60vh] whitespace-pre">
                {data.config_yaml}
              </pre>
            ) : (
              <p className="text-gray-600 text-xs italic">
                No configuration available.
              </p>
            )}
          </div>
        </div>
      )}
    </>
  )
}
