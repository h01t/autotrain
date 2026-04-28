import { useState, useCallback } from 'react'
import type { Run } from '../../api/types'
import { startRun, stopRun, restartRun, resumeRun } from '../../api/client'

const STATUS_STYLES: Record<string, string> = {
  running: 'bg-green-900/50 text-green-400 border-green-700',
  completed: 'bg-blue-900/50 text-blue-400 border-blue-700',
  budget_exhausted: 'bg-yellow-900/50 text-yellow-400 border-yellow-700',
  failed: 'bg-red-900/50 text-red-400 border-red-700',
  stopped: 'bg-gray-800 text-gray-400 border-gray-700',
}

interface HeaderProps {
  run: Run
  onRefresh?: () => void
  onRunCreated?: (runId: string) => void
}

export function Header({ run, onRefresh, onRunCreated }: HeaderProps) {
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const handleAction = useCallback(async (action: 'start' | 'stop' | 'restart' | 'resume') => {
    setActionLoading(action)
    setActionError(null)
    try {
      if (action === 'start') await startRun(run.id)
      else if (action === 'stop') await stopRun(run.id)
      else if (action === 'restart') await restartRun(run.id)
      else if (action === 'resume') {
        const result = await resumeRun(run.id, true)
        if (result.new_run_id) {
          onRefresh?.()
          onRunCreated?.(result.new_run_id)
          return
        }
      }
      onRefresh?.()
    } catch (e) {
      setActionError(String(e))
    } finally {
      setActionLoading(null)
    }
  }, [run.id, onRefresh])

  const style = STATUS_STYLES[run.status || ''] || STATUS_STYLES.stopped
  const direction = run.metric_direction === 'maximize' ? '>=' : '<='

  const showStart = run.status === 'stopped' || run.status === 'failed' || run.status === 'budget_exhausted' || run.status === 'completed'
  const showStop = run.status === 'running'
  const showRestart = run.status === 'stopped' || run.status === 'failed' || run.status === 'completed'

  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-2xl font-bold text-white">AutoTrain Monitor</h1>
        <span className={`rounded border px-2 py-0.5 text-xs font-medium ${style}`}>
          {(run.status || 'unknown').toUpperCase()}
        </span>

        {/* Run control buttons */}
        <div className="flex items-center gap-1.5 ml-2">
          {showStart && (
            <button
              onClick={() => handleAction('start')}
              disabled={actionLoading !== null}
              className="px-3 py-1 rounded text-xs font-medium bg-green-700 hover:bg-green-600 disabled:bg-gray-700 disabled:text-gray-500 text-white transition-colors"
            >
              {actionLoading === 'start' ? '...' : '▶ Start'}
            </button>
          )}
          {showStop && (
            <button
              onClick={() => handleAction('stop')}
              disabled={actionLoading !== null}
              className="px-3 py-1 rounded text-xs font-medium bg-red-700 hover:bg-red-600 disabled:bg-gray-700 disabled:text-gray-500 text-white transition-colors"
            >
              {actionLoading === 'stop' ? '...' : '■ Stop'}
            </button>
          )}
          {showRestart && (
            <button
              onClick={() => handleAction('restart')}
              disabled={actionLoading !== null}
              className="px-3 py-1 rounded text-xs font-medium bg-yellow-700 hover:bg-yellow-600 disabled:bg-gray-700 disabled:text-gray-500 text-white transition-colors"
            >
              {actionLoading === 'restart' ? '...' : '↻ Restart'}
            </button>
          )}
          {(run.status === 'failed' || run.status === 'completed' ||
            run.status === 'stopped' || run.status === 'budget_exhausted') && (
            <button
              onClick={() => handleAction('resume')}
              disabled={actionLoading !== null}
              className="px-3 py-1 rounded text-xs font-medium bg-purple-700 hover:bg-purple-600 disabled:bg-gray-700 disabled:text-gray-500 text-white transition-colors"
            >
              {actionLoading === 'resume' ? '...' : '↺ Resume'}
            </button>
          )}
        </div>
      </div>

      {actionError && (
        <p className="text-red-400 text-xs mb-2">{actionError}</p>
      )}

      <p className="text-sm text-gray-500">
        Run <code className="text-gray-400">{run.id}</code>
        {' | '}Target: <strong>{run.metric_name}</strong> {direction} {run.metric_target}
        {run.git_branch && (
          <> | Branch: <code className="text-gray-400">{run.git_branch}</code></>
        )}
      </p>
    </div>
  )
}
