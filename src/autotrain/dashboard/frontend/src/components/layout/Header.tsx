import type { Run } from '../../api/types'

const STATUS_STYLES: Record<string, string> = {
  running: 'bg-green-900/50 text-green-400 border-green-700',
  completed: 'bg-blue-900/50 text-blue-400 border-blue-700',
  budget_exhausted: 'bg-yellow-900/50 text-yellow-400 border-yellow-700',
  failed: 'bg-red-900/50 text-red-400 border-red-700',
  stopped: 'bg-gray-800 text-gray-400 border-gray-700',
}

export function Header({ run }: { run: Run }) {
  const style = STATUS_STYLES[run.status || ''] || STATUS_STYLES.stopped
  const direction = run.metric_direction === 'maximize' ? '>=' : '<='

  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-2xl font-bold text-white">AutoTrain Monitor</h1>
        <span className={`rounded border px-2 py-0.5 text-xs font-medium ${style}`}>
          {(run.status || 'unknown').toUpperCase()}
        </span>
      </div>
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
