import type { Run } from '../../api/types'

const STATUS_MARKERS: Record<string, string> = {
  running: '>>>',
  completed: '[OK]',
  budget_exhausted: '[$$]',
  failed: '[!!]',
  stopped: '[--]',
}

interface SidebarProps {
  runs: Run[]
  selectedRunId: string | undefined
  onSelectRun: (runId: string) => void
}

export function Sidebar({ runs, selectedRunId, onSelectRun }: SidebarProps) {
  return (
    <aside className="w-72 shrink-0 border-r border-gray-800 bg-gray-900 p-4 overflow-y-auto">
      <h1 className="text-lg font-bold text-white mb-1">AutoTrain</h1>
      <p className="text-xs text-gray-500 mb-4">Dashboard</p>

      <h2 className="text-sm font-semibold text-gray-400 mb-2">Runs</h2>
      <div className="space-y-1">
        {runs.map((run) => {
          const marker = STATUS_MARKERS[run.status || ''] || '[??]'
          const best = run.best_metric_value != null
            ? run.best_metric_value.toFixed(4)
            : 'N/A'
          const isSelected = run.id === selectedRunId

          return (
            <button
              key={run.id}
              onClick={() => onSelectRun(run.id)}
              className={`w-full text-left rounded px-3 py-2 text-xs font-mono transition-colors ${
                isSelected
                  ? 'bg-blue-900/50 text-blue-300 border border-blue-700'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-gray-500">{marker}</span>
                <span className="font-semibold">{run.id}</span>
              </div>
              <div className="mt-1 text-gray-500">
                {run.metric_name}={best} | {run.total_iterations}it
              </div>
            </button>
          )
        })}
      </div>

      {runs.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-800">
          {runs.map((r) => {
            const cost = `$${r.total_api_cost.toFixed(3)}`
            return (
              <p key={r.id} className="text-xs text-gray-600 mb-1">
                <code>{r.id}</code> {r.status} | {r.total_iterations}it | {cost}
              </p>
            )
          })}
        </div>
      )}
    </aside>
  )
}
