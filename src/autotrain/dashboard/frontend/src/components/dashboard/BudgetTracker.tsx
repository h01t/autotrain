import { useMemo } from 'react'
import type { Run, Iteration } from '../../api/types'

interface BudgetLimits {
  time_seconds: number | null
  api_dollars: number | null
  max_iterations: number | null
}

function parseBudget(run: Run): BudgetLimits {
  const defaults: BudgetLimits = { time_seconds: null, api_dollars: null, max_iterations: null }
  if (!run.config_snapshot) return defaults
  try {
    const cfg = JSON.parse(run.config_snapshot)
    const b = cfg.budget || {}
    return {
      time_seconds: b.time_seconds ?? null,
      api_dollars: b.api_dollars ?? null,
      max_iterations: b.max_iterations ?? null,
    }
  } catch {
    return defaults
  }
}

function Bar({ label, current, limit, unit }: {
  label: string
  current: string
  limit: string | null
  unit: string
  pct?: number
}) {
  const pct = limit != null ? Math.min(100, (parseFloat(current) / parseFloat(limit)) * 100) : null
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-mono">
          {current}{unit}{limit != null ? ` / ${limit}${unit}` : ''}
        </span>
      </div>
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        {pct != null ? (
          <div
            className={`h-full rounded-full transition-all ${
              pct >= 90 ? 'bg-red-500' : pct >= 70 ? 'bg-yellow-500' : 'bg-blue-500'
            }`}
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full rounded-full bg-gray-700" style={{ width: '100%' }} />
        )}
      </div>
    </div>
  )
}

export function BudgetTracker({ run, iterations }: { run: Run; iterations: Iteration[] }) {
  const budget = parseBudget(run)

  const { elapsed, avgDuration, avgCost } = useMemo(() => {
    const completed = iterations.filter((it) => it.duration_seconds != null)
    const totalDur = completed.reduce((s, it) => s + (it.duration_seconds || 0), 0)
    const totalCost = completed.reduce((s, it) => s + (it.api_cost || 0), 0)
    const avg = completed.length > 0 ? totalDur / completed.length : 0
    const avgC = completed.length > 0 ? totalCost / completed.length : 0

    // Elapsed = time from first iteration to now or last iteration
    let elapsed = 0
    if (run.created_at) {
      const start = new Date(run.created_at).getTime()
      const end = run.status === 'running' ? Date.now() : (run.updated_at ? new Date(run.updated_at).getTime() : Date.now())
      elapsed = (end - start) / 1000
    }

    return { elapsed, avgDuration: avg, avgCost: avgC }
  }, [iterations, run])

  const fmtTime = (s: number) => {
    if (s >= 3600) return `${(s / 3600).toFixed(1)}h`
    if (s >= 60) return `${(s / 60).toFixed(0)}m`
    return `${s.toFixed(0)}s`
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Budget</h2>
      <div className="space-y-3 rounded-lg border border-gray-800 bg-gray-900 p-4">
        <Bar
          label="Iterations"
          current={String(run.total_iterations)}
          limit={budget.max_iterations != null ? String(budget.max_iterations) : null}
          unit=""
        />
        <Bar
          label="API Cost"
          current={run.total_api_cost.toFixed(2)}
          limit={budget.api_dollars != null ? budget.api_dollars.toFixed(2) : null}
          unit="$"
        />
        <Bar
          label="Time"
          current={fmtTime(elapsed)}
          limit={budget.time_seconds != null ? fmtTime(budget.time_seconds) : null}
          unit=""
        />

        <div className="border-t border-gray-800 pt-2 flex gap-6 text-xs text-gray-500">
          {avgDuration > 0 && <span>~{fmtTime(avgDuration)}/iter</span>}
          {avgCost > 0 && <span>~${avgCost.toFixed(3)}/iter</span>}
        </div>
      </div>
    </div>
  )
}
