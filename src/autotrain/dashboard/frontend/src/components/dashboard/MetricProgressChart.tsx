import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import type { MetricSnapshot, Iteration, Run } from '../../api/types'
import { OUTCOME_COLORS } from '../../api/types'

interface Props {
  run: Run
  snapshots: MetricSnapshot[]
  iterations: Iteration[]
}

export function MetricProgressChart({ run, snapshots, iterations }: Props) {
  if (!snapshots.length) {
    return <p className="text-gray-500 text-sm">No metric data recorded yet.</p>
  }

  const outcomeMap = new Map(iterations.map((it) => [it.iteration_num, it.outcome]))
  const data = snapshots.map((s) => ({
    iteration: s.iteration_num,
    value: s.value,
    outcome: outcomeMap.get(s.iteration_num) || 'no_change',
  }))

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Metric Progress</h2>
      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="iteration"
            type="number"
            name="Iteration"
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
          />
          <YAxis
            dataKey="value"
            name={run.metric_name}
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#9ca3af' }}
            itemStyle={{ color: '#e5e7eb' }}
          />
          <ReferenceLine
            y={run.metric_target}
            stroke="#ef4444"
            strokeDasharray="6 3"
            label={{ value: `Target: ${run.metric_target}`, fill: '#ef4444', fontSize: 11 }}
          />
          {run.best_metric_value != null && (
            <ReferenceLine
              y={run.best_metric_value}
              stroke="#22c55e"
              strokeDasharray="3 3"
              label={{ value: `Best: ${run.best_metric_value.toFixed(4)}`, fill: '#22c55e', fontSize: 11 }}
            />
          )}
          <Scatter name={run.metric_name} data={data} line={{ stroke: '#3b82f6', strokeWidth: 2 }}>
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={OUTCOME_COLORS[entry.outcome] || '#94a3b8'}
                r={5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
