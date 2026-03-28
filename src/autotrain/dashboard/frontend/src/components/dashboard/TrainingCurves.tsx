import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { EpochMetric, Iteration } from '../../api/types'
import { useEpochMetrics } from '../../hooks/useRunDetail'

const PALETTE = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6']

interface Props {
  runId: string
  iterations: Iteration[]
}

export function TrainingCurves({ runId, iterations }: Props) {
  const itersWithData = iterations.filter((it) => it.iteration_num > 0)
  const latest = itersWithData.length > 0 ? itersWithData[itersWithData.length - 1].iteration_num : undefined
  const [selectedIters, setSelectedIters] = useState<number[]>(latest ? [latest] : [])

  // Fetch all epoch data for this run
  const { data: allEpochs } = useEpochMetrics(runId)

  const { chartData, metricKeys, lossKeys, scoreKeys } = useMemo(() => {
    if (!allEpochs || selectedIters.length === 0) {
      return { chartData: [], metricKeys: [], lossKeys: [], scoreKeys: [] }
    }

    const allKeys = new Set<string>()
    const byIterEpoch: Record<string, Record<string, number>> = {}

    for (const em of allEpochs) {
      if (!selectedIters.includes(em.iteration_num)) continue
      const prefix = selectedIters.length > 1 ? `Iter${em.iteration_num}_` : ''
      const key = `${em.iteration_num}_${em.epoch}`
      if (!byIterEpoch[key]) {
        byIterEpoch[key] = { epoch: em.epoch }
      }
      for (const [k, v] of Object.entries(em.metrics)) {
        const name = `${prefix}${k}`
        byIterEpoch[key][name] = v
        allKeys.add(name)
      }
    }

    const keys = Array.from(allKeys)
    const loss = keys.filter((k) => k.toLowerCase().includes('loss'))
    const score = keys.filter((k) => !k.toLowerCase().includes('loss'))

    return {
      chartData: Object.values(byIterEpoch).sort((a, b) => a.epoch - b.epoch),
      metricKeys: keys,
      lossKeys: loss,
      scoreKeys: score,
    }
  }, [allEpochs, selectedIters])

  if (itersWithData.length === 0) {
    return <p className="text-gray-500 text-sm">No per-epoch training data yet.</p>
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Training Curves</h2>

      <div className="flex flex-wrap gap-2 mb-3">
        {itersWithData.map((it) => {
          const selected = selectedIters.includes(it.iteration_num)
          return (
            <button
              key={it.iteration_num}
              onClick={() => {
                setSelectedIters((prev) =>
                  selected
                    ? prev.filter((n) => n !== it.iteration_num)
                    : [...prev, it.iteration_num],
                )
              }}
              className={`rounded px-2 py-1 text-xs font-mono transition-colors ${
                selected
                  ? 'bg-blue-900/50 text-blue-300 border border-blue-700'
                  : 'bg-gray-800 text-gray-500 border border-gray-700 hover:text-gray-300'
              }`}
            >
              Iter {it.iteration_num}
            </button>
          )
        })}
      </div>

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 10, right: 60, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis yAxisId="score" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            {lossKeys.length > 0 && (
              <YAxis yAxisId="loss" orientation="right" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            )}
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#9ca3af' }}
              itemStyle={{ color: '#e5e7eb' }}
            />
            <Legend wrapperStyle={{ color: '#9ca3af' }} />
            {scoreKeys.map((key, i) => (
              <Line
                key={key}
                yAxisId="score"
                type="monotone"
                dataKey={key}
                stroke={PALETTE[i % PALETTE.length]}
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls
              />
            ))}
            {lossKeys.map((key, i) => (
              <Line
                key={key}
                yAxisId="loss"
                type="monotone"
                dataKey={key}
                stroke={PALETTE[(i + scoreKeys.length) % PALETTE.length]}
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                connectNulls
                opacity={0.6}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
