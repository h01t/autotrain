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
import type { Iteration, EpochMetric } from '../../api/types'
import { OutcomeBadge } from '../shared/OutcomeBadge'
import { useEpochMetrics } from '../../hooks/useRunDetail'

const PALETTE = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b']

interface Props {
  runId: string
  iterations: Iteration[]
}

function IterPicker({ label, iterations, value, onChange }: {
  label: string
  iterations: Iteration[]
  value: number | null
  onChange: (n: number | null) => void
}) {
  return (
    <div>
      <label className="text-xs text-gray-500 block mb-1">{label}</label>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
        className="bg-gray-800 text-gray-300 text-sm rounded px-2 py-1 border border-gray-700"
      >
        <option value="">Select...</option>
        {iterations.map((it) => (
          <option key={it.iteration_num} value={it.iteration_num}>
            Iter {it.iteration_num} — {it.outcome || '?'} ({it.metric_value?.toFixed(4) ?? 'N/A'})
          </option>
        ))}
      </select>
    </div>
  )
}

function DeltaRow({ label, a, b, fmt, higherBetter }: {
  label: string; a: number | null; b: number | null; fmt: (v: number) => string; higherBetter?: boolean
}) {
  if (a == null || b == null) return null
  const delta = b - a
  const positive = higherBetter ? delta > 0 : delta < 0
  const color = delta === 0 ? 'text-gray-400' : positive ? 'text-green-400' : 'text-red-400'
  const sign = delta > 0 ? '+' : ''
  return (
    <tr className="text-sm">
      <td className="text-gray-500 pr-4 py-1">{label}</td>
      <td className="text-gray-300 font-mono pr-4">{fmt(a)}</td>
      <td className="text-gray-300 font-mono pr-4">{fmt(b)}</td>
      <td className={`font-mono ${color}`}>{sign}{fmt(delta)}</td>
    </tr>
  )
}

export function IterationComparison({ runId, iterations }: Props) {
  const sorted = [...iterations].sort((a, b) => a.iteration_num - b.iteration_num)
  const [iterA, setIterA] = useState<number | null>(sorted.length >= 2 ? sorted[sorted.length - 2].iteration_num : null)
  const [iterB, setIterB] = useState<number | null>(sorted.length >= 1 ? sorted[sorted.length - 1].iteration_num : null)

  const a = iterations.find((it) => it.iteration_num === iterA)
  const b = iterations.find((it) => it.iteration_num === iterB)

  const { data: allEpochs } = useEpochMetrics(runId)

  const chartData = useMemo(() => {
    if (!allEpochs || iterA == null || iterB == null) return []
    const byEpoch: Record<number, Record<string, number>> = {}
    for (const em of allEpochs) {
      if (em.iteration_num !== iterA && em.iteration_num !== iterB) continue
      if (!byEpoch[em.epoch]) byEpoch[em.epoch] = { epoch: em.epoch }
      const prefix = em.iteration_num === iterA ? 'A_' : 'B_'
      for (const [k, v] of Object.entries(em.metrics)) {
        byEpoch[em.epoch][`${prefix}${k}`] = v
      }
    }
    return Object.values(byEpoch).sort((a, b) => a.epoch - b.epoch)
  }, [allEpochs, iterA, iterB])

  const metricKeys = useMemo(() => {
    if (!chartData.length) return []
    const keys = new Set<string>()
    for (const row of chartData) {
      for (const k of Object.keys(row)) {
        if (k !== 'epoch' && k.startsWith('A_')) keys.add(k.slice(2))
      }
    }
    return Array.from(keys)
  }, [chartData])

  if (iterations.length < 2) {
    return <p className="text-gray-500 text-sm">Need at least 2 iterations to compare.</p>
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Iteration Comparison</h2>

      <div className="flex items-end gap-4 mb-4">
        <IterPicker label="Iteration A" iterations={sorted} value={iterA} onChange={setIterA} />
        <span className="text-gray-600 text-lg pb-1">vs</span>
        <IterPicker label="Iteration B" iterations={sorted} value={iterB} onChange={setIterB} />
      </div>

      {a && b && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 mb-4">
          <table>
            <thead>
              <tr className="text-xs text-gray-500">
                <th className="text-left pr-4 pb-2">Metric</th>
                <th className="text-left pr-4 pb-2">Iter {a.iteration_num}</th>
                <th className="text-left pr-4 pb-2">Iter {b.iteration_num}</th>
                <th className="text-left pb-2">Delta</th>
              </tr>
            </thead>
            <tbody>
              <DeltaRow label="Metric" a={a.metric_value} b={b.metric_value} fmt={(v) => v.toFixed(4)} higherBetter />
              <DeltaRow label="Duration" a={a.duration_seconds} b={b.duration_seconds} fmt={(v) => `${v.toFixed(0)}s`} />
              <DeltaRow label="API Cost" a={a.api_cost} b={b.api_cost} fmt={(v) => `$${v.toFixed(3)}`} />
              <tr className="text-sm">
                <td className="text-gray-500 pr-4 py-1">Outcome</td>
                <td className="pr-4"><OutcomeBadge outcome={a.outcome} /></td>
                <td className="pr-4"><OutcomeBadge outcome={b.outcome} /></td>
                <td />
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {chartData.length > 0 && metricKeys.length > 0 && (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#9ca3af' }}
              itemStyle={{ color: '#e5e7eb' }}
            />
            <Legend wrapperStyle={{ color: '#9ca3af' }} />
            {metricKeys.map((key, i) => (
              <Line
                key={`A_${key}`}
                type="monotone"
                dataKey={`A_${key}`}
                name={`A: ${key}`}
                stroke={PALETTE[i % PALETTE.length]}
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls
              />
            ))}
            {metricKeys.map((key, i) => (
              <Line
                key={`B_${key}`}
                type="monotone"
                dataKey={`B_${key}`}
                name={`B: ${key}`}
                stroke={PALETTE[i % PALETTE.length]}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 2 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
