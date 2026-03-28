import { useMemo, memo } from 'react'
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
import type { GpuSnapshot } from '../../api/types'
import { useGpuSnapshots, useGpuLatest } from '../../hooks/useRunDetail'

function GpuCard({ label, value, unit, color }: {
  label: string; value: string; unit: string; color: string
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-lg font-bold" style={{ color }}>
        {value}<span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>
      </p>
    </div>
  )
}

const GpuChart = memo(function GpuChart({ snapshots }: { snapshots: GpuSnapshot[] }) {
  const chartData = useMemo(() => {
    // Downsample to ~150 points max for smooth rendering
    const src = snapshots.length > 150
      ? snapshots.filter((_, i) => i % Math.ceil(snapshots.length / 150) === 0 || i === snapshots.length - 1)
      : snapshots

    return src.map((s) => ({
      time: s.timestamp ? new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
      util: s.utilization_pct,
      memPct: s.memory_total_mb ? (s.memory_used_mb || 0) / s.memory_total_mb * 100 : null,
      temp: s.temperature_c,
    }))
  }, [snapshots])

  if (chartData.length < 2) return null

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData} margin={{ top: 10, right: 60, left: 10, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <YAxis yAxisId="pct" domain={[0, 100]} stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
        <YAxis yAxisId="temp" orientation="right" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 12 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
          labelStyle={{ color: '#9ca3af' }}
          itemStyle={{ color: '#e5e7eb' }}
        />
        <Legend wrapperStyle={{ color: '#9ca3af' }} />
        <Line yAxisId="pct" type="monotone" dataKey="util" name="Util %" stroke="#3b82f6" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
        <Line yAxisId="pct" type="monotone" dataKey="memPct" name="Mem %" stroke="#22c55e" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
        <Line yAxisId="temp" type="monotone" dataKey="temp" name="Temp °C" stroke="#ef4444" strokeWidth={1} strokeDasharray="5 5" dot={false} connectNulls opacity={0.7} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
})

export function GpuResources({ runId }: { runId: string }) {
  const { data: snapshots } = useGpuSnapshots(runId)
  const { data: latest } = useGpuLatest(runId)

  if (!snapshots?.length && !latest) {
    return <p className="text-gray-500 text-sm">No GPU data recorded yet.</p>
  }

  const memPct = latest && latest.memory_total_mb
    ? ((latest.memory_used_mb || 0) / latest.memory_total_mb * 100).toFixed(0)
    : 'N/A'

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">GPU Resources</h2>

      {latest && (
        <div className="grid grid-cols-4 gap-3 mb-4">
          <GpuCard
            label="Utilization"
            value={latest.utilization_pct != null ? `${latest.utilization_pct}` : 'N/A'}
            unit="%"
            color="#3b82f6"
          />
          <GpuCard
            label="Memory"
            value={memPct}
            unit="%"
            color="#22c55e"
          />
          <GpuCard
            label="VRAM Used"
            value={latest.memory_used_mb != null ? `${(latest.memory_used_mb / 1024).toFixed(1)}` : 'N/A'}
            unit="GB"
            color="#f59e0b"
          />
          <GpuCard
            label="Temperature"
            value={latest.temperature_c != null ? `${latest.temperature_c}` : 'N/A'}
            unit="°C"
            color="#ef4444"
          />
        </div>
      )}

      <GpuChart snapshots={snapshots || []} />
    </div>
  )
}
