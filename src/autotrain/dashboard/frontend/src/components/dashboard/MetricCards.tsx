import type { Run } from '../../api/types'

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-bold text-white">{value}</p>
    </div>
  )
}

export function MetricCards({ run }: { run: Run }) {
  const best = run.best_metric_value != null
    ? run.best_metric_value.toFixed(4)
    : 'N/A'

  return (
    <div className="grid grid-cols-4 gap-4 mb-6">
      <Card label="Status" value={(run.status || 'unknown').toUpperCase()} />
      <Card label="Best" value={best} />
      <Card label="Iterations" value={String(run.total_iterations)} />
      <Card label="API Cost" value={`$${run.total_api_cost.toFixed(2)}`} />
    </div>
  )
}
