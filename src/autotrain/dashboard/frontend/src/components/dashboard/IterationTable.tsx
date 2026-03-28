import type { Iteration } from '../../api/types'
import { OutcomeBadge } from '../shared/OutcomeBadge'

export function IterationTable({ iterations }: { iterations: Iteration[] }) {
  if (!iterations.length) {
    return <p className="text-gray-500 text-sm">No iterations yet.</p>
  }

  const sorted = [...iterations].reverse() // newest first

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Iteration History</h2>
      <div className="overflow-x-auto rounded-lg border border-gray-800">
        <table className="w-full text-sm">
          <thead className="bg-gray-900 text-gray-400">
            <tr>
              <th className="px-3 py-2 text-left">#</th>
              <th className="px-3 py-2 text-left">Metric</th>
              <th className="px-3 py-2 text-left">Outcome</th>
              <th className="px-3 py-2 text-left">Hypothesis</th>
              <th className="px-3 py-2 text-left">Duration</th>
              <th className="px-3 py-2 text-left">Commit</th>
              <th className="px-3 py-2 text-left">Flags</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {sorted.map((it) => {
              const flags: string[] = []
              if (it.resumed_from_checkpoint) flags.push('R')
              if (it.checkpoint_path) flags.push('C')

              return (
                <tr key={it.id} className="text-gray-300 hover:bg-gray-800/50">
                  <td className="px-3 py-2 font-mono">{it.iteration_num}</td>
                  <td className="px-3 py-2 font-mono">
                    {it.metric_value != null ? it.metric_value.toFixed(4) : 'N/A'}
                  </td>
                  <td className="px-3 py-2">
                    <OutcomeBadge outcome={it.outcome} />
                  </td>
                  <td className="px-3 py-2 max-w-xs truncate text-gray-400">
                    {it.agent_hypothesis?.slice(0, 80) || ''}
                  </td>
                  <td className="px-3 py-2 text-gray-500">
                    {it.duration_seconds ? `${it.duration_seconds.toFixed(0)}s` : '-'}
                  </td>
                  <td className="px-3 py-2 font-mono text-gray-500">
                    {it.commit_hash?.slice(0, 7) || ''}
                  </td>
                  <td className="px-3 py-2 text-gray-500">{flags.join(' ')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-600 mt-1">Flags: R=resumed from checkpoint, C=checkpoint saved</p>
    </div>
  )
}
