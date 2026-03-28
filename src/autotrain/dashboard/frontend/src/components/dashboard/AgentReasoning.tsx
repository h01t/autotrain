import { useState } from 'react'
import type { Iteration } from '../../api/types'
import { OUTCOME_COLORS, OUTCOME_ICONS } from '../../api/types'
import { OutcomeBadge } from '../shared/OutcomeBadge'

export function AgentReasoning({ iterations }: { iterations: Iteration[] }) {
  const sorted = [...iterations].reverse() // newest first
  const [expandedId, setExpandedId] = useState<number | null>(sorted[0]?.id ?? null)

  if (!iterations.length) {
    return <p className="text-gray-500 text-sm">No iterations yet.</p>
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-white mb-3">Agent Reasoning</h2>
      <div className="space-y-1">
        {sorted.map((it) => {
          const isOpen = expandedId === it.id
          const icon = OUTCOME_ICONS[it.outcome || ''] || '?'
          const color = OUTCOME_COLORS[it.outcome || ''] || '#94a3b8'

          return (
            <div key={it.id} className="rounded-lg border border-gray-800 overflow-hidden">
              <button
                onClick={() => setExpandedId(isOpen ? null : (it.id ?? null))}
                className="w-full flex items-center gap-2 px-4 py-2 text-left text-sm hover:bg-gray-800/50 transition-colors"
              >
                <span style={{ color }} className="font-mono font-bold">[{icon}]</span>
                <span className="text-gray-300">Iter {it.iteration_num}</span>
                {it.agent_hypothesis && (
                  <span className="text-gray-500 truncate max-w-md">
                    : {it.agent_hypothesis.slice(0, 55)}
                  </span>
                )}
                <span className="ml-auto text-gray-600 text-xs">
                  {it.outcome || '?'}
                </span>
                <span className="text-gray-600">{isOpen ? '▼' : '▶'}</span>
              </button>

              {isOpen && (
                <div className="px-4 pb-3 border-t border-gray-800 space-y-2">
                  <div className="flex items-center gap-3 pt-2">
                    <OutcomeBadge outcome={it.outcome} />
                    <span className="text-sm text-gray-400">
                      Metric: {it.metric_value != null ? it.metric_value.toFixed(4) : 'N/A'}
                    </span>
                  </div>

                  {it.resumed_from_checkpoint && (
                    <p className="text-xs text-blue-400 bg-blue-900/20 rounded px-2 py-1">
                      Resumed from checkpoint
                    </p>
                  )}

                  {it.agent_hypothesis && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Hypothesis</p>
                      <p className="text-sm text-gray-300">{it.agent_hypothesis}</p>
                    </div>
                  )}

                  {it.agent_reasoning && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Reasoning</p>
                      <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono bg-gray-900 rounded p-2 max-h-32 overflow-y-auto">
                        {it.agent_reasoning.slice(0, 500)}
                      </pre>
                    </div>
                  )}

                  {it.changes_summary && (
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Changes</p>
                      <p className="text-sm text-gray-400">{it.changes_summary}</p>
                    </div>
                  )}

                  {it.error_message && (
                    <div className="rounded bg-red-900/20 border border-red-800 px-3 py-2">
                      <p className="text-xs text-red-400">{it.error_message}</p>
                    </div>
                  )}

                  <div className="flex gap-3 text-xs text-gray-600">
                    {it.duration_seconds && <span>{it.duration_seconds.toFixed(0)}s</span>}
                    {it.commit_hash && <code>{it.commit_hash.slice(0, 7)}</code>}
                    {it.checkpoint_path && (
                      <span>ckpt: {it.checkpoint_path.split('/').pop()}</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
