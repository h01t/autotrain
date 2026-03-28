import { OUTCOME_COLORS, OUTCOME_ICONS } from '../../api/types'

export function OutcomeBadge({ outcome }: { outcome: string | null }) {
  const key = outcome || 'no_change'
  const color = OUTCOME_COLORS[key] || '#94a3b8'
  const icon = OUTCOME_ICONS[key] || '?'

  return (
    <span
      className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium"
      style={{ backgroundColor: `${color}20`, color }}
    >
      [{icon}] {key}
    </span>
  )
}
