import { useRun, useIterations, useMetrics } from '../hooks/useRunDetail'
import { useWebSocket } from '../hooks/useWebSocket'
import { Header } from '../components/layout/Header'
import { MetricCards } from '../components/dashboard/MetricCards'
import { MetricProgressChart } from '../components/dashboard/MetricProgressChart'
import { TrainingCurves } from '../components/dashboard/TrainingCurves'
import { IterationTable } from '../components/dashboard/IterationTable'
import { AgentReasoning } from '../components/dashboard/AgentReasoning'
import { IterationComparison } from '../components/dashboard/IterationComparison'
import { BudgetTracker } from '../components/dashboard/BudgetTracker'
import { GpuResources } from '../components/dashboard/GpuResources'
import { LogsPanel } from '../components/dashboard/LogsPanel'
import { ArtifactsPanel } from '../components/dashboard/ArtifactsPanel'
import { ConfigDrawer } from '../components/dashboard/ConfigDrawer'

interface RunDashboardProps {
  runId: string
  onRefresh?: () => void
  onRunCreated?: (runId: string) => void
}

export function RunDashboard({ runId, onRefresh, onRunCreated }: RunDashboardProps) {
  useWebSocket(runId)

  const { data: run, isLoading: runLoading } = useRun(runId)
  const { data: iterations } = useIterations(runId)
  const { data: snapshots } = useMetrics(runId)

  if (runLoading) {
    return <p className="text-gray-500 p-8">Loading...</p>
  }

  if (!run) {
    return <p className="text-red-400 p-8">Run not found: {runId}</p>
  }

  const iters = iterations || []
  const snaps = snapshots || []

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <Header run={run} onRefresh={onRefresh} onRunCreated={onRunCreated} />
      <MetricCards run={run} />
      <MetricProgressChart run={run} snapshots={snaps} iterations={iters} />
      <TrainingCurves runId={runId} iterations={iters} />
      <AgentReasoning iterations={iters} />
      <IterationTable iterations={iters} />
      <IterationComparison runId={runId} iterations={iters} />
      <BudgetTracker run={run} iterations={iters} />
      <GpuResources runId={runId} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LogsPanel runId={runId} />
        <ArtifactsPanel runId={runId} />
      </div>
      <div className="flex justify-end">
        <ConfigDrawer runId={runId} />
      </div>
    </div>
  )
}
