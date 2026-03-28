import { useState, useEffect } from 'react'
import { useRuns } from './hooks/useRuns'
import { Sidebar } from './components/layout/Sidebar'
import { RunDashboard } from './pages/RunDashboard'

export default function App() {
  const { data: runs, isLoading, error } = useRuns()
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>()

  const allRuns = runs || []

  // Auto-select first run if none selected
  useEffect(() => {
    if (!selectedRunId && allRuns.length > 0) {
      setSelectedRunId(allRuns[0].id)
    }
  }, [selectedRunId, allRuns])

  if (error) {
    return (
      <div className="flex h-screen bg-gray-950 text-red-400 items-center justify-center">
        <p>API error: {String(error)}</p>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <Sidebar
        runs={allRuns}
        selectedRunId={selectedRunId}
        onSelectRun={setSelectedRunId}
      />
      <main className="flex-1 overflow-hidden flex flex-col">
        {isLoading ? (
          <p className="text-gray-500 p-8">Loading runs...</p>
        ) : selectedRunId ? (
          <RunDashboard runId={selectedRunId} />
        ) : (
          <div className="flex items-center justify-center flex-1">
            <p className="text-gray-600 text-lg">No runs found. Start a training run to see data here.</p>
          </div>
        )}
      </main>
    </div>
  )
}
