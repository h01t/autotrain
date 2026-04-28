import { useState, useEffect, useCallback } from 'react'
import { useRuns } from './hooks/useRuns'
import { Sidebar } from './components/layout/Sidebar'
import { RunDashboard } from './pages/RunDashboard'
import { NewRunModal } from './components/dashboard/NewRunModal'

export default function App() {
  const { data: runs, isLoading, error, refetch } = useRuns()
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>()
  const [newRunOpen, setNewRunOpen] = useState(false)

  const allRuns = runs || []

  // Auto-select first run if none selected
  useEffect(() => {
    if (!selectedRunId && allRuns.length > 0) {
      setSelectedRunId(allRuns[0].id)
    }
  }, [selectedRunId, allRuns])

  // Keyboard shortcut: Cmd/Ctrl+K → New Run
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault()
      setNewRunOpen(true)
    }
  }, [])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const handleRunCreated = useCallback((runId: string) => {
    refetch()
    setSelectedRunId(runId)
  }, [refetch])

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
        onNewRun={() => setNewRunOpen(true)}
      />
      <main className="flex-1 overflow-hidden flex flex-col">
        {isLoading ? (
          <LoadingSkeleton />
        ) : selectedRunId ? (
          <RunDashboard runId={selectedRunId} onRefresh={refetch} />
        ) : (
          <div className="flex flex-col items-center justify-center flex-1 gap-4">
            <p className="text-gray-600 text-lg">No runs found. Start a training run to see data here.</p>
            <button
              onClick={() => setNewRunOpen(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors"
            >
              + New Run
            </button>
            <p className="text-gray-700 text-xs">
              Press <kbd className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">⌘K</kbd> to open
            </p>
          </div>
        )}
      </main>
      <NewRunModal
        isOpen={newRunOpen}
        onClose={() => setNewRunOpen(false)}
        onRunCreated={handleRunCreated}
      />
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="p-8 space-y-6 animate-pulse">
      <div className="h-8 bg-gray-800 rounded w-1/3" />
      <div className="h-4 bg-gray-800 rounded w-1/2" />
      <div className="grid grid-cols-4 gap-4 mt-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 bg-gray-800 rounded-lg" />
        ))}
      </div>
      <div className="h-64 bg-gray-800 rounded-lg mt-4" />
      <div className="h-48 bg-gray-800 rounded-lg mt-4" />
    </div>
  )
}
