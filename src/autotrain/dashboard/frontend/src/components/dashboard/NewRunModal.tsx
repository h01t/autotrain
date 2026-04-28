import { useState, useEffect, useCallback } from 'react'
import {
  createRun,
  validateConfig,
  runPreflight,
} from '../../api/client'
import type {
  ConfigValidationError,
  PreflightResult,
  PreflightGpuInfo,
  CreateRunResponse,
} from '../../api/types'

interface NewRunModalProps {
  isOpen: boolean
  onClose: () => void
  onRunCreated: (runId: string) => void
}

const DEFAULT_CONFIG = `repo_path: /path/to/your/repo
metric:
  name: val_loss
  target: 0.1
  direction: minimize
execution:
  mode: local
  train_command: python train.py
`

export function NewRunModal({ isOpen, onClose, onRunCreated }: NewRunModalProps) {
  const [configYaml, setConfigYaml] = useState(DEFAULT_CONFIG)
  const [startImmediately, setStartImmediately] = useState(true)
  const [tab, setTab] = useState<'edit' | 'validate' | 'preflight'>('edit')

  // Validation state
  const [validationErrors, setValidationErrors] = useState<ConfigValidationError[]>([])
  const [validationWarnings, setValidationWarnings] = useState<string[]>([])
  const [isValid, setIsValid] = useState<boolean | null>(null)
  const [validating, setValidating] = useState(false)

  // Preflight state
  const [preflightResults, setPreflightResults] = useState<PreflightResult[]>([])
  const [preflightGpus, setPreflightGpus] = useState<PreflightGpuInfo[]>([])
  const [preflightPassed, setPreflightPassed] = useState<boolean | null>(null)
  const [preflightDuration, setPreflightDuration] = useState<number>(0)
  const [preflighting, setPreflighting] = useState(false)

  // Create state
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setValidationErrors([])
      setValidationWarnings([])
      setIsValid(null)
      setPreflightResults([])
      setPreflightGpus([])
      setPreflightPassed(null)
      setCreateError(null)
    }
  }, [isOpen])

  const handleValidate = useCallback(async () => {
    setValidating(true)
    setValidationErrors([])
    setValidationWarnings([])
    setIsValid(null)
    try {
      const result = await validateConfig(configYaml)
      setValidationErrors(result.errors)
      setValidationWarnings(result.warnings)
      setIsValid(result.valid)
      if (result.valid) {
        setTab('preflight')
      }
    } catch (e) {
      setValidationErrors([{ field: '(request)', message: String(e) }])
      setIsValid(false)
    } finally {
      setValidating(false)
    }
  }, [configYaml])

  const handlePreflight = useCallback(async () => {
    setPreflighting(true)
    setPreflightResults([])
    setPreflightGpus([])
    setPreflightPassed(null)
    try {
      // Parse repo_path and other fields from YAML
      const lines = configYaml.split('\n')
      let repoPath = ''
      let mode = 'local'
      let trainCommand = 'python train.py'
      for (const line of lines) {
        const trimmed = line.trim()
        if (trimmed.startsWith('repo_path:')) {
          repoPath = trimmed.replace('repo_path:', '').trim()
        }
        if (trimmed.startsWith('mode:') && mode === 'local') {
          mode = trimmed.replace('mode:', '').trim()
        }
        if (trimmed.startsWith('train_command:')) {
          trainCommand = trimmed.replace('train_command:', '').trim()
        }
      }

      const result = await runPreflight({
        repo_path: repoPath,
        mode,
        train_command: trainCommand,
      })
      setPreflightResults(result.checks)
      setPreflightGpus(result.gpus)
      setPreflightPassed(result.passed)
      setPreflightDuration(result.duration_seconds)
    } catch (e) {
      setPreflightResults([{
        check: 'request',
        passed: false,
        message: `Preflight request failed: ${e}`,
        detail: String(e),
      }])
      setPreflightPassed(false)
    } finally {
      setPreflighting(false)
    }
  }, [configYaml])

  const handleCreate = useCallback(async () => {
    setCreating(true)
    setCreateError(null)
    try {
      const result: CreateRunResponse = await createRun(configYaml, startImmediately)
      if (result.run_id) {
        onRunCreated(result.run_id)
        onClose()
      } else {
        setValidationErrors(result.config_errors)
        setIsValid(false)
        setCreateError(result.message)
      }
    } catch (e) {
      setCreateError(String(e))
    } finally {
      setCreating(false)
    }
  }, [configYaml, startImmediately, onRunCreated, onClose])

  // Keyboard: Escape to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-[900px] max-h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <h2 className="text-lg font-bold text-white">New Training Run</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 text-xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-800 px-6">
          {(['edit', 'validate', 'preflight'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                tab === t
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              {t === 'edit' && '1. Edit Config'}
              {t === 'validate' && '2. Validate'}
              {t === 'preflight' && '3. Preflight'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {tab === 'edit' && (
            <div className="space-y-4">
              <textarea
                value={configYaml}
                onChange={(e) => setConfigYaml(e.target.value)}
                className="w-full h-64 bg-gray-950 text-gray-200 font-mono text-sm p-4 rounded border border-gray-700 focus:border-blue-500 focus:outline-none resize-y"
                spellCheck={false}
              />
              <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={startImmediately}
                  onChange={(e) => setStartImmediately(e.target.checked)}
                  className="rounded bg-gray-800 border-gray-600"
                />
                Start training immediately after creation
              </label>
            </div>
          )}

          {tab === 'validate' && (
            <div className="space-y-4">
              {validating ? (
                <ValidateSkeleton />
              ) : isValid === true ? (
                <div className="rounded-lg bg-green-900/30 border border-green-700 p-4">
                  <p className="text-green-400 font-semibold">✓ Configuration is valid</p>
                  {validationWarnings.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {validationWarnings.map((w, i) => (
                        <p key={i} className="text-yellow-400 text-sm">⚠ {w}</p>
                      ))}
                    </div>
                  )}
                </div>
              ) : isValid === false ? (
                <div className="space-y-2">
                  <div className="rounded-lg bg-red-900/30 border border-red-700 p-4">
                    <p className="text-red-400 font-semibold">✗ Configuration has errors</p>
                  </div>
                  {validationErrors.map((err, i) => (
                    <div
                      key={i}
                      className="rounded-lg bg-gray-950 border border-red-900/50 p-3"
                    >
                      <p className="text-red-400 text-sm font-mono">{err.field}</p>
                      <p className="text-gray-300 text-sm mt-1">{err.message}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">
                  Click "Validate" to check your configuration.
                </p>
              )}

              <button
                onClick={handleValidate}
                disabled={validating}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded text-sm font-medium transition-colors"
              >
                {validating ? 'Validating...' : 'Validate'}
              </button>
            </div>
          )}

          {tab === 'preflight' && (
            <div className="space-y-4">
              {preflighting ? (
                <PreflightSkeleton />
              ) : preflightPassed === true ? (
                <div className="rounded-lg bg-green-900/30 border border-green-700 p-4">
                  <p className="text-green-400 font-semibold">
                    ✓ All preflight checks passed ({preflightDuration}s)
                  </p>
                </div>
              ) : preflightPassed === false ? (
                <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 mb-3">
                  <p className="text-red-400 font-semibold">
                    ✗ Some preflight checks failed
                  </p>
                </div>
              ) : null}

              {/* GPU info */}
              {preflightGpus.length > 0 && (
                <div className="rounded-lg bg-gray-950 border border-gray-700 p-4">
                  <p className="text-gray-300 text-sm font-semibold mb-2">
                    GPUs Detected
                  </p>
                  {preflightGpus.map((gpu) => (
                    <div key={gpu.index} className="flex items-center gap-3 text-sm text-gray-400 py-1">
                      <span className="text-blue-400 font-mono">GPU {gpu.index}</span>
                      <span>{gpu.name}</span>
                      <span className="text-gray-600">
                        {gpu.memory_free_mb?.toFixed(0)} MB free / {gpu.memory_total_mb?.toFixed(0)} MB
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Individual check results */}
              {preflightResults.map((check) => (
                <PreflightCheckCard key={check.check} check={check} />
              ))}

              <button
                onClick={handlePreflight}
                disabled={preflighting}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded text-sm font-medium transition-colors"
              >
                {preflighting ? 'Running preflight...' : 'Run Preflight'}
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-800">
          <div>
            {createError && (
              <p className="text-red-400 text-sm">{createError}</p>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
            >
              Cancel
            </button>
            {tab !== 'edit' && (
              <button
                onClick={() => setTab(tab === 'validate' ? 'edit' : 'validate')}
                className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
              >
                Back
              </button>
            )}
            <button
              onClick={handleCreate}
              disabled={creating || isValid !== true}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded text-sm font-semibold transition-colors"
            >
              {creating ? 'Creating...' : 'Create Run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Preflight Check Card (with expandable "Why this failed?") ── */

function PreflightCheckCard({ check }: { check: PreflightResult }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`rounded-lg border p-3 ${
        check.passed
          ? 'bg-gray-950 border-gray-700'
          : 'bg-red-950/30 border-red-800'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={check.passed ? 'text-green-400' : 'text-red-400'}>
            {check.passed ? '✓' : '✗'}
          </span>
          <span className="text-gray-300 text-sm font-medium">{check.check}</span>
        </div>
        {!check.passed && check.detail && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-400 hover:text-blue-300 shrink-0 mt-0.5"
          >
            {expanded ? 'Hide' : 'Why this failed?'}
          </button>
        )}
      </div>
      <p className="text-gray-500 text-xs mt-1 ml-6">{check.message}</p>

      {expanded && !check.passed && check.detail && (
        <div className="mt-2 ml-6 space-y-2">
          <div className="bg-gray-950 border border-gray-700 rounded p-3">
            <p className="text-gray-400 text-xs whitespace-pre-wrap">{check.detail}</p>
          </div>
          {check.suggestion && (
            <div className="bg-blue-950/30 border border-blue-800 rounded p-3">
              <p className="text-blue-400 text-xs font-medium mb-1">Suggestion</p>
              <code className="text-blue-300 text-xs">{check.suggestion}</code>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Skeleton Loaders ── */

function ValidateSkeleton() {
  return (
    <div className="space-y-3 animate-pulse">
      <div className="h-16 bg-gray-800 rounded-lg" />
      <div className="h-10 bg-gray-800 rounded-lg w-3/4" />
      <div className="h-10 bg-gray-800 rounded-lg w-1/2" />
    </div>
  )
}

function PreflightSkeleton() {
  return (
    <div className="space-y-3 animate-pulse">
      <div className="h-12 bg-gray-800 rounded-lg" />
      <div className="h-16 bg-gray-800 rounded-lg" />
      <div className="h-16 bg-gray-800 rounded-lg" />
      <div className="h-16 bg-gray-800 rounded-lg" />
      <div className="h-10 bg-gray-800 rounded-lg w-1/2" />
    </div>
  )
}
