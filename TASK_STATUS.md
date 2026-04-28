# Task Status — Phase 1.0 Milestone #2: Full Dashboard Run Creation & Control

| # | Task | Status |
|---|---|---|
| 1 | Pydantic request/response models | DONE |
| 2 | Dashboard control service layer | DONE |
| 3 | API endpoints (create/start/stop/restart/preflight/validate) | DONE |
| 4 | Backend tests for all new endpoints (30 tests) | DONE |
| 5 | Frontend types + API client | DONE |
| 6 | "New Run" modal/form component | DONE |
| 7 | Run control buttons (start/stop/restart) | DONE |
| 8 | Preflight results + expandable "Why this failed?" | DONE |
| 9 | Skeleton loaders for pending states | DONE |
| 10 | Keyboard shortcut Ctrl/Cmd+K → "New Run" | DONE |
| 11 | Full test suite + lint pass (249 tests, 0 warnings) | DONE |
| 12 | Preflight gating on Create button | DONE |
| 13 | Config endpoint + ConfigDrawer component | DONE |
| 14 | Logs endpoint + LogsPanel component | DONE |
| 15 | Artifacts endpoint + ArtifactsPanel (safety-hardened) | DONE |
| 16 | Defaults endpoint + template prefetch | DONE |
| 17 | Save-config endpoint | DONE |
| 18 | True resume flow (new run record + link) | DONE |
| 19 | Same-repo active-run guard (409 Conflict) | DONE |
| 20 | Stop endpoint preserves terminal status | DONE |
| 21 | AgentLoop accepts optional run_id parameter | DONE |
| 22 | Schema migration V5 — resumed_from_run_id column | DONE |
| **23** | **Duplicate run creation blocked (AgentLoop skips create_run)** | **DONE** |
| **24** | **Same-repo guard extended to start_run() + resume_run()** | **DONE** |
| **25** | **Resume UX auto-selects new run (no manual switch message)** | **DONE** |
| **26** | **Save-config validates via load_config() (not just YAML parse)** | **DONE** |
| **27** | **Quick-preflight gate in create_and_start/start_run/resume_run** | **DONE** |
| **28** | **Missing tests added: start conflict, resume conflict, duplicate rows, schema validation** | **DONE** |
