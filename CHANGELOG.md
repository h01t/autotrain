## [Unreleased] - 2026-04-28
### Added
- Full dashboard run creation and control (Phase 1.0 Milestone #2)
  - Create, preflight, start, stop, resume runs entirely from UI
  - Form + raw YAML modes with validation
  - Inline preflight with blockers/warnings
  - Live logs, artifacts, config drawer
  - Same-repo active-run guard (409 Conflict)
  - Proper resume semantics (new run record + link)
  - Full path safety and artifact boundary enforcement
  - 249 tests passing (+72 from Milestone #1)

## [Unreleased] - 2026-04-28
### Added
- Atomic multi-file edit safety (Phase 1.0 Milestone #1)
  - Full Pydantic schema validation with strict Literal actions
  - New `patch_validation.py` module
  - `apply_patch_set_atomically()` with git worktree isolation
  - Typed exception hierarchy (`AutoTrainEditError`)
  - 72 new tests, 197 total passing