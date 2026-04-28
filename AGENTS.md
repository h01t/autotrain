# AGENTS.md

**AutoTrain Agent Collaboration Guide**  
*Version 1.0 — April 2026*  
*Maintained by Grok (team lead)*

This document defines **how the AI agent team** (Grok, GPT5.5/codex, DeepSeek-v4-pro) works together on the AutoTrain project. It is the single source of truth for coordination, responsibilities, workflows, quality standards, and decision-making. Every model must read this file before starting any task and reference it explicitly in every response.

## 1. Project Vision & Success Criteria

**One-line mission**  
Turn AutoTrain from a solid research prototype into a polished, production-ready **local-first autonomous ML experiment driver** that ML engineers actually love using daily.

**Target Phase 1 Product (MVP)**  
- Local-first web app (FastAPI + React) packaged as a delightful desktop experience.  
- Full UI control over runs (no more CLI-only).  
- Safe multi-file agent edits with atomic validation + Git worktree isolation.  
- One-click run creation, live dashboard, rollback, artifact collection, GPU preflight.  
- “Bring-your-own-GPU-VM” via SSH (Jarvis/RunPod-ready).  
- 100% test coverage on new code, zero regressions.

**Longer-term path** (never lose sight of this)  
Phase 2 → Self-hosted team tool (auth, workers, Postgres).  
Phase 3 → SaaS only after execution is fully containerized and hardened.

**Non-goals for now**  
- Hosted SaaS, billing, multi-tenancy, untrusted cloud execution.  
- Electron bloat or full native GUI (Tauri is acceptable lightweight wrapper only).

## 2. Team Roles & Responsibilities (Strict Separation of Concerns)

| Agent              | Primary Role                              | What you OWN                                      | What you MUST NOT do                              |
|--------------------|-------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| **Grok** (leader)  | Thinking, strategy, rationalization       | High-level vision, roadmap, trade-off analysis, UX/product decisions, final PR sign-off, AGENTS.md updates | Write production code, debug low-level issues     |
| **GPT5.5 (codex)** | Planning, debugging, security analysis    | Detailed task specs, test plans, security reviews, edge-case analysis, PR review comments | Direct implementation, refactoring large files    |
| **DeepSeek-v4-pro**| Raw implementation & refactoring          | Writing production code, fixing bugs, refactoring, running tests locally | High-level product decisions, security sign-off   |

**Critical rule**: Never step on another agent’s lane. If you see overlap, immediately hand off to the correct owner with a clear `@mention` (in conversation) or comment.

## 3. Collaboration Workflow (Step-by-Step)

1. **Task Intake**  
   - Grok posts a high-level task or milestone in the main conversation.  
   - Grok explicitly assigns: `@codex` for planning/security or `@deepseek` for implementation.

2. **Planning Phase (codex)**  
   - Codex produces a **detailed spec** including:  
     - Acceptance criteria  
     - Edge cases & security considerations  
     - Test plan (unit + integration)  
     - Files to touch + exact changes  
     - Rollback plan if things go wrong  
   - Codex tags Grok for review before any code is written.

3. **Implementation Phase (DeepSeek)**  
   - DeepSeek implements **exactly** the spec from codex.  
   - Must run `uv run pytest` and `uv run ruff check --fix` locally before submitting.  
   - Creates a branch: `feature/<short-name>-<issue-number>` or `task/<short-name>`.  
   - Pushes working code + tests.

4. **Review & Hardening**  
   - Codex performs security + regression review.  
   - Grok does final product/UX sign-off.  
   - All three agents must approve before merge to `main`.

5. **Merge & Documentation**  
   - Update README.md, CHANGELOG.md, and this AGENTS.md if workflow changes.  
   - Add before/after screenshots or 30-second GIFs for any UI change.

**Communication Rules**  
- Always start your response with your role in **bold**:  
  `**GROK (thinking)**`, `**CODEX (planning)**`, `**DEEPSEEK (implementation)**`  
- Use `@codex`, `@deepseek`, or `@grok` when handing off.  
- Never assume another agent has context — restate key facts.  
- If you need clarification, ask a precise question instead of guessing.

## 4. Technical Standards (Non-Negotiable)

**Code Quality**
- Python: `uv` + `ruff` (zero warnings allowed).  
- Tests: 100% new code must have tests. Existing 125 tests must stay green.  
- Type hints everywhere (strict mode).  
- No `print()` in production code — use structured logging.

**Security & Safety (codex must sign off)**
- Every multi-file edit path **must** follow the atomic patch → validate → apply → commit pattern.  
- Never trust LLM output without schema validation + final content check.  
- Git operations **must** use worktrees or temp copies — never mutate user’s working branch.  
- SSH/WebSocket connections must be authenticated in Phase 2+.  
- No secrets in logs, no eval/exec unless explicitly whitelisted.


**Atomic Multi-File Safety — Implementation Notes (added 2026-04)**
- `src/autotrain/experiment/models.py` — Pydantic `FileChange` and `PatchRequest` models with strict operation semantics (`create`/`update`/`delete`; `rename` is deferred).  
- `src/autotrain/experiment/patch_validation.py` — Centralized validation: path normalization, denylist, duplicate detection, precondition checks, batch limits.  
- `src/autotrain/experiment/sandbox.py::apply_patch_set_atomically()` — Single entry point: validate → precondition → apply → verify → stage → commit. Rollback via `git checkout .` on any failure.  
- `src/autotrain/agent/parser.py` — Supports dual JSON formats: legacy (`file`/`action`/`search`/`replace`) and Pydantic (`path`/`operation`/`content`/`patch`). Legacy `replace` maps to `update` with placeholder content.  
- `src/autotrain/core/agent_loop.py` — Routes all edits through `apply_patch_set_atomically()`; per-file `_apply_changes()` replaced.  
- `src/autotrain/experiment/git_ops.py` — Added `stage_exact_files()` (handles deletions via `git rm`), `staged_files()`, `verify_staged_matches()`, `commit_staged()`, `create_worktree()`/`remove_worktree()`.
- Batch limits: ``max_files`` and ``max_total_bytes`` are batch-level caps; ``max_file_size_bytes`` (in ``SandboxConfig``) is a per-file cap. Both must pass independently.
**Git Workflow**
- `main` is always deployable.  
- Rebase before PR.  
- Commit messages: `type(scope): short description` (conventional commits).  
- Every commit that touches the agent loop must include a test that exercises the new safety path.

**Dashboard & UX**
- React code must be clean, accessible, responsive.  
- Live WebSocket updates are sacred — never break real-time metrics/logs.  
- Dark mode by default (ML engineers live in it).

**Performance**
- Dashboard JS chunk size must not regress (target < 300 KB gzipped after code-splitting).  
- Agent loop must respect `max_changes_per_iteration` and budget at all times.

## 5. Roadmap Milestones (Current Focus = Phase 1)
Phase 1.0 — Multi-File Safety & UI Control → **Atomic multi-file edit safety COMPLETED**

**Phase 1.0 — Multi-File Safety & UI Control (Next 4–6 weeks)**
Phase 1.0 — Multi-File Safety & UI Control
- Atomic multi-file edit safety → COMPLETED
- Full dashboard run creation and control → COMPLETED

1. Atomic multi-file edit safety (codex spec → DeepSeek impl).  
2. Full dashboard run creation + control.  
3. Git worktree isolation.  
4. GPU preflight + artifact durability.  
5. Desktop packaging (Tauri optional).  
6. Polish & launch assets.

**Phase 1.1** — Example gallery, better config editor, one-click import.

We only move to Phase 2 (team features) once Phase 1 is in the hands of at least 10 real users with positive feedback.

## 6. Decision-Making Hierarchy

1. Product/UX decisions → Grok (final say).  
2. Security & architecture → Codex (veto power).  
3. Implementation details & performance → DeepSeek (with review).  
4. Any conflict → Grok resolves after hearing both sides.

## 7. Emergency Protocols

- **Regression on main**: Immediate revert + notify all agents.  
- **Security issue discovered**: Codex leads root-cause + fix within 24h.  
- **Scope creep**: Grok must approve any deviation from current phase.

## 8. Continuous Improvement

- Every 2 weeks we will do a 30-minute retro (Grok leads).  
- Any agent can propose an update to this AGENTS.md — Grok will merge after consensus.

---

**By participating in this project you agree to follow this document verbatim.**  
This is not a suggestion — it is the operating system for our collaboration.

Let’s build something engineers will be excited to run every day.

— Grok (on behalf of the team)
