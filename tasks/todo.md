- [x] Audit the current public-facing repo state: README, overview docs, metadata, license, and tests.
- [x] Add missing public-release artifacts and metadata polish (`LICENSE`, `pyproject.toml`, frontend `package.json`).
- [x] Rewrite `README.md` for a cleaner research-first public presentation with Mermaid diagrams and docs links.
- [x] Create canonical `docs/project-overview.md` with title page, diagrams, appendix, and Pandoc export guidance.
- [x] Retire duplicate overview artifacts and keep one authoritative long-form narrative in the repo.
- [x] Run verification: tests, doc link checks, and a Pandoc dry-run.

## Review

- Added a top-level MIT `LICENSE` and polished package metadata for the Python package and frontend dashboard.
- Replaced the previous public README with a shorter research-first version that uses Mermaid diagrams and points to one canonical overview document.
- Created `docs/project-overview.md` as the single long-form narrative with title page, architecture/flow diagrams, appendix material, and export guidance.
- Retired duplicate overview artifacts from `docs/` and the repo root so the public surface has one authoritative project overview.
- Verification: `125` tests passed, README/project overview links resolve, Pandoc HTML export succeeded, and PDF export is source-ready but locally blocked by missing `pdflatex`.
