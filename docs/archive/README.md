# Documentation Archive

## Purpose

This directory preserves historical design documents, implementation plans, progress reports, and point-in-time code reviews for their forensic and design-history value. These files are **not** current guidance and are **not** maintained. They capture the state of the project as of the dates they were written and may not match current code.

Current, maintained documentation lives elsewhere: under `docs/guides/`, `docs/specs/`, `docs/concepts/`, and in the per-package `README.md` / `SPECIFICATIONS.md` files. Consult those for anything you intend to rely on today.

## Status labels

These labels are used in the per-file banners and in the inventory below.

- **Historical report** — a point-in-time progress, completion, or analysis report; accurate only as of its date.
- **Superseded** — replaced by a newer document (the canonical document is named where one exists).
- **Implemented** — described work that shipped; kept for design history.
- **Abandoned** — an approach that was explored but not adopted.
- **Unresolved review** — a code-review report whose finding statuses were last checked on a stated date and have **not** been re-verified against current `main`.

## Retention policy

Archived files are kept indefinitely for their historical value. They are not edited to track current state — only archival banners and this index's metadata are added. Empty or content-free files are removed. Where a report exists as a series of near-duplicate snapshots, the series is indexed here with a single canonical document identified, rather than deleted, so the full history is preserved.

## Referencing archived material from active docs

Active (maintained) documentation may link into this archive **only** for historical context, and any such link **must** be labelled as historical — for example, appending "(historical)" to the link text. Never cite an archived file as current guidance.

## Link-validation policy

`docs/archive/**` is excluded from the Sphinx build (`exclude_patterns` in `docs/conf.py`), so archive pages are not published and their internal links are **not** checked by the docs build or any link checker. Archive internal links are therefore best-effort, point-in-time references and may rot over time.

A contributor who adds a link **from** an active (built) doc **into** the archive is responsible for that link's validity, because the build will not catch a broken one.

## Inventory

Every remaining file is listed below, grouped by subdirectory. `Category` is one of the status labels above.

### Top-level (`docs/archive/`)

| File | Category | Notes |
|------|----------|-------|
| `COMPLETE_DOCUMENTATION_AUDIT.md` | Historical report | Documentation audit / final verification pass (companion to `DOCUMENTATION_FINAL_VERIFICATION.md`, same Dec-2025 verification effort) |
| `DIMENSION_INITIALIZATION_FIX.md` | Implemented | Dimension-initialization fix that shipped |
| `DOCS_CLEANUP_SUMMARY.md` | Historical report | December 2025 documentation cleanup summary |
| `DOCUMENTATION_CLEANUP_COMPLETE.md` | Historical report | Documentation cleanup completion report |
| `DOCUMENTATION_ENHANCEMENTS_COMPLETE.md` | Historical report | Documentation enhancement completion summary |
| `DOCUMENTATION_FINAL_VERIFICATION.md` | Historical report | Documentation final verification report (companion to `COMPLETE_DOCUMENTATION_AUDIT.md`) |
| `DOCUMENTATION_IMPROVEMENT_PLAN.md` | Historical report | Documentation improvement plan (2025-12-13) |
| `DOCUMENTATION_INDEX.md` | Historical report | Old documentation index (replaced by the current docs tree) |
| `DOCUMENTATION_REVIEW_FINAL_REPORT.md` | Historical report | Project documentation review final report |
| `DOCUMENTATION_UPDATES_2025-12-12.md` | Historical report | Documentation updates log, 2025-12-12 |
| `E2E_COVERAGE_GAPS.md` | Historical report | E2E test coverage gaps and proposals |
| `E2E_TEST_AUDIT.md` | Historical report | E2E test suite audit and improvements |
| `FIXES_COMPLETED.md` | Historical report | High-priority fixes final report (later/final snapshot; supersedes `HIGH_PRIORITY_FIXES_SUMMARY.md`) |
| `HIGH_PRIORITY_FIXES_SUMMARY.md` | Historical report | High-priority fixes completion summary (earlier snapshot of the same 8-fix effort; see `FIXES_COMPLETED.md`) |
| `JSDOC_ENHANCEMENT_GUIDE.md` | Historical report | JSDoc enhancement plan/guide for a past phase |
| `LINES_SEGMENT_CHUNKING_BUG_FIX.md` | Implemented | Lines segment spatial-index chunking bug fix that shipped |
| `METAL_SPLATTING_IMPLEMENTATION_SPEC.md` | Historical report | Metal splatting engine spec; kept for design history |
| `PERFORMANCE_OPTIMIZATION_SPEC.md` | Historical report | Canonical performance-optimization **design** spec (v3.6.0); kept for design history |
| `PHASE_1_COMPLETION_REPORT.md` | Historical report | Phase 1 completion report |
| `PHASE_2_FINAL_SUMMARY.md` | Historical report | Phase 2 complete summary |
| `PHASE_2_PROGRESS.md` | Historical report | Phase 2 progress report |
| `PHASE_3_ENHANCEMENTS_SUMMARY.md` | Historical report | Phase 3 documentation-enhancements summary |
| `PHASE_4_AUTOMATION_SUMMARY.md` | Historical report | Phase 4 documentation-tooling implementation summary |
| `SCENE_UPDATE_OPTIMIZATION.md` | Historical report | Scene-update optimization design ideas |
| `SPECIFICATION_AUDIT_REPORT.md` | Historical report | Specification audit report |
| `SPECIFICATIONS_TEMPLATE_ASSESSMENT.md` | Historical report | SPECIFICATIONS_TEMPLATE.md assessment report |
| `TASK_2.1_COMPLETE_SUMMARY.md` | Historical report | Task 2.1 (JSDoc enhancement) completion summary |
| `TEMPLATE_STRUCTURE_MAPPING.md` | Historical report | SPECIFICATIONS template structure mapping reference |
| `TEMPLATE_USAGE_SUMMARY.md` | Historical report | SPECIFICATIONS template usage summary |

### `ai-status-reports/`

The `PERFORMANCE_OPTIMIZATION_*` files are a series of near-duplicate progress snapshots. The canonical **status** record is `PERFORMANCE_OPTIMIZATION_STATUS.md` (the latest snapshot, 2025-12-27); the other five perf reports are superseded snapshots of it. (The canonical **design** spec is `PERFORMANCE_OPTIMIZATION_SPEC.md` at the archive top level.)

| File | Category | Notes |
|------|----------|-------|
| `METAL_GRADIENT_BUG_FINAL_REPORT.md` | Historical report | Metal gradient bug investigation; bug unresolved at time of writing |
| `PERFORMANCE_OPTIMIZATION_ABSOLUTELY_FINAL.md` | Superseded | Superseded snapshot of `PERFORMANCE_OPTIMIZATION_STATUS.md` |
| `PERFORMANCE_OPTIMIZATION_ALL_PHASES_COMPLETE.md` | Superseded | Superseded snapshot of `PERFORMANCE_OPTIMIZATION_STATUS.md` |
| `PERFORMANCE_OPTIMIZATION_COMPLETE.md` | Superseded | Superseded snapshot of `PERFORMANCE_OPTIMIZATION_STATUS.md` |
| `PERFORMANCE_OPTIMIZATION_FINAL_COMPLETE.md` | Superseded | Superseded snapshot of `PERFORMANCE_OPTIMIZATION_STATUS.md` |
| `PERFORMANCE_OPTIMIZATION_FINAL_REVIEW_AND_STATUS.md` | Superseded | Superseded snapshot of `PERFORMANCE_OPTIMIZATION_STATUS.md` |
| `PERFORMANCE_OPTIMIZATION_STATUS.md` | Historical report | Canonical status record for the perf work (latest snapshot, 2025-12-27) |
| `PHASE_1_COMPLETE_ALL_TYPES_DEEP_INTEGRATION.md` | Historical report | Phase 1 deep-integration completion report |
| `PHASE_4_IMPLEMENTATION_PLAN.md` | Historical report | Phase 4 GPU buffer pool implementation plan (2025-12-24) |

### `code_reviews/`

See the **Findings note** below regarding the "STILL OPEN" statuses in the review reports.

| File | Category | Notes |
|------|----------|-------|
| `2026-03-01-000718-please-investigate-all-caching-mechanisms-in-the-v.txt` | Historical report | Raw review-session transcript on caching mechanisms |
| `cache-config-integration-review.md` | Historical report | Cache configuration/types/integration review |
| `cache-system-master-review.md` | Historical report | Cache system master review summary |
| `fitting_review.md` | Unresolved review | Fitting pipeline review; statuses not re-verified |
| `gpu-buffer-pool-review.md` | Historical report | GPU buffer pool review |
| `gsplats_core_review.md` | Unresolved review | GSplats core/IO/utils review; statuses not re-verified |
| `gsplats-docs-review.md` | Historical report | GSplats SPECIFICATIONS/README documentation review |
| `l0-cache-review.md` | Historical report | L0 decompressed-chunk cache review |
| `l2-opfs-cache-review.md` | Historical report | L2 OPFS persistent cache review |
| `lru-l1-cache-review.md` | Historical report | LRU / L1 segmented-LRU cache review |
| `master-summary.md` | Historical report | Documentation review master summary |
| `models_review.md` | Unresolved review | gsplats/models & rendering review; statuses not re-verified |
| `orchestration-prefetcher-review.md` | Historical report | Two-level caching store & chunk prefetcher review |
| `python-core-docs-review.md` | Historical report | Python core SPECIFICATIONS documentation review |
| `seeds_review.md` | Unresolved review | Seeds sub-package review; statuses not re-verified |
| `test-specs-review.md` | Historical report | Test SPECIFICATIONS.md critical review |
| `viewer-docs-review.md` | Historical report | Viewer documentation review |

### `developer-archive/`

| File | Category | Notes |
|------|----------|-------|
| `GSPLATS_VIEWER_IMPLEMENTATION.md` | Implemented | GSplats viewer support implementation plan that shipped |
| `Hierarchical Tensor-Gaussian Pursuit.md` | Historical report | Research note on a seeding/initialization method |
| `SPARKJS_ANALYSIS.md` | Historical report | External analysis of the SparkJS renderer |
| `SPLAT_MODEL_METAL.md` | Abandoned | Metal splat-model approach; not adopted |
| `THEMING_IMPLEMENTATION_PLAN.md` | Implemented | Viewer theming implementation plan that shipped |

### `gsplat-rendering-fix/`

| File | Category | Notes |
|------|----------|-------|
| `GSPLAT_RENDERING_ANALYSIS_AND_FIX.md` | Implemented | GSplat rendering analysis and fix that shipped |
| `diagnose_gsplat_rendering.py` | Historical report | Diagnostic helper script accompanying the analysis (not a doc) |
| `verify_integral_factor.py` | Historical report | Verification helper script accompanying the analysis (not a doc) |

### `implementation-notes/`

| File | Category | Notes |
|------|----------|-------|
| `DATA_ACCUMULATOR_STATUS.md` | Implemented | Data accumulator; active for all geometry types |
| `DYNAMIC_OPS_INTEGRATION.md` | Implemented | Dynamic operations integration that shipped |
| `UNIFIED_LOADER_ARCHITECTURE.md` | Implemented | Unified loader architecture that shipped |
| `WASM_ANALYSIS.md` | Historical report | WASM acceleration analysis; feature deleted 2026-07 |
| `WORKER_INFRASTRUCTURE_STATUS.md` | Historical report | Worker infrastructure status; feature deleted 2026-07 |

### `testing-reports/`

| File | Category | Notes |
|------|----------|-------|
| `PYTHON_TESTING_ANALYSIS.md` | Historical report | Python testing infrastructure analysis |
| `RUST_WASM_TESTING_ANALYSIS.md` | Historical report | Rust/WASM testing infrastructure analysis |
| `TESTING_INFRASTRUCTURE_REPORT.md` | Historical report | Comprehensive testing infrastructure analysis |
| `TYPESCRIPT_TESTING_ANALYSIS.md` | Historical report | TypeScript testing infrastructure analysis |

### `viewer-status/`

| File | Category | Notes |
|------|----------|-------|
| `DOCUMENTATION_REVIEW_2025-12-15.md` | Historical report | Viewer theming documentation review (2025-12-15) |
| `LIQUID_GLASS_THEME_FIX.md` | Historical report | Liquid-glass theme fix documentation |
| `OVERMOCKING_ISSUES.md` | Unresolved review | Over-mocking issues analysis; not re-verified |
| `RECOMMENDED-IMPROVEMENTS.md` | Unresolved review | Recommended viewer improvements; not re-verified |
| `SYNC-AUDIT-SYNTHESIS.md` | Historical report | Documentation synchronization audit synthesis |
| `THEMING_IMPLEMENTATION_COMPLETE.md` | Implemented | Viewer theming system implementation completion report |

### Findings note

The code-review reports under `code_reviews/` — notably `fitting_review.md`, `models_review.md`, `seeds_review.md`, and `gsplats_core_review.md` — carry per-finding statuses such as "STILL OPEN" that were last checked on the dates recorded in their headers. The unresolved-review notes under `viewer-status/` (`OVERMOCKING_ISSUES.md`, `RECOMMENDED-IMPROVEMENTS.md`) are prioritized issue/improvement lists in the same vein. None of these have been re-verified against current `main` and must not be treated as a live task list. Verify any finding against current code before acting on it. No GitHub issues are being filed from these reports here.
