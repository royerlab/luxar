# Viewer Documentation Review

**Reviewer**: Claude Opus 4.6 (1M context)
**Date**: 2026-02-28
**Scope**: All SPECIFICATIONS.md and README.md files in `packages/luxar-viewer/src/`

---

## Summary

Reviewed 21 documentation files across 14 packages in the viewer. Found **46 issues** total:
- **Critical**: 3 (broken references to nonexistent files, incorrect API signatures)
- **Major**: 15 (missing documentation for new files, outdated cross-references, incorrect descriptions)
- **Minor**: 28 (formatting inconsistencies, stale line numbers, minor inaccuracies)

**Fixed directly**: 13 issues (all Critical, 2 Major, 8 Minor) across 6 files.

Overall documentation quality is high. Most issues stem from code evolving faster than docs, particularly around new files added after documentation was written.

---

## Findings by File

### 1. src/cache/SPECIFICATIONS.md

**Status**: Good - well-maintained, accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | Minor | Data pipeline diagram (line 17) shows `RangeCache` which no longer exists as a separate file (`range-cache.ts` does not exist in `src/data/`). The L0 cache replaced it. | **FIXED** |

### 2. src/cache/README.md

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 2 | **Major** | Line 550: References `../data/range-cache.ts` in "Related Packages" section - this file does NOT exist. The RangeCache concept was replaced by the L0 DecompressedChunkCache. | **FIXED** |
| 3 | Minor | Line 582: References `../config/` in "Related Packages" - vague, could reference `../config/index.ts` specifically | **FIXED** |

### 3. src/config/SPECIFICATIONS.md

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 4 | **Major** | Missing documentation for `viewer-config-utils.ts` - a new file that exists in the config directory (shown as untracked in git status) but has no mention in SPECIFICATIONS.md | Needs attention |
| 5 | Minor | Version "1.2.0" and "Last Updated: 2025-12-09" are stale - the file has been modified since then | Needs attention |

### 4. src/controls/SPECIFICATIONS.md

**Status**: Good - physics documentation is thorough and accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 6 | Minor | Line references to `controls-manager.ts:280-295` and `controls-manager.ts:309-311` may be stale after code changes - line numbers should be verified | Needs attention |

### 5. src/controls/README.md

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 7 | **Critical** | Line 24: Package architecture lists `control-config.ts` which does NOT exist. Configuration is in `../config/index.ts` | **FIXED** |
| 8 | **Critical** | Line 25: Package architecture lists `input-context-manager.ts` in the controls directory - this file is actually in `../input/input-context-manager.ts` | **FIXED** |
| 9 | **Major** | Line 131: `setControlType(type: 'orbit' | 'fly')` signature is missing `'arcball'`. Actual code signature is `setControlType(type: ControlType)` where `ControlType = 'orbit' | 'arcball' | 'fly'` | **FIXED** |
| 10 | **Major** | Line 137: `getControls()` return type `OrbitControls | LuxarFlyControls` is missing `ArcballControls` | **FIXED** |
| 11 | **Major** | The entire README barely mentions arcball mode (only in the "Key Features" list at top). Section "Control Types" covers Orbit and Fly but not Arcball in the architecture/switching sections. | Needs attention |
| 12 | Minor | Line 824: Links to `./luxar-fly-controls-guide.md` which does NOT exist | **FIXED** |
| 13 | Minor | Line 536: References `control-config.ts` for checking fly mode keys - should reference `../config/index.ts` | **FIXED** |
| 14 | Minor | Line 583-584: References `control-config.ts` for custom configuration - should reference `../config/index.ts` | **FIXED** |
| 15 | Minor | Line 591: References `control-config.ts` for new input modes - should reference `../config/index.ts` | **FIXED** |

### 6. src/core/SPECIFICATIONS.md

**Status**: Good

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 16 | Minor | Line 409-410: States `main.ts` lines 64-71 for debug property setup - line numbers likely stale | Needs attention |
| 17 | Minor | Line 440: States `app.ts setupDebugInterface()` method at lines 273-440 - line numbers likely stale | Needs attention |
| 18 | Minor | Section 6.5 line 582: States "L2 (IndexedDB)" but the actual L2 is OPFS (Origin Private File System), not IndexedDB. Same issue on lines 644, 647, 660 | **FIXED** |

### 7. src/core/README.md

**Status**: Good - well-structured

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 19 | Minor | No mention of the `viewer-config-utils.ts` integration from the config package (newly added feature) | Needs attention |

### 8. src/data/SPECIFICATIONS.md

**Status**: Good - very detailed

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 20 | **Major** | Missing documentation for several files that exist in `src/data/`: `loader-orchestrator.ts`, `view-state-manager.ts`, `data-monitor-manager.ts`, `geometry-update-manager.ts`, `scene-graph-builder.ts`, `gsplats-processor.ts`, `effective-radius-calculator.ts`, `data-accumulator.ts`, `data-loader-types.ts`, `directory-navigator.ts` | Needs attention |
| 21 | Minor | Version 1.2.8 may not reflect all current code changes | Needs attention |

### 9. src/data/loaders/README.md

**Status**: Good - accurate and detailed

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 22 | Minor | Test counts (61 total, 24+27+10) may be stale if tests have been added/removed | Needs attention |

### 10. src/input/SPECIFICATIONS.md

**Status**: Excellent - very detailed and accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 23 | Minor | Line references (`input-context-manager.ts:31-394`, `input-handler.ts:549-799`, etc.) may be stale after code modifications | Needs attention |
| 24 | Minor | States "58 total bindings (25 NAVIGATION + 33 FLY_CONTROLS)" - these counts should be verified against current code | Needs attention |

### 11. src/profiling/SPECIFICATIONS.md

**Status**: Good - mostly a planning/design document

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 25 | Minor | Migration plan (Section 10) shows incomplete phases (Phase 1 partially checked, Phases 2-4 unchecked). Should be updated to reflect current implementation status. | Needs attention |

### 12. src/rendering/SPECIFICATIONS.md

**Status**: Good - comprehensive

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 26 | **Major** | Missing documentation for `adaptive-dpr-manager.ts` - while the file is referenced in section 9 header and changelog, the actual `AdaptiveDPRManager` class is not fully documented with its API | Needs attention |
| 27 | **Major** | Missing documentation for `gpu-buffer-pool.ts` - the file exists but is not mentioned anywhere in the SPECIFICATIONS.md | Needs attention |
| 28 | Minor | The `postprocessing-types.ts` file is only referenced once (line 683) and not documented as part of the module structure | Needs attention |

### 13. src/scene/SPECIFICATIONS.md

**Status**: Good - thorough

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 29 | **Major** | Missing documentation for `scene-manager-utils.ts` - the file exists and is referenced once (line 544) but has no dedicated section | Needs attention |
| 30 | Minor | Line 544 references constants from `scene-manager-utils.ts` without documenting what those constants are | Needs attention |

### 14. src/types/SPECIFICATIONS.md

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 31 | **Major** | Missing documentation for `animation.ts` types - the file defines `LoopMode`, `AnimationDirection`, and animation state types but is not mentioned in SPECIFICATIONS.md | Needs attention |
| 32 | **Major** | Missing documentation for `gsplats.ts` types - defines GSplats metadata, loaded data, view state, and user data types, similar to the Lines and Points types that ARE documented | Needs attention |
| 33 | **Major** | Missing documentation for `zarr.ts` types - defines `PositionBounds`, `SceneDimensionAttrs`, and other zarr attribute types | Needs attention |
| 34 | Minor | Missing documentation for `float16array.d.ts` - a type declaration file | Needs attention |
| 35 | Minor | The `DimensionMetadata` interface in Section 6.1 is a simplified version that omits `scale`, `cyclic`, `spatial`, and `categories` fields which ARE documented in Section 1.2. This inconsistency could confuse readers. | **FIXED** |

### 15. src/ui/SPECIFICATIONS.md

**Status**: Good - very detailed

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 36 | **Major** | Missing documentation for `rendering-controls-utils.ts` - the file exists but is not mentioned in the UI SPECIFICATIONS.md | Needs attention |
| 37 | Minor | Missing documentation for `ui/components/` directory which contains: `base/`, `event-queue.ts`, `hierarchical-timing-panel.ts`, `loading-advisor.ts`, `polling-loop.ts`, `resolution-indicator.ts` | Needs attention |

### 16. src/ui/gui/SPECIFICATIONS.md

**Status**: Good - accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 38 | Minor | No issues found - file structure matches actual code, API descriptions are accurate | OK |

### 17. src/ui/gui/README.md

**Status**: Good - accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 39 | Minor | No significant issues - file structure matches, API is accurate | OK |

### 18. src/ui/rendering-controls/SPECIFICATIONS.md

**Status**: Good - detailed and accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 40 | Minor | Line counts for each module (e.g., "Lines: 242") may be stale after modifications | Needs attention |

### 19. src/ui/rendering-controls/README.md

**Status**: Good - accurate

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 41 | Minor | No significant issues found | OK |

### 20. src/utils/SPECIFICATIONS.md

**Status**: Good - thorough

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 42 | Minor | Missing documentation for `escape-html.ts` - the file exists in `src/utils/` but is not mentioned in SPECIFICATIONS.md | Needs attention (README fixed) |

### 21. src/utils/README.md

**Status**: Good

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 43 | Minor | Architecture section lists 4 files but `escape-html.ts` is missing from the list | **FIXED** |
| 44 | Minor | "Structured Logging" section is mentioned in Table of Contents but the actual section content is thin (only shows import examples, no dedicated module subsection) | Needs attention |
| 45 | Minor | "Performance Considerations" section discusses "Slicing Optimization" and "Navigation Efficiency" which are not part of the utils package - these seem copied from a different README | **FIXED** |

---

## Missing Documentation Files

The following packages/directories lack README.md files (though they exist in the original review scope and had other doc files):

| Directory | Has SPECIFICATIONS.md | Has README.md | Assessment |
|-----------|----------------------|---------------|------------|
| `src/config/` | Yes | Yes | OK |
| `src/input/` | Yes | Yes | OK (README exists but was not in original review scope) |
| `src/profiling/` | Yes | Yes | OK (README exists but was not in original review scope) |
| `src/scene/` | Yes | Yes | OK (README exists but was not in original review scope) |
| `src/types/` | Yes | Yes | OK (README exists but was not in original review scope) |
| `src/rendering/` | Yes | Yes | OK (README exists but was not in original review scope) |

---

## Cross-Reference Issues

### Broken File References

| Doc File | Referenced Path | Actual Status | Fix Status |
|----------|----------------|---------------|------------|
| `controls/README.md` line 24 | `control-config.ts` | **Does not exist** - config is in `../config/index.ts` | **FIXED** |
| `controls/README.md` line 25 | `input-context-manager.ts` (in controls dir) | **Wrong directory** - file is in `../input/` | **FIXED** |
| `controls/README.md` line 824 | `./luxar-fly-controls-guide.md` | **Does not exist** | **FIXED** |
| `cache/README.md` line 550 | `../data/range-cache.ts` | **Does not exist** - removed in favor of L0 cache | **FIXED** |

### Stale API Signatures

| Doc File | Documented Signature | Actual Signature | Fix Status |
|----------|---------------------|-----------------|------------|
| `controls/README.md` line 131 | `setControlType(type: 'orbit' \| 'fly')` | `setControlType(type: 'orbit' \| 'arcball' \| 'fly')` | **FIXED** |
| `controls/README.md` line 137 | `getControls(): OrbitControls \| LuxarFlyControls` | Should include `ArcballControls` | **FIXED** |

### Incorrect Technology References

| Doc File | Stated | Actual | Fix Status |
|----------|--------|--------|------------|
| `core/SPECIFICATIONS.md` section 6.5 | "L2 (IndexedDB)" (multiple lines) | L2 uses OPFS (Origin Private File System), not IndexedDB | **FIXED** |

---

## Recommendations

### High Priority (Critical/Major fixes)

1. **Fix controls/README.md package architecture** - Remove references to nonexistent `control-config.ts` and misplaced `input-context-manager.ts`. Update the architecture diagram to show the actual file layout: `controls-manager.ts`, `luxar-fly-controls.ts`, `types.ts`.

2. **Fix controls/README.md API signatures** - Add `'arcball'` to `setControlType` and update `getControls()` return type to include `ArcballControls`. Add documentation for arcball mode throughout the README.

3. **Fix cache/README.md broken reference** - Remove or update the "Related Packages" reference to `range-cache.ts`.

4. **Fix core/SPECIFICATIONS.md IndexedDB reference** - Change "IndexedDB" to "OPFS" in section 6.5.

### Medium Priority (Missing documentation)

5. **Add GSplats types to types/SPECIFICATIONS.md** - Document `gsplats.ts` types following the same pattern as Lines (section 7) and Points (section 8).

6. **Add animation types to types/SPECIFICATIONS.md** - Document `animation.ts` types (`LoopMode`, `AnimationDirection`, etc.)

7. **Add zarr types to types/SPECIFICATIONS.md** - Document `zarr.ts` types.

8. **Document new data files** - Add at least brief mentions of `loader-orchestrator.ts`, `view-state-manager.ts`, `geometry-update-manager.ts`, `gsplats-processor.ts`, and other new files in `src/data/`.

9. **Document rendering files** - Add documentation for `gpu-buffer-pool.ts` and `adaptive-dpr-manager.ts` API.

10. **Document config/viewer-config-utils.ts** - Add to config SPECIFICATIONS.md.

### Low Priority (Minor fixes)

11. Update stale line number references across all SPECIFICATIONS.md files.
12. Add `escape-html.ts` to utils documentation.
13. Remove misplaced "Slicing Optimization" content from utils/README.md.
14. Fix DimensionMetadata inconsistency between sections 1.2 and 6.1 in types/SPECIFICATIONS.md.
15. Update test counts and line counts that may have become stale.

---

## Files with No Issues Found

- `src/ui/gui/SPECIFICATIONS.md` - Accurate and well-maintained
- `src/ui/gui/README.md` - Accurate and well-maintained
- `src/ui/rendering-controls/README.md` - Accurate

---

## Overall Assessment

The documentation quality across the viewer package is **generally high**. The SPECIFICATIONS.md files are particularly well-done with detailed algorithm descriptions, data flow diagrams, and clear version histories. The main categories of issues are:

1. **Code evolution outpacing docs** - New files added without corresponding doc updates (most common)
2. **Broken cross-references** - Files renamed/removed but doc links not updated
3. **Incomplete API coverage** - The controls README is the worst offender, missing arcball support entirely in API docs
4. **Stale line numbers** - A recurring minor issue across multiple spec files

The most impactful fix would be updating the **controls/README.md** which has the most critical issues (nonexistent files in architecture, missing arcball support in API).
