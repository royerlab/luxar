> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# Luxar Documentation Index

**Last Updated**: 2025-12-13

This index provides a quick reference to all active documentation in the Luxar project.

---

## Quick Navigation

| I want to... | Read this... |
|-------------|--------------|
| Understand the Luxar zarr format | `docs/LUXAR_ZARR_FORMAT.md` |
| Use HDR colors | `docs/user-guides/HDR_GUIDE.md` |
| Write E2E tests | `packages/luxar-viewer/docs/PLAYWRIGHT_GUIDE.md` |
| Understand console logging style | `docs/CONSOLE_OUTPUT_STYLE.md` |
| Use the debug console | `docs/DEBUG_CONSOLE_GUIDE.md` |
| Run tests quickly | `docs/E2E_TESTING_GUIDE.md` |
| Simulate network conditions | `docs/NETWORK_SIMULATION_SPEC.md` |
| Understand cache prefetching | `docs/CACHE_PREFETCHING_SPEC.md` |
| Create new SPECIFICATIONS.md | `docs/templates/SPECIFICATIONS_TEMPLATE.md` |

---

## Documentation by Category

### Format & Specifications

**LUXAR_ZARR_FORMAT.md**
- Purpose: Overview of the Luxar zarr format
- Audience: Developers implementing readers/writers
- Cross-refs: Links to package SPECIFICATIONS.md for details

**NETWORK_SIMULATION_SPEC.md**
- Purpose: Network condition simulation specification
- Status: Fully implemented in luxar CLI
- Details: ASGI middleware for bandwidth/latency/jitter/packet-loss simulation

**CACHE_PREFETCHING_SPEC.md**
- Purpose: Intelligent chunk prefetching specification
- Status: Fully implemented in viewer cache system
- Details: Algorithm for proactive adjacent chunk loading

---

### Developer Guides

**E2E_TESTING_GUIDE.md** (Quick Reference)
- Purpose: Commands and test organization
- Use when: Running tests, checking coverage
- Complements: PLAYWRIGHT_GUIDE.md

**PLAYWRIGHT_GUIDE.md** (Comprehensive)
- Location: `packages/luxar-viewer/docs/`
- Purpose: Complete Playwright testing guide
- Audience: Developers AND AI agents (Claude Code)
- Features: AI debugging workflow, best practices, troubleshooting

**DEBUG_CONSOLE_GUIDE.md**
- Purpose: Using the in-app debug console (Ctrl+L)
- Features: Console filtering, copying, inspection
- Related: `packages/luxar-viewer/src/ui/SPECIFICATIONS.md` (Section 8)

**CONSOLE_OUTPUT_STYLE.md**
- Purpose: Logging style guide for TypeScript viewer
- Format: `[emoji] [Module] message`
- Related: `packages/luxar-viewer/src/utils/SPECIFICATIONS.md` (Section 4)

---

### User Guides

**user-guides/HDR_GUIDE.md** (NEW)
- Purpose: Complete guide to using HDR colors in Luxar
- Topics: Color ranges, Python API, browser setup, troubleshooting
- Consolidates: 3 previous HDR documents
- For: Users creating HDR visualizations

---

### Templates & Standards

**templates/SPECIFICATIONS_TEMPLATE.md**
- Purpose: Template for creating SPECIFICATIONS.md files
- Usage: Referenced in CLAUDE.md (required for all packages)
- Status: Actively used by all 28 packages
- Compliance: 82-95% adherence across codebase

---

### Planning & Improvement Tracking

**NEXT_STEPS.md**
- Purpose: E2E test suite completion tracking
- Status: Living document (update as work progresses)

**packages/luxar-viewer/docs/RECOMMENDED-IMPROVEMENTS.md**
- Purpose: Viewer improvement backlog
- Status: Active tracking of enhancement opportunities

**packages/luxar-viewer/docs/SYNC-AUDIT-SYNTHESIS.md**
- Purpose: Audit methodology and gold standard examples
- Use: Reference for future documentation audits

---

### Package-Specific Documentation

**gsplats/docs/DYNAMIC_OPS_INTEGRATION.md**
- Purpose: Dynamic operations configuration guide
- Audience: Users of gsplats fitting
- Status: Usage documentation for implemented feature

---

## Package SPECIFICATIONS.md Files (28 total)

Every package has comprehensive technical specifications. See individual packages:

**Python Packages** (13):
- `luxar/core/SPECIFICATIONS.md`
- `luxar/encoding/SPECIFICATIONS.md`
- `luxar/io/SPECIFICATIONS.md`
- `luxar/validation/SPECIFICATIONS.md`
- `luxar/typing_utils/SPECIFICATIONS.md`
- `luxar/utils/SPECIFICATIONS.md`
- `luxar/cli/SPECIFICATIONS.md`
- `luxar/gsplats/SPECIFICATIONS.md`
- `luxar/gsplats/io/SPECIFICATIONS.md`
- `luxar/gsplats/models/SPECIFICATIONS.md`
- `luxar/gsplats/fitting/SPECIFICATIONS.md`
- `luxar/gsplats/optim/SPECIFICATIONS.md`
- `luxar/gsplats/utils/SPECIFICATIONS.md`

**TypeScript Packages** (15):
- `luxar-viewer/src/core/SPECIFICATIONS.md`
- `luxar-viewer/src/scene/SPECIFICATIONS.md`
- `luxar-viewer/src/data/SPECIFICATIONS.md`
- `luxar-viewer/src/rendering/SPECIFICATIONS.md`
- `luxar-viewer/src/ui/SPECIFICATIONS.md`
- `luxar-viewer/src/input/SPECIFICATIONS.md`
- `luxar-viewer/src/config/SPECIFICATIONS.md`
- `luxar-viewer/src/types/SPECIFICATIONS.md`
- `luxar-viewer/src/utils/SPECIFICATIONS.md`
- `luxar-viewer/src/cache/SPECIFICATIONS.md`
- `luxar-viewer/src/controls/SPECIFICATIONS.md`
- `luxar-viewer/src/tests/SPECIFICATIONS.md`
- (Plus 3 gsplats subpackages: clahe, multiscale, seeds)

---

## Documentation Principles

Based on cleanup learnings:

### KEEP Documentation If:
- ✅ Actively guides current development
- ✅ Teaches timeless patterns/best practices
- ✅ References implemented features (with accurate status)
- ✅ Provides unique value not in other docs

### REMOVE Documentation If:
- ❌ Describes completed work (use git history)
- ❌ Duplicates content in canonical sources
- ❌ Contains outdated metrics or examples
- ❌ Tracks issues now resolved

### INTEGRATE Documentation If:
- 🔄 Contains valuable insights scattered across files
- 🔄 Better served living near the code (package docs)
- 🔄 Complements existing specs with missing details

---

## Maintenance Guidelines

**For New Documents:**
1. Ask: "Is this guidance (KEEP) or a work log (DELETE after completion)?"
2. Check: "Does this duplicate existing docs?"
3. Consider: "Where will developers look for this info?"
4. Plan: "How will this be kept current?"

**For Existing Documents:**
1. Review quarterly for relevance
2. Update or delete outdated information
3. Consolidate overlapping content
4. Move historical artifacts to git history

**For Bug Fixes/Features:**
1. Extract architectural insights → specs
2. Document patterns → CLAUDE.md or package docs
3. Delete the bug report/plan after extraction
4. Let git history preserve detailed timeline

---

## Current State Summary

The Luxar project now has:
- ✅ **Clean documentation structure** (9 + 3 + 1 standalone docs)
- ✅ **Single source of truth** for each topic
- ✅ **Active maintenance focus** (historical artifacts removed)
- ✅ **Clear hierarchy** (Specs → Guides → User Guides → Templates)
- ✅ **No broken references** (all cross-links verified)
- ✅ **Reduced maintenance burden** (60% fewer files to maintain)

**Ready for ongoing development with sustainable documentation practices** ✅
