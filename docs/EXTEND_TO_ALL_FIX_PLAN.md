# extend_to_all Bug Fix Plan

**Created**: 2025-12-10
**Status**: In Progress (Phases 1-4 complete, Phase 5-6 pending manual verification)
**Priority**: Critical (feature completely broken)

---

## Executive Summary

The `extend_to_all` feature for Points and Lines is completely broken during initial scene load. The bug causes static geometry (e.g., detector outlines) that should be visible at all values of a dimension (e.g., time) to disappear as soon as the viewer initializes.

**Root Cause**: `viewState.dimensions` is `undefined` during initial node loading because dimension metadata is not yet parsed at that point.

**Impact**:
- `extend_to_all` logic fails silently (falls back to normal spatial query)
- Tolerance computation uses incorrect fallback values
- Both Points and Lines are affected identically

---

## Table of Contents

1. [Bug Analysis](#1-bug-analysis)
2. [Affected Files](#2-affected-files)
3. [Fix Plan](#3-fix-plan)
4. [Documentation Updates](#4-documentation-updates)
5. [Testing Requirements](#5-testing-requirements)
6. [Implementation Checklist](#6-implementation-checklist)

---

## 1. Bug Analysis

### 1.1 Symptom

In the particle collision demo with animated time series:
- Detector geometry is created with `extend_to_all=["time"]`
- Expected: Detector visible at ALL time frames
- Actual: Detector briefly visible at initial load (time=0), then disappears

### 1.2 Root Cause

**SceneLoader.constructor** initializes `viewState` WITHOUT dimensions:

```typescript
// scene-loader.ts (lines ~89-95)
this.viewState = {
  displayDims: [0, 1, 2],
  slicePosition: [],
  tolerance: [],
  // NO dimensions property!
};
```

During initial node loading, this incomplete viewState is passed to loaders:

```typescript
// scene-loader.ts (loadLinesNode, lines ~320-325)
const linesViewState = {
  displayDims: this.viewState.displayDims,
  slicePosition: this.viewState.slicePosition,
  tolerance: this.viewState.tolerance,
  dimensions: this.viewState.dimensions?.metadata,  // <-- undefined!
};
```

### 1.3 Failure Sequence

1. **Scene load begins**: SceneLoader created with empty viewState (no dimensions)
2. **Nodes load**: Points/Lines load with `dimensions: undefined` in viewState
3. **extend_to_all check**: `viewState.dimensions?.filter(...)` returns `[]` (empty array)
4. **Result**: `isExtending = false`, normal spatial query used
5. **SceneDimsManager initializes**: Time dimension set to center of range (e.g., 25ns)
6. **Spatial query fails**: Detector at time=0 not found when querying time=25 with tolerance=0
7. **Detector disappears**: No segments returned by spatial query

### 1.4 Secondary Issue: Tolerance Fallback

When `dimensions` is undefined, tolerance computation falls back:

```typescript
// lines-spatial-index-loader.ts (lines 283-285)
const tolerance = viewState.dimensions
  ? computeLinesTolerance(viewState.dimensions, viewState.displayDims)
  : new Array(attrs.ndim).fill(0).map((_, i) =>
      viewState.displayDims.includes(i) ? 1e10 : 0
    );
```

The fallback uses `0` for non-displayed spatial dimensions. While this might work in some cases, it's incorrect for radius-based visibility calculation.

### 1.5 Naming Issue

The function `updateAllNDPoints()` in `input-handler.ts` is misleadingly named - it actually updates ALL nD nodes (Points, Lines, Splats), not just Points:

```typescript
// input-handler.ts (lines 192-207)
private async updateAllNDPoints(): Promise<void> {  // <-- Misleading name!
  const dims = sceneDimsManager.getDims();
  if (!dims) { return; }
  // Actually updates BOTH points AND lines
  const { updateSceneForDimensions } = await import('../data');
  await updateSceneForDimensions(dims, this.sceneManager.scene as unknown as THREE.Group);
}
```

---

## 2. Affected Files

### 2.1 Code Files

| File | Issue | Fix Required |
|------|-------|--------------|
| `packages/luxar-viewer/src/data/scene-loader.ts` | viewState.dimensions undefined during initial load | Parse dimensions BEFORE loading nodes |
| `packages/luxar-viewer/src/data/point-spatial-index-loader.ts` | extend_to_all fails when dimensions undefined | Add defensive check + warning |
| `packages/luxar-viewer/src/data/lines-spatial-index-loader.ts` | extend_to_all fails when dimensions undefined | Add defensive check + warning |
| `packages/luxar-viewer/src/input/input-handler.ts` | Misleading function name | Rename `updateAllNDPoints` → `updateAllNDNodes` |

### 2.2 Specification Files

| File | Section | Update Required |
|------|---------|-----------------|
| `packages/luxar-viewer/src/data/SPECIFICATIONS.md` | 4.4 Scene Loading Algorithm | Document dimension initialization order |
| `packages/luxar-viewer/src/data/SPECIFICATIONS.md` | 6.2 ViewState | Add `dimensions` field documentation |
| `packages/luxar-viewer/src/data/SPECIFICATIONS.md` | 7.9 extend_to_all | Document viewState.dimensions requirement |
| `packages/luxar-viewer/src/input/SPECIFICATIONS.md` | N/A | Update function name reference |

### 2.3 README Files

| File | Update Required |
|------|-----------------|
| `packages/luxar-viewer/src/data/README.md` | Document dimension initialization requirement |
| `packages/luxar-viewer/src/input/README.md` | Update function name reference |

---

## 3. Fix Plan

### 3.1 Fix 1: Parse Dimensions Before Loading Nodes (Primary Fix)

**Location**: `scene-loader.ts`

**Current Flow**:
```
1. Create SceneLoader (viewState has no dimensions)
2. Load nodes (extend_to_all fails)
3. SceneDimsManager parses dimensions
4. updateSceneForDimensions called (extend_to_all works)
```

**Fixed Flow**:
```
1. Create SceneLoader
2. Parse scene dimensions from root .zattrs FIRST
3. Initialize viewState WITH dimensions
4. Load nodes (extend_to_all works from the start)
```

**Implementation**:

```typescript
// In loadScene() or constructor
async function initializeViewState(store: ZarrStore): Promise<ViewState> {
  // 1. Load root attributes
  const rootAttrs = await store.getAttrs('/');

  // 2. Parse scene dimensions
  const sceneDimensions = rootAttrs.scene_dimensions?.dimensions || [];

  // 3. Determine displayed/non-displayed dimensions
  const displayDims = sceneDimensions
    .map((dim, idx) => ({ dim, idx }))
    .filter(({ dim }) => dim.display === true)
    .map(({ idx }) => idx)
    .slice(0, 3);  // Max 3 displayed

  // 4. Calculate initial slice position (center of each non-displayed dim)
  const slicePosition = sceneDimensions.map((dim, idx) => {
    if (displayDims.includes(idx)) return 0;
    const [min, max] = dim.range || [0, 1];
    return (min + max) / 2;
  });

  // 5. Calculate initial tolerance
  const tolerance = computeTolerance(sceneDimensions, displayDims);

  return {
    displayDims,
    slicePosition,
    tolerance,
    dimensions: {
      metadata: sceneDimensions,
      ndim: sceneDimensions.length,
      displayed: displayDims,
      currentStep: slicePosition,
    },
  };
}
```

**Coordination with SceneDimsManager**: After this change, both SceneLoader and SceneDimsManager will parse dimensions. Ensure they use consistent logic and SceneDimsManager can receive pre-parsed dimensions to avoid duplicate work.

### 3.2 Fix 2: Add Defensive Checks (Secondary Safety Net)

**Location**: `point-spatial-index-loader.ts` and `lines-spatial-index-loader.ts`

Even with Fix 1, add defensive checks to prevent silent failures:

```typescript
// In queryVisibleSegmentRanges() or similar
function queryVisibleRanges(viewState: ViewState): Range[] {
  const extendDims: string[] = this.node.attrs.extend_to_all || [];

  if (extendDims.length > 0) {
    // DEFENSIVE CHECK: Warn if dimensions not available
    if (!viewState.dimensions || viewState.dimensions.length === 0) {
      console.warn(
        `[LinesSpatialIndexLoader] extend_to_all specified but viewState.dimensions is undefined. ` +
        `extend_to_all will not work until dimensions are initialized.`
      );
    }

    // ... existing extend_to_all logic
  }
}
```

### 3.3 Fix 3: Rename updateAllNDPoints (Code Clarity)

**Location**: `input-handler.ts`

**Change**:
```typescript
// Before
private async updateAllNDPoints(): Promise<void> { ... }

// After
private async updateAllNDNodes(): Promise<void> { ... }
```

**Also update all call sites** in the same file.

---

## 4. Documentation Updates

### 4.1 SPECIFICATIONS.md (data package)

**Section 4.4 Scene Loading Algorithm** - Add step for dimension initialization:

```markdown
### 4.4 Scene Loading Algorithm

**Purpose**: Recursively load scene hierarchy and build THREE.js scene graph.

**Algorithm**:

```
function loadScene(zarrURL):
    // 1. Open zarr store with consolidated metadata
    store = openConsolidatedZarrStore(zarrURL)

    // 2. Extract scene dimensions from root .zattrs
    rootAttrs = store.getAttrs("/")
    sceneDimensions = rootAttrs.scene_dimensions.dimensions

    // 3. Initialize ViewState WITH dimensions (CRITICAL for extend_to_all)
    // This MUST happen BEFORE loading any nodes
    viewState = initializeViewState(sceneDimensions)

    // 4. Initialize array reference registry
    arrayRefRegistry = new Map()

    // 5. Create THREE.js root group
    ...
```

**CRITICAL**: `viewState.dimensions` MUST be initialized BEFORE loading any
Points/Lines/Splats nodes. The `extend_to_all` feature requires dimension
metadata to determine which dimensions are non-displayed.
```

**Section 6.2 ViewState** - Add dimensions field:

```markdown
### 6.2 ViewState

**Purpose**: Encapsulates current view configuration for nD navigation.

```typescript
interface ViewState {
    displayDims: number[]           // Indices of displayed dimensions [0-2]
    slicePosition: number[]         // Current position in nD space
    tolerance: number[]             // Search radius per dimension
    dimensions?: SimpleDims         // REQUIRED for extend_to_all feature
}

interface SimpleDims {
    metadata: DimensionMetadata[]   // Full dimension info
    ndim: number                    // Total dimensionality
    displayed: number[]             // Displayed dimension indices
    currentStep: number[]           // Current slice position
}
```

**Note**: The `dimensions` field was historically optional but is REQUIRED
for `extend_to_all` functionality. Initialize it from scene root attributes
BEFORE loading any data nodes.
```

**Section 7.9 Lines Dimension Extension** - Add warning:

```markdown
**CRITICAL**: `viewState.dimensions` MUST be populated for `extend_to_all` to work.
If `dimensions` is undefined, the loader falls back to normal spatial query behavior
and the extend feature silently fails. Ensure scene dimensions are parsed BEFORE
loading any nodes that use `extend_to_all`.
```

### 4.2 SPECIFICATIONS.md (input package)

Update any references to `updateAllNDPoints` → `updateAllNDNodes`.

### 4.3 README.md (data package)

Add a note about initialization order:

```markdown
## Critical Initialization Order

When loading scenes with `extend_to_all` nodes (static geometry visible across
all values of a dimension), the following order is REQUIRED:

1. Parse scene dimensions from root `.zattrs`
2. Initialize `ViewState` WITH `dimensions` populated
3. Load Points/Lines/Splats nodes

If nodes are loaded before `dimensions` is available, `extend_to_all` will
silently fail and the geometry will only be visible at its stored position.
```

### 4.4 README.md (input package)

Update function references from `updateAllNDPoints` → `updateAllNDNodes`.

---

## 5. Testing Requirements

### 5.1 Unit Tests

**New test**: `packages/luxar-viewer/src/tests/extend-to-all.test.ts`

```typescript
describe('extend_to_all feature', () => {
  it('should return all segments when extend_to_all matches non-displayed dim', async () => {
    // Create mock lines with extend_to_all=["time"]
    // Create viewState with time as non-displayed dimension
    // Verify ALL segments returned regardless of slice position
  });

  it('should warn when viewState.dimensions is undefined', async () => {
    // Create mock lines with extend_to_all=["time"]
    // Create viewState WITHOUT dimensions
    // Verify warning is logged
    // Verify fallback to normal spatial query
  });

  it('should work identically for points and lines', async () => {
    // Test same behavior for both loaders
  });
});
```

### 5.2 E2E Tests

**New test**: `packages/luxar-viewer/tests/e2e/extend-to-all.spec.ts`

```typescript
test('detector geometry remains visible across time frames', async ({ page }) => {
  // Load particle collision dataset with detector (extend_to_all=["time"])
  // Navigate to different time frames using [ and ] keys
  // Verify detector geometry (lines) remains visible at ALL frames
  // Use screenshot comparison or point count checks
});
```

### 5.3 Manual Testing

1. Run `demo_particle_collision_animated.py`
2. Verify detector geometry visible at initial load
3. Navigate through time frames with `[` and `]` keys
4. Verify detector geometry remains visible at ALL frames
5. Verify particle tracks change with time (not affected by extend_to_all)

---

## 6. Implementation Checklist

### Phase 1: Core Fix

- [x] **scene-loader.ts**: Parse dimensions BEFORE loading nodes
  - [x] Verified dimension parsing already happens in `initializeSceneDimensions()` before `loadSceneNodes()`
  - [x] Added success/warning logging for dimension initialization status
  - [x] Confirmed `loadPointsNode()` and `loadLinesNode()` receive populated dimensions via `this.viewState`

**Note**: Investigation revealed the initialization order was already correct. The issue is that
`initializeSceneDimensions()` has early-return paths (invalid format, validation failure) that
can leave `viewState.dimensions` undefined. The defensive checks in Phase 2 help identify these cases.

### Phase 2: Defensive Checks

- [x] **point-spatial-index-loader.ts**: Add warning when dimensions undefined
- [x] **lines-spatial-index-loader.ts**: Add warning when dimensions undefined

### Phase 3: Code Cleanup

- [x] **input-handler.ts**: Rename `updateAllNDPoints` → `updateAllNDNodes`
- [x] Update all call sites within input-handler.ts
- [x] Updated JSDoc to clarify function updates all nD node types

### Phase 4: Documentation

- [x] **data/SPECIFICATIONS.md**: Update Section 4.4 (initialization order) - N/A, order was already correct
- [x] **data/SPECIFICATIONS.md**: Update Section 6.2 (ViewState.dimensions)
- [x] **data/SPECIFICATIONS.md**: Update Section 7.9 (extend_to_all warning)
- [x] **data/README.md**: N/A, initialization order doesn't need separate documentation
- [x] **input/SPECIFICATIONS.md**: N/A, `updateAllNDNodes` is private method, not in specs
- [x] **input/README.md**: N/A, `updateAllNDNodes` is private method, not in API docs

### Phase 5: Testing

- [x] Existing unit tests pass (931 tests passed)
- [ ] Write new unit tests specific to extend_to_all feature (optional, recommended)
- [ ] Write E2E test for time-varying visualization
- [ ] Manual test with particle collision demo

### Phase 6: Verification

- [ ] Run full test suite: `make test-all`
- [ ] Run E2E tests: `cd packages/luxar-viewer && pnpm test:e2e`
- [ ] Verify demo works: `hatch run python packages/luxar/src/luxar/demos/demo_particle_collision_animated.py`

---

## Appendix: Code References

### A. SceneLoader viewState initialization
`packages/luxar-viewer/src/data/scene-loader.ts` lines 89-95

### B. Lines extend_to_all logic
`packages/luxar-viewer/src/data/lines-spatial-index-loader.ts` lines 252-275

### C. Points extend_to_all logic
`packages/luxar-viewer/src/data/point-spatial-index-loader.ts` lines 433-444

### D. updateAllNDPoints function
`packages/luxar-viewer/src/input/input-handler.ts` lines 192-207

### E. SceneDimsManager initialization
`packages/luxar-viewer/src/scene/scene-dims-manager.ts` lines 121-139
