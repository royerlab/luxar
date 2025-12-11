# Implementation Document: Dimension Initialization Fix

**Date**: 2025-12-11
**Issue**: Slider position mismatch on viewer startup
**Status**: Ready for implementation
**Priority**: High (UX bug affecting all nD datasets)

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Current Behavior](#current-behavior)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
6. [Testing Strategy](#testing-strategy)
7. [Risk Assessment](#risk-assessment)
8. [Rollout Plan](#rollout-plan)

---

## Problem Statement

### User-Reported Issue
When the Luxar viewer starts with an nD dataset, the dimension slider UI shows one position (e.g., at the "0" or minimum position), but the displayed data corresponds to a different slice position (e.g., the center of the range).

### Impact
- **User Experience**: Confusing and inconsistent - what you see doesn't match what the UI indicates
- **Discoverability**: Users may not realize they're looking at the middle of a time series, not the beginning
- **Navigation**: First interaction with slider causes unexpected jump in data
- **Scope**: Affects all nD datasets (time-series, multi-channel, higher-dimensional spatial data)

---

## Root Cause Analysis

### Architecture Overview

The viewer uses a centralized dimension management system with three key components:

```
┌──────────────────────────┐
│  SceneDimsManager        │  Singleton state manager
│  (dimension state)       │  - Tracks current slice positions
└────────┬─────────────────┘  - Manages dimension metadata
         │ notifyListeners()
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌──────────────────────┐         ┌─────────────────┐
│  DimensionSliders    │         │  Data Loaders   │
│  (UI components)     │         │  (Points/Lines) │
└──────────────────────┘         └─────────────────┘
```

### Timeline of Bug

**Initialization Sequence** (`packages/luxar-viewer/src/input/input-handler.ts:148-190`):

1. **Scene data loads** → Data loaders initialize (likely at default position [0,0,0,0])
2. **`sceneDimsManager.initFromScene()` called** → Initializes dimension state
3. **Dimension state created** → `currentStep[]` array populated
4. **`DimensionSliders` constructor called** → Reads `currentStep[]` values
5. **Sliders display** → Show position based on `currentStep[]`

**The Bug** (`packages/luxar-viewer/src/scene/scene-dims-manager.ts:157`):

```typescript
// After setting up this.dims with currentStep values...
this.dims = {
  ndim,
  currentStep,  // Contains center values
  displayed,
  metadata,
};

// ❌ MISSING: this.notifyListeners();
return true;
```

**Result**:
- Sliders read `currentStep[]` and display center position ✓
- But data loaders never get notified of the new position ✗
- Data remains at initial position (0) while UI shows center

### Supporting Evidence

**File**: `packages/luxar-viewer/src/scene/scene-dims-manager.ts`
- **Line 65-158**: `initFromScene()` method sets `currentStep[]` but never notifies
- **Line 221**: Only place `notifyListeners()` is called (in `setDimensionValue()`)
- **Line 233-235**: Listeners registered for updates (slider update, data re-slice)

**File**: `packages/luxar-viewer/src/ui/dimension-sliders.ts`
- **Line 294-302**: Slider correctly reads `dims.currentStep[dimIndex]` on construction
- **Line 463-485**: `update()` method called by listener to sync visuals

**File**: `packages/luxar-viewer/src/input/input-handler.ts`
- **Line 182-189**: Listener registered AFTER slider construction
- Expects `notifyListeners()` to trigger initial data load at correct position

---

## Current Behavior

### Initialization Logic (scene-dims-manager.ts:119-139)

```typescript
// Step 6: Initialize dimension positions
const currentStep = new Array(ndim).fill(0);

for (let i = 0; i < ndim; i++) {
  if (metadata[i].display !== true) {
    // Non-displayed dimensions start at center of range
    const [min, max] = this.dimensionRanges[i];
    let centerValue = (min + max) / 2;

    // For discrete dimensions, floor to nearest integer
    if (metadata[i].discrete) {
      centerValue = Math.floor(centerValue);
    }

    currentStep[i] = centerValue;  // ← ALL non-displayed dims → center
  }
  // Displayed dimensions stay at 0
}
```

**Current Policy**: All non-displayed dimensions initialize to **center** of range.

**Rationale** (from comment in code):
> "Non-displayed dimensions start at CENTER of range to maximize chance of visible data (starting at min could put us outside the actual data bounds)"

### Problems with Current Approach

1. **Wrong for time-series**: Time dimension (t=0 to t=50ns) starts at t=25ns instead of t=0
2. **Wrong for channels**: Channel dimension (0-4) starts at channel 2 instead of channel 0
3. **Wrong for categorical**: Category dimension starts in middle category
4. **Missing notification**: Even when logic is correct, loaders never get initial position

---

## Proposed Solution

### Two-Part Fix

#### Part 1: Add Missing Notification
Add `this.notifyListeners()` call after dimension initialization to trigger:
- Initial data load at correct position
- Slider visual update
- Scene render with correct data

#### Part 2: Smarter Initialization Policy
Distinguish between dimension types:

| Dimension Type | Initialization | Rationale |
|----------------|----------------|-----------|
| **Discrete** (time, frames) | **Minimum** (first position) | Start at t=0, frame 0, beginning of sequence |
| **Categorical** (channels, types) | **Minimum** (first category) | Start at channel 0, first category |
| **Continuous Spatial** (4th+ spatial dim) | **Center** | Spatial dimensions have no natural "first" |
| **Displayed** (X, Y, Z) | **0** (unchanged) | Camera controls these |

### Design Rationale

**Time-Series Data** (most common use case):
```
Time: 0 ──────────► 50 ns
      ↑ START HERE (not in middle!)
```

**Multi-Channel Data**:
```
Channels: 0 (DAPI) | 1 (GFP) | 2 (RFP) | 3 (Cy5)
          ↑ START HERE (not channel 2!)
```

**Higher-Dimensional Spatial**:
```
W dimension: -100 ←──── 0 ────→ +100
                        ↑ START HERE (center makes sense)
```

---

## Implementation Details

### File to Modify
**Path**: `packages/luxar-viewer/src/scene/scene-dims-manager.ts`

### Change 1: Update Initialization Logic (Lines 119-139)

**Before**:
```typescript
for (let i = 0; i < ndim; i++) {
  if (metadata[i].display !== true) {
    // Non-displayed dimensions start at center of range
    const [min, max] = this.dimensionRanges[i];
    let centerValue = (min + max) / 2;

    // For discrete dimensions, floor to nearest integer
    if (metadata[i].discrete) {
      centerValue = Math.floor(centerValue);
    }

    currentStep[i] = centerValue;
  }
  // Displayed dimensions start at 0 (camera will determine actual position)
}
```

**After**:
```typescript
for (let i = 0; i < ndim; i++) {
  if (metadata[i].display !== true) {
    const [min, max] = this.dimensionRanges[i];

    // Discrete/categorical dimensions: start at first position (minimum)
    // Examples: time=0, channel=0, frame=0, category=first
    if (metadata[i].discrete || metadata[i].categories) {
      currentStep[i] = min;
    } else {
      // Continuous spatial dimensions: start at center
      // Examples: 4th spatial dimension (W), higher-dimensional coordinates
      currentStep[i] = (min + max) / 2;
    }
  }
  // Displayed dimensions start at 0 (camera will determine actual position)
}
```

### Change 2: Add Missing Notification (After Line 156)

**Before**:
```typescript
// Step 8: Create the shared dimension state object
this.dims = {
  ndim,
  currentStep,
  displayed,
  metadata,
};

return true;
```

**After**:
```typescript
// Step 8: Create the shared dimension state object
this.dims = {
  ndim,
  currentStep,
  displayed,
  metadata,
};

// Notify all listeners of the initial dimension state
// This triggers:
// - Initial data load at correct slice position
// - Slider visual update to match data
// - Scene render with properly positioned data
this.notifyListeners();

return true;
```

### Updated Comment Block

Update the comment at line 118-120 to reflect new policy:

**Before**:
```typescript
// Step 6: Initialize dimension positions
// Non-displayed dimensions start at CENTER of range to maximize chance of visible data
// (starting at min could put us outside the actual data bounds)
```

**After**:
```typescript
// Step 6: Initialize dimension positions
// Policy:
// - Discrete/categorical dimensions (time, channels, frames): start at MINIMUM (first position)
// - Continuous spatial dimensions (4th+ spatial dims): start at CENTER (no natural "first")
// - Displayed dimensions (X, Y, Z): start at 0 (camera-controlled)
```

---

## Testing Strategy

### Unit Tests

**File**: `packages/luxar-viewer/src/tests/unit/scene/scene-dims-manager.test.ts` (create if needed)

```typescript
describe('SceneDimsManager.initFromScene()', () => {
  it('should initialize discrete dimensions to minimum', () => {
    // Test with time dimension (discrete=true, range=[0, 50])
    // Expect: currentStep[timeIndex] === 0
  });

  it('should initialize categorical dimensions to first category', () => {
    // Test with channel dimension (categories=['DAPI', 'GFP', 'RFP'])
    // Expect: currentStep[channelIndex] === 0
  });

  it('should initialize continuous spatial dimensions to center', () => {
    // Test with 4D spatial (discrete=false, range=[-100, 100])
    // Expect: currentStep[wIndex] === 0
  });

  it('should notify listeners after initialization', () => {
    const listener = jest.fn();
    sceneDimsManager.addListener(listener);
    sceneDimsManager.initFromScene(mockScene);
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
```

### Integration Tests

**Test 1: Time-Animated Dataset**
```typescript
// Load particle_collision_animated.zarr (time dimension: 0-50ns)
// Verify:
// - Slider shows position 0
// - Data displayed is from t=0
// - Console logs show "Query position: [0.00, 0.00, 0.00, 0.00]"
```

**Test 2: Multi-Channel Dataset**
```typescript
// Load multi-channel fluorescence data (channels: 0-3)
// Verify:
// - Slider shows channel 0
// - Data displayed is DAPI channel
// - Status bar shows "Channel: 0"
```

**Test 3: 4D Spatial Dataset**
```typescript
// Load 4D spatial dataset (W: -50 to +50)
// Verify:
// - Slider shows position 0 (center)
// - Data displayed is W=0 slice
// - Behavior same as before (unchanged for spatial dims)
```

### E2E Tests (Playwright)

**File**: `packages/luxar-viewer/src/tests/e2e/dimension-initialization.spec.ts` (create new)

```typescript
test('time dimension initializes to first frame', async ({ page }) => {
  await page.goto('?src=collision_animated.zarr&debug');
  await page.waitForSelector('#dim-slider-3'); // Time slider

  const sliderValue = await page.inputValue('#dim-slider-3');
  expect(parseFloat(sliderValue)).toBe(0);

  // Verify console shows t=0 query
  const logs = await page.evaluate(() => window.__luxarDebug.consoleInterceptor.getLogs());
  expect(logs).toContain('Query position: [0.00, 0.00, 0.00, 0.00]');
});
```

### Manual Testing Checklist

- [ ] Load time-animated dataset → starts at t=0
- [ ] Load multi-channel dataset → starts at channel 0
- [ ] Load 4D spatial dataset → starts at W=0 (center)
- [ ] Slider position matches displayed data
- [ ] No visual "jump" on first slider interaction
- [ ] Console logs show correct initial query position
- [ ] Keyboard navigation works from initial position

---

## Risk Assessment

### Low Risk Changes
- ✅ **Adding `notifyListeners()` call**: Pure addition, no logic change
- ✅ **Changing discrete initialization**: Well-scoped, clear semantics

### Potential Issues

**Issue 1**: Datasets with no data at t=0
- **Scenario**: Time dimension range is [5.0, 50.0] but no data exists at t=5.0
- **Mitigation**: This is a data quality issue, not a viewer bug. If data truly starts later, dimension range should reflect that.
- **Fallback**: User can immediately move slider to find data

**Issue 2**: Existing tests may expect center initialization
- **Scenario**: Tests assert specific slice positions
- **Impact**: Test failures (not production failures)
- **Solution**: Update test expectations to match new policy

**Issue 3**: Performance impact of initial `notifyListeners()`
- **Scenario**: Could trigger expensive data load during startup
- **Analysis**: Load would happen anyway on first render; just moves timing earlier
- **Mitigation**: Already asynchronous, won't block UI

### Breaking Changes
**None** - This is a UX improvement that changes default behavior, but:
- No API changes
- No data format changes
- Backwards compatible (old datasets work fine)
- Users can immediately navigate to any position they want

---

## Rollout Plan

### Phase 1: Implementation (Day 1)
1. ✅ Create feature branch: `fix/dimension-initialization`
2. ✅ Implement changes to `scene-dims-manager.ts`
3. ✅ Update inline comments
4. ✅ Self-test with dev build

### Phase 2: Testing (Day 1-2)
1. ✅ Run existing unit tests → fix any failures
2. ✅ Run E2E test suite → verify no regressions
3. ✅ Manual testing with various dataset types
4. ✅ Create new E2E test for initialization behavior

### Phase 3: Documentation (Day 2)
1. ✅ Update `packages/luxar-viewer/src/scene/README.md`
2. ✅ Update `docs/UI_DESIGN.md` with initialization policy
3. ✅ Add entry to `CHANGELOG.md`:
   ```markdown
   ### Fixed
   - Dimension sliders now initialize to correct position on viewer startup
   - Time/channel dimensions start at first position (0) instead of center
   - Data display now matches slider position from initial load
   ```

### Phase 4: Review & Merge (Day 2-3)
1. ✅ Code review
2. ✅ Approval
3. ✅ Merge to `main`
4. ✅ Tag release (if needed)

### Phase 5: Deployment (Day 3)
1. ✅ Build production bundle: `pnpm build`
2. ✅ Deploy to staging environment
3. ✅ Smoke test on staging
4. ✅ Deploy to production

---

## Success Criteria

### Must Have
- ✅ Time dimensions initialize to t=0
- ✅ Channel dimensions initialize to channel 0
- ✅ Slider position matches displayed data on load
- ✅ No console errors or warnings
- ✅ All existing tests pass

### Nice to Have
- ✅ New E2E test covering initialization
- ✅ Updated documentation
- ✅ Performance metrics unchanged or improved

### Validation
After deployment, verify with real users:
- "Does the viewer start showing the beginning of your time series?"
- "Does the slider position make sense for your data?"
- "Any unexpected behavior on initial load?"

---

## Appendix

### Code Locations Reference

| File | Lines | Description |
|------|-------|-------------|
| `scene-dims-manager.ts` | 65-158 | `initFromScene()` - Main initialization |
| `scene-dims-manager.ts` | 119-139 | Loop setting `currentStep[]` values |
| `scene-dims-manager.ts` | 157 | Missing `notifyListeners()` call location |
| `scene-dims-manager.ts` | 202-222 | `setDimensionValue()` - Shows correct pattern |
| `dimension-sliders.ts` | 201-353 | `createSlider()` - Reads initial values |
| `dimension-sliders.ts` | 294-302 | Initial value setting from `currentStep[]` |
| `input-handler.ts` | 148-190 | `initDimensionSliders()` - Orchestration |
| `input-handler.ts` | 182-189 | Listener registration - expects notification |

### Related Issues
- None directly, but this improves UX for all nD navigation

### Future Enhancements
- Allow user preference for initialization policy (via config)
- Add "reset to start" button for dimensions
- Remember last position per dataset (localStorage)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-11
**Author**: Claude (with user specification)
**Status**: ✅ Ready for Implementation
