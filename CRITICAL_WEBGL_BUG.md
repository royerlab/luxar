# CRITICAL: WebGL Vertex Buffer Sizing Bug

**Priority**: 🔴 CRITICAL
**Discovered**: 2025-12-01
**Status**: Under Investigation

## Symptom

Hundreds of WebGL errors when rendering datasets:
```
[.WebGL-0x13c004a8a00] GL_INVALID_OPERATION: glDrawArrays: Vertex buffer is not big enough for the draw call.
```

Repeats ~160 times until WebGL stops reporting errors.

## When It Occurs

**Dataset**: `sharpness_showcase_example.zarr` (and likely others)

**Console Output Shows**:
```
[✅] [SceneLoader] Loaded 10000 points for /MixedSharpnessCloud
[✅] [SceneLoader] Loaded 4900 points for /SharpnessGradient
[✅] [SceneLoader] Loaded 3969 points for /SharpnessWave
[✅] [SceneLoader] Loaded 20 points for /Sharpness_Linear (1.0)
... then WebGL errors start
```

## Impact

- **Visual**: Likely causing missing/corrupted geometry
- **Performance**: WebGL error reporting overhead
- **User Experience**: "Something looks off" even though data loads

## Root Cause Hypothesis

Based on the error pattern:

1. **Attribute Count Mismatch**: One or more vertex attributes (position, color, radius, sharpness) have different counts
2. **Buffer Not Updated**: Buffer allocated for N points but draw call uses M points
3. **Sharpness Attribute**: All affected objects have sharpness attribute - suspicious!

**Most Likely**: When adding sharpness attribute to geometry, the buffer is not being properly sized to match position count.

## Investigation Steps

### 1. Check Attribute Counts
Look at `src/data/point-spatial-index-loader.ts` where attributes are created:

```typescript
// Suspect: Does sharpness buffer match positions buffer?
geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
geometry.setAttribute('sharpness', new Float32BufferAttribute(sharpness, 1));

// Are positions.length/3 === sharpness.length?
```

### 2. Check Buffer Updates
When updating geometry with new data, are ALL buffers resized together?

### 3. Check Draw Calls
In the shader material, verify draw count matches buffer size:

```typescript
renderer.render(scene, camera);
// Does the draw call use the correct vertex count?
```

## E2E Test Added

Created `webgl-errors.spec.ts` to catch this:
- ✅ Detects WebGL console errors
- ✅ Checks buffer attribute sizing
- ✅ Verifies geometry consistency
- ✅ Tests all example datasets

**This test now FAILS** - which is correct! It catches the bug.

## Fix Priority

**IMMEDIATE** - This is a data corruption/rendering bug that affects multiple datasets.

## Reproduction

```bash
# Generate the example
make run-examples

# Serve it
luxar serve packages/luxar/examples/sharpness_showcase_example.zarr

# Open in browser with console open
# Navigate to: http://localhost:8000?src=http://127.0.0.1:8000

# Observe: ~160 WebGL errors in console
```

## Next Steps

1. ✅ E2E test created to catch this
2. 🔄 Investigate point-spatial-index-loader.ts buffer creation
3. 🔄 Check if sharpness attribute is being added incorrectly
4. 🔄 Verify all attributes have matching counts
5. 🔄 Fix the buffer sizing issue
6. ✅ Verify fix with new WebGL error test

---

**Action Required**: Investigate and fix the vertex buffer sizing issue in the renderer/data loading code.

This is exactly the kind of bug that E2E tests should catch - and now they will! 🎯
