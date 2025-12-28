# Multi-Type Support Fix Plan for Lines and GSplats

## Problem Summary

The Python encoder uses the same encoding pipeline for Points, Lines, and GSplats colors:
- SDR colors (0-1 range) → `rgb_uint8` encoding → stored as `uint8` with `original_dtype='uint8'`
- HDR colors (>1.0) → `float32` encoding
- LUT encoding (≤256 unique) → `lut_uint8` → stored as `uint8` indices

However, the TypeScript decoder only supports multi-type for Points. Lines and GSplats always return Float32Array, ignoring `original_dtype`.

## Impact

1. **Memory waste**: SDR colors stored as uint8 → decoded to float32 (4x memory)
2. **Inconsistent behavior**: Points correctly restore uint8, Lines/GSplats don't
3. **No functional bug**: Colors still render correctly (values are converted properly)

## Files to Modify

### 1. Type Definitions

**`src/types/lines.ts`** - Update `LoadedLinesData` interface:
```typescript
// BEFORE
colors: Float32Array | null;

// AFTER
colors: Float32Array | Uint8Array | Uint16Array | null;
```

**`src/types/gsplats.ts`** - Update `LoadedGSplatsData` interface:
```typescript
// BEFORE
colors: Float32Array | null;

// AFTER
colors: Float32Array | Uint8Array | Uint16Array | null;
```

### 2. Data Accumulators

**`src/data/data-accumulator.ts`** - Update `LinesDataAccumulator`:
```typescript
// BEFORE
private colorBuffer: Float32Array;

// AFTER
private colorBuffer: Float32Array | Uint8Array | Uint16Array;
```

Similar changes for `GSplatsDataAccumulator`.

### 3. Spatial Index Loaders

**`src/data/lines-spatial-index-loader.ts`**:
- Add `allocateOutputBuffer` method (copy from points loader)
- Add `loadDirectRanges` method for type preservation
- Add `original_dtype` restoration after RangeLoader decode
- Update variable types from `Float32Array` to union types

**`src/data/gsplats-spatial-index-loader.ts`**:
- Same changes as lines loader

### 4. GPU Buffer Pool

**`src/rendering/gpu-buffer-pool.ts`**:
- Add `LinesAttributeTypes` interface
- Add `GSplatsAttributeTypes` interface
- Update `acquireLinesGeometry` for type-aware pooling
- Update `acquireGSplatsGeometry` for type-aware pooling

### 5. Scene Loader

**`src/data/scene-loader.ts`**:
- Update Lines geometry creation for multi-type colors
- Update GSplats geometry creation for multi-type colors
- Add normalization flag for uint8/uint16 colors

## Implementation Order

1. Type definitions (lines.ts, gsplats.ts)
2. Data accumulators (data-accumulator.ts)
3. Lines spatial index loader (lines-spatial-index-loader.ts)
4. GSplats spatial index loader (gsplats-spatial-index-loader.ts)
5. GPU buffer pool (gpu-buffer-pool.ts)
6. Scene loader (scene-loader.ts)
7. Tests

## Key Code Patterns to Copy from Points Loader

### allocateOutputBuffer (point-spatial-index-loader.ts:788-815)
```typescript
private allocateOutputBuffer(
  totalElements: number,
  isEncoded: boolean,
  dtype: string
): Float32Array | Uint8Array | Uint16Array | Float16Array {
  if (isEncoded) {
    return new Float32Array(totalElements);
  }
  if (dtype === 'uint8' || dtype === '|u1' || dtype === '<u1' || dtype === '>u1') {
    return new Uint8Array(totalElements);
  }
  if (dtype === 'uint16' || dtype === '|u2' || dtype === '<u2' || dtype === '>u2') {
    return new Uint16Array(totalElements);
  }
  return new Float32Array(totalElements);
}
```

### original_dtype restoration (point-spatial-index-loader.ts:1042-1072)
```typescript
const originalDtype = attrs.encoding?.original_dtype;
let output: Float32Array | Uint8Array | Uint16Array = decodedFloat32;

if (originalDtype === 'uint8' || originalDtype === '|u1' || originalDtype === '<u1' || originalDtype === '>u1') {
  const uint8Output = new Uint8Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    uint8Output[i] = Math.round(Math.max(0, Math.min(255, decodedFloat32[i])));
  }
  output = uint8Output;
} else if (originalDtype === 'uint16' || originalDtype === '|u2' || originalDtype === '<u2' || originalDtype === '>u2') {
  const uint16Output = new Uint16Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    uint16Output[i] = Math.round(Math.max(0, Math.min(65535, decodedFloat32[i])));
  }
  output = uint16Output;
}
```

### Scene loader color handling (scene-loader.ts:1591-1599)
```typescript
if (data.colors) {
  const needsNormalization =
    data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;
  geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3, needsNormalization));
}
```

## Tests to Add/Update

1. `src/tests/unit/data/lines-spatial-index-loader.test.ts` - Add original_dtype restoration test
2. `src/tests/unit/data/gsplats-spatial-index-loader.test.ts` - Add original_dtype restoration test
3. `src/tests/unit/types/lines.test.ts` - Update for new type unions
4. `src/tests/unit/types/gsplats.test.ts` - Update for new type unions

## Verification

After implementation:
1. Run `pnpm test --run` - All unit tests pass
2. Run `pnpm typecheck` - No type errors
3. Run `pnpm test:e2e` - E2E tests pass (optional, takes ~17 min)
