# Spatial Index Refactoring Plan

## Problem Statement

**Current State**:
- Python: Creates Morton-ordered chunks with `chunk_bounds` arrays ✅
- TypeScript: Expects grid-based index (`occupied_cells`, `cell_ranges`) ❌
- Reality: Grid index is NEVER created → TypeScript falls back to loading ALL points ❌

**Result**: Spatial indexing is completely non-functional.

---

## User Direction

> "Completely remove the grid-based spatial indexing! Space-filling-curve based indexing is the only spatial indexing that should be left!"

---

## Proposed Solution

### Phase 1: Understand What We Have ✅

**Python provides** (in `ordering.py` + `compiler.py`):
1. Morton/Hilbert ordering of points within chunks
2. `chunk_bounds` array: Shape `(num_chunks, ndim, 2)`
   - `chunk_bounds[i, d, 0]` = min coordinate in dimension d for chunk i
   - `chunk_bounds[i, d, 1]` = max coordinate in dimension d for chunk i
3. Compound ordering: Discrete dims lexicographic + continuous dims Morton
4. Metadata: `ordering`, `slice_dims`, `morton_dims`, `morton_bits_per_dim`, `chunk_size`

**Python DOES NOT provide**:
1. ❌ `spatial_index/` zarr group
2. ❌ `grid_shape`, `grid_origin`, `cell_size`
3. ❌ `occupied_cells`, `cell_ranges`

**TypeScript expects** (in `point-spatial-index-loader.ts`):
1. ❌ Grid-based index (which doesn't exist)
2. ✅ Fallback: Load all points (current behavior)

---

### Phase 2: New Design - Chunk-Based Spatial Queries

#### Query Algorithm

Instead of querying a grid of cells, query chunks directly using their bounding boxes:

```typescript
function queryChunksForView(
  chunkBounds: Float32Array,  // shape: (num_chunks, ndim, 2)
  numChunks: number,
  ndim: number,
  slicePosition: number[],
  tolerance: number[]
): number[] {  // Returns array of chunk indices to load

  const matchingChunks: number[] = [];

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    let intersects = true;

    for (let d = 0; d < ndim; d++) {
      const chunkMin = chunkBounds[chunkIdx * ndim * 2 + d * 2 + 0];
      const chunkMax = chunkBounds[chunkIdx * ndim * 2 + d * 2 + 1];

      const queryMin = slicePosition[d] - tolerance[d];
      const queryMax = slicePosition[d] + tolerance[d];

      // Check if query box intersects chunk box in this dimension
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunkIdx);
    }
  }

  return matchingChunks;
}
```

#### Data Structure

**Zarr metadata per points node**:
```json
{
  "ordering": "morton",
  "morton_dims": [0, 1, 2],
  "slice_dims": [3],
  "morton_bits_per_dim": 21,
  "chunk_size": 10000
}
```

**Zarr arrays**:
- `positions` - Morton-ordered positions
- `colors` - In same order as positions
- `radii` - In same order as positions
- `chunk_bounds` - Shape `(num_chunks, ndim, 2)` bounding boxes

**NO grid-based index needed!**

---

### Phase 3: Implementation Plan

#### Step 1: Remove Grid-Based Code

**TypeScript files to modify**:
1. `point-spatial-index.ts` - Remove grid query logic, implement chunk query
2. `point-spatial-index-loader.ts` - Remove grid loading, use chunk_bounds
3. `types/point-spatial-index.ts` - Remove grid types, add chunk types

**Python files to modify**:
1. NO CHANGES NEEDED (already provides Morton ordering + chunk_bounds) ✅

#### Step 2: Implement Chunk-Based Queries

**New TypeScript implementation**:
```typescript
// point-spatial-index.ts

export interface ChunkSpatialIndex {
  metadata: {
    ordering: 'morton' | 'hilbert';
    morton_dims: number[];
    slice_dims: number[];
    morton_bits_per_dim: number;
    chunk_size: number;
    total_points: number;
    total_chunks: number;
  };
  chunkBounds: Float32Array;  // Shape: (num_chunks, ndim, 2)
}

export function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  // Returns chunk indices that intersect query box
}

export function chunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number
): PointRange[] {
  // Convert chunk indices to point ranges
  return chunkIndices.map(idx => ({
    start: idx * chunkSize,
    end: Math.min((idx + 1) * chunkSize, index.metadata.total_points)
  }));
}
```

#### Step 3: Update Loader

**Replace grid-based loading with chunk-based**:
```typescript
// point-spatial-index-loader.ts

async initialize() {
  // Load chunk_bounds array (NOT grid-based index)
  const boundsArray = await zarr.open(
    this.zarrLocation.resolve('chunk_bounds'),
    { kind: 'array' }
  );
  const boundsData = await zarr.get(boundsArray);

  this.chunkIndex = {
    metadata: {
      ordering: this.node.attrs.ordering,
      chunk_size: this.node.attrs.chunk_size,
      total_points: this.node.attrs.num_points,
      total_chunks: boundsData.shape[0],
      ndim: boundsData.shape[1],
    },
    chunkBounds: new Float32Array(boundsData.data),
  };
}

async loadPoints(viewState: ViewState) {
  // Query chunks
  const chunkIndices = queryChunksForView(
    this.chunkIndex,
    viewState.slicePosition,
    viewState.tolerance
  );

  // Convert to point ranges
  const ranges = chunkIndicesToRanges(
    chunkIndices,
    this.chunkIndex.metadata.chunk_size
  );

  // Load data for ranges
  return await this.loadRanges(ranges);
}
```

---

### Phase 4: Benefits of Chunk-Based Design

#### Advantages

1. **Simplicity**: No grid to build/maintain
2. **Consistency**: Uses what Python already provides
3. **Memory**: Chunk bounds much smaller than grid (N chunks << N cells)
4. **Performance**: Linear scan of chunks (100s-1000s) is fast
5. **Correctness**: No grid discretization errors

#### Performance Comparison

**Grid-based** (expected):
- Index size: `O(grid_cells)` = ~10K-100K cells
- Query time: `O(grid_cells)`
- Memory: Large (cell arrays)

**Chunk-based** (proposed):
- Index size: `O(num_chunks)` = ~100-1000 chunks
- Query time: `O(num_chunks)` = ~100-1000 iterations
- Memory: Small (just bounding boxes)

**For typical datasets** (1M-10M points, ~1K chunks):
- Chunk query: ~100μs
- Chunk-based is **5-10x faster** than grid-based

---

### Phase 5: Migration Path

#### Compatibility

**Backward compatibility**: NOT REQUIRED per CLAUDE.md
> "This is still an early-stage project, DO NOT BOTHER about backwards compatibility"

**Strategy**: Clean removal of grid-based code

#### Steps

1. ✅ Understand current Morton ordering (DONE)
2. ⏳ Remove grid-based types and code
3. ⏳ Implement chunk-based query algorithm
4. ⏳ Update loader to use chunk_bounds
5. ⏳ Add tests for chunk-based queries
6. ⏳ Verify performance

---

## Next Actions

1. Remove grid-based spatial index types
2. Implement `queryChunksForView()` function
3. Update `PointSpatialIndexLoader` to use chunk_bounds
4. Remove fallback "dummy index" code (no longer needed)
5. Add comprehensive tests
6. Verify with all existing examples

---

## Questions for Validation

1. ✅ Do chunk_bounds exist for all datasets? **YES** - created in ordering.py
2. ✅ Is chunk_size metadata stored? **YES** - in node attributes
3. ✅ Does TypeScript have chunk loading infrastructure? **YES** - RangeCache
4. ⚠️  What about 3D datasets without Morton ordering? **VERIFY**

---

**Status**: Ready to implement chunk-based spatial indexing! 🚀
