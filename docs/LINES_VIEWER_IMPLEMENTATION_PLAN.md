# Lines Viewer Implementation Plan

**Version**: 1.3.0
**Created**: 2025-12-09
**Last Updated**: 2025-12-09
**Status**: Draft - Complete

## Overview

This document outlines the implementation plan for adding Lines support to the luxar-viewer TypeScript package. Lines are the second major primitive type after Points, enabling visualization of connected line segments with variable width, color, and edge softness.

**Related Specifications**:
- Python Lines Spec: `packages/luxar/src/luxar/core/SPECIFICATIONS.md` (Section 6)
- Viewer Data Spec: `packages/luxar-viewer/src/data/SPECIFICATIONS.md` (Section 7)
- Viewer Types Spec: `packages/luxar-viewer/src/types/SPECIFICATIONS.md` (Section 7)
- Viewer Scene Spec: `packages/luxar-viewer/src/scene/SPECIFICATIONS.md`
- Viewer Rendering Spec: `packages/luxar-viewer/src/rendering/SPECIFICATIONS.md` (Section 7)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: Type Definitions](#2-phase-1-type-definitions)
3. [Phase 2: Spatial Index Loading](#3-phase-2-spatial-index-loading)
4. [Phase 3: Lines Data Loader](#4-phase-3-lines-data-loader)
5. [Phase 4: Line Material & Rendering](#5-phase-4-line-material--rendering)
6. [Phase 5: Scene Integration](#6-phase-5-scene-integration)
7. [Phase 6: Testing Strategy](#7-phase-6-testing-strategy)
8. [Implementation Risks & Mitigations](#8-implementation-risks--mitigations)
9. [Design Decisions](#9-design-decisions-resolved)

---

## 1. Architecture Overview

### 1.1 Data Flow

```
Python Compiler                    TypeScript Viewer
┌─────────────────┐               ┌──────────────────────────────────┐
│                 │               │                                  │
│  Lines Node     │    Zarr      │  LinesSpatialIndexLoader         │
│  ├─ vertices/   │ ──────────►  │  ├─ Load chunk indices           │
│  ├─ segments/   │              │  ├─ Query segment chunks          │
│  ├─ widths/     │              │  ├─ Load visible segments         │
│  ├─ colors/     │              │  ├─ Derive vertex chunks          │
│  ├─ sharpness/  │              │  ├─ Load vertex data              │
│  ├─ vertex_chunk_bounds/       │  └─ Remap indices (global→local) │
│  └─ segment_chunk_bounds/      │                                  │
│                 │               │         ▼                        │
└─────────────────┘               │  LineMaterial (instanced quads)  │
                                  │  ├─ Per-segment instance data    │
                                  │  ├─ Screen-space expansion       │
                                  │  └─ HDR color output             │
                                  │                                  │
                                  │         ▼                        │
                                  │  THREE.InstancedMesh             │
                                  │  └─ Scene graph integration      │
                                  └──────────────────────────────────┘
```

### 1.2 Key Differences from Points

| Aspect | Points | Lines |
|--------|--------|-------|
| Spatial Index | Single ordering (D-space) | Dual ordering (vertices: D-space, segments: 2×D-space) |
| Chunk Bounds | One array (`chunk_bounds/`) | Two arrays (`vertex_chunk_bounds/`, `segment_chunk_bounds/`) |
| Loading | Direct chunk load | Two-phase: segments → vertices |
| Geometry | `THREE.Points` with `gl_PointSize` | Instanced quads (custom geometry) |
| Index Buffer | None (implicit) | Segment indices → vertex lookup |
| Query Tolerance | `max_radius` | 0 (bounds already include width) |

### 1.3 File Structure

```
packages/luxar-viewer/src/
├── data/
│   ├── lines-chunk-spatial-index.ts    [NEW] Dual spatial index loading
│   ├── lines-spatial-index-loader.ts   [NEW] Two-phase data loader
│   └── scene-loader.ts                 [MODIFY] Add lines dispatch
├── rendering/
│   ├── line-material.ts                [NEW] Instanced quad shader
│   └── material-manager.ts             [MODIFY] Register line material
├── scene/
│   └── scene-manager.ts                [MODIFY] Bounding box, updates
└── types/
    ├── lines.ts                        [NEW] Lines type definitions
    └── index.ts                        [MODIFY] Export lines types
```

---

## 2. Phase 1: Type Definitions

### 2.1 Goal

Create TypeScript interfaces that mirror the Python Lines specification exactly.

### 2.2 File: `src/types/lines.ts`

```typescript
/**
 * Lines type definitions for luxar-viewer.
 *
 * These types mirror Python luxar.core.Lines and enable type-safe
 * handling of line data throughout the viewer.
 */

// ============================================================================
// Metadata Types (from zarr .zattrs)
// ============================================================================

/**
 * Ordering metadata structure (shared between vertices and segments)
 */
export interface OrderingMetadata {
  /** Discrete dimension indices for compound ordering */
  slice_dims: number[];

  /** Spatial dimension indices for curve ordering */
  ordering_dims: number[];

  /** Min bounds for coordinate normalization */
  ordering_min: number[];

  /** Max bounds for coordinate normalization */
  ordering_max: number[];

  /** Elements per chunk */
  chunk_size: number;

  /** Bits per dimension (implementation detail, optional) */
  ordering_bits_per_dim?: number;
}

/**
 * Lines node metadata from zarr .zattrs
 */
export interface LinesMetadata {
  /** Node type identifier */
  type: 'lines';

  /** Total vertex count */
  n_vertices: number;

  /** Total segment count */
  n_segments: number;

  /** Vertex position dimensionality */
  ndim: number;

  /** Original line type from Python API */
  original_line_type: 'segments' | 'polyline' | 'loop' | 'indexed';

  /** Maximum line width in world units */
  max_width: number;

  /** Whether colors array is present */
  has_colors: boolean;

  /** Whether sharpness array is present */
  has_sharpness: boolean;

  /** Whether spatial index exists (redundant with ordering !== 'none', but explicit) */
  has_spatial_index?: boolean;

  /** Spatial ordering method */
  ordering: 'morton' | 'hilbert' | 'none';

  /** Vertex ordering metadata (when ordering != 'none') */
  vertex_ordering?: OrderingMetadata;

  /** Segment ordering metadata (when ordering != 'none') */
  segment_ordering?: OrderingMetadata;

  /** 4x4 transform matrix (column-major for THREE.js) */
  transform?: number[];

  /** Opacity multiplier */
  opacity?: number;

  /** Gamma correction */
  gamma?: number;

  /** Blending mode */
  blending_mode?: 'additive' | 'normal';
}

// ============================================================================
// Spatial Index Types
// ============================================================================

/**
 * Chunk-based spatial index for Lines.
 * Supports dual ordering (vertices + segments).
 */
export interface LinesChunkSpatialIndex {
  /** Lines metadata from zarr attributes */
  metadata: LinesMetadata;

  /** Vertex chunk bounding boxes (num_v_chunks * ndim * 2), flattened */
  vertexChunkBounds: Float32Array;

  /** Segment chunk bounding boxes (num_s_chunks * ndim * 2), flattened */
  segmentChunkBounds: Float32Array;

  /** Computed: ceil(n_vertices / vertex_ordering.chunk_size) */
  vertexChunkCount: number;

  /** Computed: ceil(n_segments / segment_ordering.chunk_size) */
  segmentChunkCount: number;
}

// ============================================================================
// Loaded Data Types
// ============================================================================

/**
 * Lines data loaded from zarr and ready for rendering.
 * Segments use LOCAL indices into the loaded vertex arrays.
 */
export interface LoadedLinesData {
  /** Vertex positions (N vertices * ndim dimensions), flattened row-major */
  vertices: Float32Array;

  /** Segment index pairs (M segments * 2) - local indices into vertices */
  segments: Uint32Array;

  /** Vertex widths (N,) or (1,) if broadcast */
  widths: Float32Array;

  /** Vertex colors (N * 3) RGB, null if not present */
  colors: Float32Array | null;

  /** Vertex sharpness (N,) null if not present */
  sharpness: Float32Array | null;

  /** Number of segments loaded */
  segmentCount: number;

  /** Number of vertices loaded */
  vertexCount: number;

  /** Dimensionality for interpreting vertices array */
  ndim: number;
}

/**
 * Segment index range for partial loading
 */
export interface LineRange {
  /** Start segment index (inclusive) */
  start: number;

  /** End segment index (exclusive) */
  end: number;
}

// ============================================================================
// Scene Integration Types
// ============================================================================

/**
 * User data attached to THREE.InstancedMesh for Lines in scene
 */
export interface LinesUserData {
  /** Node type identifier for runtime type checking */
  nodeType: 'lines';

  /** Data loader instance */
  loader: LinesDataLoader;

  /** Zarr group attributes */
  attrs: LinesMetadata;

  /** Spatial index for queries */
  spatialIndex: LinesChunkSpatialIndex;

  /** Scene dimension metadata */
  sceneDimensions: DimensionMetadata[];

  /** Maximum line width (for bounding box expansion) */
  maxWidth: number;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if metadata is for a Lines node
 */
export function isLinesMetadata(attrs: unknown): attrs is LinesMetadata {
  return (
    typeof attrs === 'object' &&
    attrs !== null &&
    (attrs as Record<string, unknown>).type === 'lines'
  );
}

/**
 * Check if userData indicates a Lines object
 */
export function isLinesUserData(userData: unknown): userData is LinesUserData {
  return (
    typeof userData === 'object' &&
    userData !== null &&
    (userData as Record<string, unknown>).nodeType === 'lines'
  );
}

// Forward declaration for loader interface
export interface LinesDataLoader {
  loadLines(viewState: ViewState): Promise<LoadedLinesData>;
  updateView(viewState: ViewState): Promise<LoadedLinesData>;
  dispose(): void;
}

// Import dependencies (adjust paths as needed)
import type { DimensionMetadata } from './dimensions';
import type { ViewState } from '../data/data-loader-types';
```

### 2.3 Tasks

- [ ] Create `src/types/lines.ts` with all type definitions
- [ ] Export from `src/types/index.ts`
- [ ] Add JSDoc comments with examples
- [ ] Run `pnpm typecheck` to verify no errors

---

## 3. Phase 2: Spatial Index Loading

### 3.1 Goal

Load the dual spatial index (vertex + segment chunk bounds) from zarr.

### 3.2 File: `src/data/lines-chunk-spatial-index.ts`

```typescript
/**
 * Dual spatial index loading for Lines.
 *
 * Lines have two independent spatial orderings:
 * - Vertices: Ordered in D-dimensional space
 * - Segments: Ordered in (2×D)-dimensional space
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import type { LinesMetadata, LinesChunkSpatialIndex } from '../types/lines';
import { log, Modules } from '../utils/log';

/**
 * Load the dual spatial index for a Lines node.
 *
 * @param location - Zarr location of the lines group
 * @param attrs - Lines metadata from .zattrs
 * @returns LinesChunkSpatialIndex or null if no spatial ordering
 */
export async function loadLinesChunkSpatialIndex(
  location: zarr.Location<zarr.Readable>,
  attrs: LinesMetadata
): Promise<LinesChunkSpatialIndex | null> {
  // Check if spatial ordering is enabled
  if (attrs.ordering === 'none' || !attrs.vertex_ordering || !attrs.segment_ordering) {
    log.info(Modules.SPATIAL_INDEX_LOADER,
      `Lines node has no spatial ordering (ordering=${attrs.ordering})`);
    return null;
  }

  try {
    // Load vertex chunk bounds
    const vertexBoundsArray = await zarr.open(
      location.resolve('vertex_chunk_bounds'),
      { kind: 'array' }
    );
    const vertexBoundsData = await get(vertexBoundsArray);
    const vertexChunkBounds = new Float32Array(vertexBoundsData.data as ArrayBuffer);

    // Load segment chunk bounds
    const segmentBoundsArray = await zarr.open(
      location.resolve('segment_chunk_bounds'),
      { kind: 'array' }
    );
    const segmentBoundsData = await get(segmentBoundsArray);
    const segmentChunkBounds = new Float32Array(segmentBoundsData.data as ArrayBuffer);

    // Compute chunk counts
    const vertexChunkCount = Math.ceil(
      attrs.n_vertices / attrs.vertex_ordering.chunk_size
    );
    const segmentChunkCount = Math.ceil(
      attrs.n_segments / attrs.segment_ordering.chunk_size
    );

    // Validate bounds array sizes
    const expectedVertexSize = vertexChunkCount * attrs.ndim * 2;
    const expectedSegmentSize = segmentChunkCount * attrs.ndim * 2;

    if (vertexChunkBounds.length !== expectedVertexSize) {
      log.warning(Modules.SPATIAL_INDEX_LOADER,
        `Vertex bounds size mismatch: got ${vertexChunkBounds.length}, ` +
        `expected ${expectedVertexSize}`);
    }

    if (segmentChunkBounds.length !== expectedSegmentSize) {
      log.warning(Modules.SPATIAL_INDEX_LOADER,
        `Segment bounds size mismatch: got ${segmentChunkBounds.length}, ` +
        `expected ${expectedSegmentSize}`);
    }

    log.info(Modules.SPATIAL_INDEX_LOADER,
      `Loaded Lines spatial index: ${vertexChunkCount} vertex chunks, ` +
      `${segmentChunkCount} segment chunks`);

    return {
      metadata: attrs,
      vertexChunkBounds,
      segmentChunkBounds,
      vertexChunkCount,
      segmentChunkCount,
    };
  } catch (error) {
    log.error(Modules.SPATIAL_INDEX_LOADER,
      'Failed to load Lines spatial index:', error);
    return null;
  }
}

/**
 * Query segment chunks that intersect the view region.
 *
 * IMPORTANT: segment_chunk_bounds already include line width extent.
 * Tolerance for spatial dimensions should be 0.
 *
 * @param index - Lines spatial index
 * @param slicePosition - Current nD position
 * @param tolerance - Per-dimension tolerance (0 for spatial, 0.5 for discrete)
 * @returns Array of segment chunk indices
 */
export function querySegmentChunksForView(
  index: LinesChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const { segmentChunkBounds, segmentChunkCount } = index;
  const ndim = index.metadata.ndim;
  const matchingChunks: number[] = [];

  for (let chunk = 0; chunk < segmentChunkCount; chunk++) {
    let intersects = true;

    for (let dim = 0; dim < ndim; dim++) {
      const offset = chunk * ndim * 2 + dim * 2;
      const chunkMin = segmentChunkBounds[offset];
      const chunkMax = segmentChunkBounds[offset + 1];

      const queryMin = slicePosition[dim] - tolerance[dim];
      const queryMax = slicePosition[dim] + tolerance[dim];

      // AABB intersection test
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunk);
    }
  }

  return matchingChunks;
}

/**
 * Compute tolerance array for Lines queries.
 *
 * CRITICAL: segment_chunk_bounds already include line width.
 * Spatial dimensions should have tolerance = 0.
 * Discrete dimensions should have tolerance = 0.5.
 */
export function computeLinesTolerance(
  sceneDimensions: Array<{ discrete?: boolean }>,
  displayDims: number[]
): number[] {
  return sceneDimensions.map((dim, idx) => {
    if (displayDims.includes(idx)) {
      // Displayed dimensions: infinite tolerance (want all segments in view)
      return 1e10;
    }
    if (dim.discrete) {
      // Discrete dimensions: half-unit for integer matching
      return 0.5;
    }
    // Spatial dimensions: zero - bounds already include width!
    return 0;
  });
}
```

### 3.3 Tasks

- [ ] Create `src/data/lines-chunk-spatial-index.ts`
- [ ] Implement `loadLinesChunkSpatialIndex()`
- [ ] Implement `querySegmentChunksForView()`
- [ ] Implement `computeLinesTolerance()`
- [ ] Add unit tests for query algorithm

---

## 4. Phase 3: Lines Data Loader

### 4.1 Goal

Implement two-phase loading: segments → vertices with index remapping.

### 4.2 File: `src/data/lines-spatial-index-loader.ts`

**Key Algorithm: Two-Phase Loading**

```
Phase 1: Load Segments
├── Query segment_chunk_bounds for visible chunks
├── Load segment chunks from segments/ array
└── Collect unique vertex indices from loaded segments

Phase 2: Load Vertices
├── Determine which vertex chunks contain required vertices
├── Load vertex chunks (positions, widths, colors, sharpness)
└── Build index remapping for rendering
```

### 4.3 Pseudocode

```typescript
async function loadLinesForView(viewState: ViewState): Promise<LoadedLinesData> {
  // Phase 1: Load visible segments
  const segmentChunks = querySegmentChunksForView(index, slicePosition, tolerance);
  const segmentRanges = segmentChunksToRanges(segmentChunks);
  const segments = await loadSegmentRanges(segmentRanges);

  // Collect unique vertex indices
  const uniqueVertexIndices = new Set<number>();
  for (let i = 0; i < segments.length; i += 2) {
    uniqueVertexIndices.add(segments[i]);
    uniqueVertexIndices.add(segments[i + 1]);
  }

  // Phase 2: Load required vertices
  const sortedIndices = Array.from(uniqueVertexIndices).sort((a, b) => a - b);
  const vertexRanges = computeVertexRangesFromIndices(sortedIndices);

  const vertices = await loadVertexRanges('vertices', vertexRanges);
  const widths = await loadVertexRanges('widths', vertexRanges);
  const colors = metadata.has_colors
    ? await loadVertexRanges('colors', vertexRanges)
    : null;
  const sharpness = metadata.has_sharpness
    ? await loadVertexRanges('sharpness', vertexRanges)
    : null;

  // Build global → local index mapping
  const vertexIndexMap = new Map<number, number>();
  let localIdx = 0;
  for (const range of vertexRanges) {
    for (let i = range.start; i < range.end; i++) {
      vertexIndexMap.set(i, localIdx++);
    }
  }

  // Remap segment indices to local space
  const remappedSegments = new Uint32Array(segments.length);
  for (let i = 0; i < segments.length; i++) {
    remappedSegments[i] = vertexIndexMap.get(segments[i])!;
  }

  return {
    vertices,
    segments: remappedSegments,
    widths,
    colors,
    sharpness,
    segmentCount: segments.length / 2,
    vertexCount: vertexIndexMap.size,
    ndim: metadata.ndim,
  };
}
```

### 4.4 nD Slicing with Endpoint Clipping

For lines in nD space, segments may partially intersect the visible slice. We handle this with **endpoint clipping**.

**Visibility Cases**:

```
Case A: Both endpoints IN slice     → Render full segment
Case B: P1 IN, P2 OUT               → Clip P2 to slice boundary
Case C: P1 OUT, P2 IN               → Clip P1 to slice boundary
Case D: Both OUT, opposite sides    → Clip both to boundaries (segment crosses slice)
Case E: Both OUT, same side         → Don't render (segment misses slice)
```

**Algorithm**:

```typescript
interface ClippedSegment {
  p1: number[];           // Clipped start position (3D display coords)
  p2: number[];           // Clipped end position (3D display coords)
  t1: number;             // Parameter at clipped start (0-1, for attribute interpolation)
  t2: number;             // Parameter at clipped end (0-1)
  visible: boolean;       // Whether segment should be rendered
}

function clipSegmentToSlice(
  p1: number[],           // Start vertex (nD)
  p2: number[],           // End vertex (nD)
  slicePosition: number[],
  tolerance: number[],    // Per-dimension (0 for spatial, 0.5 for discrete)
  displayDims: number[]   // [d0, d1, d2] - which dims to display
): ClippedSegment {

  // For each non-displayed dimension, check if segment crosses slice
  let t1 = 0.0;  // Parameter at start
  let t2 = 1.0;  // Parameter at end

  for (let dim = 0; dim < p1.length; dim++) {
    if (displayDims.includes(dim)) continue;  // Skip displayed dimensions

    const tol = tolerance[dim];
    const sliceMin = slicePosition[dim] - tol;
    const sliceMax = slicePosition[dim] + tol;

    const v1 = p1[dim];
    const v2 = p2[dim];

    // Classify endpoints relative to slice
    const p1In = v1 >= sliceMin && v1 <= sliceMax;
    const p2In = v2 >= sliceMin && v2 <= sliceMax;

    if (p1In && p2In) {
      // Both in - no clipping needed for this dimension
      continue;
    }

    if (!p1In && !p2In) {
      // Both out - check if on same side
      if ((v1 < sliceMin && v2 < sliceMin) || (v1 > sliceMax && v2 > sliceMax)) {
        return { p1: [], p2: [], t1: 0, t2: 0, visible: false };  // Case E
      }
      // Opposite sides - clip both (Case D)
    }

    // Compute intersection parameters
    const dv = v2 - v1;
    if (Math.abs(dv) < 1e-10) continue;  // Parallel to slice

    // t where line crosses sliceMin
    const tMin = (sliceMin - v1) / dv;
    // t where line crosses sliceMax
    const tMax = (sliceMax - v1) / dv;

    // Clip t1 (entry) and t2 (exit) to valid range
    if (dv > 0) {
      // Moving from low to high
      t1 = Math.max(t1, tMin);
      t2 = Math.min(t2, tMax);
    } else {
      // Moving from high to low
      t1 = Math.max(t1, tMax);
      t2 = Math.min(t2, tMin);
    }

    if (t1 >= t2) {
      return { p1: [], p2: [], t1: 0, t2: 0, visible: false };  // No valid range
    }
  }

  // Interpolate clipped positions
  const clippedP1 = p1.map((v, i) => v + t1 * (p2[i] - v));
  const clippedP2 = p1.map((v, i) => v + t2 * (p2[i] - v));

  // Project to 3D display space
  const display1 = displayDims.map(d => clippedP1[d]);
  const display2 = displayDims.map(d => clippedP2[d]);

  return { p1: display1, p2: display2, t1, t2, visible: true };
}
```

**Attribute Interpolation**: When a segment is clipped, per-vertex attributes must be interpolated:

```typescript
// For clipped segment with t1, t2 parameters:
const clippedStartColor = lerpVec3(startColor, endColor, t1);
const clippedEndColor = lerpVec3(startColor, endColor, t2);
const clippedStartWidth = lerp(startWidth, endWidth, t1);
const clippedEndWidth = lerp(startWidth, endWidth, t2);
const clippedStartSharpness = lerp(startSharpness, endSharpness, t1);
const clippedEndSharpness = lerp(endSharpness, endSharpness, t2);

// Segment length is recalculated from clipped positions
const clippedLength = distance3D(clippedP1, clippedP2);
```

**Cap Factor Adjustment**: Clipped endpoints are NOT true segment endpoints - they should have `capFactor = 1.0` (not 0.5) since the "real" endpoint is outside the slice.

```typescript
// Track whether each end was clipped
interface ClippedSegmentData {
  // ... positions, colors, etc ...
  startWasClipped: boolean;  // If true, use capFactor=1.0 at start
  endWasClipped: boolean;    // If true, use capFactor=1.0 at end
}
```

### 4.5 Instance Buffer Construction

After loading and clipping, convert per-vertex data to per-segment GPU instance buffers.

**Input** (from loader + clipper):

```typescript
interface ProcessedLinesData {
  // Per-segment data (M segments)
  startPositions: Float32Array;  // (M, 3) - 3D display coords
  endPositions: Float32Array;    // (M, 3)
  startColors: Float32Array;     // (M, 3)
  endColors: Float32Array;       // (M, 3)
  startWidths: Float32Array;     // (M,)
  endWidths: Float32Array;       // (M,)
  startSharpness: Float32Array;  // (M,)
  endSharpness: Float32Array;    // (M,)
  segmentLengths: Float32Array;  // (M,)
  startClipped: Uint8Array;      // (M,) - 1 if start was clipped
  endClipped: Uint8Array;        // (M,) - 1 if end was clipped
  segmentCount: number;
}
```

**Transformation Algorithm**:

```typescript
function buildInstanceBuffers(
  loadedData: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ProcessedLinesData {
  const { vertices, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;

  // Pre-allocate output arrays (may be smaller after clipping)
  const maxSegments = segmentCount;
  const startPositions = new Float32Array(maxSegments * 3);
  const endPositions = new Float32Array(maxSegments * 3);
  const startColors = new Float32Array(maxSegments * 3);
  const endColors = new Float32Array(maxSegments * 3);
  const startWidths = new Float32Array(maxSegments);
  const endWidths = new Float32Array(maxSegments);
  const startSharpness = new Float32Array(maxSegments);
  const endSharpness = new Float32Array(maxSegments);
  const segmentLengths = new Float32Array(maxSegments);
  const startClipped = new Uint8Array(maxSegments);
  const endClipped = new Uint8Array(maxSegments);

  let outIdx = 0;

  for (let i = 0; i < segmentCount; i++) {
    // Get vertex indices
    const v0 = segments[i * 2];
    const v1 = segments[i * 2 + 1];

    // Extract nD positions
    const p1 = Array.from(vertices.slice(v0 * ndim, (v0 + 1) * ndim));
    const p2 = Array.from(vertices.slice(v1 * ndim, (v1 + 1) * ndim));

    // Clip to slice
    const clipped = clipSegmentToSlice(p1, p2, slicePosition, tolerance, displayDims);
    if (!clipped.visible) continue;

    // Write 3D positions
    startPositions.set(clipped.p1, outIdx * 3);
    endPositions.set(clipped.p2, outIdx * 3);

    // Interpolate and write colors
    const c0 = colors ? Array.from(colors.slice(v0 * 3, (v0 + 1) * 3)) : [1, 1, 1];
    const c1 = colors ? Array.from(colors.slice(v1 * 3, (v1 + 1) * 3)) : [1, 1, 1];
    const startC = lerpVec3(c0, c1, clipped.t1);
    const endC = lerpVec3(c0, c1, clipped.t2);
    startColors.set(startC, outIdx * 3);
    endColors.set(endC, outIdx * 3);

    // Interpolate widths
    const w0 = widths[v0];
    const w1 = widths[v1];
    startWidths[outIdx] = lerp(w0, w1, clipped.t1);
    endWidths[outIdx] = lerp(w0, w1, clipped.t2);

    // Interpolate sharpness (default 1.0 if not present)
    const s0 = sharpness ? sharpness[v0] : 1.0;
    const s1 = sharpness ? sharpness[v1] : 1.0;
    startSharpness[outIdx] = lerp(s0, s1, clipped.t1);
    endSharpness[outIdx] = lerp(s0, s1, clipped.t2);

    // Calculate 3D segment length
    segmentLengths[outIdx] = distance3D(clipped.p1, clipped.p2);

    // Track clipping for cap factor adjustment
    startClipped[outIdx] = clipped.t1 > 0 ? 1 : 0;
    endClipped[outIdx] = clipped.t2 < 1 ? 1 : 0;

    outIdx++;
  }

  // Trim arrays to actual size
  return {
    startPositions: startPositions.slice(0, outIdx * 3),
    endPositions: endPositions.slice(0, outIdx * 3),
    startColors: startColors.slice(0, outIdx * 3),
    endColors: endColors.slice(0, outIdx * 3),
    startWidths: startWidths.slice(0, outIdx),
    endWidths: endWidths.slice(0, outIdx),
    startSharpness: startSharpness.slice(0, outIdx),
    endSharpness: endSharpness.slice(0, outIdx),
    segmentLengths: segmentLengths.slice(0, outIdx),
    startClipped: startClipped.slice(0, outIdx),
    endClipped: endClipped.slice(0, outIdx),
    segmentCount: outIdx,
  };
}

// Helper functions
function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

function lerpVec3(a: number[], b: number[], t: number): number[] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function distance3D(a: number[], b: number[]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}
```

### 4.6 Tasks

- [ ] Create `src/data/lines-spatial-index-loader.ts`
- [ ] Implement `LinesSpatialIndexLoader` class
- [ ] Implement segment range loading
- [ ] Implement vertex range computation from indices
- [ ] Implement index remapping
- [ ] Implement `clipSegmentToSlice()` for nD slicing
- [ ] Implement `buildInstanceBuffers()` transformation
- [ ] Handle broadcast arrays (widths, colors)
- [ ] Handle encoded arrays (LUT, quantized)
- [ ] Add monitoring events

---

## 5. Phase 4: Line Material & Rendering

### 5.1 Goal

Create custom shader material for thick lines using instanced quads, with mathematically correct intensity for additive blending.

### 5.2 Mathematical Foundation: Semicircle Kernel

**Key insight**: Lines are "smeared points" - the intensity at any point is the convolution of a radial kernel with the line path.

**The problem with naive approach**: If we use the same intensity function as points, joints between segments would be overbright (double intensity where two segments meet).

**Solution**: Use a **semicircle kernel** that produces a **parabolic profile** after convolution:

```
Kernel function:     f(r) = √(1 - (r/R)²)     for r < R

Convolution result:  I_body(p) ∝ (1 - (p/R)²)  ← Parabola!
```

**Why this works for joints**:

The convolution integral over a line segment gives exactly **half intensity at endpoints**:

```
Body (middle of segment):  I_body(p) = full intensity
Endpoint (at segment tip): I_end(p) = I_body(p) / 2
```

When two segments meet at joint `b`:
```
Segment a-b endpoint:  0.5 × I_body
Segment b-c endpoint:  0.5 × I_body
─────────────────────────────────────
Sum (additive):        1.0 × I_body  ✓ Seamless!
```

### 5.3 Rendering Approach: Instanced Quads

WebGL `lineWidth` is always 1px on most hardware. We use **instanced quads** instead:

```
Each segment becomes a quad with cap regions:

  cap     ←───── body ─────→     cap
region          region          region
  ├─R─┤                        ├─R─┤

     startVertex ─────────────── endVertex
         │                           │
    ┌────●───────────────────────────●────┐
    │    │                           │    │  ← Quad expanded
    │    │      LINE SEGMENT         │    │    by lineWidth R
    │    │                           │    │
    └────●───────────────────────────●────┘
         │                           │
        cap                         cap
      factor=0.5                  factor=0.5
```

### 5.4 File: `src/rendering/line-material.ts`

**Instanced Attributes (per segment)**:
- `aStartPos`: vec3 - Start vertex position (3D display space)
- `aEndPos`: vec3 - End vertex position (3D display space)
- `aStartColor`: vec3 - Start vertex color (HDR)
- `aEndColor`: vec3 - End vertex color (HDR)
- `aStartWidth`: float - Start half-width (world units)
- `aEndWidth`: float - End half-width (world units)
- `aStartSharpness`: float - Start vertex sharpness
- `aEndSharpness`: float - End vertex sharpness
- `aSegmentLength`: float - Length of segment (for cap factor calculation)
- `aStartClipped`: float - 1.0 if start was clipped (force capFactor=1.0)
- `aEndClipped`: float - 1.0 if end was clipped (force capFactor=1.0)

**Static Geometry (per quad)**:
- 4 vertices with `aQuadCorner`: `(-1,-1), (1,-1), (-1,1), (1,1)`
- 6 indices: `[0, 1, 2, 2, 1, 3]`

**Vertex Shader**:
```glsl
attribute vec2 aQuadCorner;      // Static: quad corner (-1 to 1)
attribute vec3 aStartPos;        // Instanced: segment start
attribute vec3 aEndPos;          // Instanced: segment end
attribute vec3 aStartColor;      // Instanced: start color
attribute vec3 aEndColor;        // Instanced: end color
attribute float aStartWidth;     // Instanced: start width
attribute float aEndWidth;       // Instanced: end width
attribute float aStartSharpness; // Instanced: start sharpness
attribute float aEndSharpness;   // Instanced: end sharpness
attribute float aSegmentLength;  // Instanced: segment length
attribute float aStartClipped;   // Instanced: 1.0 if start was clipped
attribute float aEndClipped;     // Instanced: 1.0 if end was clipped

uniform float uFOV;
uniform vec2 uResolution;

varying vec3 vColor;
varying float vSharpness;        // Per-vertex sharpness (interpolated)
varying float vPerpNorm;         // Signed: -1 at bottom edge, +1 at top edge
varying float vCapFactor;        // 0.5 at endpoints, 1.0 in body

void main() {
    // Position along segment: 0 = start, 1 = end
    float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

    // Interpolate attributes along segment
    vec3 worldPos = mix(aStartPos, aEndPos, t);
    vColor = mix(aStartColor, aEndColor, t);
    float width = mix(aStartWidth, aEndWidth, t);
    vSharpness = mix(aStartSharpness, aEndSharpness, t);

    // Project to clip space
    vec4 clipStart = projectionMatrix * modelViewMatrix * vec4(aStartPos, 1.0);
    vec4 clipEnd = projectionMatrix * modelViewMatrix * vec4(aEndPos, 1.0);
    vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);

    // Screen-space line direction and perpendicular
    vec2 screenStart = clipStart.xy / clipStart.w;
    vec2 screenEnd = clipEnd.xy / clipEnd.w;
    vec2 lineDir = normalize(screenEnd - screenStart);
    vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

    // World-space to pixel conversion (perspective-correct)
    float dist = length((modelViewMatrix * vec4(worldPos, 1.0)).xyz);
    float pixelWidth = width * uResolution.y / (dist * tan(uFOV * 0.5));

    // Perpendicular position: -1 at bottom edge, +1 at top edge
    // GPU interpolates this across the quad, giving 0 at centerline
    vPerpNorm = aQuadCorner.y;
    clipPos.xy += perpendicular * aQuadCorner.y * pixelWidth / uResolution * clipPos.w;

    // Cap factor calculation with clipping awareness
    // Normal cap factor: 0.5 at true endpoints, 1.0 in body
    // Clipped endpoints: force 1.0 (the "real" endpoint is outside the slice)
    float distFromStart = t * aSegmentLength;
    float distFromEnd = (1.0 - t) * aSegmentLength;

    // Base cap factor from distance to nearest endpoint
    float distToNearest = min(distFromStart, distFromEnd);
    float baseCap = (distToNearest >= width) ? 1.0 : 0.5 + 0.5 * (distToNearest / width);

    // Override if the nearest endpoint was clipped
    float nearestIsStart = step(distFromEnd, distFromStart);  // 1 if closer to start
    float nearestClipped = mix(aEndClipped, aStartClipped, nearestIsStart);

    // If nearest endpoint was clipped, use full intensity (1.0)
    vCapFactor = mix(baseCap, 1.0, nearestClipped);

    gl_Position = clipPos;
}
```

**Fragment Shader**:
```glsl
uniform float uHDRMultiplier;
uniform float uOpacity;

varying vec3 vColor;
varying float vSharpness;        // Per-vertex sharpness (interpolated)
varying float vPerpNorm;         // Interpolated: 0 at centerline, ±1 at edges
varying float vCapFactor;        // 0.5 at endpoints, 1.0 in body

void main() {
    // Compute distance from centerline (0 to 1)
    float p = abs(vPerpNorm);

    // Discard pixels outside the line width
    if (p >= 1.0) discard;

    // Parabolic falloff from semicircle kernel convolution
    // Base: (1 - p²) where p = distance from centerline
    // With per-vertex sharpness: (1 - p²)^sharpness
    float perpFalloff = pow(1.0 - p * p, vSharpness);

    // Apply cap factor for correct joint intensity
    float intensity = vCapFactor * perpFalloff;

    // HDR output (gamma correction in post-processing)
    vec3 finalColor = vColor * intensity * uHDRMultiplier;

    gl_FragColor = vec4(finalColor, intensity * uOpacity);
}
```

### 5.5 Cap Factor Visualization

```
Intensity along a segment (at centerline, perpFalloff = 1.0):

capFactor:  0.5 ──────────── 1.0 ──────────── 0.5
              ╱                                 ╲
             ╱                                   ╲
            ╱                                     ╲
           ╱           FULL INTENSITY              ╲
          ╱                                         ╲
─────────●───────────────────────────────────────────●─────────
      endpoint                                    endpoint

When two segments join at a vertex:

Segment 1:  ... ────── 1.0 ────── 0.5
                                  │
Segment 2:                        0.5 ────── 1.0 ────── ...
                                  │
Sum:        ... ────── 1.0 ────── 1.0 ────── 1.0 ────── ...
                                   ↑
                              Seamless join!
```

### 5.6 Comparison: Points vs Lines

| Aspect | Points | Lines |
|--------|--------|-------|
| Kernel | `(1 - r/R)^s` | `√(1 - (r/R)²)` (semicircle) |
| Profile | `(1 - r/R)^s` (same) | `(1 - p²)^s` (parabolic) |
| Blending | Additive | Additive (same!) |
| Joint handling | N/A | Cap factor (0.5 → 1.0) |
| Shader cost | `pow(1-r, s)` | `pow(1-p*p, s)` (similar) |

### 5.7 Tasks

- [ ] Create `src/rendering/line-material.ts`
- [ ] Implement vertex shader with screen-space expansion and cap factor
- [ ] Implement fragment shader with parabolic falloff
- [ ] Create `LineMaterial` class extending `THREE.ShaderMaterial`
- [ ] Add `getLineMaterial()` to `MaterialManager`
- [ ] Test joint rendering with polyline dataset
- [ ] Verify additive blending produces seamless joints

---

## 6. Phase 5: Scene Integration

### 6.1 Goal

Integrate Lines into scene graph alongside Points.

### 6.2 Modifications to `scene-loader.ts`

```typescript
// In loadSceneNodes():
if (node.type === 'points') {
  const points = await this.loadPoints(node, parentLoc);
  if (points) parentThree.add(points);
} else if (node.type === 'lines') {
  const lines = await this.loadLines(node, parentLoc);
  if (lines) parentThree.add(lines);
}

// New method:
private async loadLines(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>
): Promise<THREE.InstancedMesh | null> {
  // 1. Create LinesSpatialIndexLoader
  // 2. Load initial lines data
  // 3. Build instanced geometry
  // 4. Create LineMaterial
  // 5. Create THREE.InstancedMesh
  // 6. Apply transform
  // 7. Set userData
  return linesObject;
}
```

### 6.3 Modifications to `scene-manager.ts`

```typescript
// In updateBoundingBox():
if (object instanceof THREE.InstancedMesh && isLinesUserData(object.userData)) {
  const geometry = object.geometry;
  if (!geometry.boundingBox) {
    geometry.computeBoundingBox();
  }
  if (geometry.boundingBox) {
    const worldBox = geometry.boundingBox.clone();
    worldBox.applyMatrix4(object.matrixWorld);

    // Expand by max line width
    const maxWidth = object.userData.maxWidth ?? 0;
    if (maxWidth > 0) {
      worldBox.expandByScalar(maxWidth);
    }

    box.union(worldBox);
  }
}
```

### 6.4 Tasks

- [ ] Add `loadLines()` method to `SceneLoader`
- [ ] Add lines branch in `loadSceneNodes()`
- [ ] Update `updateView()` to handle lines
- [ ] Update `updateBoundingBox()` in `scene-manager.ts`
- [ ] Add lines handling in dimension update propagation
- [ ] Update geometry disposal for instanced meshes

---

## 7. Phase 6: Testing Strategy

### 7.1 Unit Tests

| Test | File | Description |
|------|------|-------------|
| Type guards | `lines.test.ts` | `isLinesMetadata()`, `isLinesUserData()` |
| Chunk query | `lines-chunk-spatial-index.test.ts` | AABB intersection, tolerance computation |
| Index remapping | `lines-spatial-index-loader.test.ts` | Global→local mapping correctness |
| Vertex range computation | `lines-spatial-index-loader.test.ts` | Chunk-aligned range merging |

### 7.2 E2E Tests (Playwright)

| Test | Description |
|------|-------------|
| Basic lines rendering | Load lines dataset, verify segments visible |
| Variable width | Lines with different widths render correctly |
| Color interpolation | Gradient colors along segments |
| nD slicing | Lines visibility changes with dimension navigation |
| Transform | Lines respect parent group transforms |

### 7.3 Test Data Generation

```python
# Python script to generate test lines dataset
import numpy as np
import luxar

scene = luxar.Scene([
    luxar.Dimension("x", "um", range=(0, 100)),
    luxar.Dimension("y", "um", range=(0, 100)),
    luxar.Dimension("z", "um", range=(0, 100)),
])

# Simple polyline (square loop)
vertices = np.array([
    [0, 0, 0],
    [50, 0, 0],
    [50, 50, 0],
    [0, 50, 0],
], dtype=np.float32)

widths = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
], dtype=np.float32)

# Use scene.add_lines() API (Lines created internally)
scene.add_lines(
    "test_lines",
    vertices=vertices,
    widths=widths,
    colors=colors,
    line_type="loop",
)

luxar.compile(scene, "test_lines_example.zarr")
```

### 7.4 Tasks

- [ ] Create test fixtures in Python
- [ ] Write unit tests for spatial index
- [ ] Write unit tests for loader
- [ ] Write E2E test: basic rendering
- [ ] Write E2E test: nD slicing
- [ ] Add test to CI pipeline

---

## 8. Implementation Risks & Mitigations

### 8.1 Risk: Instanced Geometry Complexity

**Risk**: THREE.js instanced mesh setup is more complex than `THREE.Points`.

**Mitigation**:
- Start with simple static geometry (no dynamic updates)
- Reference existing THREE.js instanced examples
- Consider fallback to `THREE.LineSegments` (1px) for initial prototype

### 8.2 Risk: Index Remapping Performance

**Risk**: Building vertex index maps for large datasets could be slow.

**Mitigation**:
- Use `Map` for O(1) lookup
- Pre-allocate arrays where possible
- Consider WebWorker for heavy computation

### 8.3 Risk: Memory for Duplicate Vertex Data

**Risk**: Instanced attributes store start/end for each segment, duplicating data.

**Mitigation**:
- Accept memory trade-off for rendering performance
- Optimize by sharing vertex buffers where possible
- Document memory requirements

### 8.4 Risk: Shader Precision at Extreme Zoom

**Risk**: Screen-space line expansion may have artifacts at very high zoom.

**Mitigation**:
- Clamp minimum/maximum pixel widths
- Test with various zoom levels
- Add configurable precision settings if needed

---

## 9. Design Decisions (Resolved)

### 9.1 THREE.js Object Type ✓

**Decision**: Use `THREE.InstancedMesh` with custom geometry.

**Rationale**: This is the modern, performant approach for rendering many similar objects. THREE.js's instanced rendering is well-optimized for GPU batching.

**Implementation**: Each segment is one instance with custom instanced attributes (start/end positions, colors, widths, sharpness).

### 9.2 Geometry Update Strategy ✓

**Decision**: Dispose and recreate (same as Points).

**Rationale**: Keep it simple for initial implementation. The Points pattern works well and is easier to reason about. Optimization can be added later if performance profiling shows it's needed.

### 9.3 Line Cross-Section and Caps ✓

**Decision**: Lines behave like "smeared points" - circular cross-section with round caps.

**Rationale**: Conceptually, a line is a continuous series of overlapping points dragged along a path. This means:
- The cross-sectional falloff is based on **perpendicular distance to the centerline**
- Caps are naturally round (like point circles at endpoints)

**Visual Concept**:
```
  Points:          Line ("smeared points"):

    ●              ●────────────────●
   /|\             │    circular    │
  / | \            │   cross-section│
 /  |  \           │                │
    r              └────────────────┘
```

### 9.4 Joint Rendering Model ✓

**Decision**: Use semicircle kernel convolution model with cap factor for seamless additive blending.

**Problem**: With naive intensity functions, joints between adjacent segments (a-b-c) would have double brightness at the shared endpoint b.

**Solution**: Mathematical model based on convolution theory:

1. **Semicircle Kernel**: `f(r) = √(1 - (r/R)²)` instead of point-like `(1-r)^s`
2. **Parabolic Profile**: Convolution yields `intensity = (1 - p²)^sharpness`
3. **Cap Factor**: Endpoints naturally contribute exactly half intensity

**Why it works**:
```
Segment a-b at endpoint b:  0.5 × full intensity
Segment b-c at endpoint b:  0.5 × full intensity
─────────────────────────────────────────────────
Sum (additive blending):    1.0 × full intensity ✓
```

**Cap Factor Calculation**:
```glsl
float distFromEnd = min(t, 1.0 - t) * aSegmentLength;
vCapFactor = (distFromEnd >= width) ? 1.0 : 0.5 + 0.5 * (distFromEnd / width);
```

**Rationale**: This approach:
- Works correctly with additive blending (Luxar's default)
- Produces smooth, seamless joints without visual artifacts
- Is computationally cheap (just `1 - p*p` with optional power)
- Has clean mathematical foundation (convolution associativity)
- Differs from point kernel intentionally to solve joint problem

---

## Appendix: Implementation Checklist

### Phase 1: Types
- [ ] `src/types/lines.ts` created
- [ ] Types exported from `src/types/index.ts`
- [ ] `pnpm typecheck` passes

### Phase 2: Spatial Index
- [ ] `src/data/lines-chunk-spatial-index.ts` created
- [ ] Dual bounds loading works
- [ ] Query algorithm tested

### Phase 3: Data Loader
- [ ] `src/data/lines-spatial-index-loader.ts` created
- [ ] Two-phase loading works
- [ ] Index remapping correct
- [ ] Encoded arrays handled

### Phase 4: Rendering
- [ ] `src/rendering/line-material.ts` created
- [ ] Instanced quad geometry works
- [ ] Shaders compile and render
- [ ] Material registered

### Phase 5: Scene Integration
- [ ] `SceneLoader.loadLines()` implemented
- [ ] Bounding box includes lines
- [ ] Dimension updates work
- [ ] Transforms applied correctly

### Phase 6: Testing
- [ ] Unit tests pass
- [ ] E2E tests pass
- [ ] Test data committed
- [ ] CI green

---

## Changelog

- **v1.3.0** (2025-12-09): nD slicing and instance buffer construction
  - **ADDED**: Section 4.4 - nD Slicing with Endpoint Clipping
    - Line segment clipping algorithm for nD → 3D projection
    - 5 visibility cases: both in, one in/out, crossing, both out
    - Attribute interpolation for clipped segments
    - Cap factor adjustment for clipped endpoints
  - **ADDED**: Section 4.5 - Instance Buffer Construction
    - Complete algorithm for per-vertex → per-segment transformation
    - ProcessedLinesData interface with all GPU instance attributes
    - Clipping integration with buildInstanceBuffers()
  - **ADDED**: aStartClipped, aEndClipped instance attributes
  - Updated vertex shader with clipping-aware cap factor
  - Updated rendering spec to v1.3.3

- **v1.2.1** (2025-12-09): Critical shader bug fixes
  - **BUGFIX**: Fixed vPerpNorm always being 1.0 (was using abs() on quad corners)
    - Vertex shader now passes signed vPerpNorm to fragment shader
    - Fragment shader computes `p = abs(vPerpNorm)` for actual distance
    - GPU interpolation gives 0 at centerline, ±1 at edges
  - **CHANGE**: Per-vertex sharpness (aStartSharpness, aEndSharpness)
  - **CHANGE**: Per-vertex width (aStartWidth, aEndWidth)
  - Fixed test data generation to use `scene.add_lines()` API
  - Updated rendering spec to v1.3.2

- **v1.2.0** (2025-12-09): Semicircle kernel model for joint rendering
  - **MAJOR**: Section 5 completely rewritten with mathematical foundation
  - Added semicircle kernel convolution model: `f(r) = √(1-(r/R)²)`
  - Derived parabolic body profile: `(1 - p²)^sharpness`
  - Added cap factor algorithm for seamless joints with additive blending
  - Endpoints contribute 0.5 intensity → two adjacent segments sum to 1.0
  - Updated vertex shader with cap factor calculation
  - Updated fragment shader with parabolic falloff
  - Added Section 9.4: Joint Rendering Model design decision
  - Added intensity visualization diagrams
  - Added comparison table: Points vs Lines

- **v1.1.0** (2025-12-09): First review pass
  - Added `has_spatial_index` field to `LinesMetadata`
  - Resolved all open questions (Section 9 renamed to "Design Decisions")
  - Decision: Use `THREE.InstancedMesh` (modern, performant)
  - Decision: Dispose/recreate geometry (same as Points)
  - Decision: Round caps with circular cross-section ("smeared points" concept)
  - Verified metadata fields against Python compiler implementation
  - Verified alignment with viewer spec files

- **v1.0.0** (2025-12-09): Initial draft
  - Complete architecture overview
  - All six phases documented
  - Risks and mitigations identified
  - Open questions listed
