# Lines Viewer Implementation Plan

**Version**: 1.1.0
**Created**: 2025-12-09
**Last Updated**: 2025-12-09
**Status**: Draft - Reviewed

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

### 4.4 Tasks

- [ ] Create `src/data/lines-spatial-index-loader.ts`
- [ ] Implement `LinesSpatialIndexLoader` class
- [ ] Implement segment range loading
- [ ] Implement vertex range computation from indices
- [ ] Implement index remapping
- [ ] Handle broadcast arrays (widths, colors)
- [ ] Handle encoded arrays (LUT, quantized)
- [ ] Add monitoring events

---

## 5. Phase 4: Line Material & Rendering

### 5.1 Goal

Create custom shader material for thick lines using instanced quads.

### 5.2 Rendering Approach: Instanced Quads

WebGL `lineWidth` is always 1px on most hardware. We use **instanced quads** instead:

```
Each segment becomes a quad:

     startVertex ─────────────── endVertex
         │                           │
    ┌────┼───────────────────────────┼────┐
    │    │                           │    │  ← Quad expanded
    │    │      LINE SEGMENT         │    │    by lineWidth
    │    │                           │    │
    └────┼───────────────────────────┼────┘
         │                           │
```

### 5.3 File: `src/rendering/line-material.ts`

**Instanced Attributes (per segment)**:
- `aStartPos`: vec3 - Start vertex position
- `aEndPos`: vec3 - End vertex position
- `aStartColor`: vec3 - Start vertex color
- `aEndColor`: vec3 - End vertex color
- `aStartWidth`: float - Start half-width
- `aEndWidth`: float - End half-width
- `aStartSharpness`: float - Start falloff
- `aEndSharpness`: float - End falloff

**Static Geometry (per quad)**:
- 4 vertices with `aQuadCorner`: `(-1,-1), (1,-1), (-1,1), (1,1)`
- 6 indices: `[0, 1, 2, 2, 1, 3]`

**Vertex Shader** (key logic):
```glsl
// Determine position along segment (0 = start, 1 = end)
float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

// Interpolate attributes
vec3 worldPos = mix(aStartPos, aEndPos, t);
vec3 color = mix(aStartColor, aEndColor, t);
float width = mix(aStartWidth, aEndWidth, t);

// Calculate perpendicular in screen space
vec2 lineDir = normalize(screenEnd - screenStart);
vec2 perpendicular = vec2(-lineDir.y, lineDir.x);

// Calculate pixel width from world width
float pixelWidth = 2.0 * width * uResolution.y / (distance * tan(uFOV * 0.5));

// Offset by perpendicular * width
clipPos.xy += perpendicular * aQuadCorner.y * pixelWidth / uResolution * clipPos.w;
```

**Fragment Shader**:
```glsl
void main() {
    float intensity = pow(1.0 - vLinePos, vSharpness);
    vec3 finalColor = vColor * intensity * uHDRMultiplier;
    gl_FragColor = vec4(finalColor, intensity * uOpacity);
}
```

### 5.4 Tasks

- [ ] Create `src/rendering/line-material.ts`
- [ ] Implement vertex shader with screen-space expansion
- [ ] Implement fragment shader with falloff
- [ ] Create `LineMaterial` class extending `THREE.ShaderMaterial`
- [ ] Add `getLineMaterial()` to `MaterialManager`
- [ ] Test with simple line dataset

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
import luxar

scene = luxar.Scene([
    luxar.Dimension("x", "um", range=(0, 100)),
    luxar.Dimension("y", "um", range=(0, 100)),
    luxar.Dimension("z", "um", range=(0, 100)),
])

# Simple polyline
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

lines = luxar.Lines(
    vertices=vertices,
    widths=widths,
    colors=colors,
    line_type="loop",
)

scene.add("test_lines", lines)
luxar.write(scene, "test_lines_example.zarr")
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
- The same `intensity = pow(1.0 - r, sharpness)` formula as Points
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

**Fragment Shader**:
```glsl
// vLinePos is perpendicular distance from centerline, normalized to [0, 1]
// 0 = on centerline, 1 = at edge
float intensity = pow(1.0 - vLinePos, vSharpness);
```

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
