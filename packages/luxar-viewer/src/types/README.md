# Types Package

TypeScript type definitions for high-dimensional data visualization in Luxar. This package provides the fundamental data structures and interfaces for managing nD points, lines, and Gaussian-splat data, dimension metadata, zarr-store schemas, animation state, data-loading monitor contracts, and ambient global typings.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Core Types](#core-types)
- [Dimension Metadata](#dimension-metadata)
- [SimpleDims Interface](#simpledims-interface)
- [Utility Functions](#utility-functions)
- [Geometry Vocabulary and Capabilities](#geometry-vocabulary-and-capabilities)
- [Specialized Group Types](#specialized-group-types)
- [Zarr Types](#zarr-types)
- [Animation Types](#animation-types)
- [Data-Loading Monitor Types](#data-loading-monitor-types)
- [Float16Array Type Declaration](#float16array-type-declaration)
- [Window Debug Surface](#window-debug-surface)
- [Usage Examples](#usage-examples)
- [Type Safety](#type-safety)
- [Best Practices](#best-practices)

## Overview

The types package defines the foundational type system for Luxar's nD visualization capabilities. It provides type-safe interfaces for representing high-dimensional datasets, managing dimension states, and handling coordinate system transformations.

**Core Philosophy**: Provide precise, well-documented types that capture the mathematical and conceptual structure of high-dimensional data visualization while ensuring compile-time safety and runtime reliability.

## Key Features

- **nD Data Structures**: Complete type definitions for high-dimensional points and lines
- **Lines Types**: Type definitions for line segments with nD clipping support
- **GSplats Types**: Type definitions for Gaussian splats with anisotropic covariance
- **Dimension Metadata**: Rich semantic information for dataset dimensions
- **Navigation State**: Type-safe dimension slicing and display configuration
- **Initialization Utilities**: Functions for creating properly structured dimension objects
- **Range Computation**: Mathematical utilities for analyzing dimension bounds
- **Type Safety**: Comprehensive interfaces preventing common visualization errors

## Core Types

### DimensionMetadata Interface

Describes the semantic properties and behavior of a single dimension in an nD dataset:

```typescript
interface DimensionMetadata {
  /** Human-readable name (e.g., "Time", "X", "Channel") */
  name: string;

  /** Physical unit of measurement (e.g., "μm", "s", "nm") */
  unit: string;

  /** Scale factor for converting indices to real-world units */
  scale: number;

  /** Optional min/max bounds in real-world units */
  range?: [number, number];

  /** Whether this dimension should be displayed by default */
  display?: boolean;

  /** Whether values are discrete (integers) vs continuous */
  discrete?: boolean;

  /** Whether dimension wraps around (for angles, periodic states) */
  cyclic?: boolean;

  /** Step size for navigation in this dimension */
  step?: number;

  /** Whether points extend through this dimension (true for spatial dims) */
  spatial?: boolean;

  /** Optional category labels for categorical dimensions */
  categories?: string[];

  /** Optional human-readable description for UI tooltips */
  description?: string;
}
```

**Use Cases**:

- **Scientific Data**: Physical dimensions with units and scales
- **Time Series**: Temporal dimensions with discrete time points
- **Categorical Data**: Discrete dimensions like channels (using `categories` field)
- **Spatial Coordinates**: X, Y, Z with physical units and continuous navigation
- **Periodic Dimensions**: Angles or cyclic time (using `cyclic` field)
- **Spatial vs Categorical**: Use `spatial` field to distinguish physical extent

### SimpleDims Interface

The core state object for nD visualization, tracking current position and display configuration:

```typescript
interface SimpleDims {
  /** Total number of dimensions in the dataset */
  ndim: number;

  /** Current position/slice in each dimension */
  currentStep: number[];

  /** Indices of dimensions currently displayed (max 3) */
  displayed: number[];

  /** Optional metadata for each dimension */
  metadata?: DimensionMetadata[];
}
```

**State Management**:

- **Navigation**: `currentStep` tracks position in nD space
- **Visualization**: `displayed` determines which dimensions are visualized
- **Semantics**: `metadata` provides human-readable context

## Dimension Metadata

### Physical Units and Scales

Dimension metadata enables proper handling of physical coordinate systems:

```typescript
const dimensionMeta: DimensionMetadata = {
  name: 'X',
  unit: 'μm',
  scale: 0.1, // 0.1 μm per array index
  range: [0, 100], // 0-100 μm physical range
  display: true, // Show in 3D visualization
  discrete: false, // Continuous spatial dimension
  step: 1.0, // 1 μm navigation steps
};
```

### Discrete vs Continuous Dimensions

The type system distinguishes between discrete and continuous dimensions for proper navigation behavior:

```typescript
// Continuous dimension (spatial coordinates)
const spatialDim: DimensionMetadata = {
  name: 'Y',
  unit: 'nm',
  scale: 1.0,
  discrete: false, // Supports interpolation and smooth navigation
  step: 10.0, // Fine-grained steps
};

// Discrete dimension (time frames)
const timeDim: DimensionMetadata = {
  name: 'Time',
  unit: 'frame',
  scale: 1.0,
  discrete: true, // Exact integer matching required
  step: 1.0, // Integer frame steps
  spatial: false, // Points exist at discrete time points
};

// Cyclic dimension (angles)
const angleDim: DimensionMetadata = {
  name: 'Theta',
  unit: 'deg',
  scale: 1.0,
  range: [0, 360],
  cyclic: true, // Wraps from 360° to 0°
  discrete: false,
};

// Categorical dimension (with labels)
const channelDim: DimensionMetadata = {
  name: 'Channel',
  unit: '',
  scale: 1.0,
  range: [0, 2],
  discrete: true,
  categories: ['DAPI', 'GFP', 'RFP'], // Human-readable labels
  spatial: false, // Points don't extend through channels
  description: 'Fluorescence imaging channel',
};
```

### Display Configuration

Metadata controls which dimensions are visualized by default:

```typescript
// Typical 5D scientific dataset (T, Z, C, Y, X)
const metadata: DimensionMetadata[] = [
  { name: 'Time', unit: 's', scale: 0.1, discrete: true, display: false },
  { name: 'Z', unit: 'μm', scale: 0.2, discrete: false, display: false },
  { name: 'Channel', unit: '', scale: 1, discrete: true, display: false },
  { name: 'Y', unit: 'μm', scale: 0.1, discrete: false, display: true }, // Spatial
  { name: 'X', unit: 'μm', scale: 0.1, discrete: false, display: true }, // Spatial
];
```

## SimpleDims Interface

### Navigation State

The `SimpleDims` interface tracks the complete state of nD navigation:

```typescript
const dims: SimpleDims = {
  ndim: 5, // 5D dataset
  currentStep: [0, 2.5, 1, 0, 0], // Position in each dimension
  displayed: [3, 4], // Show Y and X dimensions
  metadata: dimensionMetadata, // Semantic information
};
```

**State Interpretation**:

- `currentStep[0] = 0`: Time frame 0
- `currentStep[1] = 2.5`: Z-slice at 2.5 μm
- `currentStep[2] = 1`: Channel 1
- `currentStep[3]` and `currentStep[4]`: Camera-controlled (displayed dimensions)

### Display Dimension Management

The `displayed` array determines which dimensions are visualized:

```typescript
// 2D visualization (common for image data)
displayed: [3, 4]; // Show Y, X

// 3D visualization (common for volume data)
displayed: [2, 3, 4]; // Show Z, Y, X

// 1D visualization (for line plots or profiles)
displayed: [4]; // Show only X
```

**Constraints**:

- Maximum 3 displayed dimensions (hardware limitation)
- Displayed dimensions are camera-controlled
- Non-displayed dimensions use slice navigation

## Utility Functions

### initializeDims()

Creates a properly initialized `SimpleDims` object from dataset properties:

```typescript
export function initializeDims(
  numPoints: number,
  totalElements: number,
  metadata?: DimensionMetadata[]
): SimpleDims;
```

**Initialization Logic**:

1. **Dimension Count**: Calculate `ndim` from array structure
2. **Default Position**: Initialize all dimensions at position 0
3. **Display Selection**: Use metadata preferences or default to last 3 dimensions
4. **Validation**: Ensure consistent array structure

**Usage Example**:

```typescript
// Initialize from points data
const dims = initializeDims(100000, 500000, metadata);
// Result: 5D dataset with 100k points, last 3 dims displayed
```

### getDimensionRanges()

Analyzes actual data values to determine dimension bounds:

```typescript
export function getDimensionRanges(
  positions: Float32Array,
  ndim: number,
  numPoints: number
): Array<[number, number]>;
```

**Analysis Process**:

1. **Scan Data**: Examine all point positions across all dimensions
2. **Min/Max Calculation**: Find actual data bounds for each dimension
3. **Return Ranges**: Array of `[min, max]` tuples for navigation setup

**Usage Example**:

```typescript
const ranges = getDimensionRanges(positions, 5, 100000);
// Result: [[0, 10], [0, 5.2], [0, 2], [-50, 50], [-30, 30]]
//         Time    Z      Chan   Y        X
```

## Points Types

The package includes type definitions for point cloud visualization in `points.ts`:

### PointsMetadata

Metadata for points nodes from zarr `.zattrs`:

```typescript
interface PointsMetadata {
  type: 'points';
  n_points: number; // Total point count
  ndim: number; // Position dimensionality
  max_radius?: number; // Maximum point radius
  has_colors?: boolean; // Whether colors array is present
  has_radii?: boolean; // Whether radii array is present
  has_sharpness?: boolean; // Whether sharpness array is present
  // ... additional properties
}
```

### Type Guards

```typescript
import { isPointsMetadata, isPointsUserData } from '../types/points';

if (isPointsMetadata(attrs)) {
  console.log(`Found ${attrs.n_points} points`);
}
```

See `points.ts` for complete interface definitions including `PointsViewState`, `PointsDataLoader`, and `PointsUserData`. The chunk-bounds index type is the canonical `ChunkSpatialIndex` from `data/loaders/spatial-query/spatial-query-builder.ts`.

## Lines Types

The package includes type definitions for line segment visualization in `lines.ts`:

### LinesMetadata

Metadata for lines nodes from zarr `.zattrs`:

```typescript
interface LinesMetadata {
  type: 'lines';
  original_line_type: LineType; // 'segments' | 'polyline' | 'loop' | 'indexed'
  n_vertices: number; // Total vertex count
  n_segments: number; // Total segment count
  ndim: number; // Position dimensionality
  max_width: number; // Maximum line width in world units
  has_colors: boolean; // Whether colors array is present
  has_sharpness: boolean; // Whether sharpness array is present
  // ... additional properties
}
```

### Type Guards

```typescript
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../types/lines';

if (isLinesMetadata(attrs)) {
  console.log(`Found ${attrs.n_segments} segments`);
}
```

See `lines.ts` for complete interface definitions including `OrderingMetadata`, `SegmentRange`, `LoadedLinesData`, `ProcessedLinesData`, `ClippedSegment`, and `LinesViewState`. The chunk-bounds index type is the canonical `ChunkSpatialIndex` from `data/loaders/spatial-query/spatial-query-builder.ts`; lines additionally carry the vertex-side bounds inside the loader.

## GSplats Types

The package includes complete type definitions for Gaussian Splats visualization in `gsplats.ts`:

### GSplatsMetadata

Metadata for gsplats nodes from zarr `.zattrs`:

```typescript
interface GSplatsMetadata {
  type: 'gsplats';
  n_splats: number; // Total splat count
  ndim: number; // Position dimensionality
  has_colors: boolean; // Whether colors array is present
  chunk_size: number; // Elements per chunk
  amplitude_range: ValueRange; // Amplitude value range
  center_bounds: CoordinateBounds; // Center coordinate bounds
  ordering: 'morton' | 'hilbert' | 'none'; // Spatial ordering method
  extend_to_all?: string[]; // Dimensions to extend visibility across
  n_lods?: number; // Number of LOD levels (v1.1 multi-LOD format)
  truncation_radius?: number; // Gaussian truncation radius (defaults to GSPLAT_DEFAULT_TRUNCATION_RADIUS)
  // ... additional properties
}
```

### LoadedGSplatsData

Raw gsplats data loaded from zarr before nD projection:

```typescript
interface LoadedGSplatsData {
  positions: Float32Array; // Splat positions (N * ndim)
  amplitudes: Float32Array; // Splat amplitudes (N,)
  choleskyFactors: Float32Array; // Packed Cholesky (N * k) where k = ndim*(ndim+1)/2
  colors: Float32Array | Uint8Array | Uint16Array | null; // RGB colors
  splatCount: number;
  ndim: number;
}
```

### Type Guards

```typescript
import {
  isGSplatsMetadata,
  isGSplatsUserData,
  choleskyPackedSize,
  CHOLESKY_SIZES,
} from '../types/gsplats';

// Check if zarr attrs is for gsplats
if (isGSplatsMetadata(attrs)) {
  console.log(`Found ${attrs.n_splats} splats`);
}

// Check if THREE.Object3D is gsplats
if (isGSplatsUserData(mesh.userData)) {
  console.log(`Visible: ${mesh.userData.visibleSplatCount}`);
}

// Compute packed Cholesky factor count for a given dimensionality
const packed = choleskyPackedSize(3); // 6 = 3*(3+1)/2

// Pre-computed sizes for common dimensions (2D-4D), keyed by dimensionality
console.log(CHOLESKY_SIZES); // { '2D': 3, '3D': 6, '4D': 10 }
```

See `gsplats.ts` for complete interface definitions including `ProcessedGSplatsData`, `GSplatsViewState`, and `GSplatsUserData`. The chunk-bounds index type is the canonical `ChunkSpatialIndex` from `data/loaders/spatial-query/spatial-query-builder.ts`.

## Geometry Vocabulary and Capabilities

`geometry-capabilities.ts` answers two different questions that look alike in code:

1. **"Is this a geometry leaf?"** -- `isGeometryType(value)`, a type guard over the
   contract-generated `GEOMETRY_TYPES`. Use it wherever the answer should follow the
   vocabulary automatically (world-bounds walks, layer fan-out, leaf detection).
2. **"Does this geometry type support feature X?"** -- `supportsLod`,
   `supportsPartition`, `isPooledGeometry`, `isDepthSortable`, all reading the
   `GEOMETRY_CAPABILITIES` table.

The distinction matters because a _subset_ spelled out as
`x === 'points' || x === 'lines' || x === 'gsplats'` is indistinguishable from the
vocabulary at a glance, so a well-meaning sweep silently enables a feature for a
type that cannot support it.

```typescript
import { isGeometryType, supportsLod, GEOMETRY_CAPABILITIES } from '../types/geometry-capabilities';

isGeometryType(node.type); // vocabulary membership
supportsLod(child.type); // may this be a kind=lod level?
```

`GEOMETRY_CAPABILITIES` is a `Record<GeometryTypeName, GeometryCapabilities>`, so
**adding a geometry type to `format-contract/contract.yaml` fails to compile here**
until its capabilities are declared -- and the runtime consumers then behave
correctly with no further edits.

A few interface fields (`LODGroupMetadata.display_type`,
`PartitionGroupMetadata.display_type`, `SceneGraphNode.displayType`,
`PooledBuffer.type`) still spell a subset out by hand, because a union cannot be
derived from a value without more machinery than it is worth. Each carries a comment
pointing at the capability it corresponds to; do not widen them to
`GeometryTypeName`.

Not re-exported from `index.ts`; import directly from `../types/geometry-capabilities`.

## Specialized Group Types

Two `Group` node variants carry a `kind` discriminant and dedicated metadata/guards so the viewer can treat a single logical layer specially. Both are authored on a group's `.zattrs` and inherit the standard compositing attrs (`opacity`, `gamma`, `intensity`, `offset`, `blending_mode`, `layer`, `visible`), `transform`, `nd_transform`, and `extend_to_all`.

### LOD Groups (`lod-group.ts`)

A `Group` whose `kind === 'lod'` selects **one** of N alternative children at runtime based on the projected bbox diagonal in pixels and each child's `coverage_fraction` threshold — a dimensionless, viewport-relative fraction in `[0, 1]` that the viewer multiplies by the current viewport diagonal to get the pixel comparison. Children are arbitrary geometry subtrees (points / lines / gsplats / nested specialized groups).

```typescript
import { type LODGroupMetadata, type LODGroupSelectorMode } from '../types/lod-group';

// The loader matches the shape inline (attrs.type === 'group' && attrs.kind === 'lod');
// attrs.selector is 'coverage'; attrs.display_type is the user-facing label;
// attrs.default_level seeds the manual-override widget (0-based, coarsest-first).
```

- **`LODGroupMetadata`** -- `{ type: 'group', kind: 'lod', selector: 'coverage', display_type?, default_level?, ... }`. Each child carries a `coverage_fraction` threshold (a `number` in `[0, 1]`), strictly monotonic increasing coarsest→finest (coarsest 0.0, finest 1.0); the selector picks the finest child whose threshold is satisfied.
- **`LODGroupSelectorMode`** -- runtime selector state: `'auto'` (view-driven, the default) or `{ lockLevel: number }` (user-locked child index, 0-based coarsest→finest).

### Partition Groups (`partition-group.ts`)

A `Group` whose `kind === 'partition'` is a compile-time decomposition of one large geometry node (10M+ elements) into multiple smaller children for per-child frustum culling and per-child LOD. Unlike `kind === 'lod'`, **all children render simultaneously** (no per-frame selector), and all must resolve to the same `display_type` (homogeneity is mandatory).

- **`PartitionGroupMetadata`** -- `{ type: 'group', kind: 'partition', display_type, max_elements, position_bounds?, ... }`. `position_bounds` is the union of the children's bounds so picking / framing / the scene-bounds cache can treat the layer as one entity. The loader matches the shape inline (`attrs.type === 'group' && attrs.kind === 'partition'`).

Neither module is re-exported from `index.ts`; import directly from `../types/lod-group` / `../types/partition-group`.

## Zarr Types

Type definitions in `zarr.ts` for Zarr store attributes and scene graph metadata. These provide proper typing for Zarr `.zattrs` data, eliminating `as any` assertions.

### Key Types

- **`ZarrViewerConfig`** -- Viewer configuration stored in the zarr root `.zattrs` by the Python API. All fields are optional and use `snake_case`. Covers camera, tone mapping, bloom, controls, post-processing, UI panel visibility, dimension navigation state, and animation state. This is also the format exported by Ctrl+Shift+S in the viewer, enabling round-trip Python-to-viewer-to-Python workflows.
- **`ZarrSceneAttrs`** -- Root scene group attributes: format version, scene dimensions, units, position bounds, and an optional `viewer_config`.
- **`ZarrNodeAttrs`** -- Per-node attributes in the scene graph: node type, 4x4 transform, nD transform, rendering properties (opacity, gamma, blending mode, etc.), position bounds, and `extend_to_all`.
- **`SceneDimensionAttrs`** -- Scene-level dimension array mirroring Python's `luxar.core.Dimension` class.
- **`PositionBounds`** -- nD bounding box with `min` and `max` arrays (one entry per dimension).

### nD Transform Types

Per-dimension transforms applied to non-displayed dimensions (see `docs/guides/specs/ND_TRANSFORMS_SPEC.md`):

- **`NdTransformAffine`** -- Affine transform for continuous/discrete dimensions: `{ scale?, offset? }`. Applied as `effective = scale * value + offset`.
- **`NdTransformPermutation`** -- Permutation for categorical dimensions: `{ permutation: number[] }`. Maps old category index to new index.
- **`NdTransformEntry`** -- Union of `NdTransformAffine | NdTransformPermutation`.
- **`NdTransformMap`** -- `Record<string, NdTransformEntry>` mapping dimension name to its transform.

### Type Guards

```typescript
import {
  hasContentsMethod,
  hasTransform,
  hasNdTransform,
  hasSceneDimensions,
  isPermutation,
} from '../types/zarr';
```

- `hasContentsMethod(store)` -- checks if a zarr store supports `contents()`.
- `hasTransform(attrs)` -- checks for a 16-element transform array.
- `hasNdTransform(attrs)` -- checks for an `nd_transform` object.
- `hasSceneDimensions(attrs)` -- checks for `scene_dimensions` with a `dimensions` array.
- `isPermutation(entry)` -- distinguishes permutation entries from affine entries.

## Animation Types

Type definitions in `animation.ts` for FPS-based dimension animation. These types are **not re-exported** from `index.ts` -- import them directly from `../types/animation`.

```typescript
import type {
  LoopMode,
  AnimationDirection,
  DimensionAnimationState,
  DimensionAnimationEvents,
} from '../types/animation';
```

- **`LoopMode`** -- `'once' | 'loop' | 'bounce'`. Controls behavior when animation reaches a dimension boundary.
- **`AnimationDirection`** -- `'forward' | 'backward'`. Current playback direction.
- **`DimensionAnimationState`** -- Full state for a single dimension's animation: `isPlaying`, `targetFPS`, `loopMode`, `direction`, FPS measurement fields (`actualFPS`, `frameCount`, timestamps).
- **`DimensionAnimationEvents`** -- Event map emitted by `DimensionAnimationManager`: `play`, `pause`, `complete`, `speedChange`, `loopModeChange`, `directionChange`, `fpsWarning`.

## Data-Loading Monitor Types

`data-monitor-types.ts` defines the event-driven contracts shared by the loader layer and the data-loading-monitor UI. These types are deliberately housed in `types/` so the data layer (`cache-setup.ts`, spatial-index loaders) and the UI layer (`ui/data-loading-monitor/*`) can both reference them without crossing into each other's modules.

Key exports:

- **`MonitorEvent` / `MonitorEventType` / `MonitorEventListener`** -- the event stream emitted by loaders: `query`, `load`, `cache-hit`, `cache-miss`, `evict`, `error`, `prefetch`.
- **`LoaderType`** -- `'point-spatial-index' | 'lines-spatial-index' | 'gsplats-spatial-index'`.
- **`LoaderMonitor`** -- the `addEventListener` / `removeEventListener` / `getMetrics` / `getActiveQueries` surface implemented by spatial-index loaders.
- **`LoaderMetrics`** -- per-loader counters (queries, loads, evictions), throughput (elements/bytes loaded — geometry-neutral: points / vertices / splats), avg query/load times, memory usage, plus optional `spatialIndex` (`SpatialIndexMetrics`, the chunk-index telemetry all three facades attach) and `optimization` (`OptimizationMetrics`) breakdowns. `OptimizationMetrics` surfaces accumulator pooling, worker offload, WASM acceleration, and GPU buffer pool stats.
- **`QueryInfo`** -- shape of an in-flight or recent query (id, status, ranges, fromCache).
- **`GlobalStats`** -- aggregate dashboard numbers across all loaders (totals for points/segments/splats, queries-per-second, `recommendations`; per-tier cache hit rates live in `CacheMetrics`, not here).
- **`Recommendation`** -- `{ id, severity, category, title, message, ... }` performance recommendations surfaced in the monitor UI.
- **`CacheMetrics`** + **`CacheTelemetryState`** + **`CacheStatusBadge`** -- the cache tab's data shape. `CacheTelemetryState` distinguishes the four operational states (`enabled`, `disabled-no-cache`, `disabled-config`, `not-wired`); `CacheStatusBadge` enumerates the chips rendered next to it (`cache-enabled`, `no-cache`, `opfs-unavailable`, `quota-constrained`, `unvalidated-external-dataset`, `cache-errors-detected`, `provider-missing`). L0/L1/L2 breakdowns are optional and only present when a `CacheStatsProvider` is wired.
- **`CacheStatsProvider`** -- the interface `MultiLevelCachingStore` implements so the monitor can read stats without depending on the cache implementation directly.
- **`SceneGraphNode` / `SceneGraphNodeType` / `SceneGraphState`** -- simplified scene-graph view used by the monitor's tree panel (separate from the runtime `SceneNode` in `data/data-loader-types.ts`). `SceneGraphState`'s per-type tallies (`nodesByType` / `totalByType` / `visibleByType`) are `GeometryCounters`, so they extend with the vocabulary rather than needing a field per type per counter.
- **`GeometryCounters`** -- `Record<GeometryTypeName, number>`, the monitor's per-geometry-type aggregation shape. The element _nouns_ (points / segments / splats) deliberately stay on `GlobalStats`' named fields: each is rendered with its own label and DOM id, so they are presentation, not aggregation.
- **`MemoryMetrics`** + **`GPUPoolStats`** + **`GPUPoolTypeStats`** + **`AccumulatorStats`** -- cross-layer memory contracts used by the Memory tab. `GPUPoolStats.byType` is broken down per geometry kind (`points`, `lines`, `gsplats`).
- **`MonitorConfig`** / **`MonitorUIState`** -- display configuration and runtime UI state (active tab, time range, expand state).
- **`TimelinePoint`** / **`GridCellState`** -- `@internal` reserved extension shapes; no current consumer.

These types are **not** re-exported from `index.ts` -- import them directly from `../types/data-monitor-types`.

## Float16Array Type Declaration

The file `float16array.d.ts` provides TypeScript type declarations for `Float16Array`, which is supported in modern browsers (Chrome 122+, Firefox 127+, Safari 17+) but lacks built-in TypeScript definitions. This allows the viewer to handle Float16-encoded zarr arrays without type errors. `Float16Array` is also a member of the `PositionArray` and `ScalarArray` unions in `points.ts`.

## Window Debug Surface

`window.d.ts` augments the global `Window` interface with the optional `__luxarDebug` namespace. The viewer attaches this object **only** when debug mode is active (`?debug` URL parameter or persisted `luxar.debug` localStorage flag); production builds without those flags leave it undefined.

Population happens in two stages:

1. `bootstrapStandalone()` (or any caller that opts in) attaches `app`, `consoleInterceptor`, and `version` before `init()` runs.
2. `LuxarApp.setupDebugInterface()` extends with runtime references after `init()` completes and flips `runtimeReady` to `true`.

Notable members on `window.__luxarDebug`:

- Runtime objects: `app`, `scene`, `camera`, `renderer` (typed as `unknown` -- may be `THREE.WebGLRenderer` or `WebGPURenderer` when opted in via `?renderer=webgpu`), `controls`, `postProcessing`, `animationController`, `inputHandler`, `renderingControls`, `recordingPanel`, `sceneDimsManager`.
- Helpers for interactive debugging and Playwright agents: `getState()`, `renderOnce()`, `getSceneLoader()`, `getPickingSystem()`, `getOverlayManager()`, `showError(message)`.
- `cache` -- `getStats()`, `listDatasets()`, `clearL0/L1/L2/All()`.
- `workers` -- `getQueueDepth()`, `getStats()` for worker-pool diagnostics.
- `lastExportedState` -- last viewer state exported via the keyboard shortcut handler.
- `injectSyntheticScene(spec)` -- debug/perf-bench-only synthetic Points/Lines/GSplats scene injection (only attached to `__luxarDebug` under `?debug`, and its generator chunk is never loaded otherwise); resolves to the discriminated union `{type, elementCount, <per-type count>, mesh}` (capacity-clamped `elementCount` plus a per-type count alias). The full spec/return signature is documented in and kept in sync with `window.d.ts`.
- `getLodLoadStats()` / `resetLodLoadStats()` -- per-stage timing snapshot (`lazy:loadGSplats` / `lazy:process` / `lazy:commit` / `lazy:release`) for lazy LOD level loads triggered by the per-frame selector, which the UpdateProfiler does not see. Debug-only.

Return shapes for the helpers are intentionally dynamic and typed as `unknown` so callers must narrow before reading.

## Usage Examples

### Scientific Dataset Setup

```typescript
import { initializeDims, DimensionMetadata } from '../types/dims';

// 4D microscopy data: Time, Z, Y, X
const metadata: DimensionMetadata[] = [
  {
    name: 'Time',
    unit: 's',
    scale: 0.5, // 0.5 seconds per frame
    discrete: true, // Discrete time points
    display: false, // Navigate via slicing
    step: 1,
  },
  {
    name: 'Z',
    unit: 'μm',
    scale: 0.2, // 200 nm Z-steps
    discrete: false, // Continuous space
    display: true, // Show as 3D depth
    step: 0.2,
  },
  {
    name: 'Y',
    unit: 'μm',
    scale: 0.065, // 65 nm pixels
    discrete: false,
    display: true,
    step: 0.065,
  },
  {
    name: 'X',
    unit: 'μm',
    scale: 0.065,
    discrete: false,
    display: true,
    step: 0.065,
  },
];

// Initialize dimension state
const dims = initializeDims(numPoints, totalElements, metadata);
```

### Multi-condition Experiment

```typescript
// 6D dataset: Condition, Time, Channel, Z, Y, X
const experimentMetadata: DimensionMetadata[] = [
  {
    name: 'Condition',
    unit: '',
    scale: 1,
    discrete: true, // Control vs Treatment
    display: false,
    step: 1,
  },
  {
    name: 'Timepoint',
    unit: 'h',
    scale: 2, // 2-hour intervals
    discrete: true,
    display: false,
    step: 1,
  },
  {
    name: 'Channel',
    unit: '',
    scale: 1, // Fluorescence channels
    discrete: true,
    display: false,
    step: 1,
  },
  // ... spatial dimensions (Z, Y, X) with continuous navigation
];
```

### Navigation Through Dimensions

**Note**: Navigation utilities are implemented in the `input` package. See `../input/README.md` for complete navigation API documentation.

```typescript
// Navigation is handled by the InputHandler class in the input package
import { InputHandler } from '../input/input-handler';

// InputHandler provides dimension navigation methods:
// - navigateDimension(dimIndex, direction): Navigate forward/backward
// - setDimensionPosition(dimIndex, position): Jump to specific position
// - getNonDisplayedDimensions(): Get list of navigable dimensions

// Example: Create input handler for dimension navigation
const inputHandler = new InputHandler(sceneManager, config);

// Navigate forward in time dimension (typically first non-displayed dimension)
const success = inputHandler.navigateDimension(0, 1);

// Check which dimensions can be keyboard-navigated
const navigableDims = inputHandler.getNonDisplayedDimensions();
// Returns indices of dimensions not currently displayed (e.g., [0, 1, 4])
```

## Type Safety

### Compile-time Validation

TypeScript interfaces prevent common errors at compile time:

```typescript
// ✅ Type-safe dimension access
const currentZ = dims.currentStep[zDimension];

// ❌ Compile error: dimension index must be number
const invalidDim = dims.currentStep['z'];

// ✅ Type-safe metadata access
const zUnit = dims.metadata?.[2]?.unit ?? 'unknown';

// ❌ Compile error: display property is boolean
dims.metadata[0].display = 'yes'; // Should be boolean
```

### Runtime Validation

Utility functions include runtime validation for data consistency:

```typescript
// Validates array structure consistency
const dims = initializeDims(numPoints, totalElements, metadata);
// Throws: "Invalid positions array: 500001 elements for 100000 points"
// if totalElements is not evenly divisible by numPoints
```

### Optional Properties

Metadata properties are carefully designed with optional fields:

```typescript
interface DimensionMetadata {
  name: string; // Required
  unit: string; // Required
  scale: number; // Required
  range?: [number, number]; // Optional: calculated from data if missing
  display?: boolean; // Optional: defaults to false
  discrete?: boolean; // Optional: defaults to false (continuous)
  step?: number; // Optional: calculated from range if missing
}
```

## Best Practices

### Dimension Ordering

Follow consistent conventions for dimension ordering:

```typescript
// ✅ Good: Standard scientific convention (T, Z, C, Y, X)
const standardOrder = ['Time', 'Z', 'Channel', 'Y', 'X'];

// ✅ Good: Physics convention (T, X, Y, Z)
const physicsOrder = ['Time', 'X', 'Y', 'Z'];

// ❌ Avoid: Inconsistent or unclear ordering
const confusingOrder = ['X', 'Time', 'Y', 'Channel', 'Z'];
```

### Metadata Completeness

Provide complete metadata for better user experience:

```typescript
// ✅ Good: Complete semantic information
const completeMetadata: DimensionMetadata = {
  name: 'Time', // Clear, descriptive name
  unit: 'min', // Standard unit with clear meaning
  scale: 2.5, // Explicit conversion factor
  discrete: true, // Navigation behavior specification
  display: false, // Clear display intent
  step: 1, // Appropriate step size
  range: [0, 120], // Expected data bounds
};

// ❌ Avoid: Minimal or unclear metadata
const poorMetadata = {
  name: 'D0', // Generic name
  unit: '', // Missing unit information
  scale: 1, // No additional context
};
```

### Type Guards and Validation

Use type guards for safe metadata access:

```typescript
// ✅ Good: Safe metadata access
function getDimensionName(dims: SimpleDims, index: number): string {
  if (index < 0 || index >= dims.ndim) {
    return `D${index}`;
  }

  return dims.metadata?.[index]?.name ?? `D${index}`;
}

// ✅ Good: Validate dimension state consistency
function validateDims(dims: SimpleDims): boolean {
  // Check displayed dimensions are within bounds
  return (
    dims.displayed.every((d) => d >= 0 && d < dims.ndim) &&
    dims.currentStep.length === dims.ndim &&
    dims.displayed.length <= 3
  );
}
```

### Performance Considerations

```typescript
// ✅ Good: Cache dimension lookups
const displayedSet = new Set(dims.displayed);
const isDisplayed = displayedSet.has(dimIndex); // O(1)

// ❌ Avoid: Linear search for each check
const isDisplayed = dims.displayed.includes(dimIndex); // O(n)

// ✅ Good: Reuse range calculations
const ranges = getDimensionRanges(positions, dims.ndim, numPoints);
// Use ranges for multiple navigation operations
```

The types package provides the type-safe foundation for all nD visualization operations in Luxar, ensuring data consistency and enabling rich semantic interpretation of high-dimensional datasets.

---

## File Index

- `index.ts` -- Barrel re-exporting the public types and helpers (`DimensionMetadata`, `SimpleDims`, `initializeDims`, `getDimensionRanges`, the Points/Lines/GSplats interface families and their type guards, `ZarrSceneAttrs`/`ZarrNodeAttrs`, `hasContentsMethod`).
- `dims.ts` -- `DimensionMetadata`, `SimpleDims`, `initializeDims()`, `getDimensionRanges()`.
- `points.ts` -- `EffectiveRadiusConfig`, `PointsMetadata`, `LoadedPointsData`, `PointRange`, `PointsViewState`, `PointsDataLoader`, `PointsUserData`, `PositionArray` / `ColorArray` / `ScalarArray` aliases, and `isPointsMetadata` / `isPointsUserData` guards.
- `lines.ts` -- `LineType`, `LinesMetadata`, `OrderingMetadata`, `SegmentRange`, `LoadedLinesData`, `ProcessedLinesData`, `ClippedSegment`, `LinesDataLoader`, `LinesViewState`, `LinesUserData`, and `isLinesMetadata` / `isLinesUserData` / `isValidLineType` guards.
- `gsplats.ts` -- `GSplatsMetadata`, `ValueRange`, `CoordinateBounds`, `SplatRange`, `LoadedGSplatsData`, `ProcessedGSplatsData`, `GSplatsDataLoader`, `GSplatsViewState`, `GSplatsUserData`, `isGSplatsMetadata` / `isGSplatsUserData` guards, plus `choleskyPackedSize()` and the `CHOLESKY_SIZES` constant.
- `zarr.ts` -- `ZarrSceneAttrs`, `ZarrNodeAttrs`, `ZarrViewerConfig`, `SceneDimensionAttrs`, `PositionBounds`, `Matrix4x4`, nD-transform types (`NdTransformAffine`, `NdTransformPermutation`, `NdTransformEntry`, `NdTransformMap`), `ZarrStoreWithContents`, and the `hasContentsMethod` / `hasTransform` / `hasNdTransform` / `hasSceneDimensions` / `isPermutation` / `isPointsNode` guards.
- `format-contract.ts` -- Generated cross-language format-contract constants (the TypeScript consumer half of the Python <-> TypeScript contract; single source of truth is `format-contract/contract.yaml`, regenerate via `make gen-contract`): `SCENE_FORMAT_VERSION` / `SUPPORTED_SCENE_VERSIONS`, `GSPLATS_FORMAT_VERSION` / `SUPPORTED_GSPLATS_FORMAT_VERSIONS`, `FORMAT_TYPE_GSPLATS`, `NODE_TYPES`, `NODE_KINDS`, `ENCODING_NAMES`, `ATTR_KEYS`, `ARRAY_NAMES`, and their corresponding union types (`SceneFormatVersion`, `GSplatsFormatVersion`, `NodeTypeName`, `NodeKind`, `EncodingName`, ...).
- `lod-group.ts` -- `LODGroupMetadata` and `LODGroupSelectorMode` (the `kind === 'lod'` specialized group; loader matches the shape inline).
- `partition-group.ts` -- `PartitionGroupMetadata` (the `kind === 'partition'` specialized group; loader matches the shape inline).
- `animation.ts` -- `LoopMode`, `AnimationDirection`, `DimensionAnimationState`, `DimensionAnimationEvents`.
- `data-monitor-types.ts` -- Data-loading monitor contracts (`MonitorEvent`, `LoaderMetrics`, `CacheMetrics`, `CacheTelemetryState`, `CacheStatusBadge`, `CacheStatsProvider`, `SceneGraphNode`, `MemoryMetrics`, `GPUPoolStats`, ...).
- `float16array.d.ts` -- Ambient `Float16Array` typing.
- `window.d.ts` -- Ambient `window.__luxarDebug` augmentation.
