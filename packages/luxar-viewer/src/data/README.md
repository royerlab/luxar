# Luxar Data Package

> High-performance Zarr data loading and nD slicing for scientific visualization

## Overview

The Luxar Data package provides the critical data loading infrastructure for visualizing massive nD datasets. It handles Zarr store access, hierarchical scene loading, dimension metadata extraction, and intelligent slicing operations for multi-dimensional data navigation.

### Key Features

- **Chunk-Based Spatial Index**: Efficient spatial queries using Morton/Hilbert-ordered chunks (optional for 3D datasets)
  - **Dimension-aware indexing**: Step-aware tolerance for discrete dimensions (time, channels)
  - **Smart chunk sizing**: Optimized for time-series data (~7× faster for animated Lines)
- **Zarr-Native Loading**: Direct integration with Zarr stores for chunked data access
- **nD Data Support**: Handle arbitrary-dimensional points with automatic slicing
- **Hierarchical Scenes**: Load nested scene structures with inheritance
- **Smart Slicing**: Radius-based hypersphere intersection for smooth navigation
- **Dimension Extension**: Explicit visibility extension of point groups across non-displayed dimensions
- **Directory Navigation**: Multi-strategy server navigation (WebDAV, S3, nginx)
- **GPU Optimization**: Automatic data format conversion for WebGL compatibility
- **Efficient Caching**: Chunk-based caching with LRU eviction

### Package Architecture

```
data/
├── zarr-loader.ts                 # Main API entry point for loading scenes
├── zarr.ts                        # Zarrita facade (only module allowed to import zarrita directly)
├── codecs/
│   └── luxar-delta.ts             # `luxar_delta_v1` zarr filter (columnar delta+zigzag on
│                                  #   quantized codes; registered by zarr.ts as
│                                  #   `numcodecs.luxar_delta_v1`; Python twin in
│                                  #   luxar/encoding/_encoders/delta_codec.py)
├── zip/                           # HTTP-range access to .zarr.zip stores (cached like any other)
│   ├── entries.ts                 # Validates flat/nested archive layouts and normalizes keys
│   ├── range-reader.ts            # Strict 206/Content-Range reader; retains + stitches the directory read
│   └── store.ts                   # Lazily opens the archive; memoizes the directory only on success
├── scene-loader.ts                # Orchestrates hierarchical scene loading (spans all geometries)
├── scene-identity-watchdog.ts     # Conditionally re-probes root attrs (zarr.json/.zattrs)
│                                  #   every 15 s locally / 120 s remotely, plus tab focus,
│                                  #   and raises the scene-identity banner when the ?src=
│                                  #   address starts serving a DIFFERENT scene (a demo/dev
│                                  #   server died and another took its port) or goes
│                                  #   unreachable; started per dataset by load-scene.ts,
│                                  #   disposed by SceneLoader.dispose
├── scene-loader-manager.ts        # Singleton manager for SceneLoader instances
├── scene-loader-monitor-port.ts   # Port interface bridging SceneLoader → DataLoadingMonitor
├── data-loader-types.ts           # TypeScript interfaces and types
├── view-state-manager.ts          # Centralized ViewState initialization and validation
├── attrs-composer.ts              # Composes per-layer attributes along the scene graph
│                                  #   (used by SceneLoader and UI panels)
├── dims-to-view-state.ts          # SimpleDims → ViewState (zarr-loader entry helper)
├── index.ts                       # Package exports
├── README.md                      # This documentation
│
├── points/                        # Points geometry — facade + decomposition + math
│   ├── points-spatial-index-loader.ts # Loads points using chunk-based spatial queries
│   ├── points-progressive-loader.ts   # Composite-pattern multi-LOD facade
│   ├── chunk-index-loader.ts          # `chunk_bounds` zarr probe + registerBounds
│   ├── handler.ts                     # Per-kind load + stage wiring for updateView (`loadAndStage` + its handler ctx)
│   ├── lod-refinement.ts              # Sequential LOD-tier refinement helpers
│   ├── projection.ts                  # nD → 3D projection (main-thread, WASM-accelerated; single impl)
│   └── effective-radius-calculator.ts # query-tolerance + should-apply helpers (+ TS-ref effective-radii)
│
├── lines/                         # Lines geometry — facade + chunk-index probe + projection math
│   ├── lines-spatial-index-loader.ts  # Loads lines with nD clipping + attribute interpolation
│   ├── lines-progressive-loader.ts    # Composite-pattern multi-LOD facade
│   ├── chunk-index-loader.ts          # Dual-bounds zarr probe + computeVertexRangesFromIndices
│   ├── handler.ts                     # Per-kind load + stage wiring for updateView (`loadAndStage` + its handler ctx)
│   ├── lod-refinement.ts              # Sequential LOD-tier refinement helpers
│   └── projection.ts                  # createEmptyLinesData only (nD→3D math lives in workers/data-worker/projection/lines.ts)
│
├── gsplats/                       # GSplats geometry — facade + chunk-index probe + processor + multi-LOD wrapper
│   ├── gsplats-spatial-index-loader.ts  # Loads Gaussian splats with nD visibility
│   ├── chunk-index-loader.ts            # `chunk_bounds` zarr probe + array-bounds prefetcher registration
│   ├── gsplats-progressive-loader.ts    # Composite-pattern multi-LOD facade (loads N LODs sequentially)
│   ├── handler.ts                       # Per-kind load + stage wiring for updateView (`loadAndStage` + its handler ctx)
│   ├── lod-refinement.ts                # Sequential LOD-tier refinement helpers
│   └── projection.ts                    # createEmptyGSplatsData only (nD→3D math lives in workers/data-worker/projection/gsplats.ts)
│
├── mesh/                          # Mesh geometry — whole-node loader (no spatial index by design; a mesh may carry an additive ladder that REVEALS a growing surface rather than coarsening it — substitutive levels are kind=lod siblings; MESH_NODE_SPEC §7/§9)
│   ├── handler.ts                        # Per-kind load + stage wiring for updateView (`loadAndStage` + its handler ctx)
│   ├── lod-refinement.ts                 # Drains the reveal ladder a level per pass after level 0 commits (wraps scene-loader/progressive/refinement.ts)
│   ├── mesh-progressive-loader.ts        # Reveal-ladder composite — wraps one whole-node loader per `additive_<i>` level (view-independent, no SliceCache)
│   ├── mesh-whole-node-loader.ts         # Whole-node loader — loads the entire mesh at once (`ordering: 'none'`, no chunk index)
│   ├── preflight.ts                      # Metadata-only Stage 1 (shapes, dtypes, encodings, attribute presence)
│   ├── projection.ts                     # nD → 3D: extract_3d_positions + whole-triangle nD cull + winding post-pass
│   ├── validate.ts                       # Load-time mesh attribute/topology validation
│   └── README.md                         # Mesh loader documentation
│
├── transforms/                    # nD transform helpers
│   └── nd-transform.ts            # Inverse-query for non-displayed dimensions
│                                  #   Given a world-space query (slicePosition + tolerance) and a
│                                  #   composed nd_transform, produces the equivalent local-space query.
│                                  #   Avoids transforming geometry data — all loader internals unchanged.
│
├── accumulators/                  # Per-geometry zero-allocation buffer pooling
│   ├── types.ts                   # Shared DataAccumulator<T> + AccumulatorStats
│   ├── points.ts                  # LoadedPointsDataAccumulator
│   ├── lines.ts                   # LinesDataAccumulator (flat per-vertex buffers)
│   └── gsplats.ts                 # GSplatsDataAccumulator
│
├── array-decoder/                 # Decodes Python luxar.encoding arrays
│   ├── decoder.ts                 # ArrayDecoder priority-dispatch body
│   ├── ref-registry.ts            # ArrayRefRegistry (array_ref dedup)
│   ├── load-and-decode.ts         # loadAndDecodeOptionalArray helper
│   └── types.ts                   # ArrayMetadata + EncodingMetadata
│
├── nav/                           # Multi-strategy server directory browsing
│   └── directory-navigator.ts
│
├── stats/                         # Scene + per-loader statistics
│   ├── scene-stats.ts             # Roll up loaded geometry per type
│   └── aggregator.ts              # Accumulator stats aggregation across loaders
│
├── scene-loader/                  # Modules composed by scene-loader.ts, organised into
│                                  #   eleven subpackages: cache/, loaders/, view-state/,
│                                  #   lifecycle/, monitor/, commit/, nodes/, prefetch/,
│                                  #   process/, progressive/, update-view/, plus the
│                                  #   top-level lod-load-stats.ts. See scene-loader/README.md.
│
├── loaders/                       # Unified loader infrastructure (see loaders/README.md)
│   ├── index.ts                   # Barrel — external callers import from here
│   ├── base-types.ts              # Common types (BaseViewState, LoadRange)
│   ├── abort-error.ts             # isAbortError — "superseded, not failed" classifier
│   ├── chunk-bounds-loader.ts     # Shared chunk_bounds zarr probe
│   ├── color-loader.ts            # Shared color-range loader (points, lines, gsplats, mesh)
│   ├── extend-to-all-preflight.ts # Resolves extend_to_all dim names → indices
│   ├── spatial-facade.ts          # Shared loadX/updateView facade orchestration (incl. S-cache restore/store)
│   ├── loader-metrics.ts          # Shared latency / event metrics (per-geometry)
│   ├── aggregate-loader-metrics.ts # Roll up per-loader metrics for the monitor
│   ├── monitor-events.ts          # Shared LoaderEventEmitter for monitor events
│   ├── once-init.ts               # Tiny once-init helper for lazy initializers
│   ├── progressive-monitor-adapter.ts # Bridges progressive loaders → monitor events
│   ├── spatial-query/             # Chunk-bounds query pipeline
│   │   ├── spatial-query-builder.ts # Canonical chunk-bounds query API
│   │   │                          #   `SpatialQueryBuilder` accepts either a `geometryType` (delegates
│   │   │                          #   tolerance to `tolerance-computer.computeTolerance`) or a
│   │   │                          #   pre-computed `tolerance: number[]`.
│   │   ├── tolerance-computer.ts  # Geometry-aware per-dimension tolerance
│   │   └── range-loader.ts        # Unified encoding dispatch (+ range-loader/ shared instance)
│   ├── picking/                   # Lazy label fetching for the picking subsystem
│   │   ├── image-label-loader.ts  # Lazy per-element image fetching from zarr
│   │   └── label-loader.ts        # Lazy CSR-style label fetching from zarr
│   ├── overlays/                  # Overlay config loading
│   │   └── overlay-loader.ts      # Reads overlay configurations from zarr store
│   └── progressive/               # Shared progressive-LOD helpers (concat, constants,
│                                  #   slice-cache-helper.ts — S-cache restore/store, see §S-cache)
│
└── (related: ../workers/)         # Web Worker infrastructure
    ├── worker-pool.ts             # Pool manager with load balancing
    └── data-worker.ts             # Worker with WASM acceleration
```

Zipped stores go through the same L1/L2 tiers as a directory store: the caching
store reads through a `ChunkSource` rather than building chunk URLs, so an
archive member is reachable without one. Caching earns more here than for a
directory store — a member costs about two requests (the zip format puts a local
file header immediately before each member's data), and a repeat read cannot
fall back to the browser's HTTP cache, because every member read is a `Range`
request against a single URL.

Two costs the chunk cache cannot absorb, because the reader pays them below it
rather than as chunk keys: the fixed directory preamble on every visit, and — on
a cold read — that second request per member. See `cache/chunk-source/` for the
port, and issue #1716 for the measurements.

### State Management Architecture

The data package uses a **clean singleton pattern** for instance management, completely avoiding global variables:

```
┌─────────────────────────────────────┐
│      Public API (zarr-loader.ts)    │
│  loadScene(), updateView(), etc.    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       SceneLoaderManager             │
│  (Singleton - manages instances)     │
│  ✓ No global variables              │
│  ✓ Multiple loader support          │
│  ✓ Clean lifecycle management       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      SceneLoader Instances          │
│  ✓ Independent instances            │
│  ✓ Use DataMonitorManager           │
│  ✓ No direct monitor creation       │
│  ✓ Proper resource cleanup          │
└─────────────────────────────────────┘
```

**Key Benefits:**

- **No Global State**: Window object remains unpolluted
- **Testability**: Easy reset functionality for testing
- **Multiple Instances**: Support for independent loaders
- **Memory Safety**: Proper cleanup and disposal
- **Centralized Management**: Single source of truth

**Note**: Datasets with Morton/Hilbert ordering (chunk_bounds) will load much faster due to efficient spatial queries. Small 3D datasets without spatial ordering will fall back to loading all points (acceptable for <100K points).

### Modular Architecture

SceneLoader is split into focused, testable modules:

```
┌─────────────────────────────────────────────────────────────┐
│                    SceneLoader (Facade)                      │
│  Coordinates loading and provides public API                │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────────────┐
    ▼          ▼          ▼                  ▼
┌─────────┐ ┌─────────┐ ┌─────────────────┐ ┌─────────────┐
│ Scene   │ │ Loader  │ │ Geometry Update │ │ Data        │
│ Graph   │ │ Orch.   │ │ Manager         │ │ Accumulators│
│ Builder │ │         │ │                 │ │             │
└─────────┘ └─────────┘ └─────────────────┘ └─────────────┘
```

**Node Factories** (`scene-loader/nodes/`):

- Creates THREE.js scene nodes (Points, Lines, GSplats, Mesh) from loaded data
- Geometry creation with proper dtype handling (Float32, Uint8, Float16)
- Material creation and colormap application
- Transform application and validation
- Picking system integration (shadow pick-node creation)
- **Empty-placeholder factories** — every loader attaches a
  placeholder before the first fetch so a transient load failure leaves
  a findable, retryable node in the scene rather than a hole. The same
  node is later populated in place by `scene-loader/commit/commit-*-geometry.ts`
  helpers.

**Data Accumulators** (`accumulators/`):

- Zero-allocation buffer pooling for Points/Lines/GSplats
- Multi-type support (Float32, Uint8, Uint16)
- Eliminates per-frame allocations, reduces GC pressure

**Benefits of Modular Design**:

- **Testability**: Each module tested independently
- **Maintainability**: Clear responsibility boundaries
- **Reduced Complexity**: `scene-loader.ts` is now ~1,180 lines (down
  from ~2,100) thanks to ongoing extraction. See `scene-loader/` for
  the extracted modules, organised into eleven thematic subpackages:
  `cache/`, `loaders/`, `view-state/`, `lifecycle/`, `monitor/`,
  `commit/`, `nodes/`, `prefetch/`, `process/`, `progressive/`, and
  `update-view/`.
  The top level of `scene-loader/` holds a single helper,
  `lod-load-stats.ts`; every other helper lives in one of the
  subpackages above.

---

## Components

### 1. Zarr Loader API

The `zarr-loader.ts` provides the main public API for loading and managing Zarr datasets.

**Core Features:**

- Hierarchical scene graph loading
- Scene-level dimension metadata extraction
- Radius-based nD slicing for visibility
- Attribute inheritance in nested structures
- GPU-optimized format conversion
- Instance-based loader management

**Data Pipeline:**

```typescript
// 1. Load scene from Zarr store (uses consolidated metadata)
const scene = await loadScene(url, config);

// 2. The loader automatically:
//    - Extracts scene dimensions from metadata
//    - Builds hierarchical scene graph
//    - Loads spatial indices for each points
//    - Creates THREE.js geometries and materials
//    - Handles transforms and attribute inheritance

// 3. Update view for dimension navigation
await updateView({
  displayDims: [0, 1, 2],
  slicePosition: currentPosition,
  tolerance: sliceTolerance,
});

// 4. Clean up when done
dispose();
```

**Key Functions:**

```typescript
// Main API functions from zarr-loader.ts
export async function loadScene(
  src: string,
  config?: LoaderConfig,
  loaderId?: string
): Promise<THREE.Group>;

export async function updateView(viewState: Partial<ViewState>, loaderId?: string): Promise<void>;

// Update scene when navigating dimensions. `opts.frameBudgetMs` is the
// per-pass LOD time budget during dimension-animation playback (loaders
// stream sub-LODs until the budget runs out, then commit); absent outside
// playback. Queued calls resolve when the requested-or-newer state's pass
// commits (the animation pacing gate awaits this).
export async function updateSceneForDimensions(
  dims: SimpleDims,
  scene: THREE.Group,
  loaderId?: string,
  opts?: { frameBudgetMs?: number; ladderDepth?: number }
): Promise<void>;
```

### 2. Chunk-Based Spatial Index

The chunk-bounds spatial-query API lives in `loaders/spatial-query/spatial-query-builder.ts` and is used by all three geometry loaders (Points, Lines, GSplats). Each loader inlines its own `chunk_bounds` zarr probe — the array names differ slightly (`chunk_bounds` for points/gsplats, `vertex_chunk_bounds` + `segment_chunk_bounds` for lines) — and then delegates the AABB scan, range conversion, and range merge to the canonical `SpatialQueryBuilder`.

**Core features:**

- **Chunk-based indexing** with Morton/Hilbert space-filling curves for spatial locality.
- **Bounding-box queries** — each chunk has a `(ndim, 2)` AABB; queries are linear scans.
- **Memory-efficient** — only chunk bounds are stored, never per-element indices.
- **nD support** — arbitrary dimensionality.
- **Cache-friendly** — elements are sorted by space-filling curve order.
- **Progressive loading** — only chunks intersecting the current slice are loaded.

**How it works (gsplats / lines geometry-aware path):**

```typescript
import { SpatialQueryBuilder, type ChunkSpatialIndex } from './loaders';

// 1. Probe chunk_bounds from zarr (loader-private; differs per geometry type).
const index: ChunkSpatialIndex = await this.loadChunkBounds(attrs);

// 2. One call: build position, compute tolerance via geometry strategy,
//    run the AABB scan, convert chunks to ranges, merge contiguous ranges.
const ranges = await new SpatialQueryBuilder(index, viewState, {
  geometryType: 'gsplats', // or 'lines'
  totalElements: attrs.n_splats,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
}).execute();

// 3. Load only the required data.
for (const range of ranges) {
  loadElementRange(range.start, range.end);
}
```

**Pre-computed tolerance path (points):** the points loader computes tolerance via
`effective-radius-calculator.calculateSpatialQueryTolerance` (which knows about
`EffectiveRadiusConfig` and the `>= 1e9` extend-to-all sentinel) and passes the
result to the builder verbatim:

```typescript
const tolerance = calculateSpatialQueryTolerance(viewState, config, fullDim);
const ranges = await new SpatialQueryBuilder(index, viewState, {
  tolerance,
  totalElements: attrs.n_points,
  chunkSize: attrs.chunk_size,
  extendDims: attrs.extend_to_all,
}).execute();
```

**Canonical type:**

```typescript
interface ChunkSpatialIndex {
  chunkBounds: Float32Array; // shape (numChunks, ndim, 2) flattened row-major
  chunkCount: number;
  metadata: {
    ndim: number;
    chunk_size?: number;
  };
}
```

**Performance benefits:**

- **10–100× faster queries** on datasets with spatial locality.
- **Reduced memory usage** — only visible elements are loaded.
- **Better cache utilization** via space-filling curve ordering.
- **One canonical API** — the same builder serves all three geometry types.

### 3. Directory Navigator

The `DirectoryNavigator` provides flexible directory browsing across different server types.

**Features:**

- Multi-strategy detection (WebDAV, HTML parsing, index files)
- Zarr dataset identification
- Server-agnostic operation
- Breadcrumb navigation support
- File size and modification time extraction

**Strategy Chain:**

```typescript
class DirectoryNavigator {
  // Try strategies in order:
  1. Check if path is a Zarr dataset (.zgroup)
  2. Try WebDAV PROPFIND request
  3. Parse HTML directory listing
  4. Look for index.json manifest
  5. Fallback to manual entry
}
```

**Usage Example:**

```typescript
const navigator = new DirectoryNavigator('http://data.server.com/');

// Navigate to a path
const result = await navigator.navigate('datasets/');

// Result includes:
// - entries: Array of files/directories/zarr datasets
// - currentPath: Current location
// - parentPath: Parent directory
// - strategy: Detection method used
// - isZarr: Whether current path is a Zarr dataset
```

---

## Data Loading Pipeline

### Loading Stages

```
URL Input
    ↓
Store Initialization (with consolidated metadata)
    ↓
Scene Dimension Extraction
    ↓
Hierarchical Group Enumeration
    ↓
Point Cloud Loading (parallel)
    ↓
nD Slicing (if dims > 3)
    ↓
GPU Format Conversion
    ↓
THREE.js Geometry Creation
    ↓
Material Assignment
    ↓
Scene Assembly
```

### Mesh: the whole-node loader

Three of the four geometry types stream: they carry a chunk-bounds spatial index, fetch
the chunks a query intersects, and grow the visible set incrementally. **Mesh does not
stream per slice** — a laddered mesh still grows its _loaded_ set level-by-level as it
loads, view-independently rather than from a spatial query; what is _drawn_ is then
decided by the slice exactly as for an unladdered mesh (`mesh-progressive-loader.ts`).
Its loader (`data/mesh/`) is whole-node — `ordering: 'none'`, no spatial index — and
that is the correct shape rather than a gap (`docs/specs/MESH_NODE_SPEC.md` §7, §9): a
surface is CONNECTED, so a chunk of triangles is not independently meaningful, and the
index buffer references vertices anywhere in the array.

Two consequences follow, and both are load-bearing:

- **An admission gate runs before any chunk is fetched.** Because the loader reads every
  array in full, a check that ran after decode would arrive too late — a hostile or
  corrupt store can declare enormous arrays and exhaust the tab first. So
  `data/mesh/preflight.ts` is a metadata-only Stage 1 (shapes, dtypes, encodings, and a
  per-node byte budget summing stored bytes, decoded bytes and the largest single chunk
  allocation), with the value-level checks that genuinely need materialized arrays in
  Stage 2 (`validate.ts`).
- **A slice move rewrites only the INDEX buffer.** The nD cull is per-vertex slab
  membership with a whole-triangle decision (§5.4) — no interpolation, no vertex
  compaction — so `compact_visible_faces` narrows `drawRange` while the vertex arrays
  stay put. That is why a mesh pick id is a VERTEX ordinal: a face ordinal would be
  renumbered on every slice change, while a vertex ordinal is invariant.

Mesh also needs its **own tolerance arm** (`computeMeshHiddenTolerance`, §5.2.1), and
only for a CONTINUOUS hidden dimension: a discrete/categorical one takes the shared
`discreteDimMembershipTolerance` like every other type. On a continuous axis, Lines'
spatial `0` works because segment clipping interpolates through the slab; with no
interpolation and no per-element extent, `0` reduces membership to exact float equality —
a measure-zero condition, so effectively nothing renders. Mesh therefore takes
`step x meshSlabTolerance` (one cell by default) instead.

### nD Slicing Algorithm

For datasets with more than 3 dimensions, visibility depends on dimension type:

#### Spatial Dimensions

Points extend through spatial dimensions as hyperspheres:

```typescript
// Hypersphere intersection for spatial dimensions only
for each point:
  // Calculate distance in non-displayed SPATIAL dimensions
  distance = 0
  for each non-displayed dimension d:
    if (dimension[d].spatial):
      delta = point[d] - currentSlice[d]
      distance += delta * delta
    // Non-spatial dimensions handled separately

  // Point visible if within radius
  radius = point.radius || defaultRadius
  if sqrt(distance) <= radius:
    include point in slice
```

#### Discrete Dimensions

Points exist at exact values in discrete dimensions:

```typescript
// Exact matching for discrete dimensions (channels, time frames)
for each non-displayed dimension d:
  if (dimension[d].discrete):
    // Use exact matching (tolerance = 0)
    if point[d] != currentSlice[d]:
      exclude point from slice
```

**Note**: Non-displayed, non-spatial dimensions are automatically discrete.

### Dimension Extension

Dimension extension allows both points and lines to appear across all values of specified non-displayed dimensions without data duplication:

```typescript
// Extended dimensions are explicitly specified in zarr attributes
// Set via Python API:
//   scene.add_points(..., extend_to_all=["Time", "Channel"])
//   scene.add_lines(..., extend_to_all=["Time"])  // e.g., static detector geometry
const extendDims = attrs.extend_to_all || [];

// During navigation, extended groups are handled specially
if (extendDims.includes(current_dimension)) {
  // Load all data once and display at every dimension value
  // This avoids duplicating data across all dimension values
  loadAllData(); // Works for both points and lines
} else {
  // Normal slicing based on current dimension value
  loadSlice(dimension_value);
}
```

Benefits:

- **Memory Efficient**: No data duplication needed (e.g., static detector geometry visible at all time points without replication)
- **Explicit Control**: Clear API for specifying extension behavior
- **Works for Points and Lines**: Both primitive types support extend_to_all
- **Cache Aware**: Proper cache key isolation prevents data corruption

### Attribute Composition

Rendering attributes **compose** along the scene-graph hierarchy (root → leaf),
as implemented in `packages/luxar/src/luxar/core/`. Unset values are treated
as identity:

```typescript
effective_opacity   = clamp(∏ opacity_i,   0, 1)
effective_gamma     = clamp(∏ gamma_i,     0.1, 10)
effective_intensity = max(0, ∏ intensity_i)
effective_offset    = Σ offset_i
effective_blending  = nearest ancestor that sets blending_mode, else 'additive'
effective_colormap  = nearest ancestor that sets colormap, else undefined
                      (its `customLutBytes` travel with the name, from the
                       SAME node — never a name/LUT pair from two nodes)
```

Composition is applied by `SceneLoader.applyEffectiveAttrs()` (uses
`data/attrs-composer.ts`) before each leaf material is created. The
Layers panel recomposes on every slider change so live edits to a group
or ancestor layer are reflected in every affected descendant.

### Web Worker Offloading

CPU-intensive operations are offloaded to Web Workers for parallel execution:

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LOADERS                             │
│  (point/lines/gsplats-spatial-index-loader.ts)              │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │      WorkerPool       │                       │
│              │   (1-N DataWorkers)   │                       │
│              │           │           │                       │
│              │           ▼           │                       │
│              │  ┌─────────────────┐  │                       │
│              │  │   WASM Module   │  │                       │
│              │  │  (or TypeScript │  │                       │
│              │  │    fallback)    │  │                       │
│              │  └─────────────────┘  │                       │
│              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**Operations offloaded to workers**:

- Encoded data decoding — six tasks: `decodeQuantized`, `decodeLogScalar`,
  `decodeGeologScalar`, `decodePerChannel`, `decodeLUT`, `decodeBroadcasted`
- nD → 3D projection of Lines and GSplats (`projectLinesTo3D` /
  `projectGSplatsTo3D`); per-element nD visibility (clip mask, effective
  radius, attenuation) is computed inside these projection kernels

**Kept on the main thread**: spatial index queries (chunk bounding box
intersection) and Points projection (WASM-accelerated in
`data/points/projection.ts` with a zero-alloc accumulator).

**Performance benefits**:

- Non-blocking main thread (smooth UI during data loading)
- WASM acceleration (3-5x faster than TypeScript)
- Automatic fallback if WASM unavailable

See `src/wasm/rust/README.md` for WASM details.

---

## Data Formats

### Zarr Structure

Expected Zarr store structure:

```
dataset.luxar.zarr/
├── .zattrs                # Scene metadata
├── .zgroup                # Zarr group marker
├── .zmetadata            # Consolidated metadata (recommended)
└── points/
    ├── .zattrs           # Node attributes (ordering, chunk_size, etc.)
    ├── .zgroup
    ├── positions/        # Float32[N, D] - required, Morton-ordered
    ├── colors/           # Float32[N, 3] - optional
    ├── radii/            # Float32[N] - optional
    ├── sharpness/        # Float32[N] - optional
    └── chunk_bounds/     # Float32[num_chunks, D, 2] - chunk bounding boxes
```

### Attribute Schema

```typescript
interface ZarrGroupAttrs {
  // Type identification
  type?: 'points' | 'group' | 'scene';

  // Transformation
  transform?: number[]; // 16-element 4x4 matrix

  // Rendering
  opacity?: number; // 0.0 to 1.0
  gamma?: number; // 0.1 to 10.0
  blending_mode?: BlendingMode;

  // Dimensions
  scene_dimensions?: {
    dimensions: DimensionMetadata[];
  };

  // Dimension extension
  extend_to_all?: string[]; // Dimension names to extend visibility across

  // Physical units
  units?: string; // e.g., 'um', 'nm'
}
```

### Dimension Metadata

```typescript
interface DimensionMetadata {
  name: string; // e.g., 'x', 'time'
  unit: string; // e.g., 'μm', 'ms'
  range: [number, number];
  step?: number; // Navigation step size
  discrete?: boolean; // Whether dimension is discrete
  display: boolean; // Whether to display this dimension
}
```

---

## Performance Optimization

### Chunk-Based Spatial Index for Efficient Queries

The chunk-based spatial index dramatically improves performance for large nD datasets:

```typescript
// Chunk index is loaded automatically by SceneLoader
// For nD datasets with Morton ordering, uses chunk_bounds for fast queries
// For 3D datasets without Morton ordering, falls back to loading all points

// The PointsSpatialIndexLoader delegates to the canonical builder:
const ranges = await new SpatialQueryBuilder(index, viewState, {
  tolerance, // pre-computed via calculateSpatialQueryTolerance for points
  totalElements: attrs.n_points,
  chunkSize: attrs.chunk_size,
}).execute();
const visiblePoints = await loadRanges(ranges);

// Query performance:
// With chunk index: O(num_chunks) - scan chunk bounds only (~100-1000 chunks)
// Without chunk index: Loads all points (acceptable for <100K 3D datasets)
```

**When Chunk Index is Used:**

- Datasets with Morton/Hilbert ordering
- High-dimensional data (4D+)
- Datasets with chunk_bounds array

**How It Works:**

```python
# In Python during compilation
# Morton ordering and chunk_bounds are created automatically
scene.add_points(
  "points",
  positions=data,  # Will be Morton-ordered automatically
)

# The compiler:
# 1. Sorts points by Morton code
# 2. Divides into chunks
# 3. Computes bounding box for each chunk
# 4. Stores in chunk_bounds array
```

### Chunk Index Performance

The chunk-based approach provides excellent performance for large datasets:

```typescript
// The SceneLoader automatically uses chunk indices for efficient loading
// You don't need to interact with the chunk index directly - it's handled internally

// When you update the view:
await updateView({
  displayDims: [0, 1, 2],
  slicePosition: [x, y, z, t],
  tolerance: [0, 0, 0, radius],
});
// The loader automatically queries chunk bounds and loads only visible chunks
```

**Benefits:**

- Load only visible points from massive datasets
- Efficient nD queries without scanning all points
- Smooth navigation through temporal/dimensional slices
- Automatic caching of frequently accessed ranges

**How It Works:**

1. **Initial Load**: Queries chunk index for visible chunks
2. **Navigation**: As user navigates, queries update to find new visible chunks
3. **Caching**: Recently accessed ranges are cached for fast re-access
4. **Memory Management**: Automatic eviction of least-recently-used cached ranges

### Slice Cache (S-cache) and t+1 Prefetch

One level above the chunk caches, decoded slices are cached whole: the
shared `SliceCache` (`src/cache/slice-cache.ts`) stores each node's
decoded per-slice LOD ladder keyed by a view signature (displayDims,
slicePosition, tolerance, dimensions). The key/snapshot/lookup logic is
shared by the plain spatial-index loaders and the progressive loaders via
`loaders/progressive/slice-cache-helper.ts` (the restore/store hook sits
in `loaders/spatial-facade.ts`), so revisiting a slice — e.g. scrubbing
back through time — restores the decoded ladder without re-fetching or
re-decoding chunks.

During dimension playback, `SceneLoader.prefetchSlice(viewState, budgetMs)`
(driven by `zarr-loader.ts::prefetchSceneForDimensions`) warms the NEXT
tick's ladder into the S-cache using shadow loaders
(`scene-loader/prefetch/slice-prefetcher.ts`). The pass is
fire-and-forget, aborted at the top of every foreground `updateView`, and
its resources are freed via `releasePrefetchResources()` when playback
ends.

### Chunking Strategy

Optimal chunk sizes for different scenarios:

```typescript
// Recommended chunk configurations
const CHUNK_CONFIGS = {
  small: {
    // < 100K points
    chunkSize: [10000, 3], // 10K points per chunk
    compression: 'blosc',
  },
  medium: {
    // 100K - 10M points
    chunkSize: [100000, 3], // 100K points per chunk
    compression: 'blosc',
  },
  large: {
    // > 10M points
    chunkSize: [1000000, 3], // 1M points per chunk
    compression: 'zstd',
  },
};
```

### Memory Management

```typescript
// Efficient attribute loading
async function loadAttributes(store, path, count) {
  // Load in parallel for speed
  const [colors, radii, sharpness] = await Promise.all([
    loadOptionalArray(store, `${path}/colors`),
    loadOptionalArray(store, `${path}/radii`),
    loadOptionalArray(store, `${path}/sharpness`),
  ]);

  // Apply defaults only where needed
  return {
    colors: colors || createDefaultColors(count),
    radii: radii || createDefaultRadii(count),
    sharpness: sharpness || null, // Optional
  };
}
```

### Streaming Considerations

For very large datasets:

1. **Progressive Loading**: Load visible chunks first
2. **Level of Detail**: Use decimated versions at distance
3. **Culling**: Skip chunks outside view frustum
4. **Caching**: Keep recently used chunks in memory

---

## Usage Examples

### Basic Loading

```typescript
import { loadScene, updateView } from '../data';

// Load a Zarr dataset (point spatial index required)
const scene = await loadScene('http://server.com/data/points.luxar.zarr');

// Add to THREE.js scene
threeScene.add(scene);

// Update view when navigating dimensions
await updateView({
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, timeStep, channel],
  tolerance: [0, 0, 0, 0.1, 0.1],
});
```

### nD Dataset Loading

```typescript
import { loadScene, updateSceneForDimensions } from '../data';
import type { SimpleDims } from '../types/dims';

// Load 5D dataset (x, y, z, time, channel)
const scene = await loadScene('http://server.com/data/5d-points.luxar.zarr');

// Dimensions are stored ONLY at scene level (single source of truth)
// Individual nodes (Points, Lines, Splats) access dims via ViewState
// Navigate through dimensions
const dims: SimpleDims = {
  ndim: 5,
  displayed: [0, 1, 2], // Show x, y, z
  currentStep: [0, 0, 0, 10, 2], // Position in 5D space
  metadata: scene.userData.sceneDimensions?.dimensions,
};

// Update for dimension changes
await updateSceneForDimensions(dims, scene);
```

### Directory Navigation

```typescript
import { DirectoryNavigator } from '../data';

const navigator = new DirectoryNavigator('http://data.server.com/');

// Browse datasets
const result = await navigator.navigate('datasets/');

// Display entries
result.entries.forEach((entry) => {
  if (entry.type === 'zarr') {
    console.log(`📊 Zarr dataset: ${entry.name}`);
  } else if (entry.type === 'directory') {
    console.log(`📁 Directory: ${entry.name}`);
  }
});

// Load selected Zarr dataset
const selected = result.entries.find((e) => e.type === 'zarr');
if (selected) {
  const scene = await loadScene(selected.path);
  threeScene.add(scene);
}
```

### Multiple Loader Instances

```typescript
import { loadScene, updateView, dispose } from '../data';

// Create multiple independent loaders for different datasets
const scene1 = await loadScene('http://server.com/data1.luxar.zarr', config, 'loader1');
const scene2 = await loadScene('http://server.com/data2.luxar.zarr', config, 'loader2');

// Update each loader independently
await updateView(viewState1, 'loader1');
await updateView(viewState2, 'loader2');

// Dispose specific loader
dispose('loader1');

// Or dispose all loaders
dispose();
```

### Cache Management

```typescript
import { SceneLoaderManager } from '../data';

// Monitor cache usage for default loader
const loader = SceneLoaderManager.getInstance().getDefaultLoader();
const stats = loader?.getCacheStats();
if (stats) {
  console.log(`L1 hit rate: ${(stats.l1?.hitRate ?? 0) * 100}%`);
  console.log(`L2 hit rate: ${(stats.l2?.hitRate ?? 0) * 100}%`);
}

// Clear the L2 (persistent / OPFS) cache for a specific loader
await loader?.clearL2Cache();

// Or clear every cache tier (L0 + L1 + L2)
await loader?.clearAllCaches();
```

The public `zarr-loader.ts` API no longer exports `getCacheStats` /
`clearCaches`; reach the loader through `SceneLoaderManager` (or call
`loader.getCacheStats()` / `clearL2Cache()` / `clearAllCaches()`
directly).

### Advanced Instance Management

```typescript
// The data/ and ui/ layers are kept separate: SceneLoader receives a
// SceneLoaderMonitorPort factory at construction (wired in
// core/app.ts) instead of importing DataMonitorManager directly.
// Production code resolves the monitor through that port; the example
// below shows direct registry access for diagnostic/advanced cases.
import { SceneLoaderManager } from '../data';
// DataMonitorManager lives in ui/ now; only import it from there if you
// truly need to inspect the monitor singleton (rare).
import { DataMonitorManager } from '../ui/data-monitor-manager';

// Direct access to manager for advanced use cases
const loaderManager = SceneLoaderManager.getInstance();
const monitorManager = DataMonitorManager.getInstance();

// Check active loaders
console.log(`Active loaders: ${loaderManager.getLoaderCount()}`);

// Get specific loader instance
const loader = loaderManager.getLoader('myLoader');
if (loader) {
  // Direct loader manipulation
  loader.showMonitor();
  await loader.clearAllCaches();
}

// Reset for testing
SceneLoaderManager.disposeInstance();
DataMonitorManager.disposeInstance();
```

---

## Error Handling

### Common Issues and Solutions

**Problem: CORS errors when loading Zarr**

```typescript
// Solution: Configure server headers
// nginx.conf:
location /data/ {
  add_header Access-Control-Allow-Origin *;
  add_header Access-Control-Allow-Methods "GET, OPTIONS";
}
```

**Problem: Large 3D dataset loading slowly**

```typescript
// For 3D datasets without Morton ordering, all points are loaded at once.
// This is acceptable for <100K points but may be slow for larger datasets.

// Solution: For nD (4D+) datasets, Morton ordering is automatic and enables
// chunk-based spatial queries. 3D datasets without additional dimensions
// fall back to loading all points - this is fine for small datasets.
```

**Problem: Memory usage too high**

```typescript
// Pass ?no-cache or { noCache: true } to disable caching entirely.
const scene = await loadScene(url, { noCache: true });

// Or inspect the live cache stats via SceneLoaderManager and clear
// the persistent L2 tier (or every tier) when needed.
import { SceneLoaderManager } from '../data';
const loader = SceneLoaderManager.getInstance().getDefaultLoader();
const stats = loader?.getCacheStats();
if (stats?.l2 && stats.l2.bytes > threshold) {
  await loader?.clearL2Cache();
}
```

Cache size limits live with the `cache/` package (multi-level caching
store and OPFS backend) — not with `LoaderConfig`. See
`../cache/README.md`.

---

## Configuration

### Loader Configuration

```typescript
interface LoaderConfig {
  /** Enable debug logging */
  debug?: boolean;

  /** Enable data loading monitor UI */
  enableMonitor?: boolean;

  /** Disable both L1/L2 cache tiers and L0 decompressed cache. */
  noCache?: boolean;

  /** Verbose cache logging. */
  cacheDebug?: boolean;

  /** Clear caches on init. */
  clearCache?: boolean;

  /** Disable adjacent-chunk prefetching. */
  noPrefetch?: boolean;

  /** Verbose prefetch logging. */
  prefetchDebug?: boolean;
}

// Usage
const scene = await loadScene(url, {
  enableMonitor: true,
  debug: false,
});
```

Cache size limits and eviction strategy are controlled by the
multi-level caching store and `cache/` package — not by `LoaderConfig`.
See `../cache/README.md`.

### Server Configuration

Recommended server setup for optimal performance:

**nginx:**

```nginx
location /data/ {
  # Enable CORS
  add_header Access-Control-Allow-Origin *;

  # Enable range requests
  add_header Accept-Ranges bytes;

  # Compression
  gzip on;
  gzip_types application/octet-stream;

  # Caching
  expires 1h;
  add_header Cache-Control "public, immutable";
}
```

**Apache:**

```apache
<Directory /var/www/data>
  Header set Access-Control-Allow-Origin "*"
  Header set Accept-Ranges "bytes"

  <FilesMatch "\.(zarr|zarray|zattrs|zgroup|zmetadata)$">
    Header set Cache-Control "max-age=3600, public"
  </FilesMatch>
</Directory>
```

---

## API Reference

### Main API (zarr-loader.ts)

| Function                                                  | Description                                                                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `loadScene(url, config?, loaderId?)`                      | Load complete Zarr dataset with chunk-based indexing                                                                                             |
| `updateView(viewState, loaderId?)`                        | Update all points for new view state                                                                                                             |
| `updateSceneForDimensions(dims, scene, loaderId?, opts?)` | Update scene when navigating dimensions (opts.frameBudgetMs = playback LOD budget)                                                               |
| `prefetchSceneForDimensions(dims, scene, loaderId, opts)` | Fire-and-forget t+1 slice prefetch for a PREDICTED dimension state (playback) — routes to `SceneLoader.prefetchSlice`, never moves the real view |
| `releasePrefetchResources(loaderId?)`                     | Release the loader's t+1 prefetch resources (shadow loaders) when playback ends                                                                  |
| `dispose(loaderId?)`                                      | Clean up resources (specific or all)                                                                                                             |

Cache inspection and clearing are not on the `zarr-loader.ts` surface;
get the loader via `SceneLoaderManager.getDefaultLoader()` (or
`getLoader(id)`) and call `getCacheStats()`, `clearL2Cache()`, or
`clearAllCaches()` directly on the `SceneLoader` instance.

### Instance Management (scene-loader-manager.ts)

| Class/Method                                    | Description                                                                                    |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `SceneLoaderManager`                            | Singleton manager for SceneLoader instances                                                    |
| `getInstance()`                                 | Get the singleton manager instance                                                             |
| `createLoader(id, config?, setAsDefault?)`      | Create a new loader instance                                                                   |
| `createLoaderAsync(id, config?, setAsDefault?)` | Like `createLoader` but awaits the previous same-ID loader's full disposal (dataset switches). |
| `getLoader(id)`                                 | Get a specific loader by ID                                                                    |
| `getDefaultLoader()`                            | Get the default loader instance                                                                |
| `getAllLoaders()`                               | Get all active loader instances                                                                |
| `destroyLoader(id)`                             | Dispose and remove a specific loader (fire-and-forget).                                        |
| `destroyLoaderAsync(id)`                        | Like `destroyLoader` but awaits full disposal (dataset switches).                              |
| `destroyAll()`                                  | Dispose all loaders and reset manager (fire-and-forget).                                       |
| `destroyAllAsync()`                             | Like `destroyAll` but awaits every loader's `dispose()` to settle.                             |
| `hasLoader(id)` / `getLoaderCount()`            | Lookup and count helpers for the loader registry.                                              |
| `setMonitorFactory(factory)`                    | Inject a `SceneLoaderMonitorFactory` (called from `core/app.ts`).                              |
| `setLODGroupRegistryFactory(factory)`           | Inject the LOD-group registry factory (app init pipeline owns SceneManager + camera).          |
| `setRequestRender(callback)`                    | Inject the render-loop wake-up forwarded to every created loader.                              |
| `getProfiler()`                                 | Return the shared `UpdateProfiler` singleton.                                                  |
| `disposeInstance()`                             | Dispose the singleton (call from app dispose; preserved for re-init)                           |

### Scene Loading (scene-loader.ts)

| Class/Method                                            | Description                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| `SceneLoader`                                           | Main scene loader orchestrator                                           |
| `constructor(config?, id?, profiler?, monitorFactory?)` | Create loader; profiler + monitor factory are injected by the manager.   |
| `loadScene(url)`                                        | Load complete scene from zarr store                                      |
| `updateView(viewState)`                                 | Update all loaders with new view state                                   |
| `getCacheStats()`                                       | Return a `CacheStatsSnapshot` covering L0/L1/L2 + network/demand/health. |
| `listCachedDatasets()`                                  | List the datasets currently held in the L2 (persistent) cache.           |
| `clearL2Cache()`                                        | Clear the persistent L2 cache tier (e.g. OPFS) for this loader.          |
| `clearAllCaches()`                                      | Clear every cache tier (L0 + L1 + L2) for this loader.                   |
| `showMonitor()`                                         | Show data loading monitor UI                                             |
| `hideMonitor()`                                         | Hide data loading monitor UI                                             |
| `toggleMonitor()`                                       | Toggle data loading monitor UI                                           |
| `retryFailedLoader(path)`                               | Retry a failed loader or lazy LOD branch (identified by zarr path).      |
| `retryAllFailedLoaders()`                               | Retry all failed loaders and latched lazy LOD branches.                  |
| `prefetchSlice(viewState, budgetMs)`                    | Fire-and-forget t+1 S-cache warm pass via shadow loaders (playback).     |
| `releasePrefetchResources()`                            | Free the prefetcher's shadow loaders + accumulators (playback end).      |
| `dispose()`                                             | Async — clean up all resources, drain caches, await teardown.            |

### Node Factories (scene-loader/nodes/)

The node factories that build the THREE.js scene from loaded data now
live under `scene-loader/nodes/` (one file per geometry kind:
`load-points-node.ts`, `load-lines-node.ts`, `load-gsplats-node.ts`),
plus `build-scene-graph.ts`, `enumerate-store.ts`, and the supporting
helpers. See `scene-loader/` for details — there is no single
`NodeFactory` class to import from this folder.

### Points Spatial Index Loader (points/points-spatial-index-loader.ts)

| Class/Method                                                                                    | Description                                                                 |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `PointsSpatialIndexLoader`                                                                      | Loader using chunk-based spatial indexing                                   |
| `constructor(zarrLocation, node, refRegistry?, zarrStore?, l0Cache?, prefetcher?, sliceCache?)` | Create loader; caches/prefetcher are injected by `loader-factory`.          |
| `initialize()`                                                                                  | Async — load the `chunk_bounds` array and prepare the spatial index.        |
| `loadPoints(viewState, session?)`                                                               | Load points for the given view state; returns a `LoadedPointsData` payload. |
| `updateView(viewState, session?)`                                                               | Re-query the spatial index for a new view state.                            |
| `prefetchChunks(viewState)`                                                                     | Background-fetch adjacent chunks predicted to be visible next.              |
| `dispose()`                                                                                     | Release the accumulator and other resources owned by this loader.           |

Cache statistics are aggregated by the parent `SceneLoader` via the
`loader-metrics.ts` event bus — per-loader cache APIs were removed in
favour of a single `SceneLoader.getCacheStats()` snapshot.

### Spatial-query API (loaders/spatial-query/spatial-query-builder.ts)

| Symbol                                            | Description                                                                             |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `class SpatialQueryBuilder`                       | Canonical chunk-bounds query API used by all three geometry loaders.                    |
| `interface ChunkSpatialIndex`                     | Canonical index shape: `{chunkBounds, chunkCount, metadata}`.                           |
| `executeSpatialQuery(params)`                     | AABB scan (lower-level helper).                                                         |
| `chunkIndicesToRanges(indices, chunkSize, total)` | Convert chunk indices to load ranges.                                                   |
| `mergeRanges(ranges)`                             | Merge overlapping or adjacent ranges.                                                   |
| `buildQueryPosition(viewState, ndim)`             | Pad/truncate `viewState.slicePosition` to `ndim`.                                       |
| `shouldExtendVisibility(extendDims, viewState)`   | True if any extend-to-all dim is currently hidden.                                      |
| `createLoadAllRange(totalElements)`               | Single range covering the whole dataset.                                                |
| `formatTolerance(t)`                              | Format one tolerance for the `?debug` query log (sub-0.01 → exponential, `1e10` → `∞`). |

### Tolerance computer (loaders/spatial-query/tolerance-computer.ts)

| Symbol                                                               | Description                                                                                     |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `computeTolerance(geometryType, displayDims, ndim, dims?, options?)` | Geometry-aware per-dimension tolerance. Used by `SpatialQueryBuilder`'s geometry-aware path.    |
| `gsplatsContinuousDimTolerance(dimInfo, truncationRadius?)`          | The gsplats hidden-continuous float-safety epsilon, `max(1e-3 × step, T × 1e-5)` — not a reach. |

### Monitor Port (scene-loader-monitor-port.ts)

| Symbol                      | Description                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `SceneLoaderMonitorPort`    | Interface — the subset of `DataLoadingMonitor`'s surface that `SceneLoader` consumes. |
| `SceneLoaderMonitorFactory` | Factory type injected by `core/app.ts` via `SceneLoaderManager.setMonitorFactory`.    |
| `L0CacheProviderPort`       | Provider injected via `setL0CacheProvider`.                                           |
| `GPUBufferPoolProviderPort` | Provider injected via `setGPUBufferPoolProvider`.                                     |
| `AccumulatorProviderPort`   | Per-geometry accumulator stats provider.                                              |

The concrete `DataMonitorManager` (a UI panel manager) lives in
`../ui/data-monitor-manager.ts` and is intentionally outside the
`data/` layer — production code only reaches it through the monitor
port above.

### Directory Navigation (nav/directory-navigator.ts)

| Class/Method           | Description                             |
| ---------------------- | --------------------------------------- |
| `DirectoryNavigator`   | Multi-strategy directory browser        |
| `navigate(path)`       | Navigate to directory or dataset        |
| `getFullUrl(path)`     | Get full URL for a path                 |
| `canListDirectories()` | Check if directory listing is supported |

### Zarr Facade (zarr.ts)

Every production import of `zarrita` is funnelled through `zarr.ts`,
which re-exports the types and helpers the rest of the viewer needs
(`openStore`, `openGroup`, `openArray`, `readArray`, `slice`,
`isNotFoundError`, `createFetchStore`, `codecRegistry`, plus the
`Location`, `Group`, `Array`, `DataType`, `TypedArray` aliases). Future
backend swaps or zarrita API moves stay isolated to this one file.

### Attribute Composition (attrs-composer.ts)

| Symbol                             | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `composeAttrs(chain)`              | Compose a root-to-leaf chain of `ComposableAttrs` into `EffectiveAttrs`. |
| `collectAncestorNodes(root, path)` | Walk the scene graph and return the chain of ancestor `SceneNode`s.      |
| `collectAncestorAttrs(root, path)` | Convenience — collect the chain as `ComposableAttrs[]`.                  |
| `getEffectiveAttrs(root, path)`    | Compose the effective attrs for a target path in one call.               |
| `collectDataDescendants(start)`    | Collect every data-leaf (points/lines/gsplats/mesh) under `start`.       |

### Dims → ViewState (dims-to-view-state.ts)

| Function                               | Description                                                               |
| -------------------------------------- | ------------------------------------------------------------------------- |
| `simpleDimsToViewState(dims, options)` | Convert a `SimpleDims` navigation snapshot into a per-loader `ViewState`. |

---

## Best Practices

### Data Preparation

1. **Use consolidated metadata**: Run `zarr.consolidate_metadata()`
2. **Choose appropriate chunks**: Balance size vs. granularity
3. **Enable compression**: Use blosc with zstd for best ratio
4. **Include dimension metadata**: Define coordinate systems
5. **Validate structure**: Use `luxar info` to verify

### Loading Strategy

1. **Check dataset size first**: Adjust strategy for large data
2. **Use progressive loading**: Load visible data first
3. **Implement caching**: Reuse loaded chunks
4. **Handle errors gracefully**: Provide fallbacks
5. **Monitor memory usage**: Dispose unused geometries

### Performance Tips

1. **Batch parallel loads**: Load multiple chunks simultaneously
2. **Use typed arrays**: Avoid unnecessary conversions
3. **Minimize slicing**: Cache sliced results
4. **Optimize chunk access**: Align requests with chunk boundaries
5. **Profile loading times**: Identify bottlenecks

---

## Troubleshooting

### Debug Mode

Enable detailed logging:

```typescript
// Enable debug output in loader config
const scene = await loadScene(url, { debug: true });

// Or enable global debug mode
if (typeof window !== 'undefined') {
  (window as any).__luxarDebug = { data: true };
}

// Logs will include:
// - Store initialization
// - Spatial index loading
// - Scene graph construction
// - View state updates
// - Cache hits/misses
// - Loading times
```

### Common Error Messages

| Error                  | Cause                   | Solution                          |
| ---------------------- | ----------------------- | --------------------------------- |
| "Failed to open store" | Invalid URL or CORS     | Check URL and server CORS headers |
| "No positions array"   | Missing required data   | Ensure positions array exists     |
| "Dimension mismatch"   | Incompatible dimensions | Verify dimension metadata         |
| "Chunk decode failed"  | Corrupted data          | Re-generate Zarr dataset          |

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._

## Subpackages

- [accumulators](./accumulators/) — Zero-allocation buffer pools for
  Points/Lines/GSplats.
- [array-decoder](./array-decoder/) — Decoders for the Python
  `luxar.encoding` array formats.
- [gsplats](./gsplats/README.md) — GSplats spatial-index loader,
  progressive multi-LOD, projection math.
- [lines](./lines/README.md) — Lines spatial-index loader, dual chunk
  index, projection.
- [loaders](./loaders/README.md) — Unified loader infrastructure
  (`SpatialQueryBuilder`, `RangeLoader`, tolerance).
- [mesh](./mesh/README.md) — Mesh whole-node loader, metadata preflight,
  whole-triangle nD cull, reveal-ladder progressive loader.
- [nav](./nav/) — Multi-strategy directory navigation
  (`DirectoryNavigator`).
- [points](./points/README.md) — Points spatial-index loader and
  effective-radius math.
- [scene-loader](./scene-loader/) — Modules composed by
  `scene-loader.ts`, grouped into eleven thematic subpackages
  (cache, loaders, view-state, lifecycle, monitor, commit, nodes,
  prefetch, process, progressive, update-view).
- [stats](./stats/) — Scene and per-loader statistics aggregation for
  the monitor.
- [transforms](./transforms/) — nD inverse-query helper for
  non-displayed dimensions.
