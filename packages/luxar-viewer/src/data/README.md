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
├── scene-loader.ts                # Orchestrates hierarchical scene loading (points + lines)
├── scene-loader-manager.ts        # Singleton manager for SceneLoader instances
├── chunk-spatial-index.ts         # Chunk-based spatial index queries (points)
├── point-spatial-index-loader.ts  # Loads points using chunk-based spatial queries
├── lines-spatial-index-loader.ts  # Loads lines with nD clipping and attribute interpolation
├── lines-chunk-spatial-index.ts   # Dual spatial index for lines (vertices + segments)
├── gsplats-spatial-index-loader.ts # Loads Gaussian splats with nD visibility
├── array-decoder.ts               # Decodes Python luxar.encoding arrays
├── data-monitor-manager.ts        # Singleton manager for monitoring UI instances
├── directory-navigator.ts         # Multi-strategy server directory browsing
├── nd-transform.ts                # nD transform inverse-query for non-displayed dimensions
│                                  #   Given a world-space query (slicePosition + tolerance) and a
│                                  #   composed nd_transform, produces the equivalent local-space query.
│                                  #   Avoids transforming geometry data — all loader internals unchanged.
├── gsplats-chunk-spatial-index.ts  # Spatial index loading and querying for GSplats
├── gsplats-progressive-loader.ts  # Progressive multi-LOD GSplats loader (Composite pattern)
├── effective-radius-calculator.ts # Calculates effective radii for nD slicing
├── data-loader-types.ts           # TypeScript interfaces and types
├── data-accumulator.ts            # Zero-allocation buffer pooling
├── scene-graph-builder.ts         # Scene hierarchy builder (extracted from SceneLoader)
├── view-state-manager.ts          # Centralized ViewState initialization and validation
├── stats-aggregator.ts            # Accumulator stats aggregation across loaders
├── loader-registry.ts             # Lifecycle management for geometry loaders (Points, Lines, GSplats)
├── tolerance-computer.ts          # Unified tolerance computation for spatial queries
├── index.ts                       # Package exports
├── README.md                      # This documentation
│
├── loaders/                       # Unified loader infrastructure (see loaders/README.md)
│   ├── base-types.ts              # Common types (BaseViewState, LoadRange)
│   ├── range-loader.ts            # Unified encoding dispatch
│   ├── spatial-query-builder.ts   # Spatial query utilities
│   └── transferable-accumulator.ts # Zero-allocation buffer management
│
└── (related: ../workers/)         # Web Worker infrastructure
    ├── worker-pool.ts             # Pool manager with load balancing
    └── data-worker.ts             # Worker with WASM acceleration
```

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

### Architecture Refactoring (Modular Design)

The SceneLoader has been refactored into focused, testable modules:

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

**SceneGraphBuilder** (`scene-graph-builder.ts`):

- Builds hierarchical scene structure from Zarr metadata
- Pure data structure building (no THREE.js dependencies)
- Enumerates store contents

**NodeFactory** (`node-factory.ts`):

- Creates THREE.js scene nodes (Points, Lines, GSplats) from loaded data
- Geometry creation with proper dtype handling (Float32, Uint8, Float16)
- Material creation and colormap application
- Transform application and validation
- Picking system integration (shadow pick-node creation)

**Data Accumulators** (`data-accumulator.ts`):

- Zero-allocation buffer pooling for Points/Lines/GSplats
- Multi-type support (Float32, Uint8, Uint16)
- Eliminates per-frame allocations, reduces GC pressure

**Benefits of Modular Design**:

- **Testability**: Each module tested independently (88+ new tests)
- **Maintainability**: Clear responsibility boundaries
- **Reduced Complexity**: SceneLoader reduced from ~2000 to ~800 lines

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
): Promise<THREE.Group>

export async function updateView(
  viewState: Partial<ViewState>,
  loaderId?: string
): Promise<void>

export async function updateSceneForDimensions(
  dims: SimpleDims,
  scene: THREE.Group,
  loaderId?: string
): Promise<void>
  // Update scene when navigating dimensions
}
```

### 2. Chunk-Based Spatial Index

The `chunk-spatial-index.ts` provides efficient nD point queries using Morton/Hilbert-ordered chunks with bounding boxes.

**Core Features:**

- **Chunk-Based Indexing**: Uses Morton/Hilbert space-filling curves for spatial locality
- **Bounding Box Queries**: Each chunk has a bounding box for fast intersection tests
- **Memory Efficient**: Only stores chunk bounds, not individual point indices
- **nD Support**: Works with arbitrary dimensional data
- **Cache-Friendly**: Points sorted by space-filling curve for better cache utilization
- **Progressive Loading**: Load only chunks that intersect the current slice

**How It Works:**

```typescript
// 1. Load chunk spatial index from zarr (chunk_bounds array)
const chunkIndex = await loadChunkSpatialIndex(location, attrs);

// 2. Query for chunks near slice position
const chunkIndices = queryChunksForView(
  chunkIndex,
  slicePosition, // Current position in nD space
  tolerance // Search radius per dimension
);

// 3. Convert chunk indices to point ranges
const ranges = chunkIndicesToRanges(chunkIndex, chunkIndices);

// 4. Merge adjacent ranges for efficient loading
const merged = mergePointRanges(ranges);

// 5. Load only required data
for (const range of merged) {
  loadPointRange(range.start, range.end);
}
```

**Chunk Index Structure:**

```typescript
interface ChunkSpatialIndex {
  metadata: {
    ordering: 'morton' | 'hilbert'; // Space-filling curve (default: hilbert)
    ordering_dims: number[]; // Dimensions used for curve ordering
    slice_dims: number[]; // Dimensions used for slicing
    ordering_bits_per_dim: number; // Bits per dimension for encoding
    chunk_size: number;
    total_points: number;
    total_chunks: number;
    ndim: number;
  };
  chunkBounds: Float32Array; // Shape: (num_chunks, ndim, 2) - min/max per dimension
}
```

**Key Functions:**

```typescript
export async function loadChunkSpatialIndex(
  location: any,
  attrs: any
): Promise<ChunkSpatialIndex | null> {
  // Load chunk spatial index from zarr chunk_bounds array
}

export function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePos: number[],
  tolerance: number[]
): number[] {
  // Query for chunks that intersect the query box
}

export function chunkIndicesToRanges(
  index: ChunkSpatialIndex,
  chunkIndices: number[]
): PointRange[] {
  // Convert chunk indices to point ranges
}

export function mergePointRanges(ranges: PointRange[]): PointRange[] {
  // Merge overlapping/adjacent ranges
}
```

**Performance Benefits:**

- **10-100x faster queries** for large datasets with spatial locality
- **Reduced memory usage** by loading only visible points
- **Better cache utilization** through space-filling curve ordering
- **Simple architecture**: No complex grid structures, just bounding box checks

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

### Attribute Inheritance

Rendering attributes cascade through the scene hierarchy:

```typescript
// Child inherits from parent unless overridden
opacity: child.opacity ?? parent.opacity ?? 1.0;
gamma: child.gamma ?? parent.gamma ?? 1.0;
blending_mode: child.blending_mode ?? parent.blending_mode ?? 'additive';
```

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

- Spatial index queries (chunk bounding box intersection)
- nD visibility computation (hypersphere/ellipsoid intersection)
- Encoded data decoding (quantized, LUT, broadcasted)
- nD → 3D projection with visibility filtering

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
dataset.zarr/
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
  gamma?: number; // 0.2 to 2.0
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

// The PointSpatialIndexLoader uses the index internally:
const chunkIndices = queryChunksForView(index, slicePos, tolerance);
const ranges = chunkIndicesToRanges(index, chunkIndices);
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
import { loadScene, updateView } from 'luxar-viewer/data';

// Load a Zarr dataset (point spatial index required)
const scene = await loadScene('http://server.com/data/points.zarr');

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
import { loadScene, updateSceneForDimensions } from 'luxar-viewer/data';
import type { SimpleDims } from '@luxar/player/types';

// Load 5D dataset (x, y, z, time, channel)
const scene = await loadScene('http://server.com/data/5d-points.zarr');

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
import { DirectoryNavigator } from 'luxar-viewer/data';

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
import { loadScene, updateView, dispose } from 'luxar-viewer/data';

// Create multiple independent loaders for different datasets
const scene1 = await loadScene('http://server.com/data1.zarr', config, 'loader1');
const scene2 = await loadScene('http://server.com/data2.zarr', config, 'loader2');

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
import { getCacheStats, clearCaches } from 'luxar-viewer/data';

// Monitor cache usage for default loader
const stats = getCacheStats();
if (stats) {
  for (const [path, cacheInfo] of stats) {
    console.log(`${path}: ${cacheInfo.hitRate * 100}% hit rate`);
  }
}

// Clear caches for specific loader
clearCaches('loader1');

// Or clear default loader cache
clearCaches();
```

### Advanced Instance Management

```typescript
import { SceneLoaderManager, DataMonitorManager } from 'luxar-viewer/data';

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
  loader.clearCaches();
}

// Reset for testing
SceneLoaderManager.reset();
DataMonitorManager.reset();
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
// Solution: Configure cache limits
const scene = await loadScene(url, {
  maxMemoryMB: 200, // Limit cache to 200MB
  evictionStrategy: 'lfu', // Use least-frequently-used eviction
});

// Monitor and clear cache as needed
const stats = getCacheStats();
if (stats.get('/points')?.totalMemory > threshold) {
  clearCaches();
}
```

---

## Configuration

### Loader Configuration

```typescript
interface LoaderConfig {
  // Memory management
  maxMemoryMB?: number; // Maximum memory for caching (default: 500)

  // Cache strategy
  evictionStrategy?: 'lru' | 'lfu'; // Cache eviction strategy (default: 'lru')

  // UI
  enableMonitor?: boolean; // Enable data loading monitor UI (default: true)

  // Debug
  debug?: boolean; // Enable debug logging (default: false)
}

// Usage
const scene = await loadScene(url, {
  maxMemoryMB: 1000,
  evictionStrategy: 'lru',
  enableMonitor: true,
  debug: false,
});
```

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

| Function                                           | Description                                          |
| -------------------------------------------------- | ---------------------------------------------------- |
| `loadScene(url, config?, loaderId?)`               | Load complete Zarr dataset with chunk-based indexing |
| `updateView(viewState, loaderId?)`                 | Update all points for new view state                 |
| `updateSceneForDimensions(dims, scene, loaderId?)` | Update scene when navigating dimensions              |
| `getCacheStats(loaderId?)`                         | Get cache statistics for monitoring                  |
| `clearCaches(loaderId?)`                           | Clear caches to free memory                          |
| `dispose(loaderId?)`                               | Clean up resources (specific or all)                 |

### Instance Management (scene-loader-manager.ts)

| Class/Method                               | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `SceneLoaderManager`                       | Singleton manager for SceneLoader instances |
| `getInstance()`                            | Get the singleton manager instance          |
| `createLoader(id, config?, setAsDefault?)` | Create a new loader instance                |
| `getLoader(id)`                            | Get a specific loader by ID                 |
| `getDefaultLoader()`                       | Get the default loader instance             |
| `getAllLoaders()`                          | Get all active loader instances             |
| `destroyLoader(id)`                        | Dispose and remove a specific loader        |
| `destroyAll()`                             | Dispose all loaders and reset manager       |
| `reset()`                                  | Reset singleton (for testing)               |

### Scene Loading (scene-loader.ts)

| Class/Method                | Description                               |
| --------------------------- | ----------------------------------------- |
| `SceneLoader`               | Main scene loader orchestrator            |
| `constructor(config?, id?)` | Create loader with config and optional ID |
| `loadScene(url)`            | Load complete scene from zarr store       |
| `updateView(viewState)`     | Update all loaders with new view state    |
| `getCacheStats()`           | Get cache statistics from all loaders     |
| `clearCaches()`             | Clear all loader caches                   |
| `showMonitor()`             | Show data loading monitor UI              |
| `hideMonitor()`             | Hide data loading monitor UI              |
| `toggleMonitor()`           | Toggle data loading monitor UI            |
| `retryFailedLoader(nodeId)` | Retry a failed loader                     |
| `retryAllFailedLoaders()`   | Retry all failed loaders                  |
| `dispose()`                 | Clean up all resources                    |

### Scene Graph Builder (scene-graph-builder.ts)

| Class/Method                  | Description                            |
| ----------------------------- | -------------------------------------- |
| `SceneGraphBuilder`           | Builds scene hierarchy from Zarr       |
| `constructor(store, rootLoc)` | Create builder with store and location |
| `buildSceneGraph(rootAttrs)`  | Build complete scene graph             |
| `getNodesOfType(node, type)`  | Get all nodes of a specific type       |

### Node Factory (node-factory.ts)

| Class/Method                                    | Description                                   |
| ----------------------------------------------- | --------------------------------------------- |
| `NodeFactory`                                   | Creates THREE.js scene nodes from loaded data  |
| `createPointsNode(path, attrs, data, loader)`   | Create Points node with geometry and material  |
| `createLinesNode(path, attrs, data, loader)`    | Create instanced Lines mesh                    |
| `createGSplatsNode(path, attrs, data, loader)`  | Create instanced GSplats mesh                  |
| `createPointsGeometry(data, maxR, maxS)`        | Create THREE.js geometry for points            |
| `createPointsMaterial(attrs, rScale, sScale)`   | Create shader material for points              |
| `applyTransform(object, transform)`             | Apply 4x4 column-major transform               |
| `validateTransformFormat(transform)`            | Detect row-major vs column-major format         |

### Point Spatial Index Loader (point-spatial-index-loader.ts)

| Class/Method                          | Description                               |
| ------------------------------------- | ----------------------------------------- |
| `PointSpatialIndexLoader`             | Loader using chunk-based spatial indexing |
| `constructor(location, node, config)` | Create loader with chunk index support    |
| `updateView(viewState)`               | Update for new view (reloads currently)   |
| `getCacheStats()`                     | Get cache statistics                      |
| `clearCache()`                        | Clear cached data                         |
| `dispose()`                           | Clean up resources                        |

### Chunk Spatial Index Functions (chunk-spatial-index.ts)

| Function                                    | Description                                     |
| ------------------------------------------- | ----------------------------------------------- |
| `loadChunkSpatialIndex(location, attrs)`    | Load chunk spatial index from zarr chunk_bounds |
| `queryChunksForView(index, pos, tol)`       | Query chunks that intersect the given box       |
| `chunkIndicesToRanges(index, chunkIndices)` | Convert chunk indices to point ranges           |
| `mergePointRanges(ranges)`                  | Merge overlapping or adjacent ranges            |

### Monitoring Management (data-monitor-manager.ts)

| Class/Method                            | Description                                |
| --------------------------------------- | ------------------------------------------ |
| `DataMonitorManager`                    | Singleton manager for monitor UI instances |
| `getInstance()`                         | Get the singleton manager instance         |
| `createMonitor(id, container, config?)` | Create a new monitor UI instance           |
| `getMonitor(id)`                        | Get a specific monitor by ID               |
| `getDefaultMonitor()`                   | Get the default monitor instance           |
| `showMonitor(id?)`                      | Show specific or default monitor           |
| `hideMonitor(id?)`                      | Hide specific or default monitor           |
| `toggleMonitor(id?)`                    | Toggle specific or default monitor         |
| `destroyMonitor(id)`                    | Dispose and remove a specific monitor      |
| `reset()`                               | Reset singleton (for testing)              |

### Directory Navigation (directory-navigator.ts)

| Class/Method           | Description                             |
| ---------------------- | --------------------------------------- |
| `DirectoryNavigator`   | Multi-strategy directory browser        |
| `navigate(path)`       | Navigate to directory or dataset        |
| `getFullUrl(path)`     | Get full URL for a path                 |
| `canListDirectories()` | Check if directory listing is supported |

### Utility Functions (zarr-loader-utils.ts)

| Function                                        | Description                             |
| ----------------------------------------------- | --------------------------------------- |
| `normalizeZarrPath(path, baseUrl?)`             | Normalize path to valid Zarr URL        |
| `extractDimensionMetadata(attrs)`               | Extract dimensions from zarr attributes |
| `inheritRenderingAttributes(attrs, parent)`     | Apply attribute inheritance             |
| `validatePointsData(positions, expected, ndim)` | Validate points data                    |
| `calculateInitialSlicePosition(dims)`           | Calculate initial nD slice position     |
| `isPointsGroup(attrs, name)`                    | Check if group contains points          |
| `calculateBoundingBox(positions, ndim)`         | Calculate nD bounding box               |
| `processTransformAttribute(transform)`          | Process transform from zarr metadata    |
| `estimatePointsMemory(n, ndim, ...)`            | Estimate memory usage in MB             |
| `validateRenderingAttributes(attrs)`            | Validate and apply defaults             |
| `determineLoadingStrategy(n, memory)`           | Choose loading strategy based on size   |

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
