# Luxar Data Package

> High-performance Zarr data loading and nD slicing for point cloud visualization

## Overview

The Luxar Data package provides the critical data loading infrastructure for visualizing massive nD point cloud datasets. It handles Zarr store access, hierarchical scene loading, dimension metadata extraction, and intelligent slicing operations for multi-dimensional data navigation.

### Key Features

- **Spatial Index (Required)**: All datasets MUST have spatial indices for loading
- **Zarr-Native Loading**: Direct integration with Zarr stores for chunked data access
- **nD Data Support**: Handle arbitrary-dimensional point clouds with automatic slicing
- **Hierarchical Scenes**: Load nested scene structures with inheritance
- **Smart Slicing**: Radius-based hypersphere intersection for smooth navigation
- **Auto-Broadcasting**: Intelligent replication of point groups across non-displayed dimensions
- **Directory Navigation**: Multi-strategy server navigation (WebDAV, S3, nginx)
- **GPU Optimization**: Automatic data format conversion for WebGL compatibility
- **Efficient Caching**: Range-based caching with LRU/LFU eviction strategies

### Package Architecture

```
data/
├── zarr-loader.ts              # Main API entry point for loading scenes
├── scene-loader.ts             # Orchestrates hierarchical scene loading
├── scene-loader-manager.ts     # Singleton manager for SceneLoader instances
├── spatial-index-loader.ts     # Loads data using spatial index queries (required)
├── spatial-index.ts            # Core spatial index query implementation
├── range-cache.ts              # Intelligent caching for range-based queries
├── data-monitor-manager.ts     # Singleton manager for monitoring UI instances
├── dimension-update-manager.ts # Bridges dimension navigation with loaders
├── directory-navigator.ts      # Multi-strategy server directory browsing
├── data-loader-types.ts        # TypeScript interfaces and types
├── zarr-loader-utils.ts        # Pure utility functions for data processing
└── README.md                   # This documentation
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

**Note**: All datasets must be generated with the Python Luxar compiler to include spatial indices. The TypeScript viewer requires spatial indices and will not load datasets without them.

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
//    - Loads spatial indices for each point cloud
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

### 2. Spatial Index (Required as of v0.3+)

The `spatial-index.ts` provides efficient nD point queries using grid-based spatial partitioning.

⚠️ **IMPORTANT**: Spatial indices are now MANDATORY for all datasets. The system will throw an error if a dataset lacks a spatial index.

**Core Features:**

- **Grid-Based Partitioning**: Divides nD space into regular grid cells
- **Sparse Storage**: Only stores occupied cells for memory efficiency
- **Fast Range Queries**: O(occupied_cells) query complexity instead of O(total_points)
- **nD Support**: Works with arbitrary dimensional data
- **Cache-Friendly**: Points sorted by spatial locality for better cache utilization
- **Progressive Loading**: Load only points near current slice position

**How It Works:**

```typescript
// 1. Load spatial index from zarr
const index = await loadSpatialIndex(group);

// 2. Query for points near slice position
const ranges = querySpatialIndex(
  index,
  slicePosition, // Current position in nD space
  tolerance // Search radius per dimension
);

// 3. Merge adjacent ranges for efficient loading
const merged = mergePointRanges(ranges);

// 4. Calculate which zarr chunks to load
const chunks = calculateChunksToLoad(merged, chunkSize);

// 5. Load only required data
for (const range of merged) {
  loadPointRange(range.start, range.end);
}
```

**Spatial Index Structure:**

```typescript
interface SpatialIndex {
  metadata: {
    grid_shape: number[]; // Grid dimensions [nx, ny, nz, ...]
    grid_origin: number[]; // Minimum coordinate per dimension
    cell_size: number[]; // Size of each cell
    num_occupied: number; // Number of non-empty cells
    dimensions: number; // Number of dimensions
  };
  occupiedCells: Uint32Array; // Flattened nD grid coordinates
  cellRanges: BigUint64Array; // [start, end] ranges per cell
}
```

**Key Functions:**

```typescript
export async function loadSpatialIndex(
  group: any,
  signal?: AbortSignal
): Promise<SpatialIndex | null> {
  // Load spatial index from zarr group
}

export function querySpatialIndex(
  index: SpatialIndex,
  slicePos: number[],
  tolerance: number[]
): PointRange[] {
  // Query for points within tolerance of position
}

export function mergePointRanges(ranges: PointRange[]): PointRange[] {
  // Merge overlapping/adjacent ranges
}

export function calculateChunksToLoad(ranges: PointRange[], chunkSize: number): Set<number> {
  // Determine which zarr chunks are needed
}
```

**Performance Benefits:**

- **10-100x faster queries** for large datasets with spatial locality
- **Reduced memory usage** by loading only visible points
- **Better cache utilization** through spatial sorting
- **Scalable to billions of points** with appropriate grid sizing

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

### Broadcasting

Broadcasting allows point groups to appear across all values of specified non-displayed dimensions without data duplication:

```typescript
// Broadcast dimensions are explicitly specified in zarr attributes
// Set via Python API: scene.add_points(..., broadcast_dims=["Time", "Channel"])
const broadcastDims = attrs.broadcast_dims || [];

// During navigation, broadcast groups are handled specially
if (group.broadcast_dims.includes(current_dimension)) {
  // Load all points once and display them at every dimension value
  // This avoids duplicating data across all dimension values
  loadAllPoints();
} else {
  // Normal slicing based on current dimension value
  loadSlice(dimension_value);
}
```

Benefits:

- **Memory Efficient**: No data duplication needed (66% reduction for 3 time points)
- **Explicit Control**: Clear API for specifying broadcast behavior
- **Backwards Compatible**: Works transparently with existing datasets
- **Cache Aware**: Proper cache key isolation prevents data corruption

### Attribute Inheritance

Rendering attributes cascade through the scene hierarchy:

```typescript
// Child inherits from parent unless overridden
opacity: child.opacity ?? parent.opacity ?? 1.0;
gamma: child.gamma ?? parent.gamma ?? 1.0;
blending_mode: child.blending_mode ?? parent.blending_mode ?? 'additive';
```

---

## Data Formats

### Zarr Structure

Expected Zarr store structure:

```
dataset.zarr/
├── .zattrs                # Scene metadata
├── .zgroup                # Zarr group marker
├── .zmetadata            # Consolidated metadata (recommended)
└── point_cloud/
    ├── .zattrs           # Node attributes
    ├── .zgroup
    ├── spatial_index/    # Spatial index group - REQUIRED
    │   ├── .zattrs       # Index metadata
    │   ├── .zgroup
    │   ├── occupied_cells/  # Grid cell coordinates
    │   └── cell_ranges/     # Point ranges per cell
    ├── positions/        # Float32[N, D] - required
    ├── colors/           # Float32[N, 3] - optional
    ├── radii/            # Float32[N] - optional
    └── sharpness/        # Float32[N] - optional
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

  // Broadcasting
  broadcast_dims?: string[]; // Dimension names to auto-broadcast across

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

### Spatial Index for Efficient Queries (Required)

The spatial index dramatically improves performance for large nD datasets:

```typescript
// Spatial index is loaded automatically by SceneLoader
// If missing, will throw: "[❌] No spatial index found for /path.
// Please rebuild the dataset with spatial index support."

// The SpatialIndexLoader uses the index internally:
const ranges = querySpatialIndex(index, slicePos, tolerance);
const visiblePoints = await loadRanges(ranges);

// Query performance:
// With index: O(m) - scan only occupied cells (m << n)
// Without index: Not supported - datasets must have spatial indices
```

**When to Use Spatial Index:**

- Datasets with >100K points
- High-dimensional data (4D+)
- Sparse point distributions
- Time-series or multi-channel data

**Grid Size Selection:**

```python
# In Python during compilation
scene.add_points(
  "points",
  positions=data,
  grid_shape=(10, 10, 10, 5)  # Grid per dimension
)

# Automatic grid sizing
# If not specified, uses heuristic based on:
# - Number of points
# - Dimensional extents
# - Target cells per dimension (10-20)
```

### Spatial Index Requirement

⚠️ **IMPORTANT**: All datasets MUST have spatial indices. Datasets without spatial indices will fail to load with an error.

The spatial index enables:

```typescript
// The SceneLoader automatically uses spatial indices for efficient loading
// You don't need to interact with the spatial index directly - it's handled internally

// When you update the view:
await updateView({
  displayDims: [0, 1, 2],
  slicePosition: [x, y, z, t],
  tolerance: [0, 0, 0, radius],
});
// The loader automatically queries the spatial index and loads only visible points
```

**Benefits:**

- Load only visible points from massive datasets
- Efficient nD queries without scanning all points
- Smooth navigation through temporal/dimensional slices
- Automatic caching of frequently accessed ranges

**How It Works:**

1. **Initial Load**: Queries spatial index for visible points
2. **Navigation**: As user navigates, queries update to find new visible points
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
import { loadScene, updateView } from '@luxar/player/data';

// Load a Zarr dataset (spatial index required)
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
import { loadScene, updateSceneForDimensions } from '@luxar/player/data';
import type { SimpleDims } from '@luxar/player/types';

// Load 5D dataset (x, y, z, time, channel)
const scene = await loadScene('http://server.com/data/5d-points.zarr');

// Dimensions are automatically extracted from zarr metadata
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
import { DirectoryNavigator } from '@luxar/player/data';

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
import { loadScene, updateView, dispose } from '@luxar/player/data';

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
import { getCacheStats, clearCaches } from '@luxar/player/data';

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
import { SceneLoaderManager, DataMonitorManager } from '@luxar/player/data';

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

**Problem: Dataset without spatial index**

```typescript
// Error: "[❌] No spatial index found for /point_cloud. Please rebuild the dataset with spatial index support."
// Solution: Regenerate dataset with Python compiler

// Python code:
from luxar import Scene
scene = Scene()
scene.add_points("points", positions=data)
scene.compile("output.zarr")  # Spatial index created automatically
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

| Function                                           | Description                                   |
| -------------------------------------------------- | --------------------------------------------- |
| `loadScene(url, config?, loaderId?)`               | Load complete Zarr dataset with spatial index |
| `updateView(viewState, loaderId?)`                 | Update all point clouds for new view state    |
| `updateSceneForDimensions(dims, scene, loaderId?)` | Update scene when navigating dimensions       |
| `getCacheStats(loaderId?)`                         | Get cache statistics for monitoring           |
| `clearCaches(loaderId?)`                           | Clear caches to free memory                   |
| `dispose(loaderId?)`                               | Clean up resources (specific or all)          |

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
| `dispose()`                 | Clean up all resources                    |

### Spatial Index Loading (spatial-index-loader.ts)

| Class/Method                          | Description                              |
| ------------------------------------- | ---------------------------------------- |
| `SpatialIndexLoader`                  | Loader using spatial indices             |
| `constructor(location, node, config)` | Create loader with spatial index support |
| `updateView(viewState)`               | Update for new view (reloads currently)  |
| `getCacheStats()`                     | Get cache statistics                     |
| `clearCache()`                        | Clear cached data                        |
| `dispose()`                           | Clean up resources                       |

### Spatial Index Functions (spatial-index.ts)

| Function                              | Description                               |
| ------------------------------------- | ----------------------------------------- |
| `loadSpatialIndex(group)`             | Load spatial index from zarr group        |
| `querySpatialIndex(index, pos, tol)`  | Query points within tolerance of position |
| `mergePointRanges(ranges)`            | Merge overlapping or adjacent ranges      |
| `calculateChunksToLoad(ranges, size)` | Calculate which zarr chunks to load       |
| `estimateMemoryUsage(ranges, bytes)`  | Estimate memory for loading point ranges  |
| `debugSpatialIndex(index)`            | Create debug summary of spatial index     |

### Range Cache (range-cache.ts)

| Class/Method                   | Description                           |
| ------------------------------ | ------------------------------------- |
| `RangeCache`                   | Cache for spatial index range queries |
| `get(arrayPath, ranges)`       | Get cached data if available          |
| `set(arrayPath, ranges, data)` | Store data in cache                   |
| `has(arrayPath, ranges)`       | Check if ranges are cached            |
| `getStats()`                   | Get cache statistics                  |
| `clear()`                      | Clear all cached data                 |
| `getMemoryInfo()`              | Get memory usage information          |

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

| Function                                            | Description                             |
| --------------------------------------------------- | --------------------------------------- |
| `normalizeZarrPath(path, baseUrl?)`                 | Normalize path to valid Zarr URL        |
| `extractDimensionMetadata(attrs)`                   | Extract dimensions from zarr attributes |
| `inheritRenderingAttributes(attrs, parent)`         | Apply attribute inheritance             |
| `validatePointCloudData(positions, expected, ndim)` | Validate point cloud data               |
| `calculateInitialSlicePosition(dims)`               | Calculate initial nD slice position     |
| `isPointCloudGroup(attrs, name)`                    | Check if group contains point cloud     |
| `calculateBoundingBox(positions, ndim)`             | Calculate nD bounding box               |
| `processTransformAttribute(transform)`              | Process transform from zarr metadata    |
| `estimatePointCloudMemory(n, ndim, ...)`            | Estimate memory usage in MB             |
| `validateRenderingAttributes(attrs)`                | Validate and apply defaults             |
| `determineLoadingStrategy(n, memory)`               | Choose loading strategy based on size   |

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
