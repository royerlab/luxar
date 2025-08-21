# Luxar Data Package

> High-performance Zarr data loading and nD slicing for point cloud visualization

## Overview

The Luxar Data package provides the critical data loading infrastructure for visualizing massive nD point cloud datasets. It handles Zarr store access, hierarchical scene loading, dimension metadata extraction, and intelligent slicing operations for multi-dimensional data navigation.

### Key Features

- **Zarr-Native Loading**: Direct integration with Zarr stores for chunked data access
- **nD Data Support**: Handle arbitrary-dimensional point clouds with automatic slicing
- **Hierarchical Scenes**: Load nested scene structures with inheritance
- **Smart Slicing**: Radius-based hypersphere intersection for smooth navigation
- **Auto-Broadcasting**: Intelligent replication of point groups across non-displayed dimensions
- **Directory Navigation**: Multi-strategy server navigation (WebDAV, S3, nginx)
- **GPU Optimization**: Automatic data format conversion for WebGL compatibility
- **Streaming Ready**: Progressive loading for massive datasets

### Package Architecture

```
data/
├── zarr-loader.ts         # Core Zarr loading and nD slicing
├── lazy-data-manager.ts   # Intelligent chunk caching and memory management
├── directory-navigator.ts # Server-agnostic directory browsing
└── README.md             # This documentation
```

---

## Components

### 1. Zarr Loader

The `zarr-loader.ts` provides comprehensive Zarr data loading with nD slicing capabilities.

**Core Features:**

- Hierarchical scene graph loading
- Scene-level dimension metadata extraction
- Radius-based nD slicing for visibility
- Attribute inheritance in nested structures
- GPU-optimized format conversion
- Fallback handling for optional attributes

**Data Pipeline:**

```typescript
// 1. Load Zarr store with consolidated metadata
const store = await openStore(url);

// 2. Extract scene dimensions
const dims = extractSceneDimensions(store);

// 3. Load point clouds hierarchically
const pointClouds = await loadPointClouds(store, dims);

// 4. Perform nD slicing
const sliced = sliceToDisplayDimensions(pointClouds, dims);

// 5. Create GPU-ready geometry
const geometry = createBufferGeometry(sliced);
```

**Key Functions:**

```typescript
export async function loadFromZarr(
  url: string,
  dims?: SimpleDims
): Promise<{
  objects: THREE.Object3D[];
  sceneDims?: SimpleDims;
}> {
  // Main entry point for loading Zarr datasets
}

async function loadPointCloud(
  store: ZarrStore,
  path: string,
  attrs: ZarrGroupAttrs,
  dims?: SimpleDims
): Promise<THREE.Points> {
  // Load individual point cloud with slicing
}

function sliceToDisplayDimensions(
  positions: Float32Array,
  colors: Float32Array | null,
  radii: Float32Array | null,
  dims: SimpleDims
): SlicedData {
  // Perform nD to 3D slicing
}
```

### 2. Lazy Data Manager

The `lazy-data-manager.ts` provides intelligent chunk-based loading and caching for massive datasets.

**Core Features:**

- **Memory-Aware Caching**: Automatic memory limit detection and management
- **Smart Chunk Loading**: Only loads visible data chunks on demand
- **Preloading Strategy**: Anticipates navigation and preloads adjacent chunks
- **LRU Eviction**: Keeps most recently used chunks in cache
- **Concurrent Load Protection**: Prevents duplicate requests for same chunk
- **Dynamic Memory Adjustment**: Adapts to available system memory

**Memory Management:**

```typescript
// Automatic memory detection
const memory = detectMemory();
// Uses 80% of available heap memory for safety
const cacheSize = memory.recommendedCacheMB;

// Cache eviction when limit reached
if (totalCacheSize + chunkSize > maxMemory) {
  evictOldestChunks();
}
```

**Chunk Loading Pipeline:**

```typescript
// 1. Check if chunk is in cache
const cached = cache.get(chunkId);
if (cached) return cached.data;

// 2. Check if chunk is currently loading
const pending = loadingChunks.get(chunkId);
if (pending) return pending;

// 3. Load chunk from Zarr store
const promise = loadChunkFromStore(array, indices);
loadingChunks.set(chunkId, promise);

// 4. Add to cache when loaded
const data = await promise;
addToCache(chunkId, data);

// 5. Preload adjacent chunks
preloadAdjacentChunks(indices);
```

**Key Functions:**

```typescript
class LazyDataManager {
  // Load a specific chunk
  async loadChunk(
    array: ZarrArray,
    arrayPath: string,
    chunkIndices: number[]
  ): Promise<ArrayBuffer | TypedArray>;

  // Preload chunks around current position
  async preloadChunks(
    array: ZarrArray,
    arrayPath: string,
    currentPosition: number[],
    fullyLoadedDims: number[]
  ): Promise<void>;

  // Get cache statistics
  getCacheStats(): {
    numChunks: number;
    totalSizeMB: number;
    maxSizeMB: number;
    utilizationPercent: number;
  };

  // Clear all cached data
  clearCache(): void;
}
```

**Intelligent Preloading:**

The system automatically determines which chunks to preload based on:

1. **Current Position**: The active chunk being viewed
2. **Array Dimensions**: Which dimensions are fully vs partially loaded
3. **Preload Radius**: How many adjacent chunks to preload (default: 1)
4. **Memory Constraints**: Only preloads if memory available

```typescript
// For a positions array with shape [n_points, n_coords]
// - Dimension 0 (points): Load current chunk ± preloadRadius
// - Dimension 1 (coords): Always fully loaded
const requiredChunks = calculateRequiredChunks(
  arrayShape,
  chunkShape,
  currentPosition,
  fullyLoadedDimensions,
  preloadRadius
);
```

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

For datasets with more than 3 dimensions:

```typescript
// Hypersphere intersection for point visibility
for each point:
  // Calculate distance in non-displayed dimensions
  distance = 0
  for each non-displayed dimension d:
    delta = point[d] - currentSlice[d]
    distance += delta * delta

  // Point visible if within radius
  radius = point.radius || defaultRadius
  if sqrt(distance) <= radius:
    include point in slice
```

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
├── .zmetadata            # Consolidated metadata
└── point_cloud/
    ├── .zattrs           # Node attributes
    ├── .zgroup
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

### Lazy Loading for Large Datasets

For datasets exceeding available memory, the lazy loading system automatically activates:

```typescript
// Automatic detection based on dataset size
const dataSize = calculateDatasetSize(shape, dtype);
const availableMemory = detectMemory().recommendedCacheMB;

if (dataSize > availableMemory * 0.5) {
  // Enable lazy loading
  userData.isLazyLoaded = true;
  userData.lazyDataManager = new LazyDataManager(config);
}
```

**Benefits:**

- Load 100GB+ datasets on systems with 8GB RAM
- Smooth navigation through temporal/dimensional slices
- No manual configuration required
- Automatic memory management

**How It Works:**

1. **Initial Load**: Only loads the current visible slice
2. **Navigation**: As user navigates, loads new chunks and caches them
3. **Preloading**: Anticipates user movement and preloads adjacent slices
4. **Eviction**: Automatically removes least-recently-used chunks when memory fills

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
import { loadFromZarr } from './data/zarr-loader';

// Load a Zarr dataset
const { objects, sceneDims } = await loadFromZarr('http://server.com/data/points.zarr');

// Add to scene
objects.forEach((obj) => scene.add(obj));

// Use scene dimensions for navigation
if (sceneDims) {
  initializeDimensionNavigation(sceneDims);
}
```

### nD Dataset Loading

```typescript
// Load 5D dataset (x, y, z, time, channel)
const dims: SimpleDims = {
  ndim: 5,
  displayed: [0, 1, 2], // Show x, y, z
  currentStep: new Float32Array([0, 0, 0, 10, 2]),
  metadata: [
    { name: 'x', unit: 'μm', range: [-100, 100], display: true },
    { name: 'y', unit: 'μm', range: [-100, 100], display: true },
    { name: 'z', unit: 'μm', range: [-50, 50], display: true },
    { name: 'time', unit: 'ms', range: [0, 1000], display: false },
    { name: 'channel', unit: '', range: [0, 4], display: false },
  ],
};

const { objects } = await loadFromZarr(url, dims);
```

### Directory Navigation

```typescript
import { DirectoryNavigator } from './data/directory-navigator';

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
  await loadFromZarr(selected.path);
}
```

### Custom Slicing

```typescript
import { slicePoints } from '../utils/slicing';

// Custom slicing for time-series data
function sliceTimePoint(positions, colors, timeIndex, timeRange) {
  const indices = [];

  for (let i = 0; i < positions.length / 4; i++) {
    const t = positions[i * 4 + 3]; // Time is 4th dimension
    if (Math.abs(t - timeIndex) <= timeRange) {
      indices.push(i);
    }
  }

  return extractIndices(positions, colors, indices);
}
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

**Problem: Missing consolidated metadata**

```typescript
// Solution: Fall back to standard loading
try {
  const store = await openConsolidated(url);
} catch (e) {
  console.warn('No consolidated metadata, using standard loading');
  const store = await openStore(url);
}
```

**Problem: Large dataset performance**

```typescript
// Solution: Implement progressive loading
async function loadProgressive(url, viewport) {
  // Load only visible chunks
  const visibleChunks = calculateVisibleChunks(viewport);
  return loadChunks(url, visibleChunks);
}
```

---

## Configuration

### Loading Options

```typescript
interface LoadOptions {
  // Performance
  maxPoints?: number; // Limit total points loaded
  chunkBatchSize?: number; // Chunks to load in parallel

  // Slicing
  sliceRadius?: number; // Default radius for nD slicing
  sliceTolerance?: number; // Distance tolerance

  // Defaults
  defaultColor?: [number, number, number];
  defaultRadius?: number;
  defaultOpacity?: number;

  // Optimization
  useConsolidated?: boolean; // Use .zmetadata if available
  enableCaching?: boolean; // Cache loaded chunks
}
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

### zarr-loader.ts

| Function                                    | Description                                |
| ------------------------------------------- | ------------------------------------------ |
| `loadFromZarr(url, dims?)`                  | Load complete Zarr dataset                 |
| `openStore(url)`                            | Open Zarr store with consolidated metadata |
| `extractSceneDimensions(store)`             | Extract dimension metadata from scene      |
| `loadPointCloud(store, path, attrs, dims)`  | Load individual point cloud                |
| `sliceToDisplayDimensions(data, dims)`      | Perform nD to 3D slicing                   |
| `inheritRenderingAttributes(attrs, parent)` | Apply attribute inheritance                |

### lazy-data-manager.ts

| Method                                  | Description                          |
| --------------------------------------- | ------------------------------------ |
| `constructor(config)`                   | Initialize with memory configuration |
| `loadChunk(array, path, indices)`       | Load specific chunk from Zarr array  |
| `preloadChunks(array, path, pos, dims)` | Preload chunks around position       |
| `calculateRequiredChunks(...)`          | Determine which chunks to load       |
| `getCacheStats()`                       | Get current cache utilization        |
| `clearCache()`                          | Clear all cached chunks              |
| `dispose()`                             | Clean up resources and memory        |

### directory-navigator.ts

| Method                   | Description                      |
| ------------------------ | -------------------------------- |
| `navigate(path)`         | Navigate to directory or dataset |
| `checkIfZarr(url)`       | Detect if URL is a Zarr dataset  |
| `tryWebDAV(url)`         | Attempt WebDAV PROPFIND          |
| `parseHTMLListing(html)` | Extract entries from HTML        |
| `loadIndexManifest(url)` | Load index.json listing          |

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
// Enable debug output
window.__luxarDebug = { data: true };

// Logs will include:
// - Store initialization
// - Dimension extraction
// - Group enumeration
// - Slicing operations
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
