# Luxar Viewer - Architecture Diagrams

**Version**: 2.0.0
**Last Updated**: 2026-05-08

Visual reference for understanding the Luxar viewer architecture, data flow, and key algorithms.

The authoritative layer order lives in `.dependency-cruiser.cjs`:
`types → config → cache → rendering → data → scene → input → ui → core`,
with `utils`, `themes`, `wasm`, `workers`, `profiling`, and `controls`
as cross-cutting concerns importable from any layer.

---

## Table of Contents

1. [Data Flow Pipeline](#1-data-flow-pipeline)
2. [Component Dependency Graph](#2-component-dependency-graph)
3. [nD Slicing Visualization](#3-nd-slicing-visualization)
4. [Rendering Pipeline](#4-rendering-pipeline)
5. [Cache Architecture](#5-cache-architecture)
6. [Input Event Routing](#6-input-event-routing)

---

## 1. Data Flow Pipeline

### Overview: Python → Zarr → TypeScript → WebGL

The pipeline is split per geometry type (Points, Lines, GSplats,
Mesh), each with its own spatial-index loader — except Mesh, whose
loader is whole-node. The `SceneLoader` orchestrates them and a
`WorkerPool` offloads decode + projection.

```mermaid
graph TB
    subgraph "Python (luxar.io)"
        A[Points / Lines / GSplats / Mesh<br/>nD attributes per geometry]
        B[ArrayEncoder<br/>Broadcasting, LUT, Quantization]
        C[SpatialIndex Builder<br/>Morton/Hilbert ordering]
        D[ZarrWriter<br/>Chunked, compressed]
    end

    subgraph "Zarr Archive"
        E[.zgroup<br/>Scene metadata]
        F[geometry-typed arrays<br/>positions/vertices/centers]
        G[chunk_bounds/<br/>Spatial index]
        H[Attribute arrays<br/>colors/radii/widths/cholesky]
    end

    subgraph "TypeScript Loader (data/)"
        I[SceneLoader<br/>Hierarchical, _updateInProgress mutex]
        J1[PointSpatialIndexLoader]
        J2[LinesSpatialIndexLoader]
        J3[GSplatsSpatialIndexLoader]
        K[ChunkSpatialIndex<br/>nD bbox queries]
        L[WorkerPool<br/>Decode + project off-thread]
    end

    subgraph "Rendering (rendering/)"
        M[THREE.BufferGeometry<br/>GPU buffers]
        N[Point/Line/GSplat materials<br/>Camera-aware, HDR]
        O[MaterialManager<br/>Cache, registration, dispose]
        P[PostProcessingManager<br/>HDR + mega-shader pipeline]
    end

    subgraph "Display"
        Q[WebGL Framebuffer<br/>GPU rendering]
        R[Canvas<br/>User display]
    end

    A --> B --> C --> D
    D --> E & F & G & H
    E & F & G & H --> I
    I --> J1 & J2 & J3
    J1 & J2 & J3 --> K
    J1 & J2 & J3 -.via Comlink.-> L
    L --> M --> N
    N --> O --> P --> Q --> R

    style A fill:#e1f5ff
    style D fill:#ffe1e1
    style I fill:#fff5e1
    style P fill:#e1ffe1
    style R fill:#f0f0f0
```

### Key Transformations

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Encoding** | Float32 arrays | uint8/uint16 + metadata | Compression (2-10x) |
| **Spatial Ordering** | Random order | Morton/Hilbert order | Locality optimization |
| **Chunking** | Full arrays | 10K-point chunks | Streaming support |
| **Indexing** | Point positions | Chunk bounding boxes | Fast queries |
| **Decoding** | Encoded chunks | Float32 arrays | Restore original values |
| **Projection** | nD positions | 3D display coords | Dimensionality reduction |
| **GPU Upload** | CPU arrays | GPU buffers | Hardware acceleration |
| **Shading** | Vertex positions | Pixel colors | Visual representation |

---

## 2. Component Dependency Graph

### Application Architecture

The dependency direction follows the layer order in
`.dependency-cruiser.cjs` strictly. Where the data-layer needs UI
behaviour (e.g. monitor events, dimension-slider construction), the
seam is inverted via a port: `SceneLoaderMonitorPort`,
`DimensionSlidersFactory`. The data layer knows the port; the UI
layer registers the implementation via `app.ts`.

```mermaid
graph TB
    subgraph "Entry Point"
        Main[main.ts<br/>Console interceptor<br/>Config validation]
    end

    subgraph "Orchestrator (core/)"
        App[LuxarApp<br/>Wiring + dispose<br/>safeDispose per component]
    end

    subgraph "ui/"
        RenderUI[RenderingControls]
        Recording[RecordingPanel]
        Layers[LayersPanel]
        Browser[DatasetBrowser]
        Monitor[DataLoadingMonitor]
        DimSliders[DimensionSliders]
    end

    subgraph "input/ + scene/"
        Input[InputHandler]
        Ctx[InputContextManager]
        Scene[SceneManager]
        Anim[AnimationController]
        Picking[PickingSystem]
    end

    subgraph "data/"
        SceneLoader[SceneLoader<br/>_updateInProgress mutex]
        Points[points/PointSpatialIndexLoader]
        Lines[lines/LinesSpatialIndexLoader]
        GSplats[gsplats/GSplatsSpatialIndexLoader]
        Pool[workers/WorkerPool<br/>initGeneration + pendingWorkers]
    end

    subgraph "rendering/ + cache/"
        PostProc[PostProcessingManager<br/>Mega-shader + bloom + FXAA]
        Materials[MaterialManager<br/>register/unregister]
        Cache[MultiLevelCachingStore<br/>L0 → L1 → L2 → network]
    end

    Main --> App
    App --> RenderUI & Recording & Layers & Browser & Monitor
    App --> Scene & Anim & Input & Picking
    App --> SceneLoader

    Scene --> PostProc & Materials
    Input --> Ctx
    Input -.factory port.-> DimSliders

    SceneLoader --> Points & Lines & GSplats
    Points & Lines & GSplats --> Cache
    Points & Lines & GSplats -.Comlink.-> Pool
    SceneLoader -.monitor port.-> Monitor

    classDef entry fill:#e1f5ff
    classDef orchestrator fill:#ffe1e1
    classDef ui fill:#ffe1f5
    classDef in fill:#fff5e1
    classDef data fill:#f5e1ff
    classDef rd fill:#e1ffe1

    class Main entry
    class App orchestrator
    class RenderUI,Recording,Layers,Browser,Monitor,DimSliders ui
    class Input,Ctx,Scene,Anim,Picking in
    class SceneLoader,Points,Lines,GSplats,Pool data
    class PostProc,Materials,Cache rd
```

### Layer Order (`.dependency-cruiser.cjs`)

```
types → config → cache → rendering → data → scene → input → ui → core
```

Plus the cross-cutting layers — importable from any layer:
`utils`, `themes`, `wasm`, `workers`, `profiling`, `controls`.

**Inverted seams** (data does not depend on ui):
- `SceneLoaderMonitorPort` — data emits events through a port the UI
  registers against.
- `DimensionSlidersFactory` — `input/` constructs sliders via a
  factory provided by `ui/`.

**Principle**: imports flow left-to-right in the layer order; no
circular dependencies (rule severity `error`).

---

## 3. nD Slicing Visualization

### Hypersphere Intersection for nD Point Visibility

#### Concept: 4D Example

```
Imagine 4D space: (x, y, z, t)
Display dimensions: [x, y, z]  (3D visualization)
Slice dimension: [t]           (navigate through time)

Current slice position: t = 5.0
Slice tolerance (radius): r = 1.0

Question: Which 4D points are visible?
Answer: Points within hypersphere distance r from slice plane
```

#### Mathematical Formula

```
For point at position p = (p_x, p_y, p_z, p_t):

1. Compute distance in non-displayed dimensions:
   d_slice = |p_t - 5.0|  (distance from slice in time dimension)

2. Compute effective radius in display dimensions:
   r_effective = sqrt(r² - d_slice²)  (Pythagorean theorem in 4D)

3. Visibility test:
   if d_slice > r:
     Point INVISIBLE (outside slice tolerance)
   else:
     Point VISIBLE with radius = r_effective

4. Project to 3D:
   Display position = (p_x, p_y, p_z)
   Display radius = r_effective
```

#### Visual Representation

```
                t-axis (non-displayed, slice dimension)
                  │
         t=6.0 ───┤─── Slice boundary (top)
                  │
         t=5.0 ───●─── Slice position (current view)
                  │
         t=4.0 ───┤─── Slice boundary (bottom)
                  │
      (tolerance = 1.0)

Point A at t=4.5: Distance 0.5 < 1.0 → VISIBLE ✓
  r_effective = sqrt(1.0² - 0.5²) = sqrt(0.75) = 0.87

Point B at t=3.0: Distance 2.0 > 1.0 → INVISIBLE ✗
  (Too far from slice)

Point C at t=5.0: Distance 0.0 → VISIBLE with full radius ✓
  r_effective = sqrt(1.0² - 0²) = 1.0
```

#### Code Implementation

```typescript
function calculateEffectiveRadius(
  pointPosition: number[],    // nD position
  slicePosition: number[],    // Current slice in each dim
  displayDims: number[],      // Which dims are displayed [0,1,2]
  maxRadius: number           // Original point radius
): number {
  let squaredDistance = 0;

  // Sum squared distances in NON-displayed dimensions
  for (let d = 0; d < pointPosition.length; d++) {
    if (!displayDims.includes(d)) {
      const diff = pointPosition[d] - slicePosition[d];
      squaredDistance += diff * diff;
    }
  }

  // Pythagorean theorem in nD
  const radiusSquared = maxRadius * maxRadius;
  const effectiveRadiusSquared = radiusSquared - squaredDistance;

  if (effectiveRadiusSquared <= 0) {
    return 0; // Point outside slice → invisible
  }

  return Math.sqrt(effectiveRadiusSquared);
}
```

#### Example: 5D Dataset Navigation

```
Dataset: (Time, Z, Channel, Y, X)  [5D]
Display: [Y, X, Z]                 [3D visualization]
Navigate: Time and Channel         [Slice dimensions]

Current state:
- Time = 10 (frames)
- Channel = 1 (GFP)
- Camera shows Y, X, Z in 3D

Point visibility:
- Point at (t=10, z=5, c=1, y=10, x=20):
  → Distance in (t,c) = 0 → FULL radius → VISIBLE ✓

- Point at (t=12, z=5, c=1, y=10, x=20):
  → Distance in time = 2 frames
  → If radius < 2 → INVISIBLE ✗
  → If radius = 3 → Effective radius = sqrt(9-4) = 2.24 → VISIBLE ✓

- Point at (t=10, z=5, c=2, y=10, x=20):
  → Distance in channel = 1
  → Depends on radius...
```

---

## 4. Rendering Pipeline

### HDR Post-Processing Flow

Materials registered with `MaterialManager` implement
`CameraAwareMaterial`. On every camera change the manager fans
`updateCameraParams(fov, resolution, isOrtho, nearCull)` out to
every registered material — no scene traversal. Picking materials
register too and unregister (without disposal) on context-restore
so a context-loss cycle doesn't leak registrations.

```mermaid
graph LR
    subgraph "Scene Rendering"
        A[3D Geometry<br/>Points, Lines, GSplats, Mesh]
        B[Camera-aware Materials<br/>register w/ MaterialManager]
        C[Vertex Shader<br/>Position + size calc]
        D[Fragment Shader<br/>Gaussian / per-geometry<br/>HDR-correct output]
    end

    subgraph "HDR Buffer"
        E[HalfFloatType<br/>16-bit float<br/>Range: 0-65504]
    end

    subgraph "Post-Processing (mega-shader)"
        F[Bloom pre-pass<br/>threshold + mip pyramid]
        H[MegaShader fullscreen pass<br/>fused per-pixel effects]
        I0[Optional FXAA<br/>edge-detect on LDR]
    end

    subgraph "Display"
        I[sRGB Canvas<br/>8-bit output]
    end

    A --> B --> C --> D --> E
    E --> F --> H --> I0 --> I

    style E fill:#ffeeaa
    style H fill:#aaffee
    style I fill:#aaffaa
```

### Effect Ordering

Single fused fragment shader. The mega-shader runs the per-pixel
effects in canonical order:

```
ChromaticLensDistortion → Bloom (additive) → DetectorNoise →
ToneMapping (with EOG) → Vignette → sRGB encode
```

Bloom remains a separate pre-pass (neighbor reads needed for
threshold + downsample/upsample). FXAA remains a separate post-pass
(edge detection runs on the tone-mapped LDR output). SMAA, DoF, and
SSAO are intentionally absent: SMAA and DoF need additional passes,
and SSAO needs surface normals that Luxar's point, line, and gsplat
primitives do not provide.

---

## 5. Cache Architecture

### Multi-Level Caching System

The cascade is L0 → L1 → L2 → network. L0 caches **decompressed**
zarr chunks (avoids repeated Blosc decompression); L1 caches
**compressed** zarr bytes (zarrita's `AsyncReadable` view); L2 is
OPFS for cross-session persistence; L3 is the origin fetch.

```mermaid
graph TB
    subgraph "Request Flow"
        A[zarr getChunk / fetch<br/>e.g., positions/c/0/1/2]
    end

    subgraph "L0: Decompressed Chunk Cache"
        B0{In L0?}
        D0[Return decoded array<br/>⚡ ~1μs, no decompression]
    end

    subgraph "L1: Memory Cache (~100MB)"
        B{In L1?}
        D[Return compressed bytes<br/>~1μs + Blosc decode ~2ms]
    end

    subgraph "L2: OPFS Cache (~2GB)"
        E{In L2?}
        G[Promote to L1<br/>🔄 ~1ms]
    end

    subgraph "L3: Network (Origin Server)"
        H[HTTP Fetch<br/>🌐 ~100ms<br/>retry + AbortSignal aware]
        I[Store in L2 + L1]
    end

    subgraph "Prefetcher (Background)"
        K[ChunkPrefetcher<br/>Adjacent chunks<br/>±1 in each displayed dim]
    end

    A --> B0
    B0 -->|Hit| D0
    B0 -->|Miss| B
    B -->|Hit| D --> D0
    B -->|Miss| E
    E -->|Hit| G --> D --> D0
    E -->|Miss| H --> I --> D --> D0

    G -.Trigger.-> K
    H -.Trigger.-> K
    K -.suppressPrefetch=true.-> A

    style D0 fill:#aaffaa
    style D fill:#aaeeaa
    style G fill:#ffffaa
    style H fill:#ffaaaa
```

### Stats Surfaces

`MultiLevelCachingStore.getStats()` exposes per-tier counters
(L1 hits/misses/evictions, L2 reads/writes/misses, network bytes
and request count). The data-loading monitor's
`cache-metrics-aggregator` reads from an optional `L0Provider` for
L0 stats and from `cacheStatsProvider` for L1/L2/network, falling
back to per-loader metrics when no providers are wired.

### Cache Key Structure

```
Format: dataset_hash/bucket_hash/base64_filename

Example:
abc123/5f/cG9pbnRzL3Bvc2l0aW9ucy9jLzAvMS8y

Where:
- abc123: Content hash of dataset (from .zmetadata)
- 5f: Bucket (2-char hex, one of 256 buckets)
- cG9p...: Base64 of "points/positions/c/0/1/2"

Benefits:
- dataset_hash: Isolate different datasets
- bucket_hash: Distribute files across 256 directories (avoid FS limits)
- base64: Handle special characters in paths
```

---

## 6. Input Event Routing

### Context-Based Priority System

```mermaid
graph TB
    A[Keyboard Event<br/>e.g., 'w' key press]
    B{Typing in<br/>text field?}
    C[Pass to browser<br/>⛔ Block shortcuts]

    D{Current Context?}
    E1[TYPING<br/>Priority: 10]
    E2[UI_INTERACTION<br/>Priority: 5]
    E3[DIMENSION_NAV<br/>Priority: 2]
    E4[FLY_CONTROLS<br/>Priority: 1]
    E5[NAVIGATION<br/>Priority: 0]

    F{Key allowed<br/>in context?}
    G[Execute Handler<br/>✅]
    H{Passthrough?}
    I[Try lower priority<br/>context]
    J[Ignore event<br/>⛔]

    A --> B
    B -->|Yes| C
    B -->|No| D

    D --> E1 & E2 & E3 & E4 & E5
    E1 & E2 & E3 & E4 & E5 --> F

    F -->|Yes| G
    F -->|No| H
    H -->|Yes| I
    H -->|No| J

    I --> F

    style C fill:#ffaaaa
    style G fill:#aaffaa
    style J fill:#ffaaaa
```

### Context Examples

| Key | NAVIGATION | FLY_CONTROLS | TYPING | Result |
|-----|------------|--------------|---------|--------|
| **W** | ❌ Blocked | ✅ Forward | ❌ Blocked | Depends on context |
| **F** | ✅ Toggle center | ✅ Toggle center | ❌ Blocked | Letter 'f' typed |
| **[** | ✅ Dim nav | ✅ Dim nav | ❌ Blocked | Navigate dimension |
| **Esc** | ✅ Reset | ✅ Exit fly | ✅ Unfocus + dispatch | Always handled |

**Escape-from-typing**: pressing **Esc** while in TYPING blurs the
focused element and re-routes the same event through the underlying
context, so a typing field can be exited without a separate
keystroke. The chain is `TYPING → UI_INTERACTION → DIMENSION_NAV →
NAVIGATION` (recursion depth-capped to prevent infinite passthrough
loops).

---

## 7. Initialization Sequence

### Application Startup Flow

```mermaid
sequenceDiagram
    participant Main as main.ts
    participant App as LuxarApp
    participant Scene as SceneManager
    participant Anim as AnimationController
    participant Input as InputHandler
    participant UI as RenderingControls
    participant Data as SceneLoader

    Main->>Main: Import console interceptor (FIRST!)
    Main->>Main: Validate config
    Main->>App: new LuxarApp()
    Main->>App: init(src)

    App->>Scene: new SceneManager()
    App->>Scene: init()
    Scene->>Scene: Setup canvas, renderer, camera
    Scene->>Scene: Setup controls, post-processing
    Scene-->>App: Ready

    App->>Anim: new AnimationController(...)
    App->>Input: new InputHandler(...)
    App->>Input: init()
    App->>UI: new RenderingControls(...)

    App->>App: Cross-link components
    App->>Anim: startAnimation()

    App->>Data: loadSceneData(src)
    Data->>Data: Load zarr metadata
    Data->>Data: Build scene graph
    Data->>Data: Load initial points
    Data-->>App: Scene ready

    App->>App: Setup cleanup handlers
    App->>App: Setup debug interface
    App-->>Main: Initialized ✅
```

### Critical Order Requirements

1. **Console interceptor MUST be first import** (captures all output)
2. **SceneManager before AnimationController** (needs renderer/camera)
3. **AnimationController before data loading** (enables progress rendering)
4. **Cross-linking after all components created** (avoid undefined refs)
5. **Animation start before data load** (user sees progress)

---

## 8. Spatial Index Query Algorithm

### Chunk-Based nD Query

```mermaid
graph TB
    A[Query: Position + Tolerance<br/>e.g., t=5.0 ± 1.0, c=1 ± 0.5]
    B[Load Chunk Bounds<br/>Float32Array: num_chunks × ndim × 2]
    C{For each chunk<br/>Test intersection}

    D[Get chunk bbox<br/>min, max per dim]
    E{Dimension loop<br/>d = 0..ndim}
    F[Query box:<br/>min = pos - tol<br/>max = pos + tol]
    G{Boxes overlap?}

    H1[chunkMax < queryMin<br/>OR<br/>chunkMin > queryMax]
    H2[No overlap ✗<br/>Skip chunk]
    H3[Overlap ✓<br/>Continue checking]

    I{All dims<br/>overlap?}
    J[Add to results<br/>✅ Chunk matched]
    K[Reject chunk<br/>❌ No match]

    L[Convert chunk indices<br/>to point ranges]
    M[Load ranges from zarr]
    N[Decode and merge]
    O[Return visible points]

    A --> B --> C
    C --> D --> E
    E --> F --> G
    G --> H1
    H1 -->|True| H2
    H1 -->|False| H3
    H3 --> E
    E -->|Done| I
    I -->|Yes| J
    I -->|No| K

    C -->|All chunks tested| L
    L --> M --> N --> O

    style A fill:#e1f5ff
    style J fill:#aaffaa
    style K fill:#ffaaaa
    style O fill:#aaffaa
```

### Performance Characteristics

```
Given:
- N = total points (e.g., 10M)
- C = num chunks (e.g., 1,000)
- D = dimensions (e.g., 5D)
- M = matching chunks (e.g., 10)

Complexity:
- Query: O(C × D) = O(1,000 × 5) = ~5,000 comparisons
- Load: O(M × chunk_size) = O(10 × 10,000) = ~100K points
- vs Full Scan: O(N × D) = O(10M × 5) = ~50M comparisons

Speedup: 10,000x faster than full scan
Memory: O(C × D × 2) = ~40KB for index (vs 400MB for full data)
```

---

## 9. Material Management & Caching

### Material Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Request: getPointMaterial(props)

    Request --> CacheCheck: Generate cache key
    CacheCheck --> CacheHit: Material exists
    CacheCheck --> CreateNew: Cache miss

    CacheHit --> Register: Return cached
    CreateNew --> Register: Create PointMaterial

    Register --> Active: Register for global updates
    Active --> Active: Update HDR/FOV/resolution
    Active --> Unregister: dispose() called

    Unregister --> CacheRemove: Unregister from manager
    CacheRemove --> GPUCleanup: Free GPU resources
    GPUCleanup --> [*]

    note right of CacheCheck
        Cache Key Format:
        point_additive_o100_g10_r1000_s1000

        o: opacity (0-100)
        g: gamma (0-30)
        r: radiusScale (0-?)
        s: sharpnessScale (0-?)
    end note

    note right of Active
        Global Updates:
        - updateCameraParams(fov, res)
        - updateHDRMultiplier(val)

        Updates ALL registered materials
        No scene traversal needed
    end note
```

### Cache Benefits

**Without Caching**:
```typescript
// Creates new material every time (memory leak!)
for (let i = 0; i < 100; i++) {
  const material = new PointMaterial({ opacity: 1.0 });
  // 100 materials created, 99 are duplicates
}
```

**With Caching**:
```typescript
// Reuses same material (memory efficient)
for (let i = 0; i < 100; i++) {
  const material = materialManager.getPointMaterial({ opacity: 1.0 });
  // Only 1 material created, returned 100 times
}
```

---

## 10. Observable Pattern for nD Navigation

### SceneDimsManager → Components

```mermaid
graph TB
    subgraph "Central State (Singleton)"
        A[SceneDimsManager<br/>currentStep: number[]<br/>displayed: number[]]
    end

    subgraph "Observers"
        B[DimensionSliders<br/>Update slider positions]
        C[PointSpatialIndexLoader<br/>Re-query visible points]
        D[StatusDisplay<br/>Show current position]
    end

    subgraph "User Actions"
        E[Keyboard: Press ']'<br/>Move forward in dimension]
        F[Slider: Drag<br/>Set position directly]
    end

    E --> A
    F --> A
    A -.notify().-> B
    A -.notify().-> C
    A -.notify().-> D

    B -.UI update.-> B
    C -.Reload data.-> C
    D -.Update text.-> D

    style A fill:#ffe1e1
    style B fill:#e1f5ff
    style C fill:#fff5e1
    style D fill:#e1ffe1
```

### Code Pattern

```typescript
// Register observer
sceneDimsManager.addListener(() => {
  const dims = sceneDimsManager.getDims();
  updatePointsForNewSlice(dims.currentStep);
});

// Change dimension (triggers all observers)
sceneDimsManager.setDimensionValue(0, 5.2);
// → All listeners called automatically
// → Sliders update
// → Points reload
// → Display updates
```

---

## 11. Error Recovery Flow

### Graceful Degradation Strategy

```mermaid
graph TB
    A[Scene Load Request]
    B{Spatial Index<br/>Available?}
    C[Use Spatial Queries<br/>✅ Efficient]
    D[Load All Points<br/>⚠️ Fallback]

    E{Points Load<br/>Success?}
    F[Create Geometry<br/>Display points]
    G{Critical Error<br/>or Recoverable?}

    H[Log Error<br/>Track in failedLoaders]
    I[Continue with<br/>other loaders]
    J[Show Error UI<br/>User notification]
    K[Animation continues<br/>App still usable]

    A --> B
    B -->|Yes| C
    B -->|No| D

    C --> E
    D --> E

    E -->|Success| F
    E -->|Failure| G

    G -->|Recoverable| H --> I --> K
    G -->|Critical| J --> K

    style F fill:#aaffaa
    style H fill:#ffffaa
    style J fill:#ffaaaa
    style K fill:#aaddff
```

### Error Handling Principles

1. **Isolation**: Component failures don't crash app
2. **Tracking**: All errors logged and tracked
3. **User Communication**: Clear, actionable error messages
4. **Graceful Fallback**: Degrade to simpler behavior
5. **Recovery**: Allow retry without reload

**Example**:
```
Scenario: One point cloud fails to load in a multi-object scene

Instead of: ❌ Entire scene fails, black screen, app crashes
We do: ✅ Other objects load, error logged, user notified, retry possible
```

---

## 12. Memory Management Strategy

### Disposal Chain

```mermaid
graph LR
    A[LuxarApp.cleanup]
    B[AnimationController.dispose<br/>Stop rendering]
    C[InputHandler.dispose<br/>Remove event listeners]
    D[RenderingControls.dispose<br/>Destroy custom GUI library]
    E[SceneManager.dispose<br/>WebGL cleanup]

    F[PostProcessing.dispose<br/>Render targets]
    G[Controls.dispose<br/>Mouse/touch events]
    H[MaterialManager.dispose<br/>Shader programs]
    I[Scene.traverse<br/>Geometry & materials]

    A --> B --> C --> D --> E
    E --> F & G & H & I

    style A fill:#ffe1e1
    style B fill:#ffffaa
    style E fill:#ffaaaa
    style I fill:#aaffaa
```

### Critical Cleanup Operations

**Why Manual Disposal is Required**:
- WebGL resources are NOT garbage collected automatically
- Textures, buffers, shaders stay in GPU memory
- Event listeners keep references alive
- Proper cleanup prevents memory leaks in long-running apps

**Cleanup Order** (Reverse of initialization):
1. Stop animation (prevents new work)
2. Remove event listeners (prevents callbacks)
3. Dispose UI components (free DOM)
4. Dispose WebGL resources (free GPU memory)
5. Traverse scene (geometry, materials)

---

## 13. World-Space Point Sizing

### Physical Size Calculation

```
Goal: Two points with radius r at distance 2r should just touch

Formula (implemented in vertex shader):

pixelSize = 2 × radius × resolution.y / (distance × tan(FOV/2))

Where:
- radius: Physical world-space radius
- resolution.y: Framebuffer height in pixels
- distance: Camera distance to point (view space)
- FOV/2: Half field of view angle (radians)
- tan(FOV/2): Pre-computed for performance

Why it works:
1. Angular size in radians: α = 2 × atan(radius / distance)
2. Pixels per radian: resolution.y / (2 × tan(FOV/2))
3. Combine: pixels = α × (resolution.y / (2 × tan(FOV/2)))
4. Simplify: pixels = 2 × radius × resolution.y / (distance × tan(FOV/2))
```

### Visual Proof

```
Side view (camera looking along Z-axis):

      Camera
        📷
         |\
         | \
    FOV  |  \
         |   \
         |    ● Point (radius r, distance d)
         |   /
         |  /
         | /
         |/

Angular size seen by camera: 2 × atan(r/d)
Screen pixels: angular_size × (screen_height / FOV_height)

For two points at distance 2r apart:
- Each has radius r
- They just touch (r + r = 2r separation)
- Physical accuracy maintained regardless of FOV or resolution
```

---

## 14. Event Flow Diagram

### User Interaction → Render

```mermaid
sequenceDiagram
    participant User
    participant Input as InputHandler
    participant Ctx as ContextManager
    participant Controls
    participant Dims as SceneDimsManager
    participant Loader as PointSpatialIndexLoader
    participant Anim as AnimationController
    participant Scene as SceneManager
    participant GPU as WebGL

    User->>Input: Mouse drag / Key press
    Input->>Ctx: Check context & filters

    alt Navigation Keys (F, R, etc.)
        Ctx->>Controls: Update camera
        Controls->>Scene: Camera changed
    else Dimension Nav ([, ])
        Ctx->>Dims: Step dimension
        Dims->>Dims: Update currentStep
        Dims-->>Loader: Notify observers
        Loader->>Loader: Query new ranges
        Loader->>Scene: Update geometry
    end

    Scene->>Anim: Trigger change event
    Anim->>Anim: Reset idle timer
    Anim->>Anim: Start animation loop

    loop Every frame
        Anim->>Controls: update()
        Anim->>Scene: render()
        Scene->>GPU: WebGL draw calls
        GPU-->>User: Display updated
    end

    Note over Anim: Auto-stop after 2s idle
```

---

## 15. Package Interaction Map

### Cross-Package Dependencies

```
         types/        config/       utils/
           │              │            │
           └──────┬───────┴────────────┘
                  │
              cache/  ←─────┐
                  │         │
         ┌────────┴────┐    │
         │             │    │
    rendering/      data/ ──┘
         │             │
         └──────┬──────┘
                │
            controls/
                │
         ┌──────┴──────┐
         │             │
      scene/        input/
         │             │
         └──────┬──────┘
                │
              ui/
                │
             core/
                │
             main.ts

Legend:
  │ = depends on
  Solid boxes = no external deps
  Higher = more foundational
```

### Dependency Rules

**Allowed**:
- Lower level → Higher level ✅
- Same level (with caution) ⚠️

**Forbidden**:
- Higher level → Lower level ❌
- Circular dependencies ❌

**Example**:
- ✅ scene/ can import from rendering/ (lower level)
- ✅ core/ can import from scene/ (lower level)
- ❌ rendering/ cannot import from scene/ (higher level)
- ❌ data/ cannot import from ui/ (higher level)

---

## Usage

These diagrams are rendered automatically in:
- GitHub markdown viewers
- VS Code with Mermaid extension
- Documentation sites (Docusaurus, MkDocs with mermaid plugin)

**To view locally**:
```bash
# Install mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG/SVG
mmdc -i ARCHITECTURE-DIAGRAMS.md -o diagrams.pdf
```

---

## Maintenance

**Update these diagrams when**:
- Adding new packages
- Changing initialization order
- Modifying data flow
- Changing architecture

**Keep aligned with**: each package's README.md
