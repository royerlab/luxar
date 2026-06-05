# Luxar Scene Package

> Core 3D scene management and orchestration for nD scientific visualization

## Overview

The Luxar Scene package provides comprehensive scene management, animation control, and nD dimension coordination for complex multi-geometry visualizations (points, lines, Gaussian splats). It handles the THREE.js scene graph, camera management, rendering pipeline integration, and multi-dimensional data navigation.

### Key Features

- **Scene Graph Management**: Hierarchical 3D scene organization
- **Animation Control**: Smart rendering loop with idle detection
- **nD Dimension Support**: Coordinate system for n-dimensional data
- **Camera Management**: Integrated control system orchestration
- **Bounding Box Computation**: Automatic scene bounds calculation
- **Center Point Management**: Native vs bounding box centering

### Package Architecture

```
scene/
├── scene-manager.ts                # Main scene orchestrator
├── scene-manager/                  # Focused helpers owned by SceneManager
│   ├── camera/                     # camera-framing, camera-setup, camera-materials, camera-mode
│   ├── clipping/                   # bounds-math, scene-bounds-cache, clipping-policy
│   ├── render-pipeline/            # renderer-setup, post-processing-setup, scene-disposal, webgl-context-recovery
│   └── viewport/                   # dpr-policy, resize-orchestrator
├── animation/                      # animation-controller, dimension-animation-manager
├── scene-dims-manager.ts           # nD dimension coordination
├── lod-group-registry.ts           # Per-frame LOD-group selector + VRAM-budget LRU
├── synthetic-scene.ts              # Synthetic perf-bench scene generators (lines)
└── README.md                       # This documentation
```

---

## Components

### 1. Scene Manager

The `SceneManager` is the central hub for all 3D scene operations.

**Responsibilities:**

- THREE.js scene initialization
- Camera and renderer setup
- Control system integration
- Scene graph manipulation
- Bounding box calculations
- Center point management

**Core Features:**

```typescript
class SceneManager extends THREE.EventDispatcher {
  // Scene setup (populated by init())
  scene: THREE.Scene;
  camera: LuxarCamera; // PerspectiveCamera | OrthographicCamera
  renderer: Renderer; // WebGLRenderer or WebGPURenderer
  capabilities: RendererCapabilities;
  controls: ControlsManager;
  postProcessing: PostProcessingManager;

  // Lifecycle
  init(options: {
    canvas: HTMLCanvasElement;
    debug?: boolean;
    renderer?: 'webgl' | 'webgpu';
    webgpuForceWebGL?: boolean;
    perfTimestamp?: boolean;
  }): Promise<void>;
  loadSceneData(src: string, loaderConfig?: LoaderConfig): Promise<void>;
  updateSize(): void;
  dispose(): void;

  // Camera control
  updateFOV(delta: number): void;
  updateClippingPlanes(near: number, far: number): void;
  autoAdjustClippingPlanes(): { near: number; far: number };
  setDynamicClipping(enabled: boolean): void;
  getDynamicClippingState(): { enabled: boolean; near: number; far: number };
  setControlType(type: 'orbit' | 'fly' | 'ortho'): void;
  getControlType(): ControlType;

  // Centering
  centerCameraOnScene(): void;
  toggleCentering(): void;
  getCurrentCenter(): THREE.Vector3;

  // WebGL context-loss
  isWebGLContextLost(): boolean;
}
```

Scene mutation goes through the standard Three.js API on the
`scene` field (e.g. `sceneManager.scene.add(group)`); SceneManager
itself does not expose a wrapping `addToScene` method.

**Usage Example:**

```typescript
const canvas = document.getElementById('app') as HTMLCanvasElement;
const sceneManager = new SceneManager();
await sceneManager.init({ canvas });

// Add objects directly through the THREE.Scene field
sceneManager.scene.add(pointCloud);

// Update camera FOV
sceneManager.updateFOV(10); // Increase FOV by 10 degrees

// Toggle between native and bbox center
sceneManager.toggleCentering();
```

### 2. Animation Controller

The `AnimationController` manages the render loop with intelligent idle detection.

**Features:**

- Automatic pause when scene is static
- Performance monitoring integration
- Event-driven animation triggers
- Configurable idle timeout
- Frame timing management

**Animation States:**

```typescript
// Start animation (triggered by events)
animationController.startAnimation();

// Animation automatically pauses after idle timeout
// Default: 2000ms of no activity

// Manual control
animationController.stopAnimation();
animationController.startAnimation();
```

**Render Loop:**

```typescript
private animate = (): void => {
  if (!this.isAnimating) return;

  requestAnimationFrame(this.animate);

  // Update controls
  this.sceneManager.controls.update();

  // Update performance stats
  this.performanceStats?.begin();

  // Render scene
  this.postProcessing.render();

  this.performanceStats?.end();

  // Check for idle timeout
  this.checkIdleTimeout();
}
```

### 3. Scene Dimensions Manager

The `SceneDimsManager` coordinates n-dimensional navigation across all scene objects.

**Capabilities:**

- Unified dimension system for all nD objects
- Dimension metadata management
- Slice position coordination
- Step size calculations
- Display status tracking

**Dimension Structure:**

```typescript
interface SimpleDims {
  ndim: number; // Total dimensions
  displayed: number[]; // Indices of displayed dims (e.g., [0,1,2])
  currentStep: number[]; // Current position in each dimension
  metadata?: DimensionMetadata[]; // Names, units, ranges
}

interface DimensionMetadata {
  name: string; // e.g., "time", "x", "wavelength"
  unit: string; // e.g., "ms", "μm", "nm"
  scale: number; // Array-index → real-world units
  range?: [number, number]; // Optional min and max values
  step?: number; // Step size for navigation
  discrete?: boolean; // Whether dimension is discrete
  cyclic?: boolean; // Whether dimension wraps (angles, etc.)
  spatial?: boolean; // Whether points extend through this dim
  categories?: string[]; // Optional categorical labels
  display?: boolean; // Show in the 3D scene by default
  description?: string; // UI tooltip
}
```

**Usage:**

```typescript
// Initialize from scene
sceneDimsManager.initFromScene(scene);

// Get unified dimensions
const dims = sceneDimsManager.getDims();

// Navigate dimensions
sceneDimsManager.setDimensionValue(dimIndex, value);

// Listen for changes
sceneDimsManager.addListener(() => {
  updateVisualization();
});
```

### 4. Dimension Animation Manager

The `DimensionAnimationManager` provides automated playback through dimension ranges.

**Responsibilities:**

- FPS-based animation control
- Loop mode management (once, loop, bounce)
- Per-dimension animation state
- Frame throttling and measurement
- Integration with AnimationController

**Key Features:**

```typescript
class DimensionAnimationManager extends THREE.EventDispatcher {
  // Playback control
  play(dimIndex: number, options?: PlayOptions): boolean;
  pause(dimIndex: number): boolean;
  togglePlay(dimIndex: number): boolean;
  stop(dimIndex: number): void;

  // Speed control
  setTargetFPS(dimIndex: number, fps: number): void;
  increaseSpeed(dimIndex: number): void;
  decreaseSpeed(dimIndex: number): void;

  // Configuration
  setLoopMode(dimIndex: number, loopMode: LoopMode): void;

  // State queries
  isAnimating(dimIndex: number): boolean;
  getState(dimIndex: number): DimensionAnimationState | undefined;

  // Lifecycle
  dispose(): void;
}

interface PlayOptions {
  targetFPS?: number; // Default: 10
  loopMode?: 'once' | 'loop' | 'bounce'; // Default: 'loop'
  direction?: 'forward' | 'backward'; // Default: 'forward'
}
```

**Usage:**

```typescript
// Create animation manager
const animManager = new DimensionAnimationManager(sceneDimsManager, animationController);

// Start animating time dimension at 10 FPS
animManager.play(3, { targetFPS: 10, loopMode: 'loop' });

// Adjust speed
animManager.increaseSpeed(3); // Next preset: 15 FPS
animManager.setTargetFPS(3, 30); // Set exact FPS

// Change loop mode
animManager.setLoopMode(3, 'bounce'); // Reverse at boundaries

// Listen for events
animManager.addEventListener('play', (e) => {
  console.log(`Dimension ${e.dimIndex} started`);
});

animManager.addEventListener('complete', (e) => {
  console.log(`Dimension ${e.dimIndex} finished (loop: once)`);
});

// Pause/stop
animManager.pause(3);
animManager.stop(3); // Also removes state
```

**Animation Modes:**

- **Loop**: Wrap to start when reaching end (continuous playback)
- **Once**: Stop at end boundary (single pass)
- **Bounce**: Reverse direction at boundaries (ping-pong)

**FPS Presets:** 1, 2, 5, 10, 15, 30, 60 Hz

### 5. LOD Group Registry

The `LODGroupRegistry` is the per-frame selector for `lod_group`
scene-graph nodes. For each registered group it picks which child
(level of detail) renders, kicks deferred geometry loads for lazy
levels, and bounds resident VRAM with an LRU eviction pass.

**Per-frame algorithm** (one entry):

1. Fold each child's nD `positionBounds` into a cached local-space
   `BoundingBox`, mapping nD axes onto X/Y/Z via the current
   `displayDims`.
2. Lift that box to world space through the group's `matrixWorld`
   (`transformBoundingBox`).
3. **Off-screen gate**: if the world box is entirely outside the
   camera frustum, hold the group at its coarsest *ready* level
   instead of loading a fine level the renderer would frustum-cull.
4. Otherwise, project the 8 world corners to NDC and measure the
   diagonal of the screen-space AABB in pixels
   (`projectBoxDiagonalPx`).
5. Pick the finest child whose `minPixelSize` threshold is satisfied,
   with 10% asymmetric, spacing-aware hysteresis on the downgrade
   direction to suppress threshold-edge flicker
   (`pickChildWithHysteresis`).
6. Swap visibility atomically when the desired child differs; lazy
   targets that are not yet committed kick `ensureLoaded()` and swap
   on a later frame once `ready` flips true.

**Selector modes:**

```typescript
// Auto (view-driven) — the default after register()
registry.setSelectorMode(path, 'auto');

// Lock to a specific level (0-based, coarsest→finest).
// Out-of-range lockLevel is clamped into [0, n-1] with a warning.
registry.setSelectorMode(path, { lockLevel: 2 });
```

**Retention + VRAM budget:** swaps never `release()` outgoing
geometry — retention keeps loaded levels resident so swapping back is
a sub-millisecond visibility toggle. Memory is bounded once per frame
by `enforceResidentByteBudget`, which (only when the GPU pool's live
resident byte total exceeds the shared budget) demotes evictable
levels — off-screen first, then furthest-from-camera, then
coldest-`lastVisibleTick`. The visible level of each group and eager
fallback levels (no `release` thunk) are never evicted.

**Wiring:** the SceneLoader instantiates one registry per scene and
hooks `evaluatePerFrame()` into `AnimationController` alongside the
dynamic-clipping callback. The injected `LODGroupRegistryDeps` supply
the camera, viewport size, `displayDims`, and the optional resident
byte budget / measurement — omitting the budget accessors yields pure
retention (the unit-test default). `projectBoxDiagonalPx` and
`pickChildWithHysteresis` are exported as pure functions for testing.

---

## Scene Graph Structure

### Hierarchy

```
Scene (THREE.Scene)
├── Ambient Light
├── Data Groups
│   ├── Point Cloud Groups
│   │   ├── Transform Group
│   │   │   └── Points (THREE.Points)
│   │   └── Direct Points
│   └── Lines Groups
│       └── Lines (THREE.Mesh with InstancedBufferGeometry)
├── Camera (managed separately)
└── Helper Objects (grids, axes, etc.)
```

**Note:** Lines use `THREE.Mesh` with `InstancedBufferGeometry` (not `THREE.InstancedMesh` or `THREE.LineSegments`) to enable thick lines with world-space width while staying under WebGL's 16 attribute location limit.

### Object Management

**Adding Objects:**

```typescript
// Add directly through the THREE.Scene field
sceneManager.scene.add(object);

// Batch operations
const group = new THREE.Group();
group.add(points1, points2, points3);
sceneManager.scene.add(group);
```

**Clearing Scene:**

`loadSceneData` internally clears prior loaded content (lights and
background are preserved) before adding the new dataset's root
group. Manual disposal helpers used by that path are in
`scene-manager/render-pipeline/scene-disposal.ts`
(`clearLoadedSceneContent`, `disposeSceneGraphResources`).

---

## Camera Management

### FOV Control

The scene manager provides comprehensive FOV management:

```typescript
// Adjust FOV with mouse wheel + shift
sceneManager.updateFOV(delta);

// FOV limits: 10° to 170° (must be <180°)
// Sensitivity: 0.05 per wheel unit
```

**Professional FOV Presets** (35mm equivalent, horizontal FOV):

- **28mm Wide** (75°): Ultra-wide for landscapes
- **35mm** (63°): Wide angle for environmental shots
- **50mm Normal** (47°): Natural human vision equivalent
- **85mm Portrait** (29°): Telephoto for subject isolation
- **135mm Tele** (18°): Strong telephoto for extreme focus

### Clipping Plane Control

Advanced Z-buffer management for optimal rendering precision:

```typescript
// Manual clipping plane adjustment
sceneManager.updateClippingPlanes(near, far);

// Auto-calculate optimal planes from scene bounds
const { near, far } = sceneManager.autoAdjustClippingPlanes();

// Enable dynamic clipping (auto-adjusts each frame)
sceneManager.setDynamicClipping(true); // enable sphere-based clipping

// Trigger an explicit dynamic update (normally called per-frame
// by AnimationController)
sceneManager.updateDynamicClippingPlanes();

// Get current dynamic clipping state
const state = sceneManager.getDynamicClippingState();
// { enabled: boolean, near: number, far: number }
```

**Z-Buffer Best Practices:**

- Keep near/far ratio under 10,000:1 for best precision
- Use auto-adjust after loading new datasets
- Lower near values see closer objects but reduce precision

### Dynamic Clipping Planes

The scene manager supports automatic per-frame clipping plane adjustment:

**How It Works:**

1. Each frame, computes a bounding sphere from the cached scene bounds (with safety margin)
2. Near plane = `max(MIN_NEAR_PLANE, distToCenter - radius)` -- smoothly transitions to minimum as camera enters the sphere
3. Far plane = `distToCenter + radius` -- distance to farthest point on the sphere
4. The bounds cache is invalidated on scene load/clear; zero per-frame scene-graph traversal in steady state

**Benefits:**

- Always-optimal Z-buffer precision as camera moves
- **Direction-independent clipping** -- no sharp jumps at bounding box edges
- Near plane continuously drops to minimum as camera enters the scene
- Eliminates need for manual clipping adjustment
- Perfect for exploring large-scale scenes from any viewpoint

**Configuration:**

```typescript
// Enable sphere-based dynamic clipping
sceneManager.setDynamicClipping(true);
```

### Centering Modes

Two centering modes for camera focus:

1. **Native Center**: Scene's inherent center point
2. **Bounding Box Center**: Computed from all objects

```typescript
// Toggle with 'C' key
sceneManager.toggleCentering();

// Get current center
const center = sceneManager.getCurrentCenter();
controls.lookAt(center);
```

---

## Animation System

### Performance Optimization

The animation controller implements smart rendering:

```typescript
// Only renders when needed
- User interaction (mouse, keyboard)
- Control updates (movement, rotation)
- Data changes (dimension navigation)
- Programmatic updates

// Automatically pauses after idle
- Default timeout: 2000ms
- Saves GPU/CPU resources
- Resumes instantly on activity
```

### Event Triggers

Animation is triggered by various events:

```typescript
// Control events
controls.addEventListener('change', startAnimation);
controls.addEventListener('start', startAnimation);

// User interaction
canvas.addEventListener('mousedown', startAnimation);
canvas.addEventListener('touchstart', startAnimation);

// Data updates
sceneDimsManager.addListener(startAnimation);
```

---

## nD Visualization

### Dimension Slicing

For data with >3 dimensions, the system provides slicing:

```typescript
// 5D data example
const dims = {
  ndim: 5,
  displayed: [0, 1, 2], // Show x, y, z
  currentStep: [0, 0, 0, 50, 100], // Position in 5D space
};

// Points visible if within radius of slice
const radius = point.radius || 1.0;
const distance = calculateDistance(point.position, slicePosition);
point.visible = distance <= radius;
```

### Dimension Initialization

**Initial Slice Position** (on viewer startup):

When nD datasets load, dimensions initialize based on their type:

| Dimension Type          | Initial Position  | Example                                            |
| ----------------------- | ----------------- | -------------------------------------------------- |
| **Time/Frames**         | Start (t=0)       | Time-series starts at first frame                  |
| **Channels**            | First channel (0) | Multi-channel data shows first channel (DAPI, GFP) |
| **Discrete dimensions** | Minimum value     | Frame 0, category 0                                |
| **Continuous spatial**  | Center (0)        | 4th spatial dimension (W) centers at 0             |

**Behavior**:

```typescript
// Time-series (discrete, range 0-50ns)
Initial position: t = 0 ns ✓ (not 25ns)

// Multi-channel (discrete, 0-3)
Initial position: channel = 0 ✓ (not channel 2)

// 4D spatial (continuous, -100 to +100)
Initial position: W = 0 ✓ (center makes sense)
```

**Why It Matters**:

- Slider position matches displayed data from first load
- Time-series animations start at beginning, not middle
- Multi-channel data shows first channel first
- No confusing mismatch between UI and visualization

**Implementation Details**:

The initialization happens in two phases:

1. `sceneDimsManager.initFromScene()` sets up dimension state
2. `inputHandler.initDimensionSliders()` triggers initial data load

See `scene-dims-manager.ts` (`initFromScene`) for the initialization
logic and `input/input-handler.ts` (`initDimensionSliders`) for the
initial update trigger.

### Keyboard Navigation

Navigate through dimensions with keyboard:

```typescript
// Select dimension
'1'-'9': Select dimension 1-9

// Navigate selected dimension
'[': Move backward in dimension
']': Move forward in dimension

// Step sizes
- Discrete dims: Use defined step
- Continuous dims: 1% of range
```

### nD Transform Inverse-Query

The viewer uses an **inverse-query** approach for `nd_transform`: instead of transforming millions of point coordinates forward (O(N)), the query (slicePosition + tolerance) is inverse-transformed from world to local space once (O(1)). This is implemented in `data/transforms/nd-transform.ts` and applied transparently during geometry slicing. No loader internals change.

### Auto-Ranging from Position Bounds

When `Dimension.range` is not explicitly set, `SceneDimsManager.initFromScene()` uses `positionBounds` from the zarr root attrs as the slider range for that dimension. These bounds are pre-computed by the Python compiler with nd_transform expansion applied, so non-displayed dimensions reflect world-space extents. Fallback order: explicit `Dimension.range` > `positionBounds` > `[0, 1]`.

---

## Performance Monitoring

### Integration with Performance Stats

```typescript
// Initialize with performance monitoring
const performanceStats = new PerformanceMonitor(renderer);
animationController.setPerformanceStats(performanceStats);

// Automatic frame timing
- Begin() before render
- End() after render
- Updates FPS display
- Tracks frame times
```

### Optimization Techniques

1. **Frustum Culling**: Automatic via THREE.js
2. **Idle Detection**: Pause rendering when static
3. **LOD Support**: Ready for level-of-detail
4. **Batch Operations**: Group scene updates
5. **Disposal Management**: Proper cleanup

---

---

## WebGL Context Loss Recovery

### What is Context Loss?

WebGL contexts can be lost due to:

- GPU driver crashes or resets
- System sleep/hibernate
- Too many WebGL contexts (browser limit ~16)
- Out of GPU memory
- GPU overheating

When this happens, all WebGL resources are lost and rendering stops.

### How Luxar Handles It

The SceneManager automatically handles context loss and restoration:

```typescript
// Automatic recovery - no user action needed
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault(); // Allow restoration
  // Show user message: "Graphics context lost - attempting to restore..."
});

canvas.addEventListener('webglcontextrestored', async () => {
  // Recreate WebGL resources
  renderer.resetState();
  // Trigger re-render
  // Show success message: "Graphics context restored"
});
```

### User Experience

**During Context Loss**:

1. Rendering stops
2. User sees message: "Graphics context lost - attempting to restore..."
3. Loading indicator appears

**During Restoration**:

1. WebGL resources automatically recreated
2. Rendering resumes
3. User sees: "Graphics context successfully restored"

**If Restoration Fails**:

- Error message: "Failed to restore graphics. Please refresh the page."
- User can continue using UI, but rendering disabled

### For Developers

Check context status:

```typescript
// Check if context is lost
if (sceneManager.isWebGLContextLost()) {
  // Skip rendering operations
  return;
}

// Safe to render
sceneManager.render();
```

**Best Practice**: Don't create WebGL-dependent operations during context loss. Wait for restoration.

## Usage Examples

### Complete Setup

```typescript
import { SceneManager } from './scene/scene-manager';
import { AnimationController } from './scene/animation/animation-controller';
import { sceneDimsManager } from './scene/scene-dims-manager';

// Initialize scene
const canvas = document.getElementById('app') as HTMLCanvasElement;
const sceneManager = new SceneManager();
await sceneManager.init({ canvas });
const animationController = new AnimationController(
  sceneManager.controls,
  sceneManager.postProcessing
);

// Setup dimensions for nD data
sceneDimsManager.initFromScene(sceneManager.scene);

// Start render loop
animationController.startAnimation();
```

### Loading a Zarr Scene

```typescript
// One-shot: clears prior content, loads zarr scene tree,
// applies any viewer_config camera overrides, auto-frames the
// camera, and auto-adjusts clipping planes.
await sceneManager.loadSceneData(zarrUrl);

// Manual recentering on bbox (also bound to 'F' key in app)
sceneManager.centerCameraOnScene();
```

### Handling Window Resize

```typescript
window.addEventListener('resize', () => {
  // updateSize() schedules an rAF-coalesced resize through the
  // ResizeOrchestrator, which propagates to renderer, camera
  // aspect, post-processing, and material uniforms.
  sceneManager.updateSize();
});
```

---

## Configuration

### Scene Settings

```typescript
const sceneConfig = {
  backgroundColor: 0x111111, // Dark gray background
  ambientLight: {
    color: 0xffffff,
    intensity: 1.0,
  },
  camera: {
    fov: 60,
    near: 0.1,
    far: 1000,
    position: { x: 0, y: 0, z: 8 },
  },
};
```

### Animation Settings

```typescript
const animationConfig = {
  idleTimeoutMs: 2000, // Pause after 2 seconds
  targetFPS: 60, // Target frame rate
  adaptiveQuality: true, // Reduce quality if FPS drops
};
```

---

## Best Practices

### Scene Organization

1. **Use Groups**: Organize related objects
2. **Name Objects**: Set meaningful names for debugging
3. **Clean Disposal**: Always dispose geometries/materials
4. **Batch Updates**: Group scene modifications
5. **Cache Bounds**: Update bbox only when needed

### Performance Tips

1. **Minimize Draw Calls**: Use instanced rendering
2. **Optimize Geometries**: Merge when possible
3. **Texture Atlas**: Combine textures
4. **Frustum Culling**: Let THREE.js handle it
5. **Smart Animation**: Use idle detection

### Memory Management

```typescript
// Proper disposal
function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry?.dispose();
      child.material?.dispose();
    }
  });

  // Remove from parent
  object.parent?.remove(object);
}
```

---

## Troubleshooting

### Common Issues

**Problem: Scene appears black**

- Check lighting setup
- Verify camera position
- Ensure objects are in view frustum

**Problem: Poor performance**

- Enable idle detection
- Reduce point count
- Check for memory leaks
- Profile with Chrome DevTools

**Problem: Objects not visible**

- Check bounding box
- Verify camera near/far planes
- Ensure materials are configured

**Problem: Dimension navigation not working**

- Initialize sceneDimsManager
- Check dimension metadata
- Verify keyboard focus

---

## API Reference

### SceneManager

| Method                                           | Description                                                    |
| ------------------------------------------------ | -------------------------------------------------------------- |
| `init(options)`                                  | Build renderer, scene, camera, controls, post-processing       |
| `loadSceneData(src, loaderConfig?)`              | Load zarr scene; clears prior content, auto-frames, auto-clips |
| `centerCameraOnScene()`                          | Frame camera on scene bounding box                             |
| `toggleCentering()`                              | Switch center mode (origin ↔ bbox)                             |
| `getCurrentCenter()`                             | Get active center point                                        |
| `updateFOV(delta)`                               | Adjust field of view                                           |
| `updateClippingPlanes(near, far)`                | Set camera clipping planes                                     |
| `autoAdjustClippingPlanes()`                     | Calculate optimal clipping from scene                          |
| `setDynamicClipping(enabled)`                    | Enable/disable per-frame clipping update                       |
| `getDynamicClippingState()`                      | Get current dynamic clipping state                             |
| `updateDynamicClippingPlanes()`                  | Manually trigger dynamic clipping update                       |
| `setControlType(type)` / `getControlType()`      | Switch / read current control type                             |
| `setAdaptivePixelRatio(dpr)`                     | Set DPR override (AdaptiveDPRManager)                          |
| `updateExposure/GlobalOffset/GlobalGamma(value)` | Update post-processing tone-mapping uniforms                   |
| `isWebGLContextLost()`                           | Query WebGL context-loss state                                 |
| `updateSize()`                                   | Handle resize (rAF-debounced)                                  |
| `dispose()`                                      | Clean up resources                                             |

### AnimationController

| Method                           | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| `startAnimation()`               | Begin render loop                                         |
| `stopAnimation()`                | Stop render loop                                          |
| `addPerFrameCallback(id, fn)`    | Add named per-frame callback (e.g., for dynamic clipping) |
| `removePerFrameCallback(id)`     | Remove per-frame callback by ID                           |
| `hasPerFrameCallback(id)`        | Check if callback exists                                  |
| `setAdaptiveDPRManager(manager)` | Set adaptive DPR manager for dynamic resolution           |
| `dispose()`                      | Clean up resources                                        |

### SceneDimsManager

| Method                        | Description           |
| ----------------------------- | --------------------- |
| `initFromScene(scene)`        | Initialize dimensions |
| `getDims()`                   | Get dimension info    |
| `setDimensionValue(idx, val)` | Update position       |
| `addListener(callback)`       | Subscribe to changes  |
| `reset()`                     | Clear dimensions      |

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._

---

## Top-Level Files

- `scene-manager.ts` — `SceneManager` orchestrator: renderer, camera,
  controls, post-processing, resize, disposal.
- `scene-dims-manager.ts` — Singleton dimension state across all nD
  objects in the scene (exported as both class `SceneDimsManager`
  and lazy-Proxy singleton `sceneDimsManager`).
- `lod-group-registry.ts` — `LODGroupRegistry`: per-frame `lod_group`
  child selector (screen-space-diagonal pick + frustum off-screen gate
  + asymmetric hysteresis), lazy-load gating, and shared-VRAM-budget
  LRU eviction. Exports pure helpers `projectBoxDiagonalPx` and
  `pickChildWithHysteresis` for unit testing.
- `synthetic-scene.ts` — Mulberry32-seeded synthetic line-segment
  scene generator (`generateSyntheticLines`) used by the perf bench
  and the `__luxarDebug.injectSyntheticScene` debug API.

## Subpackages

- [animation](./animation/README.md) — `AnimationController` render
  loop and `DimensionAnimationManager` per-dimension playback.
- [scene-manager](./scene-manager/README.md) — Extracted SceneManager
  helpers (camera setup/framing/materials/mode, clipping policy and
  bounds cache, render-pipeline setup, viewport DPR and resize).
