# luxar-viewer.scene - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2025-12-09

## Purpose

The `luxar-viewer.scene` package manages the THREE.js scene graph, animation loop, camera controls, and nD dimension coordination for point cloud visualization. It serves as the central orchestrator for all 3D rendering operations.

**Core Responsibility**: Maintain the 3D scene state, coordinate camera and controls, manage the render loop with intelligent idle detection, and synchronize nD dimension navigation across all scene objects.

**Related Specifications**:

- `luxar-viewer.data` - Data loading managed by scene (see `../data/SPECIFICATIONS.md`)
- `luxar-viewer.rendering` - Post-processing pipeline (see `../rendering/SPECIFICATIONS.md`)
- `luxar-viewer.controls` - Camera control systems (see `../controls/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Scene Management](#scene-management)
2. [Animation System](#animation-system)
3. [Bounding Box Calculation](#bounding-box-calculation)
4. [Camera Centering](#camera-centering)
5. [Window Resize Handling](#window-resize-handling)
6. [Dimension Coordination](#dimension-coordination)

---

## 1. Scene Management

### 1.1 Scene Initialization

**Purpose**: Set up THREE.js scene with camera, lighting, and rendering context.

**Initialization Sequence**:

```
1. Create Scene
   - THREE.Scene instance
   - Dark background (0x111111)
   - Ambient white light (intensity 1.0)

2. Create Camera
   - PerspectiveCamera with configurable FOV
   - Initial position: (0, 0, 8)
   - Aspect ratio from canvas

3. Create Renderer
   - WebGLRenderer with HDR support
   - HalfFloatType render targets
   - Proper color space (LinearSRGB → SRGB pipeline)

4. Initialize Controls
   - OrbitControls (default) or FlyControls
   - Target at scene center

5. Initialize Post-Processing
   - EffectComposer with HDR buffers
   - Bloom, tone mapping, optional effects

6. Start Animation Loop
   - Idle detection enabled
   - Initial render triggered
```

### 1.2 Scene Graph Structure

**Hierarchy**:

```
Scene (THREE.Scene)
├── Ambient Light (THREE.AmbientLight)
├── Loaded Data Groups (THREE.Group)
│   ├── Points A (THREE.Points)
│   ├── Points B (THREE.Points)
│   └── Nested Groups
│       └── More Points
└── Helper Objects (optional)
    ├── Grid Helper
    └── Axes Helper
```

**Object Metadata** (stored in `userData`):

```typescript
interface PointsUserData {
  loader: PointSpatialIndexLoader; // Data loader instance
  attrs: ZarrGroupAttrs; // Node attributes from zarr
  spatialIndex: PointSpatialIndex; // Spatial index for queries
  sceneDimensions: DimensionMetadata[]; // Scene coordinate system
}
```

### 1.3 Object Management

**Adding Objects**:

```typescript
function addToScene(object: THREE.Object3D): void {
  // 1. Add to scene graph
  scene.add(object);

  // 2. Update bounding box to include new object
  updateBoundingBox();

  // 3. Trigger render
  animationController.startAnimation();
}
```

**Removing Objects**:

```typescript
function removeFromScene(object: THREE.Object3D): void {
  // 1. Dispose geometries and materials
  object.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });

  // 2. Remove from scene
  scene.remove(object);

  // 3. Recalculate bounding box
  updateBoundingBox();
}
```

**Clearing Scene**:

```typescript
function clearScene(): void {
  // Remove all children except lights
  const objectsToRemove = scene.children.filter((child) => !(child instanceof THREE.Light));

  objectsToRemove.forEach((obj) => removeFromScene(obj));
}
```

---

## 2. Animation System

### 2.1 Idle Detection

**Purpose**: Automatically pause rendering when scene is static to save GPU resources.

**Algorithm**:

```typescript
class AnimationController {
  private isAnimating: boolean = false;
  private lastActivityTime: number = 0;
  private idleTimeoutMs: number = 2000; // 2 seconds

  private animate = (): void => {
    if (!this.isAnimating) return;

    requestAnimationFrame(this.animate);

    // Update controls (returns true if changed)
    const controlsChanged = this.controls.update();

    if (controlsChanged) {
      this.lastActivityTime = Date.now();
    }

    // Render frame
    this.render();

    // Check for idle timeout
    this.checkIdleTimeout();
  };

  private checkIdleTimeout(): void {
    const idleTime = Date.now() - this.lastActivityTime;

    if (idleTime > this.idleTimeoutMs) {
      this.pause();
    }
  }

  public startAnimation(): void {
    if (this.isAnimating) return;

    this.lastActivityTime = Date.now();
    this.isAnimating = true;
    this.animate();
  }

  public pause(): void {
    this.isAnimating = false;
  }
}
```

**Activity Sources** (reset idle timer):

- Mouse movement during drag
- Keyboard input
- Control changes (rotation, zoom, pan)
- Data loading completion
- Dimension navigation
- Window focus/visibility change

### 2.2 Render Loop

**Frame Sequence**:

```
1. Check if animation active
   ↓
2. Request next animation frame
   ↓
3. Update controls (OrbitControls.update() or FlyControls.update(delta))
   ↓
4. Begin performance monitoring
   ↓
5. Render scene through post-processing
   ↓
6. End performance monitoring
   ↓
7. Check idle timeout
   ↓
8. Loop if still animating
```

**Performance Monitoring Integration**:

```typescript
private render(): void {
    // Begin frame timing
    this.performanceStats?.begin()

    // Render through post-processing pipeline
    this.postProcessing.render()

    // End frame timing
    this.performanceStats?.end()
}
```

### 2.3 Render Triggering

**Manual Triggers**:

```typescript
// Force single frame render (useful after data changes)
animationController.startAnimation();

// Ensure rendering continues (e.g., during continuous control input)
animationController.resetIdleTimer();
```

**Automatic Triggers**:

- Control system reports changes
- Window gains focus
- Document becomes visible
- New data loaded
- Dimension slider moved

---

## 3. Bounding Box Calculation

### 3.1 Purpose

Calculate the axis-aligned bounding box (AABB) encompassing all scene objects for camera positioning and frustum calculations.

### 3.2 Bounding Box Algorithm

**Input**: Scene graph with multiple point clouds

**Output**: `THREE.Box3` representing AABB

**Algorithm**:

```typescript
function updateBoundingBox(): THREE.Box3 {
  const box = new THREE.Box3();

  // Traverse all objects in scene
  scene.traverse((object) => {
    // Only process point clouds
    if (!(object instanceof THREE.Points)) {
      return;
    }

    // Get geometry
    const geometry = object.geometry;
    if (!geometry) return;

    // Compute bounding box if not present
    if (!geometry.boundingBox) {
      geometry.computeBoundingBox();
    }

    // Transform to world space and expand scene box
    if (geometry.boundingBox) {
      const worldBox = geometry.boundingBox.clone();
      worldBox.applyMatrix4(object.matrixWorld);
      box.union(worldBox);
    }
  });

  // Handle empty scene
  if (box.isEmpty()) {
    box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  }

  return box;
}
```

**Optimization**: Cache bounding boxes and only recalculate when objects are added/removed.

### 3.3 Bounding Sphere Calculation

**Purpose**: Derive bounding sphere from bounding box for camera calculations.

```typescript
function getBoundingSphere(): THREE.Sphere {
  const box = getBoundingBox();
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  return sphere;
}
```

**Usage**:

- Camera distance calculations
- Auto-zoom to fit scene
- Clipping plane adjustments

---

## 4. Camera Centering

### 4.1 Center Point Modes

**Two centering modes**:

1. **Native Center**: Scene's inherent center (0, 0, 0) or specified center point
2. **Bounding Box Center**: Geometric center of all loaded objects

**Mode Toggle**:

```typescript
enum CenterMode {
  NATIVE = 'native',
  BOUNDING_BOX = 'bbox',
}

let currentCenterMode: CenterMode = CenterMode.BOUNDING_BOX;

function toggleCentering(): void {
  currentCenterMode =
    currentCenterMode === CenterMode.NATIVE ? CenterMode.BOUNDING_BOX : CenterMode.NATIVE;

  const center = getCurrentCenter();
  controls.target.copy(center);
}
```

### 4.2 Center Calculation

**Native Center**:

```typescript
function getNativeCenter(): THREE.Vector3 {
  // Use scene's origin or custom center point
  return sceneCenter ?? new THREE.Vector3(0, 0, 0);
}
```

**Bounding Box Center**:

```typescript
function getBoundingBoxCenter(): THREE.Vector3 {
  const box = getBoundingBox();
  const center = new THREE.Vector3();
  box.getCenter(center);
  return center;
}
```

**Current Center** (respects mode):

```typescript
function getCurrentCenter(): THREE.Vector3 {
  return currentCenterMode === CenterMode.NATIVE ? getNativeCenter() : getBoundingBoxCenter();
}
```

### 4.3 Auto-Focus

**Purpose**: Automatically position camera to view entire scene.

**Algorithm**:

```typescript
function focusCamera(): void {
  // Get bounding sphere
  const sphere = getBoundingSphere();

  // Calculate required distance based on FOV
  const fov = camera.fov * (Math.PI / 180); // Convert to radians
  const distance = sphere.radius / Math.tan(fov / 2);

  // Add padding (20%)
  const targetDistance = distance * 1.2;

  // Position camera along current view direction
  const direction = new THREE.Vector3();
  camera.getWorldDirection(direction);
  direction.negate(); // Look at center, not away

  const newPosition = sphere.center.clone();
  newPosition.addScaledVector(direction, targetDistance);

  // Smooth transition to new position
  camera.position.copy(newPosition);
  controls.target.copy(sphere.center);
  controls.update();

  animationController.startAnimation();
}
```

---

## 5. Window Resize Handling

### 5.1 Resize Algorithm

**Purpose**: Maintain correct aspect ratio and rendering resolution when window size changes.

**Algorithm**:

```typescript
function handleResize(): void {
  // 1. Get new canvas dimensions
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  // 2. Update camera aspect ratio
  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  // 3. Update renderer size
  renderer.setSize(width, height, false); // false = don't update canvas CSS

  // 4. Update post-processing composer
  postProcessing.setSize(width, height);

  // 5. Trigger render
  animationController.startAnimation();
}
```

**Critical Detail**: Call `updateProjectionMatrix()` after changing camera aspect ratio to rebuild the projection matrix.

### 5.2 Pixel Ratio Handling

**Purpose**: Support high-DPI displays (Retina, 4K).

```typescript
function updatePixelRatio(): void {
  const pixelRatio = window.devicePixelRatio || 1;
  renderer.setPixelRatio(Math.min(pixelRatio, 2)); // Cap at 2x for performance
}
```

**Called**:

- During initialization
- On window DPI change (rare)

### 5.3 Resize Debouncing (Performance Optimization)

**Purpose**: Prevent excessive GPU buffer reallocations during window dragging.

**Problem**: Window resize events fire rapidly (60+ times/sec during drag). Each resize triggers expensive WebGL buffer reallocations, causing stuttering and poor UX.

**Solution**: Debounce resize events using `requestAnimationFrame` coalescing.

**Implementation**:

```typescript
class SceneManager {
  private resizeRAF: number | null = null;
  private pendingResize: { width: number; height: number } | null = null;

  updateSize(): void {
    // Store latest dimensions (multiple rapid events update this)
    this.pendingResize = {
      width: window.innerWidth,
      height: window.innerHeight,
    };

    // Cancel any pending resize
    if (this.resizeRAF !== null) {
      cancelAnimationFrame(this.resizeRAF);
    }

    // Schedule resize for next frame (coalesces multiple events)
    this.resizeRAF = requestAnimationFrame(() => {
      if (!this.pendingResize) return;

      this.doUpdateSize(this.pendingResize.width, this.pendingResize.height);
      this.pendingResize = null;
      this.resizeRAF = null;
    });
  }

  private doUpdateSize(width: number, height: number): void {
    // Actual resize logic (called once per frame at most)
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    postProcessing.resize(width, height);
    materialManager.updateCameraParams(fov, resolution);
  }
}
```

**Benefits**:
- **Coalescing**: Multiple resize events in one frame → single GPU reallocation
- **Smooth UX**: No stuttering during window drag
- **Resource Efficiency**: Prevents memory thrashing
- **Frame-Synchronized**: Resizes happen on frame boundaries

**Performance Impact**:
- Without debouncing: 60+ buffer reallocations/sec during drag
- With debouncing: 1 buffer reallocation/frame (~60/sec max, typically much less)
- Memory savings: Prevents temporary buffer duplication

**Implementation Location**: `scene-manager.ts:647-692`

---

### 5.4 Fullscreen Handling

**Fullscreen Entry**:

```typescript
function enterFullscreen(): void {
  const element = canvas.parentElement || canvas;

  if (element.requestFullscreen) {
    element.requestFullscreen();
  }

  // Resize will trigger automatically via fullscreenchange event
}
```

**Fullscreen Exit**:

```typescript
function exitFullscreen(): void {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  }
}
```

**Event Handling**:

```typescript
document.addEventListener('fullscreenchange', () => {
  handleResize();
  updateSize(); // Ensure everything updates
});
```

---

---

## 7. WebGL Context Loss Handling

### 7.1 Purpose

Handle WebGL context loss and restoration gracefully to prevent application crashes when GPU resets occur.

**Common Causes of Context Loss**:

- GPU driver crashes or resets
- System sleep/hibernate
- Too many WebGL contexts (browser limit)
- Out of GPU memory
- GPU overheating or hardware issues

### 7.2 Event Handling Setup

**Initialization**:

```typescript
class SceneManager {
  private isContextLost: boolean = false;
  private contextLostHandler: ((event: Event) => void) | null = null;
  private contextRestoredHandler: ((event: Event) => void) | null = null;

  private setupContextLossHandling(): void {
    const canvas = this.canvasElement;

    // Context loss handler
    this.contextLostHandler = (event: Event) => {
      event.preventDefault(); // CRITICAL: Required to allow restoration
      this.isContextLost = true;

      log.error(
        Modules.SCENE_MANAGER,
        'WebGL context lost! GPU driver issue, system sleep, or memory pressure.'
      );

      showError('Graphics context lost - attempting to restore...');
    };

    // Context restoration handler
    this.contextRestoredHandler = async (_event: Event) => {
      log.info(Modules.SCENE_MANAGER, 'WebGL context restored - recreating resources...');

      try {
        this.isContextLost = false;

        // Force renderer to recreate internal state
        this.renderer.resetState();

        // Trigger render to force resource recreation
        this.dispatchEvent({ type: 'change' });

        hideLoadingIndicator();
        log.success(Modules.SCENE_MANAGER, 'WebGL context successfully restored');
      } catch (error) {
        log.error(Modules.SCENE_MANAGER, 'Failed to restore WebGL context:', error);
        showError('Failed to restore graphics context. Please refresh the page.');
      }
    };

    // Register event listeners
    canvas.addEventListener('webglcontextlost', this.contextLostHandler, false);
    canvas.addEventListener('webglcontextrestored', this.contextRestoredHandler, false);
  }
}
```

### 7.3 Context Loss Detection

**Public API**:

```typescript
isWebGLContextLost(): boolean {
  return this.isContextLost;
}
```

**Usage**:

```typescript
// Check before rendering
if (sceneManager.isWebGLContextLost()) {
  // Skip render, wait for restoration
  return;
}

// Normal rendering
sceneManager.render();
```

### 7.4 Resource Recreation

**What Needs Recreation**:

- WebGL internal state (renderer.resetState())
- Render targets (post-processing buffers)
- Shaders and programs (automatic on first use)
- Textures (reupload data)
- Buffers (reupload geometry data)

**Automatic Recreation**:
THREE.js handles most recreation automatically when rendering after context restoration. The key is:

1. Call `event.preventDefault()` in contextlost handler
2. Call `renderer.resetState()` after restoration
3. Trigger a render to force resource recreation

### 7.5 Error Recovery

**Failure Scenarios**:

```typescript
// If restoration fails:
// 1. Log error with details
// 2. Show user-friendly error message
// 3. Prompt page refresh

catch (error) {
  log.error(Modules.SCENE_MANAGER, 'Context restoration failed:', error);
  showError('Failed to restore graphics. Please refresh the page to continue.');
}
```

**User Experience**:

- Immediate feedback on context loss ("Graphics context lost...")
- Progress indication during restoration
- Success confirmation or error guidance
- No application crash - graceful degradation

### 7.6 Cleanup

**Disposal**:

```typescript
dispose(): void {
  // Remove context loss handlers
  if (this.contextLostHandler) {
    this.canvasElement.removeEventListener('webglcontextlost', this.contextLostHandler, false);
    this.contextLostHandler = null;
  }

  if (this.contextRestoredHandler) {
    this.canvasElement.removeEventListener('webglcontextrestored', this.contextRestoredHandler, false);
    this.contextRestoredHandler = null;
  }

  // ... other cleanup
}
```

---

## 6. Dimension Coordination

### 6.1 Scene Dimensions Manager

**Purpose**: Maintain unified dimension state for all nD objects in the scene.

**State**:

```typescript
class SceneDimsManager {
  private dims: SimpleDims | null = null;
  private listeners: Set<() => void> = new Set();

  initFromScene(scene: THREE.Scene): void {
    // Extract dimensions from first points object with metadata
    scene.traverse((object) => {
      if (object.userData.sceneDimensions) {
        this.dims = initializeDims(numPoints, totalElements, object.userData.sceneDimensions);
        this.notifyListeners();
        return; // Found dimensions, stop traversal
      }
    });
  }

  getDims(): SimpleDims | null {
    return this.dims;
  }

  setDimensionValue(dimIndex: number, value: number): boolean {
    if (!this.dims) return false;

    const changed = jumpToDimension(this.dims, dimIndex, value, this.getRanges());

    if (changed) {
      this.notifyListeners();
    }

    return changed;
  }

  addListener(callback: () => void): void {
    this.listeners.add(callback);
  }

  private notifyListeners(): void {
    this.listeners.forEach((cb) => cb());
  }
}
```

### 6.2 Dimension Update Propagation

**Flow**:

```
1. User changes dimension slider
   ↓
2. SceneDimsManager.setDimensionValue(dim, value)
   ↓
3. Update SimpleDims.currentStep[dim]
   ↓
4. Notify all listeners
   ↓
5. Data loaders update for new slice
   ↓
6. Points visibility recalculated
   ↓
7. Geometries updated
   ↓
8. Render triggered
```

**Listener Example**:

```typescript
sceneDimsManager.addListener(() => {
  const dims = sceneDimsManager.getDims();

  // Update all points objects for new slice
  scene.traverse((object) => {
    if (object instanceof THREE.Points && object.userData.loader) {
      updatePointsForDimensions(object, dims);
    }
  });

  // Trigger render
  animationController.startAnimation();
});
```

### 6.3 Multi-Object Synchronization

**Invariant**: All point clouds in a scene share the **same** dimension system.

**Enforcement**:

```typescript
function validateSceneDimensions(scene: THREE.Scene): boolean {
  let referenceDims: DimensionMetadata[] | null = null;

  scene.traverse((object) => {
    if (object.userData.sceneDimensions) {
      if (!referenceDims) {
        referenceDims = object.userData.sceneDimensions;
      } else {
        // Verify dimensions match
        if (!dimensionsEqual(referenceDims, object.userData.sceneDimensions)) {
          throw new Error('Scene contains objects with incompatible dimension systems');
        }
      }
    }
  });

  return true;
}
```

---

## Data Structures

### SceneManager

```typescript
interface SceneManager {
  // Core THREE.js objects
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls | ArcballControls | FlyControls;

  // Post-processing
  postProcessing: PostProcessingManager;

  // State
  boundingBox: THREE.Box3;
  centerMode: CenterMode;
  sceneCenter: THREE.Vector3 | null;

  // Methods
  init(): Promise<void>;
  addToScene(object: THREE.Object3D): void;
  clearScene(): void;
  updateBoundingBox(): void;
  toggleCentering(): void;
  updateFOV(delta: number): void;
  updateSize(): void;
  dispose(): void;
}
```

### AnimationController

```typescript
interface AnimationController {
  // State
  isAnimating: boolean;
  lastActivityTime: number;
  idleTimeoutMs: number;

  // Methods
  startAnimation(): void;
  pause(): void;
  resume(): void;
  resetIdleTimer(): void;
  setPerformanceStats(stats: PerformanceMonitor): void;
  dispose(): void;
}
```

### SceneDimsManager

```typescript
interface SceneDimsManager {
  dims: SimpleDims | null;
  listeners: Set<() => void>;

  initFromScene(scene: THREE.Scene): void;
  getDims(): SimpleDims | null;
  setDimensionValue(dimIndex: number, value: number): boolean;
  addListener(callback: () => void): void;
  removeListener(callback: () => void): void;
  reset(): void;
}
```

---

## Changelog

- **v1.1.0** (2025-12-09): Resize debouncing optimization
  - **ADDED**: Section 5.3 documenting resize debouncing with requestAnimationFrame
  - Documents performance optimization that prevents GPU buffer thrashing
  - Explains coalescing algorithm and benefits
  - Implementation at scene-manager.ts:647-692
  - No functional changes - documentation only

- **v1.0.1** (2025-12-08): WebGL context loss handling
  - Added `setupContextLossHandling()` method in SceneManager
  - Added event listeners for `webglcontextlost` and `webglcontextrestored`
  - Added `isWebGLContextLost()` public API method
  - Proper cleanup in dispose() method
  - Graceful recovery from GPU resets, system sleep, and memory pressure
  - User-friendly error messages and recovery flow
  - See: `scene-manager.ts:197-250`

- **v1.0.0** (2025-01-30): Initial specification
  - Scene management with THREE.js integration
  - Animation system with idle detection
  - Bounding box calculation algorithms
  - Dual-mode camera centering (native and bbox)
  - Window resize handling with DPI support
  - nD dimension coordination system
