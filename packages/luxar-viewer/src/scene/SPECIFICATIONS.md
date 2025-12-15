# luxar-viewer.scene - Technical Specification

**Version**: 1.5.0
**Last Updated**: 2025-12-15

## Purpose

The `luxar-viewer.scene` package manages the THREE.js scene graph, animation loop, camera controls, and nD dimension coordination for n-dimensional scene visualization. It handles points, lines, Gaussian splats, and serves as the central orchestrator for all 3D rendering operations.

**Core Responsibility**: Maintain the 3D scene state, coordinate camera and controls, manage the render loop with intelligent idle detection, and synchronize nD dimension navigation across all scene objects (points, lines, splats, and future primitives).

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
5. [Dynamic Clipping Planes](#dynamic-clipping-planes)
6. [Window Resize Handling](#window-resize-handling)
7. [WebGL Context Loss Handling](#webgl-context-loss-handling)
8. [Dimension Coordination](#dimension-coordination)

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
│   ├── Lines A (THREE.Mesh with InstancedBufferGeometry)
│   ├── Lines B (THREE.Mesh with InstancedBufferGeometry)
│   └── Nested Groups
│       ├── More Points
│       └── More Lines
└── Helper Objects (optional)
    ├── Grid Helper
    └── Axes Helper
```

**Object Metadata - Points** (stored in `userData`):

```typescript
interface PointsUserData {
  nodeType: 'points';
  loader: PointSpatialIndexLoader; // Data loader instance
  attrs: ZarrGroupAttrs; // Node attributes from zarr
  spatialIndex?: PointSpatialIndex; // Spatial index for queries (optional)
}
```

**Object Metadata - Lines** (stored in `userData`):

```typescript
interface LinesUserData {
  nodeType: 'lines';
  loader: LinesSpatialIndexLoader; // Lines data loader instance
  attrs: LinesMetadata; // Node attributes from zarr
  spatialIndex?: LinesChunkSpatialIndex; // Dual spatial index (optional)
  maxWidth: number; // Maximum line width (for bounding box expansion)
}
```

> **IMPORTANT: Dimension Architecture**
> Dimension metadata (`sceneDimensions`) is stored ONLY at the Scene level (on the
> `LuxarScene` root group), never on individual data nodes. This ensures a single
> source of truth. Nodes access dimension info via `ViewState` or `SceneDimsManager`.

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

**Input**: Scene graph with point clouds and line objects

**Output**: `THREE.Box3` representing AABB

**Algorithm**:

```typescript
function updateBoundingBox(): THREE.Box3 {
  const box = new THREE.Box3();

  // Traverse all objects in scene
  scene.traverse((object) => {
    // Process point clouds
    if (object instanceof THREE.Points) {
      const geometry = object.geometry;
      if (!geometry) return;

      if (!geometry.boundingBox) {
        geometry.computeBoundingBox();
      }

      if (geometry.boundingBox) {
        const worldBox = geometry.boundingBox.clone();
        worldBox.applyMatrix4(object.matrixWorld);
        box.union(worldBox);
      }
    }

    // Process line objects (Lines use THREE.Mesh with InstancedBufferGeometry)
    if (object instanceof THREE.Mesh && object.userData.nodeType === 'lines') {
      const geometry = object.geometry;
      if (!geometry) return;

      if (!geometry.boundingBox) {
        geometry.computeBoundingBox();
      }

      if (geometry.boundingBox) {
        const worldBox = geometry.boundingBox.clone();
        worldBox.applyMatrix4(object.matrixWorld);

        // Expand by max line width (stored in userData)
        const maxWidth = object.userData.maxWidth ?? 0;
        if (maxWidth > 0) {
          worldBox.expandByScalar(maxWidth);
        }

        box.union(worldBox);
      }
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

**Note**: Line bounding boxes are expanded by `maxWidth` to account for line thickness in world space.

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

## 5. Dynamic Clipping Planes

### 5.1 Purpose

Automatically adjust near and far clipping planes as the camera moves to maintain optimal Z-buffer precision and prevent clipping artifacts.

**Problem**: Static clipping planes set at load time become suboptimal as the user navigates:

- **Zooming in close**: Objects clip against a near plane that's too far
- **Moving far away**: Z-fighting artifacts from excessive far/near ratio
- **Flying to scene edges**: Far plane clips objects behind the new camera position

**Solution**: Smoothly interpolate clipping planes toward optimal values each frame using exponential filtering.

### 5.2 Algorithm

**Per-Frame Update**:

```typescript
class SceneManager {
  // Dynamic clipping state
  private dynamicClippingEnabled: boolean = true;
  private clippingAdaptSpeed: number = 0.1; // 0.01 to 0.5
  private smoothedNear: number = 0.1;
  private smoothedFar: number = 1000;

  updateDynamicClippingPlanes(): void {
    if (!this.dynamicClippingEnabled) return;

    // Get scene bounds
    const bounds = this.getSceneBoundsFromMetadata();
    if (!bounds) return;

    // Calculate distances from camera to bounding box
    const { nearDist, farDist, isInside } = this.calculateDistancesToBounds(bounds);

    // Calculate optimal planes with margin of sqrt(3)-1+0.1 ≈ 0.832
    // sqrt(3)-1 accounts for cube diagonal (corner distance is sqrt(3)× face distance)
    // +0.1 adds extra 10% safety buffer
    const margin = Math.sqrt(3) - 1 + 0.1; // ≈ 0.832

    // When camera is inside the bounding box, use minimum near plane directly
    // to avoid clipping nearby points. When outside, apply margin to corner distance.
    const optimalNear = isInside ? 0.001 : Math.max(0.001, nearDist * (1 - margin));
    const optimalFar = farDist * (1 + margin); // ~183.2% of distance

    // Exponential smoothing: new = (1-α)*current + α*optimal
    const α = this.clippingAdaptSpeed;
    this.smoothedNear = (1 - α) * this.smoothedNear + α * optimalNear;
    this.smoothedFar = (1 - α) * this.smoothedFar + α * optimalFar;

    // Apply safety clamps
    this.smoothedNear = Math.max(0.001, this.smoothedNear);

    // Prevent excessive far/near ratio (Z-buffer precision)
    const maxRatio = 100000;
    if (this.smoothedFar / this.smoothedNear > maxRatio) {
      this.smoothedNear = this.smoothedFar / maxRatio;
    }

    // Only update camera if values changed significantly (>0.1%)
    const nearChanged = Math.abs(this.camera.near - this.smoothedNear) / this.camera.near > 0.001;
    const farChanged = Math.abs(this.camera.far - this.smoothedFar) / this.camera.far > 0.001;

    if (nearChanged || farChanged) {
      this.camera.near = this.smoothedNear;
      this.camera.far = this.smoothedFar;
      this.camera.updateProjectionMatrix();
    }
  }
}
```

### 5.3 Distance Calculation

**Calculate distances from camera to bounding box, handling inside-box case**:

```typescript
private calculateDistancesToBounds(bounds: BoundingBox): {
  nearDist: number;
  farDist: number;
  isInside: boolean;
} {
  const cameraPos = this.camera.position;

  // Check if camera is inside the bounding box
  const isInside = isPointInBoundingBox(
    { x: cameraPos.x, y: cameraPos.y, z: cameraPos.z },
    bounds
  );

  // Get all 8 corners of bounding box
  const corners = [
    new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.min.z),
    new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
    new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
    new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
    new THREE.Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
    new THREE.Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
    new THREE.Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
    new THREE.Vector3(bounds.max.x, bounds.max.y, bounds.max.z),
  ];

  // Find min and max distances to corners
  let nearDist = Infinity;
  let farDist = 0;

  for (const corner of corners) {
    const dist = cameraPos.distanceTo(corner);
    nearDist = Math.min(nearDist, dist);
    farDist = Math.max(farDist, dist);
  }

  // If camera is inside the bounding box, use minimum near distance
  // to avoid clipping nearby points
  if (isInside) {
    nearDist = 0.001; // Minimum practical near plane
  } else {
    // Ensure minimum near distance for outside case
    nearDist = Math.max(0.001, nearDist);
  }

  return { nearDist, farDist, isInside };
}
```

### 5.4 Integration with Animation Loop

**AnimationController calls the update each frame**:

```typescript
class AnimationController {
  private animate = (): void => {
    if (!this.isAnimating) return;

    this.performanceMonitor.begin();
    this.animationId = requestAnimationFrame(this.animate);

    // Update camera controls
    this.controls.update();

    // Update dynamic clipping planes (smooth interpolation)
    this.sceneManager.updateDynamicClippingPlanes();

    // Render through post-processing pipeline
    this.postProcessing.render();

    this.performanceMonitor.end();
  };
}
```

### 5.5 Configuration

**Settings in RenderingSettings**:

```typescript
interface RenderingSettings {
  // ... existing settings ...

  // Dynamic clipping planes
  dynamicClippingEnabled: boolean; // Default: true
  clippingAdaptSpeed: number; // Default: 0.1, range: 0.01-0.5
}
```

**UI Controls**:

- **Auto Clipping** (checkbox): Enables/disables dynamic adjustment
- **Adapt Speed** (slider): Controls responsiveness (0.01 = slow/smooth, 0.5 = fast/responsive)

**Behavior when disabled**: Near/Far sliders become editable for manual control.

### 5.6 Performance Considerations

**Computational Cost**: Minimal (~0.01ms per frame)

- 8 distance calculations (corners to camera)
- 2 exponential smoothing operations
- Conditional projection matrix update

**Optimization**: Only update projection matrix when values change >0.1%, avoiding unnecessary GPU state changes.

**Smoothing Factor Guidelines**:

- `0.01`: Very smooth, ~5 seconds to 95% convergence (cinematic)
- `0.1`: Balanced, ~0.5 seconds to 95% convergence (default)
- `0.5`: Fast response, ~0.1 seconds to 95% convergence (responsive)

---

## 6. Window Resize Handling

### 6.1 Resize Algorithm

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

### 6.2 Pixel Ratio Handling

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

### 6.3 Resize Debouncing (Performance Optimization)

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

### 6.4 Fullscreen Handling

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

## 8. Dimension Coordination

### 8.1 Scene Dimensions Manager

**Purpose**: Maintain unified dimension state for all nD objects in the scene.

**State**:

```typescript
class SceneDimsManager {
  private dims: SimpleDims | null = null;
  private listeners: Set<() => void> = new Set();

  initFromScene(scene: THREE.Scene): boolean {
    // Extract dimensions from scene level (single source of truth)
    // 1. Check scene.userData.sceneDimensions
    // 2. Check immediate children (LuxarScene root group)
    // 3. Check scene.getObjectByName('LuxarScene')
    let sceneDimensions = scene.userData.sceneDimensions;

    if (!sceneDimensions) {
      const luxarScene = scene.getObjectByName('LuxarScene');
      sceneDimensions = luxarScene?.userData.sceneDimensions;
    }

    if (sceneDimensions?.dimensions) {
      this.dims = initializeDims(sceneDimensions.dimensions);
      // Note: notifyListeners() NOT called here because listeners
      // haven't been registered yet. Initial update triggered manually
      // in input-handler.ts after listener registration.
      return true;
    }
    return false;
  }

  getDims(): SimpleDims | null {
    return this.dims;
  }

  setDimensionValue(dimIndex: number, value: number): void {
    if (!this.dims) return;

    this.dims.currentStep[dimIndex] = value;
    this.notifyListeners(); // Trigger reactive updates
  }

  addListener(callback: () => void): void {
    this.listeners.add(callback);
  }

  private notifyListeners(): void {
    this.listeners.forEach((cb) => cb());
  }
}
```

### 8.1.1 Dimension Initialization Policy

**Policy** (implemented in `scene-dims-manager.ts:118-140`):

When initializing non-displayed dimensions, the system uses a type-aware strategy:

| Dimension Type           | Initial Value                | Rationale                                         | Examples                                                  |
| ------------------------ | ---------------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| **Discrete/Categorical** | **Minimum (first position)** | Sequence data starts at beginning, not middle     | Time series (t=0), channels (channel 0), frames (frame 0) |
| **Continuous Spatial**   | **Center**                   | No natural "first" position in spatial dimensions | 4th spatial dimension (W=0), hyperspatial coordinates     |
| **Displayed (X,Y,Z)**    | **0**                        | Camera-controlled, not slice-controlled           | Displayed X, Y, Z axes                                    |

**Implementation** (simplified from actual code):

```typescript
// Initialize dimension positions
const currentStep = new Array(ndim).fill(0);

for (let i = 0; i < ndim; i++) {
  if (metadata[i].display !== true) {
    const [min, max] = this.dimensionRanges[i];

    // Discrete/categorical: start at first position (minimum)
    if (metadata[i].discrete || metadata[i].categories) {
      currentStep[i] = min; // e.g., t=0, channel=0
    } else {
      // Continuous spatial: start at center
      currentStep[i] = (min + max) / 2; // e.g., W=0 for range [-100, 100]
    }
  }
  // Displayed dimensions start at 0 (camera controls actual view)
}
```

**Why This Matters**:

- **Time-series datasets** start at t=0 (beginning) instead of mid-timeline
- **Multi-channel data** starts at first channel (DAPI, GFP channel 0)
- **Spatial 4D+** still centers (W=0 makes sense for symmetric spatial data)
- **Slider position matches displayed data** on initial viewer load

**User Experience Impact**:

```
Before fix:
  Time slider shows: 25 ns (middle)
  Data displayed:     0 ns (start)
  Result: Confusing mismatch!

After fix:
  Time slider shows: 0 ns (start)
  Data displayed:    0 ns (start)
  Result: Perfect match!
```

### 8.1.2 Initial Update Trigger

**Critical Implementation Detail**:

The `initFromScene()` method sets up dimension state but does NOT call `notifyListeners()` because listeners haven't been registered yet. The initial update is triggered manually in `input-handler.ts` after listener registration:

```typescript
// From input-handler.ts:191-194
sceneDimsManager.addListener(() => {
  this.updateAllNDNodes();
  // ... slider updates, animation trigger
});

// Trigger initial update now that listener is registered
this.updateAllNDNodes();
this.animationController.startAnimation();
```

**Sequence**:

```
1. sceneDimsManager.initFromScene(scene)
   → Dimension state created with initial positions
   → Listeners list empty, so no notifications

2. sceneDimsManager.addListener(callback)
   → Listener registered for future updates

3. inputHandler.updateAllNDNodes()  ← INITIAL TRIGGER
   → Data loads at correct initial position
   → First render with proper slice

4. User changes slider
   → sceneDimsManager.setDimensionValue()
   → notifyListeners() → callback() → updateAllNDNodes()
```

This two-phase initialization ensures data loads at the correct initial position while avoiding premature notifications to unregistered listeners.

### 8.2 Dimension Update Propagation

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

  // Update all data objects for new slice
  scene.traverse((object) => {
    // Update Points
    if (object instanceof THREE.Points && object.userData.nodeType === 'points') {
      updatePointsForDimensions(object, dims);
    }

    // Update Lines (Lines use THREE.Mesh with InstancedBufferGeometry)
    if (object instanceof THREE.Mesh && object.userData.nodeType === 'lines') {
      updateLinesForDimensions(object, dims);
    }
  });

  // Trigger render
  animationController.startAnimation();
});
```

**Lines Dimension Update**:

```typescript
async function updateLinesForDimensions(linesObject: THREE.Mesh, dims: SimpleDims): Promise<void> {
  const { loader, spatialIndex } = linesObject.userData as LinesUserData;

  // Calculate slice position and tolerance
  const slicePosition = getSlicePosition(dims);
  // NOTE: segment_chunk_bounds already include line width, so spatial tolerance = 0
  const tolerance = computeLinesTolerance(dims.dimensions);

  // Load visible lines data
  const linesData = await loader.loadForView(slicePosition, tolerance);

  // Update geometry
  const geometry = linesObject.geometry as THREE.BufferGeometry;

  // Update position attribute
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linesData.vertices, 3));

  // Update index (for indexed line segments)
  geometry.setIndex(new THREE.Uint32BufferAttribute(linesData.segments, 1));

  // Update other attributes
  if (linesData.colors) {
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(linesData.colors, 3));
  }
  if (linesData.widths) {
    geometry.setAttribute('lineWidth', new THREE.Float32BufferAttribute(linesData.widths, 1));
  }

  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}
```

### 8.3 Multi-Object Synchronization

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

- **v1.5.0** (2025-12-11): Dimension initialization fix documentation
  - **ADDED**: Section 8.1.1 "Dimension Initialization Policy" documenting type-aware initialization
  - **ADDED**: Section 8.1.2 "Initial Update Trigger" documenting two-phase initialization sequence
  - **UPDATED**: Section 8.1 "Scene Dimensions Manager" with accurate implementation notes
  - **IMPROVED**: Discrete/categorical dimensions now initialize to minimum (t=0, channel=0) not center
  - **IMPROVED**: Documentation now matches actual implementation in scene-dims-manager.ts:118-194 and input-handler.ts:191-194
  - Documents fixes from commits 3c548d5 and d619d82

- **v1.4.0** (2025-12-10): Camera-inside-bounding-box handling
  - **IMPROVED**: Dynamic clipping now detects when camera is inside the bounding box
  - **IMPROVED**: When inside, near plane is set to minimum (0.001) to avoid clipping nearby points
  - **UPDATED**: `calculateDistancesToBounds()` now returns `isInside` flag
  - **UPDATED**: `updateDynamicClippingPlanes()` uses `isInside` to skip margin calculation when inside
  - Eliminates clipping artifacts when exploring inside point clouds

- **v1.3.0** (2025-12-09): Dynamic clipping planes
  - **ADDED**: Section 5 "Dynamic Clipping Planes" with exponential smoothing algorithm
  - **ADDED**: Per-frame clipping plane adjustment based on camera-to-bounds distance
  - **ADDED**: Configurable adapt speed parameter (0.01-0.5)
  - **ADDED**: Safety clamps for near plane minimum and far/near ratio
  - **UPDATED**: AnimationController integration for per-frame updates
  - **UPDATED**: Section numbering (Window Resize → 6, WebGL Context Loss → 7, Dimension Coordination → 8)
  - Smooth camera navigation without clipping artifacts

- **v1.2.0** (2025-12-09): Lines support
  - **ADDED**: Lines to scene graph structure (THREE.Mesh with InstancedBufferGeometry)
  - **ADDED**: `LinesUserData` interface for lines metadata
  - **UPDATED**: Bounding box algorithm to include lines with width expansion
  - **UPDATED**: Dimension update propagation to include lines
  - **ADDED**: `updateLinesForDimensions()` algorithm
  - Lines now fully integrated into scene management

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
