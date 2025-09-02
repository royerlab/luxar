# Luxar Scene Package

> Core 3D scene management and orchestration for point cloud visualization

## Overview

The Luxar Scene package provides comprehensive scene management, animation control, and nD dimension coordination for complex point cloud visualizations. It handles the THREE.js scene graph, camera management, rendering pipeline integration, and multi-dimensional data navigation.

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
├── scene-manager.ts       # Main scene orchestrator
├── animation-controller.ts # Render loop management
├── scene-dims-manager.ts  # nD dimension coordination
└── README.md             # This documentation
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
class SceneManager {
  // Scene setup
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ControlsManager;

  // Scene manipulation
  addToScene(object: THREE.Object3D): void;
  clearScene(): void;
  updateBoundingBox(): void;

  // Camera control
  updateFOV(delta: number): void;
  setControlType(type: 'orbit' | 'fly'): void;

  // Centering
  toggleCentering(): void;
  getCurrentCenter(): THREE.Vector3;

  // Lifecycle
  updateSize(): void;
  dispose(): void;
}
```

**Usage Example:**

```typescript
const sceneManager = new SceneManager('canvas-id');

// Add objects to scene
sceneManager.addToScene(pointCloud);

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
animationController.pause();
animationController.resume();
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
  currentStep: Float32Array; // Current position in each dimension
  metadata?: DimensionMetadata[]; // Names, units, ranges
}

interface DimensionMetadata {
  name: string; // e.g., "time", "x", "wavelength"
  unit: string; // e.g., "ms", "μm", "nm"
  range: [number, number]; // Min and max values
  step?: number; // Step size for discrete dimensions
  discrete?: boolean; // Whether dimension is discrete
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

---

## Scene Graph Structure

### Hierarchy

```
Scene (THREE.Scene)
├── Ambient Light
├── Point Cloud Groups
│   ├── Transform Group
│   │   └── Points (THREE.Points)
│   └── Direct Points
├── Camera (managed separately)
└── Helper Objects (grids, axes, etc.)
```

### Object Management

**Adding Objects:**

```typescript
// Add with automatic bbox update
sceneManager.addToScene(object);

// Batch operations
const group = new THREE.Group();
group.add(points1, points2, points3);
sceneManager.addToScene(group);
```

**Clearing Scene:**

```typescript
// Remove all objects except lights
sceneManager.clearScene();

// Dispose of geometries and materials
object.traverse((child) => {
  if (child.geometry) child.geometry.dispose();
  if (child.material) child.material.dispose();
});
```

---

## Camera Management

### FOV Control

The scene manager provides comprehensive FOV management:

```typescript
// Adjust FOV with mouse wheel + shift
sceneManager.updateFOV(delta);

// FOV limits: 10° to 200°
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
```

**Z-Buffer Best Practices:**

- Keep near/far ratio under 10,000:1 for best precision
- Use auto-adjust after loading new datasets
- Lower near values see closer objects but reduce precision

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

## Usage Examples

### Complete Setup

```typescript
import { SceneManager } from './scene/scene-manager';
import { AnimationController } from './scene/animation-controller';
import { sceneDimsManager } from './scene/scene-dims-manager';

// Initialize scene
const sceneManager = new SceneManager('canvas');
const animationController = new AnimationController(sceneManager, postProcessing);

// Setup dimensions for nD data
sceneDimsManager.initFromScene(sceneManager.scene);

// Start render loop
animationController.startAnimation();
```

### Loading Point Clouds

```typescript
// Load from zarr
const pointClouds = await loadFromZarr(url);

// Add to scene
pointClouds.forEach((points) => {
  sceneManager.addToScene(points);
});

// Update bounds
sceneManager.updateBoundingBox();

// Center camera on data
const center = sceneManager.getCurrentCenter();
sceneManager.controls.lookAt(center);
```

### Handling Window Resize

```typescript
window.addEventListener('resize', () => {
  sceneManager.updateSize();
  postProcessing.resize(window.innerWidth, window.innerHeight);
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

| Method                            | Description                           |
| --------------------------------- | ------------------------------------- |
| `addToScene(object)`              | Add object to scene                   |
| `clearScene()`                    | Remove all objects                    |
| `updateBoundingBox()`             | Recalculate bounds                    |
| `toggleCentering()`               | Switch center mode                    |
| `getCurrentCenter()`              | Get active center point               |
| `updateFOV(delta)`                | Adjust field of view                  |
| `updateClippingPlanes(near, far)` | Set camera clipping planes            |
| `autoAdjustClippingPlanes()`      | Calculate optimal clipping from scene |
| `setControlType(type)`            | Switch control mode                   |
| `updateSize()`                    | Handle resize                         |
| `dispose()`                       | Clean up resources                    |

### AnimationController

| Method                       | Description       |
| ---------------------------- | ----------------- |
| `startAnimation()`           | Begin render loop |
| `pause()`                    | Pause rendering   |
| `resume()`                   | Resume rendering  |
| `setPerformanceStats(stats)` | Attach monitor    |
| `dispose()`                  | Clean up          |

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
