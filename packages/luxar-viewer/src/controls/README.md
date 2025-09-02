# Luxar Controls Package

> Comprehensive 3D navigation and interaction system for the Luxar Player

## Overview

The Luxar Controls package provides a sophisticated, extensible control system for navigating and interacting with 3D points visualizations. It implements multiple control paradigms, seamless mode switching, and a robust input management system designed for both scientific exploration and cinematic presentation.

### Key Features

- **Multiple Control Modes**: Orbit (trackball), Arcball (quaternion-based), and Fly (free-flight) controls
- **Seamless Mode Switching**: Hot-swap between control types with state preservation
- **Unified Configuration**: Single source of truth for all control parameters
- **Input Context Management**: Intelligent key routing to prevent conflicts
- **Physics-Based Movement**: Smooth, momentum-based navigation with configurable damping
- **Quaternion-Based Rotation**: Gimbal-lock-free rotation with unlimited freedom

### Package Architecture

```
controls/
├── controls-manager.ts      # Central control system orchestrator
├── luxar-fly-controls.ts   # Free-flight 6DOF controller
├── control-config.ts       # Centralized configuration and defaults
├── input-context-manager.ts # Input routing and conflict resolution
├── types.ts                # TypeScript type definitions
└── README.md               # This documentation
```

---

## Control Types

### 1. Orbit Controls (Default)

The **OrbitControls** provide intuitive trackball-style navigation, ideal for examining objects from the outside. Based on THREE.js OrbitControls with enhancements for the Luxar system.

**Features:**

- Mouse drag to rotate around target
- Scroll wheel to zoom in/out
- Right-click drag to pan
- Auto-rotation for presentations
- Configurable zoom limits and damping
- **Limitation**: Gimbal lock at poles (cannot rotate past vertical)

**Best for:**

- Examining points from outside
- Presentations and demos
- Traditional 3D manipulation

### 2. Arcball Controls

The **ArcballControls** provide quaternion-based rotation without gimbal lock limitations, allowing unlimited rotation in any direction. Based on THREE.js ArcballControls.

**Features:**

- Full quaternion-based rotation (no gimbal lock)
- Unlimited rotation in any direction
- Mouse drag to rotate around target
- Scroll wheel to zoom in/out
- Right-click drag to pan
- Smooth damping for natural movement

**Best for:**

- Complex rotations without restrictions
- Scientific visualization requiring all orientations
- Users who need complete rotational freedom

**Controls:**

- 🖱️ **Left drag**: Rotate view around target
- 🖱️ **Right drag**: Pan camera
- 🖱️ **Scroll**: Zoom in/out
- **Double-click**: Focus on point

**Configuration:**

```typescript
{
  autoRotate: false,        // Enable auto-rotation
  autoRotateSpeed: 0.25,    // Rotation speed (rev/min)
  enableDamping: true,      // Smooth damping
  dampingFactor: 0.05,      // Damping strength
  minDistance: 0.1,         // Min zoom distance
  maxDistance: 1000,        // Max zoom distance
}
```

### 3. Fly Controls (Advanced)

The **LuxarFlyControls** provide quaternion-based free-flight navigation with 6 degrees of freedom, perfect for exploring points from within.

**Features:**

- True 6DOF movement (forward/back, left/right, up/down, pitch, yaw, roll)
- Quaternion-based rotation (no gimbal lock)
- Inertial physics with configurable damping
- Speed boost for rapid traversal
- Airplane-style controls (consistent at any orientation)

**Best for:**

- Exploring inside points
- Flying through data
- Cinematic camera movements

**Controls:**

- **WASD**: Move forward/back/left/right
- **Alt+W/S**: Move up/down (world space)
- **Q/E**: Roll left/right (barrel roll)
- **Shift**: 2x speed boost
- **↑↓←→**: Look up/down/left/right
- 🖱️ **Drag**: Free look
- **I**: Toggle inertial mode

---

## Control System Architecture

### ControlsManager

The `ControlsManager` class orchestrates the entire control system:

```typescript
class ControlsManager {
  // Switch between control types
  setControlType(type: 'orbit' | 'fly'): void;

  // Get current control type
  getControlType(): ControlType;

  // Access specific controls
  getControls(): OrbitControls | LuxarFlyControls;
  getFlyControls(): LuxarFlyControls | null;

  // Configuration
  setAutoRotate(enabled: boolean): void;
  setFlyMovementSpeed(speed: number): void;
  setFlyInertialMode(inertial: boolean): void;

  // Camera control
  lookAt(target: Vector3, smooth?: boolean): void;
  reset(): void;

  // Update loop
  update(): void;
}
```

**Key responsibilities:**

- Manages control instance lifecycle
- Preserves camera state during switches
- Provides unified configuration interface
- Handles control-specific updates
- Dispatches control events

### Input Context Management

The `InputContextManager` prevents input conflicts between different UI systems:

```typescript
enum InputContext {
  NAVIGATION, // Default 3D navigation
  FLY_CONTROLS, // Fly mode active
  TYPING, // Text input active
  DIMENSION_NAV, // nD dimension navigation
  UI_OVERLAY, // UI panels open
}
```

**Features:**

- Context-aware key filtering
- Context stack for nested states
- Automatic focus management
- Debug mode for troubleshooting

**Example usage:**

```typescript
// When entering fly mode
contextManager.setContext(InputContext.FLY_CONTROLS);

// When opening a text input
contextManager.pushContext(InputContext.TYPING);

// When closing the input
contextManager.popContext();
```

### Configuration System

All control parameters are centralized in the main config (`config/index.ts`):

```typescript
// In config/index.ts
controls: {
  fly: {
    inertialMode: { default: true },
    movement: {
      speed: { min: 0.5, max: 50.0, default: 5.0, step: 0.1 },
      damping: { min: 0.9, max: 0.99999, default: 0.999, step: 0.0001 },
      acceleration: { min: 0.1, max: 2.0, default: 0.5, step: 0.1 },
    },
    rotation: {
      speed: { min: 0.1, max: 5.0, default: 1.5, step: 0.1 },
      damping: { min: 0.9, max: 0.9999, default: 0.99, step: 0.0001 },
    },
    physics: {
      velocityThreshold: 1e-4,
      angularVelocityThreshold: 1e-4,
    },
  },
  orbit: {
    autoRotate: { speed: { min: 0.1, max: 5.0, default: 0.25 } },
    zoom: { minDistance: 0.1, maxDistance: 1000 },
  },
};
```

**Benefits:**

- Single source of truth for defaults
- Type-safe configuration
- UI-ready min/max/step values
- Consistent across all systems

---

## Fly Controls Deep Dive

### Mathematical Foundation

The fly controls use a quaternion-based physics simulation for smooth, gimbal-lock-free navigation.

#### Coordinate Frames

- **World frame** `W`: Global coordinates (Y-up by convention)
- **Camera frame** `B`: Local to camera
  - `e_x = (1,0,0)` → right
  - `e_y = (0,1,0)` → up
  - `e_z = (0,0,-1)` → forward (THREE.js convention)

#### Physics Model

**Translation:**

```
v ← v + a·Δt                    // Apply acceleration
v ← v · damping^(Δt·60)         // Frame-rate independent damping
p ← p + v·Δt                    // Update position
```

**Rotation:**

```
ω ← ω + τ·Δt                    // Apply torque (inertial mode)
ω ← ω · damping^(Δt·60)         // Angular damping
Δq = exp(ω·Δt/2)                // Exponential map
q ← Δq ⊗ q                      // PRE-multiply (world-space ω)
```

### Key Implementation Details

#### World-Space Angular Velocity

The angular velocity `ω` is maintained in **world space** and integrated using **pre-multiplication**:

```typescript
// Build torque in camera space
const cameraRight = new Vector3(1, 0, 0).applyQuaternion(orientation);
const cameraUp = new Vector3(0, 1, 0).applyQuaternion(orientation);

// Apply torque to world-space angular velocity
angularVelocity.addScaledVector(cameraRight, pitch);
angularVelocity.addScaledVector(cameraUp, yaw);

// Integrate with PRE-multiply
const deltaRotation = new Quaternion().setFromAxisAngle(axis, angle);
orientation.premultiply(deltaRotation); // Critical: premultiply for world-space
```

This approach prevents the "yaw becomes roll at ±90° pitch" problem common in naive implementations.

#### Inertial vs Non-Inertial Modes

**Inertial Mode** (default):

- Low damping (0.999 for translation, 0.99 for rotation)
- Momentum-based movement
- Smooth, cinematic feel
- Good for exploration

**Non-Inertial Mode**:

- High effective damping (~0.5)
- Direct velocity control
- Immediate response
- Good for precise positioning

#### Frame-Rate Independence

All physics use timestep-corrected damping:

```typescript
const dampingFactor = Math.pow(damping, delta * 60);
velocity.multiplyScalar(dampingFactor);
```

This ensures consistent behavior at any frame rate.

---

## System Architecture

### Data Flow

The control system follows a clean, event-driven architecture:

```
User Input
    ↓
InputHandler
    ↓
InputContextManager (filters based on context)
    ↓
ControlsManager (routes to active control)
    ↓
Control Implementation (Orbit/Fly)
    ↓
Camera Updates
    ↓
Scene Rendering
```

### Key Architectural Features

#### Seamless Mode Switching

- Camera state preserved during transitions
- Orbit target converted to fly look direction
- Fly orientation converted to orbit target
- Zero-downtime control swapping

#### External Input Management

Fly controls support external input management for better integration:

```typescript
flyControls.setExternalInputManagement(true);
// InputHandler now manages keyboard events
flyControls.handleKeyDown(event); // Called by InputHandler
```

#### Continuous Rendering Optimization

- **Orbit mode**: Renders only during interaction
- **Fly mode (inertial)**: Continues rendering while velocity > threshold
- **Automatic pause**: Stops rendering when stationary
- **Event-driven**: Updates triggered by control changes

### Input Conflict Resolution

The system prevents keyboard conflicts through sophisticated routing:

1. **Context Stack**: Manages nested input contexts
2. **Priority System**: Higher priority contexts override lower ones
3. **Typing Detection**: Automatically disables shortcuts when typing
4. **Mode-Specific Keys**: WASD disabled in orbit mode, enabled in fly

Example context switching:

```typescript
// When entering a text field
inputContext.pushContext(InputContext.TYPING);

// WASD keys now type text instead of moving camera

// When leaving the text field
inputContext.popContext();
```

---

## Usage Examples

### Basic Setup

```typescript
import { ControlsManager } from './controls/controls-manager';

const controlsManager = new ControlsManager(camera, canvas);

// In render loop
function animate() {
  controlsManager.update();
  renderer.render(scene, camera);
}
```

### Switching Control Modes

```typescript
// Switch to fly mode
controlsManager.setControlType('fly');

// Configure fly controls
controlsManager.setFlyMovementSpeed(10);
controlsManager.setFlyInertialMode(true);
controlsManager.setFlyDamping(0.99);

// Switch back to orbit
controlsManager.setControlType('orbit');
```

### Programmatic Camera Control

```typescript
// Look at a specific point
const target = new THREE.Vector3(10, 5, 0);
controlsManager.lookAt(target, true); // Smooth transition

// Reset to default view
controlsManager.reset();

// Save and restore view state
controlsManager.saveState();
// ... user navigates ...
controlsManager.reset(); // Return to saved state
```

### Custom Configuration

```typescript
// Override defaults during initialization
const controls = new LuxarFlyControls(camera, canvas, {
  inertialMode: true,
  movementSpeed: 10,
  rotationSpeed: 2.0,
  damping: 0.95,
  rotationDamping: 0.9,
  acceleration: 1.0,
});
```

---

## Common Patterns and Best Practices

### 1. Mode-Specific UI

Show/hide UI elements based on active control type:

```typescript
controlsManager.addEventListener('change', (event) => {
  if (event.controlType) {
    updateUI(event.controlType);
  }
});
```

### 2. Context-Aware Input

Prevent control conflicts with UI elements:

```typescript
// When opening a modal
inputContext.pushContext(InputContext.UI_OVERLAY);

// When closing
inputContext.popContext();
```

### 3. Performance Optimization

```typescript
// Only update when needed
let needsUpdate = false;

controlsManager.addEventListener('change', () => {
  needsUpdate = true;
});

function animate() {
  if (needsUpdate) {
    controlsManager.update();
    renderer.render(scene, camera);
    needsUpdate = false;
  }
}
```

### 4. Smooth Transitions

```typescript
// Animate to target over multiple frames
function smoothLookAt(target: Vector3, duration: number) {
  const start = Date.now();

  function update() {
    const progress = (Date.now() - start) / duration;
    if (progress < 1) {
      controlsManager.lookAt(target, true);
      requestAnimationFrame(update);
    }
  }
  update();
}
```

---

## Troubleshooting

### Common Issues

**Problem: Controls feel sluggish**

- Solution: Decrease damping values (try 0.95 for movement, 0.9 for rotation)

**Problem: Controls too sensitive**

- Solution: Reduce speed values or increase damping

**Problem: Yaw becomes roll at extreme pitch**

- Solution: Ensure using premultiply() for quaternion integration

**Problem: Keys not working in fly mode**

- Solution: Check that keys are in `flyModeKeys` array in control-config.ts

**Problem: Inertial mode not persisting**

- Solution: Settings are stored in ControlsManager config, ensure using getFlyConfig()

### Debug Mode

Enable debug logging:

```typescript
// In browser console
window.__luxarDebug = { controls: true };
```

This will log all control state changes and input events.

---

## Extension Guide

### Adding a New Control Type

1. Create control class implementing base interface:

```typescript
class CustomControls {
  enabled: boolean;
  update(delta?: number): void;
  dispose(): void;
  addEventListener(type: string, listener: Function): void;
  removeEventListener(type: string, listener: Function): void;
}
```

2. Add to ControlsManager:

```typescript
// In ControlsManager.setControlType()
case 'custom':
  this.createCustomControls();
  break;
```

3. Update configuration system:

```typescript
// In control-config.ts
custom: {
  // Add configuration parameters
}
```

### Adding New Input Modes

1. Define new keys in control-config.ts:

```typescript
customModeKeys: ['x', 'y', 'z'],
```

2. Add input context:

```typescript
enum InputContext {
  // ...
  CUSTOM_MODE,
}
```

3. Handle in input system:

```typescript
if (customMode && customModeKeys.includes(event.key)) {
  // Handle custom input
}
```

---

## API Reference

### ControlsManager

| Method                         | Description                      |
| ------------------------------ | -------------------------------- |
| `setControlType(type)`         | Switch control mode              |
| `getControlType()`             | Get current mode                 |
| `getControls()`                | Get active control instance      |
| `setAutoRotate(enabled)`       | Toggle auto-rotation             |
| `setFlyMovementSpeed(speed)`   | Set fly movement speed           |
| `setFlyInertialMode(inertial)` | Toggle inertial physics          |
| `setFlyDamping(damping)`       | Set movement damping             |
| `lookAt(target, smooth)`       | Point camera at target           |
| `reset()`                      | Reset to default state           |
| `update()`                     | Update controls (call per frame) |
| `dispose()`                    | Clean up resources               |

### LuxarFlyControls

| Method                             | Description               |
| ---------------------------------- | ------------------------- |
| `update(delta)`                    | Update physics simulation |
| `setInertialMode(inertial)`        | Toggle momentum mode      |
| `lookAtSmooth(target, smoothness)` | Smooth look-at            |
| `reset()`                          | Zero velocities           |
| `handleKeyDown(event)`             | Process key press         |
| `handleKeyUp(event)`               | Process key release       |
| `dispose()`                        | Remove event listeners    |

### Events

All controls dispatch these events:

| Event    | Description              |
| -------- | ------------------------ |
| `start`  | User started interacting |
| `change` | Control state changed    |
| `end`    | User stopped interacting |

---

## Performance Considerations

### Optimization Tips

1. **Use requestAnimationFrame**: Only render when controls change
2. **Batch updates**: Update all controls in single pass
3. **Limit update rate**: Cap delta time to prevent instability
4. **Dead zones**: Zero tiny velocities to prevent micro-updates
5. **LOD switching**: Reduce scene complexity during movement

### Benchmarks

Typical performance on modern hardware:

- Control update: < 0.1ms
- Event processing: < 0.05ms per event
- Mode switching: < 1ms
- Memory overhead: ~10KB per control instance

---

## Testing Strategy

### Test Coverage

The control system has comprehensive test coverage:

- **ControlsManager**: 30 tests covering mode switching, configuration, events
- **LuxarFlyControls**: 27 tests for movement, physics, input handling
- **InputContextManager**: 29 tests for context switching, key filtering

### Test Categories

1. **Initialization**: Default states and configurations
2. **Mode Switching**: State preservation and transitions
3. **Input Handling**: Keyboard and mouse events
4. **Physics Simulation**: Inertial movement and damping
5. **Context Management**: Key filtering and priority
6. **Event Propagation**: Control events and state changes

### Running Tests

```bash
# Run all TypeScript tests
pnpm test

# Run with coverage
pnpm test:coverage

# Watch mode for development
pnpm test:watch
```

---

## Common Patterns

### Frame-Rate Independent Physics

All physics calculations are normalized to 60fps:

```typescript
// Damping that works at any frame rate
velocity *= Math.pow(damping, delta * 60);

// Capped delta to prevent instability
const safeDelta = Math.min(delta, 1 / 30);
```

### Context-Aware Input Handling

```typescript
// Check if user is typing before handling shortcuts
if (this.isTypingInInput()) {
  return; // Don't handle shortcuts while typing
}

// Check active control mode
if (controlType === 'fly' && flyModeKeys.includes(key)) {
  flyControls.handleKeyDown(event);
}
```

### Smooth State Transitions

```typescript
// Smooth camera transitions in fly mode
lookAtSmooth(target: Vector3, smoothness: number) {
  const targetQuat = calculateTargetQuaternion(target);
  this.orientation.slerp(targetQuat, 1 - smoothness);
}
```

---

## Contributing

When contributing to the controls system:

1. **Maintain backwards compatibility** where possible
2. **Add tests** for new functionality
3. **Update this documentation** for API changes
4. **Follow existing patterns** for consistency
5. **Consider performance** - controls run every frame
6. **Test across different frame rates**
7. **Ensure input context compatibility**

### Code Style

- Use TypeScript strict mode
- Prefer composition over inheritance
- Document physics/math with comments
- Add JSDoc for public APIs
- Keep frame-rate independence in mind
- Use type guards instead of runtime checks
- Clean up event listeners properly

---

## Future Enhancements

### Planned Features

- **Touch/Mobile Support**: Touch gestures for fly controls
- **Gamepad Support**: Xbox/PlayStation controller input
- **Configurable Key Bindings**: User-customizable controls
- **Motion Paths**: Predefined camera animations
- **VR/AR Modes**: Immersive navigation
- **Multi-User Sync**: Shared camera state

### Architecture Improvements

- **Plugin System**: Extensible control modules
- **Record/Replay**: Camera path recording
- **Gesture Recognition**: Complex input patterns
- **Accessibility**: Keyboard-only navigation
- **Performance Profiling**: Built-in metrics

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

## Changelog

### v2.0.0 (2025-01)

- Added quaternion-based fly controls
- Implemented roll controls (Q/E)
- Added speed boost (Shift)
- Fixed frame of reference issues
- Centralized configuration system
- Added comprehensive test coverage
- Improved documentation

### v1.0.0 (2024-12)

- Initial control system
- Orbit controls integration
- Basic input management

---

_For detailed implementation notes on the fly controls mathematics and physics, see the [Fly Controls Implementation Guide](./luxar-fly-controls-guide.md)._
