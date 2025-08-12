# Control System Architecture

## Overview

The Luxar Player control system provides a flexible, extensible framework for 3D navigation with support for multiple control modes, input conflict resolution, and smooth transitions between different navigation paradigms.

## Architecture Components

### 1. Controls Manager (`controls-manager.ts`)

The `ControlsManager` class serves as the central coordinator for all camera control functionality:

- **Control Mode Switching**: Seamlessly transitions between Orbit and Fly control modes
- **State Preservation**: Maintains camera position and orientation during mode switches
- **Configuration Management**: Stores and applies control-specific settings
- **Event Forwarding**: Propagates control events to the scene manager

```typescript
// Example usage
const controlsManager = new ControlsManager(camera, domElement, scene);
controlsManager.setControlType('fly');
controlsManager.setFlyInertialMode(true);
```

### 2. Control Implementations

#### Orbit Controls (Three.js Built-in)
- Traditional 3D viewer controls
- Rotates around a target point
- Mouse-based interaction with damping
- Auto-rotation support

#### Luxar Fly Controls (`luxar-fly-controls.ts`)
Custom implementation providing first-person navigation:

- **Movement Modes**:
  - Direct: Immediate velocity control (stops when key released)
  - Inertial: Physics-based with momentum and damping

- **Input Handling**:
  - WASD keys for movement (forward/back/strafe)
  - Alt+W/S for vertical movement
  - Arrow keys for camera look
  - Mouse drag for free look

- **Physics Simulation**:
  ```typescript
  // Inertial mode physics
  velocity += acceleration * delta;
  velocity *= damping^(delta * 60);  // Frame-rate independent damping
  position += velocity * delta;
  ```

### 3. Input Context Management (`input-context-manager.ts`)

Sophisticated input routing system that prevents keyboard conflicts:

- **Context Stack**: Manages nested input contexts
- **Priority System**: Higher priority contexts override lower ones
- **Key Filtering**: Allows/blocks keys based on context
- **Typing Detection**: Automatically disables shortcuts when typing

```typescript
enum InputContext {
  NAVIGATION = 'navigation',      // Default 3D navigation
  FLY_CONTROLS = 'fly_controls',  // WASD movement active
  TYPING = 'typing',               // Text input active
  UI_INTERACTION = 'ui_interaction', // UI panels open
  DIMENSION_NAV = 'dimension_nav'  // nD navigation active
}
```

### 4. Configuration System (`control-config.ts`)

Centralized configuration with TypeScript type safety:

```typescript
export const CONTROL_CONFIG = {
  fly: {
    movement: {
      speed: { min: 0.5, max: 50.0, default: 5.0 },
      acceleration: { min: 0.1, max: 2.0, default: 0.5 },
      damping: { min: 0.9, max: 0.9999, default: 0.999 }
    },
    physics: {
      velocityThreshold: 1e-5,  // Stop threshold
      dampingPower: 60          // Frame-rate normalization
    }
  }
};
```

## Data Flow

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

## Key Features

### 1. Seamless Mode Switching
- Camera state preserved during transitions
- Orbit target converted to fly look direction
- Fly orientation converted to orbit target

### 2. Inertial Physics
- Realistic momentum-based movement
- Frame-rate independent damping
- Configurable acceleration and deceleration
- Automatic stop detection (velocity < 1e-5)

### 3. Input Conflict Resolution
- WASD keys disabled in orbit mode
- Shortcuts disabled while typing
- Context-aware key routing
- Priority-based override system

### 4. External Input Management
Fly controls support external input management for better integration:

```typescript
flyControls.setExternalInputManagement(true);
// InputHandler now manages keyboard events
flyControls.handleKeyDown(event);  // Called by InputHandler
```

### 5. Continuous Rendering Optimization
- Orbit mode: Renders only during interaction
- Fly mode (inertial): Continues rendering while velocity > threshold
- Automatic pause when stationary

## Testing Strategy

### Unit Tests Coverage
- **ControlsManager**: 30 tests covering mode switching, configuration, events
- **LuxarFlyControls**: 27 tests for movement, physics, input handling
- **InputContextManager**: 29 tests for context switching, key filtering

### Test Categories
1. **Initialization**: Default states and configurations
2. **Mode Switching**: State preservation and transitions
3. **Input Handling**: Keyboard and mouse events
4. **Physics Simulation**: Inertial movement and damping
5. **Context Management**: Key filtering and priority

## Performance Considerations

### Rendering Efficiency
- Animation loop only active during movement
- Velocity threshold prevents micro-movements
- Event-driven architecture minimizes CPU usage

### Memory Management
- Single control instance active at a time
- Proper cleanup of event listeners on dispose
- Cached configuration values

## Extension Points

### Adding New Control Modes
1. Implement control class extending `THREE.EventDispatcher`
2. Add type to `ControlType` union
3. Update `ControlsManager.setControlType()`
4. Add configuration to `CONTROL_CONFIG`

### Custom Input Contexts
1. Add context to `InputContext` enum
2. Define context configuration in `InputContextManager`
3. Register key bindings for context
4. Set appropriate priority level

## Best Practices

### Type Safety
- Use type guards instead of runtime checks
- Define comprehensive interfaces for configurations
- Avoid `any` types - use proper type assertions

### Event Management
- Always clean up event listeners
- Use bound handlers for proper `this` context
- Forward events through proper channels

### State Management
- Centralize state in appropriate managers
- Use event-driven updates for reactive UI
- Maintain single source of truth

## Common Patterns

### Smooth Transitions
```typescript
// Fly controls smooth look-at
lookAtSmooth(target: Vector3, smoothness: number) {
  const targetAngles = calculateAngles(target);
  this.angles = lerp(this.angles, targetAngles, 1 - smoothness);
}
```

### Frame-Rate Independence
```typescript
// Damping normalized to 60fps
velocity *= Math.pow(damping, delta * 60);
```

### Context-Aware Input
```typescript
if (this.isTypingInInput()) {
  return;  // Don't handle shortcuts while typing
}
```

## Troubleshooting

### Common Issues

1. **WASD keys not working in fly mode**
   - Check if InputContextManager is in FLY_CONTROLS context
   - Verify fly controls are enabled
   - Ensure not typing in input field

2. **Inertial movement not stopping**
   - Check velocity threshold configuration
   - Verify damping value is < 1.0
   - Ensure update() is called with proper delta time

3. **UI not syncing with keyboard shortcuts**
   - Call `renderingControls.syncCurrentState()` after changes
   - Ensure controllers have updateDisplay() called

## Future Enhancements

### Planned Features
- Touch/mobile support for fly controls
- Gamepad/controller input
- Configurable key bindings
- Motion paths and animations
- VR/AR control modes

### Architecture Improvements
- Plugin system for custom controls
- Record/replay functionality
- Multi-user synchronization
- Gesture recognition