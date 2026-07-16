# Luxar Controls Package

> Camera navigation and interaction system for the Luxar Viewer

## Overview

The Luxar Controls package provides a comprehensive control system for navigating nD scientific visualizations. It implements three control modes with seamless switching, unified configuration, and input context management.

### Key Features

- **Unified Orbit Controls**: Quaternion-based rotation with no gimbal lock (replaces both OrbitControls and ArcballControls)
- **Orthographic Mode**: 2D pan+zoom with optional view-axis roll
- **Fly Controls**: Free-flight 6DOF navigation with inertial physics
- **Seamless Mode Switching**: Hot-swap between control types with camera state preservation
- **Input Context Management**: Intelligent key routing to prevent conflicts

### Package Architecture

```
controls/
├── controls-manager.ts                 # Central control system orchestrator
├── controls-manager/                   # Orchestrator helpers
│   ├── factories.ts                   #   createOrbit/Fly/Ortho + naturalDragButtonMap
│   ├── camera-state.ts                #   save/restore camera state across mode switches
│   ├── event-forwarders.ts            #   wire change/start/end → manager
│   └── scene-scale.ts                 #   diagonal → distance/zoom/flySpeed math
│
├── luxar-orbit-controls.ts             # Quaternion orbit + ortho controls
├── luxar-orbit-controls/               # Per-class helpers
│   ├── camera-application.ts          #   applyToCamera, initializeFromCamera
│   ├── update.ts                      #   per-frame update sequencer
│   ├── math/                          #   pure-math cluster
│   │   ├── trackball.ts
│   │   ├── pan.ts
│   │   └── zoom.ts
│   └── input/                         #   DOM-event handler cluster
│       ├── pointer.ts
│       ├── touch.ts
│       └── keyboard.ts
│
├── luxar-fly-controls.ts               # Free-flight 6DOF controller
├── luxar-fly-controls/                 # Per-class helpers (parallel to orbit)
│   ├── camera-application.ts          #   initializeFromCamera, updateOrientation, lookAtSmooth
│   ├── physics.ts                     #   integrateTranslation, integrateRotation
│   ├── listeners.ts                   #   attachListeners → disposer
│   └── input/                         #   DOM-event handler cluster
│       ├── keyboard.ts
│       ├── mouse.ts
│       └── wheel.ts
│
├── types.ts                            # TypeScript type definitions
└── README.md                           # This documentation
```

The orchestrator files at the package root are the public API (imported
from `scene/`, `ui/`, etc.). Each orchestrator delegates body work to
helpers in its sibling `<orchestrator-name>/` subfolder, with thematic
subgroups for `math/` (pure functions) and `input/` (DOM-event handler
bodies). Event dispatch sites stay on the orchestrator so the listener
contract (`change` / `start` / `end`) is unchanged from outside the
package.

### Subpackages

- [`controls-manager/`](./controls-manager/README.md) — focused helpers
  split out from `ControlsManager`: per-mode factory functions, camera
  state save/restore across mode swaps, event-forwarder plumbing, and
  the scale-derived limit math. Stateless over `ctx` bundles.
- [`luxar-orbit-controls/`](./luxar-orbit-controls/README.md) — pure
  helpers extracted from `LuxarOrbitControls`: the per-frame update
  sequencer (`runUpdateStep`) and the camera-write step
  (`applyToCamera` / `initializeFromCamera`), delegating to thematic
  `math/` (trackball / pan / zoom) and `input/` (pointer / touch /
  keyboard) subgroups.
- [`luxar-fly-controls/`](./luxar-fly-controls/README.md) — per-class
  helpers for `LuxarFlyControls`: camera ↔ orientation sync, physics
  integration (`integrateTranslation` / `integrateRotation`), DOM
  listener wiring, plus the `input/` subfolder for keyboard / mouse /
  wheel event-handler bodies.

---

## Control Types

### 1. Orbit Controls (Default)

`LuxarOrbitControls` provides quaternion-based orbit navigation with no gimbal lock. It combines the best of THREE.js OrbitControls (smooth damping, proven pan/zoom math) and ArcballControls (quaternion rotation via virtual trackball).

**Controls:**

| Input             | Action                         |
| ----------------- | ------------------------------ |
| Left drag         | Pan camera                     |
| Right drag        | Rotate view (trackball)        |
| Shift + left drag | Rotate view (alternative)      |
| Scroll            | Zoom in/out                    |
| Shift + scroll    | Roll (rotate around view axis) |
| Ctrl/Cmd + scroll | Adjust field of view           |
| Arrow keys        | Pan                            |

The table shows the default CAD/Blender mapping (left-drag pans, right-drag
rotates). "Natural drag" mode (default on macOS, see `setNaturalDrag`) swaps
LEFT ↔ RIGHT so one-finger drag rotates and right-drag pans — touchpad
ergonomics. In both mappings `Shift+left` performs the opposite action.

**Features:**

- Quaternion-based rotation (no gimbal lock at any angle)
- Exponential damping for smooth interaction
- Auto-rotation around the screen-vertical axis
- Configurable mouse button mapping (CAD/Blender vs. natural-drag)
- Touch support (1-finger rotate, 2-finger pinch-zoom + pan)

### 2. Fly Controls

`LuxarFlyControls` provides quaternion-based free-flight navigation with 6 degrees of freedom.

**Controls:**

| Input          | Action                          |
| -------------- | ------------------------------- |
| Left drag      | Strafe (screen-space translate) |
| Right drag     | Rotate (free look)              |
| Scroll         | Move forward/backward           |
| Shift + scroll | Roll (rotate around view axis)  |
| WASD           | Move forward/back/left/right    |
| Alt+W/S        | Move up/down                    |
| Q/E            | Roll left/right                 |
| Shift          | 2x speed boost                  |
| Arrow keys     | Look up/down/left/right         |
| I              | Toggle inertial mode            |

### 3. Ortho Controls

Orthographic projection with 2D pan+zoom. Uses `LuxarOrbitControls` with rotation disabled and an `OrthographicCamera`.

**Controls:**

| Input          | Action                         |
| -------------- | ------------------------------ |
| Left drag      | Pan camera                     |
| Scroll         | Zoom in/out                    |
| Shift + scroll | Roll (rotate around view axis) |

When switching to ortho, the camera resets to a clean front view (looking along -Z, up = Y).

---

## Mode Switching

Press **V** to cycle: **Orbit → Fly → Ortho → Orbit**

Camera state (position, target) is preserved across switches. The ortho mode resets to a front view for clean 2D viewing.

---

## Control System Architecture

### ControlsManager

The `ControlsManager` class orchestrates the control system:

```typescript
class ControlsManager extends THREE.EventDispatcher {
  // Control type management
  setControlType(type: 'orbit' | 'fly' | 'ortho'): void;
  getControlType(): ControlType;
  getControls(): LuxarOrbitControls | LuxarFlyControls | null;
  setCamera(camera: LuxarCamera): void;
  setEnabled(enabled: boolean): void;

  // Orbit configuration
  setAutoRotate(enabled: boolean): void;
  setAutoRotateSpeed(speed: number): void;
  getAutoRotate(): boolean;
  setEnableZoom(enabled: boolean): void;
  setNaturalDrag(enabled: boolean): void; // swap LEFT↔RIGHT (orbit only)
  getNaturalDrag(): boolean;

  // Fly configuration
  setFlyMovementSpeed(speed: number): void;
  setFlyRotationSpeed(speed: number): void;
  setFlyInertialMode(inertial: boolean): void;
  setFlyDamping(damping: number): void;
  setFlyRotationDamping(damping: number): void;
  getFlyControls(): LuxarFlyControls | null;
  getFlyConfig(): { inertialMode; damping; rotationDamping; movementSpeed; rotationSpeed };

  // Camera control
  lookAt(target: Vector3, smooth?: boolean): void;
  setTarget(target: Vector3): void;
  getFocusTarget(): Vector3;
  reinitialize(): void;
  reset(): void;
  saveState(): void;

  // Scale-aware parameters
  setSceneScale(diagonal: number): void;
  getSceneScale(): number;
  setDistanceLimits(min: number, max: number): void;
  setZoomLimits(min: number, max: number): void;

  // Update loop (call every frame)
  update(): void;
  dispose(): void;
}
```

### LuxarOrbitControls

```typescript
class LuxarOrbitControls extends EventDispatcher {
  target: Vector3; // Orbit center
  enabled: boolean;
  enableDamping: boolean;
  dampingFactor: number; // 0.25 default (exponential decay)
  rotateSpeed: number; // 3.0 default
  panSpeed: number; // 1.0 default
  zoomSpeed: number; // 1.0 default
  enableRotate: boolean;
  enablePan: boolean;
  enableZoom: boolean;
  autoRotate: boolean;
  autoRotateSpeed: number; // 0.25 = 4 min/rotation
  mouseButtons: { LEFT; MIDDLE; RIGHT };

  update(deltaTime?: number): boolean;
  reinitialize(): void;
  enableViewAxisRotation(speed?: number): void;
  listenToKeyEvents(element: HTMLElement | Window): void;
  saveState(): void;
  reset(): void;
  dispose(): void;
}
```

### Type Definitions (`types.ts`)

Key types and type guards exported from this package:

- `ControlType` — `'orbit' | 'fly' | 'ortho'`
- `ControlInstance` — Union of `LuxarOrbitControls | LuxarFlyControls`
- `isOrbitControls(control)` — Type guard for orbit controls
- `isFlyControls(control)` — Type guard for fly controls

(GUI-controller reference types such as `RenderingControllers` live alongside
their consumers in `ui/rendering-controls/types.ts`, not here.)

### Events

All controls dispatch:

| Event    | Description              |
| -------- | ------------------------ |
| `start`  | User started interacting |
| `change` | Camera state changed     |
| `end`    | User stopped interacting |

---

## Fly Controls Deep Dive

### Mathematical Foundation

The fly controls use quaternion-based physics for smooth, gimbal-lock-free navigation.

**Translation:**

```
v <- v + a * dt              // Apply acceleration
v <- v * damping^(dt*60)     // Frame-rate independent damping
p <- p + v * dt              // Update position
```

**Rotation:**

```
w <- w + torque * dt         // Apply torque (inertial mode)
w <- w * damping^(dt*60)     // Angular damping
dq = exp(w * dt / 2)        // Exponential map
q <- dq * q                 // PRE-multiply (world-space w)
```

### Inertial vs Non-Inertial Modes

- **Inertial** (default): Low damping, momentum-based. Smooth cinematic feel.
- **Non-Inertial**: High damping, direct velocity control. Immediate response.

---

## Configuration

All control parameters are centralized in `config/index.ts`:

```typescript
controls: {
  fly: {
    inertialMode: { default: true },
    movement: { speed, damping, acceleration },
    rotation: { speed, damping },
  },
  orbit: {
    autoRotate: { speed: { default: 0.25 } },
    zoom: { minDistance: 0.1, maxDistance: 1000 },
  },
  scaleMultipliers: {
    minDistanceFactor: 0.001,
    maxDistanceFactor: 10000,
    flySpeedFactor: 0.05,
  },
}
```

---

## Testing

```bash
pnpm test --run src/tests/unit/controls/
```

Test files:

- `controls-manager.test.ts` — Mode switching, configuration, events
- `luxar-orbit-controls.test.ts` — Rotation, pan, zoom, damping, trackball math
- `luxar-fly-controls.test.ts` — Movement, physics, input handling

---

## Dependencies

- Internal: `config`, `ui/gui`, `utils/log`.
- External: `three`.
