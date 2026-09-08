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
├── types.ts                            # TypeScript type definitions + cycle helper
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
- Auto-rotation around a camera-frame axis or a fixed scene axis — see
  `AutoRotateAxis`
- Auto-dolly: a sinusoidal in-and-out motion along the view direction (the
  turntable's radial sibling), gated on `enableZoom` so it is alive in ortho
  too. Composes with the user's own zoom rather than fighting it — see
  `luxar-orbit-controls/math/auto-dolly.ts`
- Configurable mouse button mapping (CAD/Blender vs. natural-drag)
- Touch support — the same finger vocabulary the mobile UI is built on:
  1-finger drag rotates (arcball); 2-finger pinch zooms, 2-finger drag pans
  and 2-finger twist rolls around the view axis, all in one gesture. Lifting
  one finger out of a pinch re-seeds a one-finger rotate from the surviving
  finger's current position instead of ending the gesture. Touch
  `pointerdown` is `preventDefault`ed so the browser's compatibility mouse
  events never double-drive the mouse-only paths. A pen counts as a finger
  on a coarse-pointer device (iPad + Pencil, primary tip; the barrel button
  keeps the mouse mapping) and as a mouse on a fine-pointer desktop —
  `utils/input-capabilities.isTouchLikePointer`. The canvas must be
  `touch-action: none` for any of this to reach the controls
  (`core/app/interaction/canvas-gesture-ownership.ts`)

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
| 1-finger drag  | Look (touch)                    |
| 2-finger drag  | Strafe (touch)                  |
| Pinch          | Move forward/backward (touch)   |
| 2-finger twist | Roll (touch)                    |

Touch uses the orbit controls' complete finger vocabulary (one finger turns,
two fingers translate / zoom / roll) and drives the SAME physics state as the
mouse path (`velocity`, `angularVelocity`, `orientation`), so inertia and
damping behave identically — `luxar-fly-controls/input/touch.ts`. Pinch thrust
is logarithmic in the finger distance, so pinch-in exactly undoes pinch-out. A
touch `pointerdown` is `preventDefault`ed so the browser's compatibility mouse
events cannot start a phantom left-drag strafe under a tap. The canvas must be
`touch-action: none` for multi-touch gestures to reach the controls
(`core/app/interaction/canvas-gesture-ownership.ts`); mouse-typed pointer events
are ignored by the touch path and keep the legacy mouse listeners.

### 3. Ortho Controls

Orthographic projection with 2D pan+zoom. Uses `LuxarOrbitControls` with rotation disabled and an `OrthographicCamera`.

**Controls:**

| Input          | Action                         |
| -------------- | ------------------------------ |
| Left drag      | Pan camera                     |
| Scroll         | Zoom in/out                    |
| Shift + scroll | Roll (rotate around view axis) |

When switching to ortho, the camera pose is preserved verbatim (position, orientation and up); only the projection changes to orthographic, sized to match the perspective frustum at the pivot depth.

---

## Mode Switching

Press **V** to cycle: **Orbit → Fly → Ortho → Orbit**

Camera state (position, target, orientation and FOV) is preserved across switches. A full round trip with no interaction lands the camera back at its exact starting pose: the perspective↔orthographic swaps are pose-preserving inverses (ortho copies the perspective pose and stashes its FOV; the return swap restores that FOV and dollies to match apparent size at the pivot).

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
  isAutoRotateActive(): boolean;
  isGestureActive(): boolean; // active pointer gesture keeps the render loop awake
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
  zoomSpeed: number; // 1.0 default (per-scene; wheel steps are further
  // scaled by the global config.controls.wheelZoomSensitivity — Settings > Input)
  enableRotate: boolean;
  enablePan: boolean;
  enableZoom: boolean;
  autoRotate: boolean;
  autoRotateSpeed: number; // REVOLUTIONS PER MINUTE; a turn takes 60/speed s
  // (0.25 = 4 min/turn). Shown in the UI as a period.
  // camera frame ('vertical' | 'horizontal' | 'view') or a fixed scene axis
  // ('world-x' | 'world-y' | 'world-z'); also the default axis of
  // applyOrbitRotation, so recorded turntables match the preview
  autoRotateAxis: AutoRotateAxis;
  autoDolly: boolean;
  autoDollyAmplitude: number; // FRACTION of distance (0.15 = ±15%)
  autoDollyPeriod: number; // seconds per full in-and-out cycle
  mouseButtons: { LEFT; MIDDLE; RIGHT };

  update(deltaTime?: number): boolean;
  // Absolute-phase dolly for frame-indexed recording, the radial counterpart
  // of applyOrbitRotation; uses the configured amplitude so an export
  // breathes like its preview.
  applyOrbitDolly(phase: number): void;
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
- `nextControlType(current)` — Orbit → Fly → Ortho cycle helper

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
    autoDolly: { amplitudePercent: { default: 15 }, period: { default: 10 } },
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
