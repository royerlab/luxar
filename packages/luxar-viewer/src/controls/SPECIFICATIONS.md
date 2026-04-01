# luxar-viewer.controls - Technical Specification

**Version**: 2.1.0
**Last Updated**: 2026-01-16

## Purpose

The `luxar-viewer.controls` package provides comprehensive 3D navigation systems for nD scientific visualization, including quaternion-based orbit controls, fly controls with 6 degrees of freedom, and orthographic pan+zoom controls.

**Core Responsibility**: Enable intuitive camera navigation through multiple control paradigms (orbit, fly, ortho) with seamless mode switching, physics-based movement, and gimbal-lock-free rotation.

---

## Table of Contents

1. [Control Types](#control-types)
2. [Fly Controls Physics](#fly-controls-physics)
3. [Quaternion Rotation](#quaternion-rotation)
4. [Mode Switching](#mode-switching)
5. [Input Management](#input-management)

---

## 1. Control Types

### 1.1 Orbit Controls

**Type**: Quaternion-based rotation around a target point (no gimbal lock)

**Features**:

- Rotate: Left mouse drag
- Pan: Right mouse drag
- Zoom: Scroll wheel
- Roll: Shift+scroll

**Properties**:

- No gimbal lock (quaternion-based turntable)
- Smooth damping
- Camera always faces target point
- Auto-rotate support

**Use Cases**: Examining objects from outside, presentations, scientific visualization requiring arbitrary orientations

### 1.2 Ortho Controls

**Type**: Orthographic pan + zoom (2D viewing mode)

**Features**:

- Pan: Left mouse drag (Napari/Google Maps convention)
- Zoom: Scroll wheel (mouse middle button)
- Roll: Shift+scroll
- Rotation disabled

**Use Cases**: 2D viewing of slices, plan-view inspection

### 1.3 Fly Controls

**Type**: Free-flight 6DOF navigation with physics simulation

**Mouse mapping** (consistent with orbit/ortho):

- Strafe: Left mouse drag (screen-space translation)
- Rotate: Right mouse drag (free look)
- Forward/back: Scroll wheel (velocity impulse)
- Roll: Shift+scroll (view-axis rotation)

**Keyboard**:

- WASD movement in camera-local frame
- Q/E for roll
- Arrow keys for rotation
- Shift for speed boost

**Physics**: Inertial mode (momentum + damping) or non-inertial mode (direct response)

**Use Cases**: Exploring data from inside, cinematic camera paths

---

## 2. Fly Controls Physics

### 2.1 Coordinate Frames

**World Frame** `W`:

- Global coordinate system
- Y-up convention
- Basis vectors: `e_x = (1,0,0)`, `e_y = (0,1,0)`, `e_z = (0,0,1)`

**Camera Frame** `B` (body frame):

- Local to camera
- Basis vectors in THREE.js convention:
  - `e_x = (1,0,0)` → right
  - `e_y = (0,1,0)` → up
  - `e_z = (0,0,-1)` → forward (negative Z in THREE.js)

### 2.2 Translational Physics

**State Variables**:

- `position`: vec3 - Camera position in world frame
- `velocity`: vec3 - Linear velocity in world frame (m/s)

**Update Equations**:

```
// Acceleration phase (from keyboard input)
a = Σ (direction_i × speed)     // direction in camera frame

// Convert to world frame
a_world = R(q) × a              // q = camera orientation quaternion

// Integrate velocity
v ← v + a_world × Δt

// Apply damping (frame-rate independent)
damping_factor = damping^(Δt × 60)
v ← v × damping_factor

// Integrate position
p ← p + v × Δt

// Stop if velocity below threshold
if |v| < ε:
    v ← 0
```

**Parameters**:

- `speed`: Base movement speed (default 5.0)
- `damping`: Velocity decay (default 0.999 for inertial, 0.5 for direct)
- `acceleration`: Acceleration multiplier (default 0.5)
- `ε`: Velocity threshold for stopping (default 1e-4)

### 2.3 Rotational Physics

**State Variables**:

- `orientation`: Quaternion - Camera orientation in world frame
- `angularVelocity`: vec3 - Angular velocity in **world frame** (rad/s)

**Critical Design Decision**: Angular velocity maintained in **world space** (not body frame) to avoid gimbal lock.

**Update Equations**:

```
// 1. Build torque in camera frame from mouse/keyboard
τ_camera = pitch × camera_right + yaw × camera_up + roll × camera_forward

// 2. Convert torque to world frame
camera_right = R(q) × (1,0,0)
camera_up = R(q) × (0,1,0)
camera_forward = R(q) × (0,0,-1)

τ_world = pitch × camera_right + yaw × camera_up + roll × camera_forward

// 3. Integrate angular velocity (in world frame)
ω ← ω + τ_world × Δt

// 4. Apply angular damping
ω ← ω × angularDamping^(Δt × 60)

// 5. Convert angular velocity to quaternion increment
//    Using exponential map: Δq = exp(ω × Δt / 2)

angle = |ω| × Δt
if angle > 0:
    axis = ω / |ω|
    Δq = Quaternion.fromAxisAngle(axis, angle)

    // 6. Integrate orientation (PRE-multiply for world-space ω)
    q ← Δq ⊗ q

// 7. Stop if angular velocity below threshold
if |ω| < ε_angular:
    ω ← 0
```

**Why Pre-Multiply?**

For world-space angular velocity `ω`, the rotation must be applied **before** the current orientation:

```
q_new = exp(ω Δt / 2) ⊗ q_old
      = Δq ⊗ q
```

This prevents the "yaw becomes roll at 90° pitch" problem in post-multiply implementations.

**Parameters**:

- `rotationSpeed`: Base rotation speed (default 1.5)
- `rotationDamping`: Angular decay (default 0.99)
- `ε_angular`: Angular velocity threshold (default 1e-4 rad/s)

### 2.4 Frame-Rate Independence

**Problem**: Physics must behave identically at 30 FPS, 60 FPS, or 144 FPS.

**Solution**: Normalize all updates to 60 FPS equivalent.

**Damping Formula**:

```typescript
// Correct: Frame-rate independent damping
const effectiveDamping = Math.pow(baseDamping, delta * 60);
velocity *= effectiveDamping;

// Incorrect: Frame-rate dependent
// velocity *= baseDamping  // Only correct at 60 FPS
```

**Derivation**:

At 60 FPS, `Δt = 1/60`, so we apply damping once per frame:

```
v_new = v_old × damping
```

At 30 FPS, `Δt = 1/30 = 2 × (1/60)`, so we should apply damping **twice**:

```
v_new = v_old × damping²
```

Generalized: `v_new = v_old × damping^(Δt × P)`

Where `P` is the damping power (configurable via `config.controls.fly.physics.dampingPower`, typically 60)

---

## 3. Quaternion Rotation

### 3.1 Quaternion Representation

**Format**: `q = (x, y, z, w)` where:

- `(x, y, z)` = vector part (imaginary)
- `w` = scalar part (real)

**Unit Quaternion** (rotation):

```
|q| = 1  ⟹  x² + y² + z² + w² = 1
```

### 3.2 Axis-Angle to Quaternion

**Input**: Rotation axis `v = (v_x, v_y, v_z)` with `|v| = 1`, angle `θ` (radians)

**Output**: Quaternion `q`

**Formula**:

```
q = (sin(θ/2) × v_x, sin(θ/2) × v_y, sin(θ/2) × v_z, cos(θ/2))
```

**Implementation**:

```typescript
function quaternionFromAxisAngle(axis: vec3, angle: number): Quaternion {
  const halfAngle = angle / 2;
  const s = Math.sin(halfAngle);
  const c = Math.cos(halfAngle);

  return new Quaternion(axis.x * s, axis.y * s, axis.z * s, c);
}
```

### 3.3 Quaternion Multiplication

**Pre-multiplication**: `q_result = q_a ⊗ q_b`

**Formula** (Hamilton product):

```
x = a.w × b.x + a.x × b.w + a.y × b.z - a.z × b.y
y = a.w × b.y - a.x × b.z + a.y × b.w + a.z × b.x
z = a.w × b.z + a.x × b.y - a.y × b.x + a.z × b.w
w = a.w × b.w - a.x × b.x - a.y × b.y - a.z × b.z
```

**Order Matters**: `q_a ⊗ q_b ≠ q_b ⊗ q_a` (non-commutative)

**Usage**:

```typescript
// Rotate camera by delta rotation
orientation.premultiply(deltaRotation); // World-space rotation
```

### 3.4 Vector Rotation

**Purpose**: Rotate a vector by a quaternion.

**Formula**: `v' = q ⊗ v ⊗ q*`

Where:

- `v` = pure quaternion `(v_x, v_y, v_z, 0)`
- `q*` = conjugate of `q` = `(-x, -y, -z, w)`

**Implementation** (THREE.js):

```typescript
const rotated = vector.applyQuaternion(quaternion);
```

---

## 4. Mode Switching

### 4.1 Orbit ↔ Fly Transition

**Challenge**: Preserve camera position and view direction during mode switch.

**Orbit → Fly**:

```typescript
function orbitToFly() {
  // Save camera state
  const position = camera.position.clone();
  const target = orbitControls.target.clone();

  // Dispose orbit controls
  orbitControls.dispose();

  // Create fly controls
  flyControls = new FlyControls(camera, canvas);

  // Restore position
  camera.position.copy(position);

  // Set orientation to look at previous target
  const direction = target.sub(position).normalize();
  const orientation = quaternionFromDirection(direction);
  flyControls.setOrientation(orientation);
}
```

**Fly → Orbit**:

```typescript
function flyToOrbit() {
  // Save camera state
  const position = camera.position.clone();
  const direction = new Vector3(0, 0, -1).applyQuaternion(flyControls.orientation);

  // Dispose fly controls
  flyControls.dispose();

  // Create orbit controls
  orbitControls = new OrbitControls(camera, canvas);

  // Restore position
  camera.position.copy(position);

  // Set target along view direction
  const targetDistance = 10.0; // Default target distance
  orbitControls.target.copy(position).addScaledVector(direction, targetDistance);
}
```

### 4.2 State Preservation

**Invariants** during mode switch:

- Camera position unchanged
- View direction preserved
- Zoom level maintained (where applicable)

### 4.3 Ortho Mode Switching

**Orbit/Fly → Ortho**: Camera position and target are preserved. Rotation is disabled. Left-click is remapped to pan (Napari/Google Maps convention).

**Ortho → Orbit/Fly**: Camera position and target are restored. Rotation is re-enabled.

---

## 5. Input Management

### 5.1 Keyboard Bindings

**Fly Mode**:

| Key     | Action       | Formula                    |
| ------- | ------------ | -------------------------- |
| `W`     | Forward      | `v += forward × speed`     |
| `S`     | Backward     | `v -= forward × speed`     |
| `A`     | Left         | `v -= right × speed`       |
| `D`     | Right        | `v += right × speed`       |
| `Alt+W` | Up (world)   | `v += (0,1,0) × speed`     |
| `Alt+S` | Down (world) | `v -= (0,1,0) × speed`     |
| `Q`     | Roll left    | `ω += forward × rollSpeed` |
| `E`     | Roll right   | `ω -= forward × rollSpeed` |
| `Shift` | Speed boost  | `speed × 2`                |

**Direction Vectors** (camera frame):

```typescript
// Camera basis vectors
const forward = new Vector3(0, 0, -1).applyQuaternion(orientation);
const right = new Vector3(1, 0, 0).applyQuaternion(orientation);
const up = new Vector3(0, 1, 0).applyQuaternion(orientation);
```

### 5.2 Mouse Look

**Input**: Mouse delta `(dx, dy)` in pixels

**Conversion to Angular Velocity**:

```typescript
// Sensitivity factor (radians per pixel)
const sensitivity = 0.002;

// Convert pixel delta to rotation
const yaw = -dx * sensitivity; // Negative for natural direction
const pitch = -dy * sensitivity;

// Apply to angular velocity (world frame)
const cameraRight = new Vector3(1, 0, 0).applyQuaternion(orientation);
const cameraUp = new Vector3(0, 1, 0).applyQuaternion(orientation);

angularVelocity.addScaledVector(cameraRight, pitch);
angularVelocity.addScaledVector(cameraUp, yaw);
```

### 5.3 Inertial vs Non-Inertial Modes

**Inertial Mode** (default):

- Low damping (0.999 translation, 0.99 rotation)
- Momentum-based movement
- Smooth, cinematic feel
- Continues moving after input stops

**Non-Inertial Mode**:

- High effective damping (~0.5)
- Direct velocity control (no momentum)
- Immediate response
- Stops immediately when input released

**Mode Toggle**:

```typescript
function setInertialMode(enabled: boolean): void {
  if (enabled) {
    translationDamping = 0.999;
    rotationDamping = 0.99;
  } else {
    translationDamping = 0.5;
    rotationDamping = 0.5;
  }
}
```

---

## Data Structures

### ControlsManager

```typescript
interface ControlsManager {
  currentControlType: 'orbit' | 'fly' | 'ortho';
  currentControls: LuxarOrbitControls | LuxarFlyControls | null;

  setControlType(type: 'orbit' | 'fly' | 'ortho'): void;
  getControlType(): 'orbit' | 'fly' | 'ortho';
  getControls(): LuxarOrbitControls | LuxarFlyControls | null;
  update(): void;
  dispose(): void;
}
```

**Note**: Both orbit and ortho modes use `LuxarOrbitControls` internally (ortho disables rotation and remaps left-click to pan).

### FlyControls State

```typescript
interface FlyControlsState {
  // Position
  position: THREE.Vector3; // World space

  // Orientation (quaternion)
  orientation: THREE.Quaternion; // World space

  // Linear motion
  velocity: THREE.Vector3; // World space (m/s)
  acceleration: number; // Acceleration multiplier

  // Angular motion
  angularVelocity: THREE.Vector3; // World space (rad/s)

  // Input state
  keysPressed: Set<string>;
  mouseDown: boolean;
  shiftPressed: boolean; // Speed boost

  // Configuration
  movementSpeed: number; // Base speed
  rotationSpeed: number; // Base rotation speed
  damping: number; // Translation damping
  rotationDamping: number; // Angular damping
  inertialMode: boolean;
}
```

### FlyControls Update Loop

```typescript
function update(delta: number): boolean {
  // 1. Process keyboard input → acceleration
  const accel = this.processKeyboardInput();

  // 2. Update linear velocity
  this.velocity.add(accel.multiplyScalar(delta));
  this.velocity.multiplyScalar(Math.pow(this.damping, delta * 60));

  // 3. Update position
  this.position.addScaledVector(this.velocity, delta);

  // 4. Process mouse input → angular acceleration
  const torque = this.processMouseInput();

  // 5. Update angular velocity (world space)
  this.angularVelocity.add(torque.multiplyScalar(delta));
  this.angularVelocity.multiplyScalar(Math.pow(this.rotationDamping, delta * 60));

  // 6. Integrate orientation
  const angle = this.angularVelocity.length() * delta;
  if (angle > 0) {
    const axis = this.angularVelocity.clone().normalize();
    const deltaQ = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    this.orientation.premultiply(deltaQ); // World-space rotation
  }

  // 7. Update camera transform
  camera.position.copy(this.position);
  camera.quaternion.copy(this.orientation);

  // 8. Check if still moving
  const isMoving = this.velocity.length() > 1e-4 || this.angularVelocity.length() > 1e-4;

  return isMoving;
}
```

---

## Changelog

- **v3.0.0** (2026-03-31): Arcball removal, ortho mode addition
  - **REMOVED**: Arcball as a separate control type (orbit is now quaternion-based, no gimbal lock)
  - **ADDED**: Ortho mode (orthographic pan + zoom, 2D viewing)
  - **UPDATED**: Control types are now: orbit, fly, ortho
  - **UPDATED**: Mode cycle (V key): orbit -> fly -> ortho -> orbit
  - **UPDATED**: ControlsManager data structures to match actual implementation
  - **UPDATED**: Section 4.3 now documents ortho mode switching (replaces arcball section)

- **v2.1.0** (2026-01-16): Version alignment with README
  - Aligned specification version with README v2.x scheme
  - No functional changes

- **v2.0.0** (2025-01-30): Initial specification (aligns with README v2.0.0)
  - Three control types: Orbit, Fly, Ortho
  - Fly controls physics with quaternion-based rotation
  - World-space angular velocity (prevents gimbal artifacts)
  - Frame-rate independent physics with timestep correction
  - Inertial and non-inertial modes
  - Seamless mode switching with state preservation
  - Pre-multiply quaternion integration for world-space rotation
