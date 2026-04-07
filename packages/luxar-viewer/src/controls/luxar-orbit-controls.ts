/**
 * Unified orbit controls with quaternion-based rotation (no gimbal lock)
 * and exponential damping (smooth interaction feel).
 *
 * Combines the best of THREE.js OrbitControls (damping, pan, zoom math)
 * and ArcballControls (quaternion rotation via virtual trackball).
 *
 * Mouse mapping:
 * - 3D mode: left-drag = pan, right-drag = rotate, Shift+left = rotate, scroll = zoom
 * - Ortho mode: left-drag = pan, scroll = zoom, Shift+scroll = view-axis rotate
 *
 * Supports both PerspectiveCamera and OrthographicCamera.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../scene/camera-utils';

export interface LuxarOrbitControlsConfig {
  enableDamping?: boolean;
  dampingFactor?: number;
  rotateSpeed?: number;
  panSpeed?: number;
  zoomSpeed?: number;
  enableRotate?: boolean;
  enablePan?: boolean;
  enableZoom?: boolean;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  minDistance?: number;
  maxDistance?: number;
  minZoom?: number;
  maxZoom?: number;
  screenSpacePanning?: boolean;
  trackballRadius?: number;
}

export type ControlAction = 'rotate' | 'pan' | 'zoom' | 'none';

const _IDENTITY_QUAT = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q1 = new THREE.Quaternion();

export class LuxarOrbitControls extends THREE.EventDispatcher<{
  change: {};
  start: {};
  end: {};
}> {
  // --- Public API ---
  public enabled: boolean = true;
  public target: THREE.Vector3;

  // Configuration
  public enableDamping: boolean;
  public dampingFactor: number;
  public rotateSpeed: number;
  public panSpeed: number;
  public zoomSpeed: number;
  public enableRotate: boolean;
  public enablePan: boolean;
  public enableZoom: boolean;
  public autoRotate: boolean;
  public autoRotateSpeed: number;
  public screenSpacePanning: boolean;

  // Constraints
  public minDistance: number;
  public maxDistance: number;
  public minZoom: number;
  public maxZoom: number;

  // Mouse button mapping
  public mouseButtons: {
    LEFT: THREE.MOUSE | null;
    MIDDLE: THREE.MOUSE | null;
    RIGHT: THREE.MOUSE | null;
  };

  // --- Internal state ---
  private camera: LuxarCamera;
  private domElement: HTMLElement;

  // Quaternion orbit state
  private orientation = new THREE.Quaternion();
  private distance: number = 1;

  // Damping accumulators (applied fractionally each frame, then decayed)
  private rotationDelta = new THREE.Quaternion(); // identity = no pending rotation
  private panDelta = new THREE.Vector3();
  private zoomDelta: number = 0;
  private rollDelta: number = 0; // view-axis rotation (radians, damped)

  // Pointer state
  private pointers: PointerEvent[] = [];
  private pointerPositions = new Map<number, THREE.Vector2>();
  private state: ControlAction = 'none';
  private rotateStart = new THREE.Vector2();
  private panStart = new THREE.Vector2();
  private dollyStart = new THREE.Vector2();
  private trackballRadius: number;

  // Change detection
  private lastPosition = new THREE.Vector3();
  private lastQuaternion = new THREE.Quaternion();

  // Saved state for reset()
  private target0 = new THREE.Vector3();
  private position0 = new THREE.Vector3();
  private orientation0 = new THREE.Quaternion();
  private zoom0: number = 1;

  // Bound event handlers (for cleanup)
  private boundOnPointerDown: (e: PointerEvent) => void;
  private boundOnPointerMove: (e: PointerEvent) => void;
  private boundOnPointerUp: (e: PointerEvent) => void;
  private boundOnWheel: (e: WheelEvent) => void;
  private boundOnContextMenu: (e: Event) => void;

  // Ortho view-axis rotation
  private viewAxisRotationHandler: ((e: WheelEvent) => void) | null = null;

  // Keyboard pan
  public keyPanSpeed: number = 7; // pixels per arrow key press
  private boundOnKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private keyListenElement: HTMLElement | null = null;

  constructor(camera: LuxarCamera, domElement: HTMLElement, config?: LuxarOrbitControlsConfig) {
    super();

    this.camera = camera;
    this.domElement = domElement;
    this.target = new THREE.Vector3();

    // Apply configuration with defaults
    this.enableDamping = config?.enableDamping ?? true;
    this.dampingFactor = config?.dampingFactor ?? 0.25;
    this.rotateSpeed = config?.rotateSpeed ?? 3.0;
    this.panSpeed = config?.panSpeed ?? 1.0;
    this.zoomSpeed = config?.zoomSpeed ?? 1.0;
    this.enableRotate = config?.enableRotate ?? true;
    this.enablePan = config?.enablePan ?? true;
    this.enableZoom = config?.enableZoom ?? true;
    this.autoRotate = config?.autoRotate ?? false;
    this.autoRotateSpeed = config?.autoRotateSpeed ?? 0.25;
    this.screenSpacePanning = config?.screenSpacePanning ?? true;
    this.trackballRadius = config?.trackballRadius ?? 1.0;

    this.minDistance = config?.minDistance ?? 0.01;
    this.maxDistance = config?.maxDistance ?? Infinity;
    this.minZoom = config?.minZoom ?? 0.01;
    this.maxZoom = config?.maxZoom ?? Infinity;

    // Default: left=pan, right=rotate, scroll=zoom
    this.mouseButtons = {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    };

    // Initialize orbit state from camera
    this.initializeFromCamera();

    // Ensure camera matrix is up-to-date (needed by pan math which reads matrix columns)
    this.camera.updateMatrixWorld();

    // Save initial state for reset()
    this.saveState();

    // Bind event handlers
    this.boundOnPointerDown = this.onPointerDown.bind(this);
    this.boundOnPointerMove = this.onPointerMove.bind(this);
    this.boundOnPointerUp = this.onPointerUp.bind(this);
    this.boundOnWheel = this.onWheel.bind(this);
    this.boundOnContextMenu = (e: Event) => e.preventDefault();

    // Attach listeners
    this.domElement.addEventListener('pointerdown', this.boundOnPointerDown);
    this.domElement.addEventListener('wheel', this.boundOnWheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this.boundOnContextMenu);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Update controls. Call every frame.
   * @param deltaTime - Optional time since last frame (seconds). Used for frame-rate independent auto-rotation.
   * @returns true if camera moved (useful for render-on-demand).
   */
  public update(deltaTime?: number): boolean {
    // 1. Auto-rotation: around the camera's screen-up axis (always appears vertical to the viewer)
    // Speed=1.0 → one full rotation in 60 seconds (matches THREE.js OrbitControls convention)
    if (this.autoRotate && this.enableRotate) {
      const dt = deltaTime ?? 1 / 60;
      const angle = ((2 * Math.PI) / 60) * this.autoRotateSpeed * dt;
      // Inline quaternion math (applyOrbitRotation also calls applyToCamera, redundant in update())
      _v2.set(0, 1, 0).applyQuaternion(this.orientation);
      _q1.setFromAxisAngle(_v2, angle);
      this.orientation.premultiply(_q1);
      this.orientation.normalize();
    }

    // 2. Apply trackball rotation with damping (local frame)
    if (this.enableDamping) {
      _q1.slerpQuaternions(_IDENTITY_QUAT, this.rotationDelta, this.dampingFactor);
      this.orientation.multiply(_q1);
      this.orientation.normalize();
      this.rotationDelta.slerp(_IDENTITY_QUAT, this.dampingFactor);
    } else {
      this.orientation.multiply(this.rotationDelta);
      this.orientation.normalize();
      this.rotationDelta.identity();
    }

    // 3. Apply view-axis roll with damping
    if (Math.abs(this.rollDelta) > 1e-6) {
      _v2.set(0, 0, -1).applyQuaternion(this.orientation).normalize();
      if (this.enableDamping) {
        const rollApply = this.rollDelta * this.dampingFactor;
        _q1.setFromAxisAngle(_v2, rollApply);
        this.orientation.premultiply(_q1);
        this.orientation.normalize();
        this.rollDelta *= 1 - this.dampingFactor;
      } else {
        _q1.setFromAxisAngle(_v2, this.rollDelta);
        this.orientation.premultiply(_q1);
        this.orientation.normalize();
        this.rollDelta = 0;
      }
    }

    // 4. Apply pan with damping
    if (this.enableDamping) {
      this.target.addScaledVector(this.panDelta, this.dampingFactor);
      this.panDelta.multiplyScalar(1 - this.dampingFactor);
    } else {
      this.target.add(this.panDelta);
      this.panDelta.set(0, 0, 0);
    }

    // 5. Apply zoom with damping
    if (Math.abs(this.zoomDelta) > 1e-8) {
      if (this.enableDamping) {
        const zoomApply = 1 + this.zoomDelta * this.dampingFactor;
        this.applyZoomScale(zoomApply);
        this.zoomDelta *= 1 - this.dampingFactor;
      } else {
        this.applyZoomScale(1 + this.zoomDelta);
        this.zoomDelta = 0;
      }
    }

    // 6. Clamp distance
    this.distance = THREE.MathUtils.clamp(this.distance, this.minDistance, this.maxDistance);

    // 7. Clamp ortho zoom
    if (this.camera instanceof THREE.OrthographicCamera) {
      this.camera.zoom = THREE.MathUtils.clamp(this.camera.zoom, this.minZoom, this.maxZoom);
      this.camera.updateProjectionMatrix();
    }

    // 8. Apply to camera
    this.applyToCamera();

    // 9. Change detection
    const moved =
      !this.camera.position.equals(this.lastPosition) ||
      !this.camera.quaternion.equals(this.lastQuaternion);

    if (moved) {
      this.dispatchEvent({ type: 'change' });
      this.lastPosition.copy(this.camera.position);
      this.lastQuaternion.copy(this.camera.quaternion);
    }

    return moved;
  }

  /** Save current state for reset(). */
  public saveState(): void {
    this.target0.copy(this.target);
    this.position0.copy(this.camera.position);
    this.orientation0.copy(this.orientation);
    this.zoom0 = this.camera instanceof THREE.OrthographicCamera ? this.camera.zoom : 1;
  }

  /** Restore to last saved state. */
  public reset(): void {
    this.target.copy(this.target0);
    this.orientation.copy(this.orientation0);
    this.distance = Math.max(this.position0.distanceTo(this.target0), 0.001);
    if (this.camera instanceof THREE.OrthographicCamera) {
      this.camera.zoom = this.zoom0;
      this.camera.updateProjectionMatrix();
    }
    this.rotationDelta.identity();
    this.panDelta.set(0, 0, 0);
    this.zoomDelta = 0;
    this.rollDelta = 0;
    this.applyToCamera();
    this.update();
  }

  /**
   * Apply an orbit rotation by the given angle (radians).
   *
   * @param angle - Rotation angle in radians (positive = counter-clockwise when looking along the axis).
   * @param axis  - World-space axis to rotate around. Defaults to the camera's screen-up direction
   *               (same axis used by auto-rotation), which always appears vertical on screen.
   *
   * This is the same quaternion math that auto-rotation uses — call it from turntable
   * recording or any other code that needs to orbit the camera programmatically.
   */
  public applyOrbitRotation(angle: number, axis?: THREE.Vector3): void {
    const rotAxis = axis ?? new THREE.Vector3(0, 1, 0).applyQuaternion(this.orientation);
    const q = new THREE.Quaternion().setFromAxisAngle(rotAxis, angle);
    this.orientation.premultiply(q);
    this.orientation.normalize();
    this.applyToCamera();
  }

  /**
   * Re-derive orientation and distance from the current camera position and target.
   * Call after changing the target externally to keep the orbit state consistent.
   */
  public reinitialize(): void {
    this.initializeFromCamera();
    this.rotationDelta.identity();
    this.panDelta.set(0, 0, 0);
    this.zoomDelta = 0;
    this.rollDelta = 0;
  }

  /**
   * Enable Shift+scroll view-axis rotation (roll around the viewing axis).
   * Uses capture phase to intercept before other wheel handlers (zoom, FOV).
   * Works in both orbit and ortho modes.
   */
  public enableViewAxisRotation(speed: number = 0.0005): void {
    if (this.viewAxisRotationHandler) return; // Already enabled

    this.viewAxisRotationHandler = (event: WheelEvent) => {
      if (!event.shiftKey || !this.enabled) return;
      event.preventDefault();
      event.stopImmediatePropagation();

      // Accumulate into rollDelta — damping is applied in update()
      this.rollDelta += event.deltaY * speed;

      // Wake up animation loop (rollDelta is applied in update())
      this.dispatchEvent({ type: 'change' });
    };

    this.domElement.addEventListener('wheel', this.viewAxisRotationHandler, {
      capture: true,
      passive: false,
    });
  }

  /** Clean up all event listeners. */
  public dispose(): void {
    this.domElement.removeEventListener('pointerdown', this.boundOnPointerDown);
    this.domElement.removeEventListener('pointermove', this.boundOnPointerMove);
    this.domElement.removeEventListener('pointerup', this.boundOnPointerUp);
    this.domElement.removeEventListener('pointercancel', this.boundOnPointerUp);
    this.domElement.removeEventListener('wheel', this.boundOnWheel);
    this.domElement.removeEventListener('contextmenu', this.boundOnContextMenu);

    if (this.viewAxisRotationHandler) {
      this.domElement.removeEventListener('wheel', this.viewAxisRotationHandler, {
        capture: true,
      } as any);
      this.viewAxisRotationHandler = null;
    }

    this.stopListenToKeyEvents();

    // Release any active pointer captures
    for (const pointer of this.pointers) {
      try {
        this.domElement.releasePointerCapture(pointer.pointerId);
      } catch {
        /* ignore */
      }
    }
    this.pointers.length = 0;
    this.pointerPositions.clear();
  }

  // ---------------------------------------------------------------------------
  // Trackball math (vendored from ArcballControls / Shoemake)
  // ---------------------------------------------------------------------------

  /** Project NDC coordinates onto virtual trackball (sphere + hyperboloid). */
  private projectOnTrackball(ndcX: number, ndcY: number): THREE.Vector3 {
    const r = this.trackballRadius;
    const r2 = r * r;
    const d2 = ndcX * ndcX + ndcY * ndcY;
    let z: number;
    if (d2 <= r2 * 0.5) {
      z = Math.sqrt(r2 - d2); // On the sphere
    } else {
      z = (r2 * 0.5) / Math.sqrt(d2); // On the hyperboloid (smooth falloff at edges)
    }
    return new THREE.Vector3(ndcX, ndcY, z).normalize();
  }

  /** Compute rotation quaternion from arcball drag (start → end in NDC). */
  private computeArcballRotation(startNDC: THREE.Vector2, endNDC: THREE.Vector2): THREE.Quaternion {
    const p1 = this.projectOnTrackball(startNDC.x, startNDC.y);
    const p2 = this.projectOnTrackball(endNDC.x, endNDC.y);

    const axis = new THREE.Vector3().crossVectors(p1, p2);
    if (axis.lengthSq() < 1e-10) return new THREE.Quaternion(); // No rotation

    axis.normalize();
    const angle = Math.acos(THREE.MathUtils.clamp(p1.dot(p2), -1, 1)) * this.rotateSpeed;

    // Negate angle: camera orbits opposite to the drag direction
    // (dragging right rotates the view rightward = camera moves left around target)
    return new THREE.Quaternion().setFromAxisAngle(axis, -angle);
  }

  // ---------------------------------------------------------------------------
  // Pan math (vendored from OrbitControls)
  // ---------------------------------------------------------------------------

  private panLeft(distance: number, objectMatrix: THREE.Matrix4): void {
    _v.setFromMatrixColumn(objectMatrix, 0); // camera X axis
    _v.multiplyScalar(-distance);
    this.panDelta.add(_v);
  }

  private panUp(distance: number, objectMatrix: THREE.Matrix4): void {
    if (this.screenSpacePanning) {
      _v.setFromMatrixColumn(objectMatrix, 1); // camera Y axis
    } else {
      _v.setFromMatrixColumn(objectMatrix, 0);
      _v.crossVectors(this.camera.up, _v);
    }
    _v.multiplyScalar(distance);
    this.panDelta.add(_v);
  }

  private pan(deltaX: number, deltaY: number): void {
    if (this.camera instanceof THREE.PerspectiveCamera) {
      const fovRad = this.camera.fov * (Math.PI / 180);
      const height = 2 * this.distance * Math.tan(fovRad / 2);
      this.panLeft(
        (deltaX * height * this.panSpeed) / this.domElement.clientHeight,
        this.camera.matrix
      );
      this.panUp(
        (deltaY * height * this.panSpeed) / this.domElement.clientHeight,
        this.camera.matrix
      );
    } else {
      const cam = this.camera as THREE.OrthographicCamera;
      this.panLeft(
        (deltaX * (cam.right - cam.left) * this.panSpeed) / cam.zoom / this.domElement.clientWidth,
        this.camera.matrix
      );
      this.panUp(
        (deltaY * (cam.top - cam.bottom) * this.panSpeed) / cam.zoom / this.domElement.clientHeight,
        this.camera.matrix
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Zoom math (vendored from OrbitControls)
  // ---------------------------------------------------------------------------

  private getZoomScale(delta: number): number {
    const normalizedDelta = Math.abs(delta * 0.01);
    return Math.pow(0.95, this.zoomSpeed * normalizedDelta);
  }

  private applyZoomScale(scale: number): void {
    if (this.camera instanceof THREE.PerspectiveCamera) {
      this.distance *= scale;
    } else {
      const cam = this.camera as THREE.OrthographicCamera;
      cam.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, cam.zoom / scale));
      cam.updateProjectionMatrix();
    }
  }

  // ---------------------------------------------------------------------------
  // Camera application
  // ---------------------------------------------------------------------------

  /** Apply orientation + distance + target to camera transform. */
  private applyToCamera(): void {
    _v.set(0, 0, this.distance).applyQuaternion(this.orientation);
    this.camera.position.copy(this.target).add(_v);
    this.camera.up.set(0, 1, 0).applyQuaternion(this.orientation);
    this.camera.lookAt(this.target);
    // Ensure camera.matrix is up-to-date (needed by pan math which reads matrix columns)
    this.camera.updateMatrixWorld();
  }

  /** Extract orientation and distance from current camera state. */
  private initializeFromCamera(): void {
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
    this.distance = Math.max(offset.length(), 0.001);

    // Derive up from camera quaternion rather than camera.up — the quaternion
    // is always authoritative, whereas camera.up may be stale (fly controls
    // only update quaternion, not up). Prevents roll loss on fly→orbit switch.
    // lookAt() has a singularity when the view direction is parallel to the up vector.
    // Detect this and use a fallback up vector to prevent NaN.
    const viewDir = offset.clone().normalize();
    let up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    const upDot = Math.abs(viewDir.dot(up));
    if (upDot > 0.999) {
      // Near singularity: pick a fallback up vector perpendicular to view direction
      up = Math.abs(viewDir.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
    }

    const lookMatrix = new THREE.Matrix4().lookAt(this.camera.position, this.target, up);
    this.orientation.setFromRotationMatrix(lookMatrix);
  }

  // ---------------------------------------------------------------------------
  // Pointer event handling
  // ---------------------------------------------------------------------------

  private getPointerNDC(event: PointerEvent): THREE.Vector2 {
    const rect = this.domElement.getBoundingClientRect();
    return new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
  }

  private getMouseAction(button: number, shiftKey: boolean): ControlAction {
    let mapping: THREE.MOUSE | null = null;
    if (button === 0) mapping = this.mouseButtons.LEFT;
    else if (button === 1) mapping = this.mouseButtons.MIDDLE;
    else if (button === 2) mapping = this.mouseButtons.RIGHT;

    if (mapping === null) return 'none';

    // Shift+left inverts the primary action:
    // If left=pan → Shift+left=rotate; if left=rotate → Shift+left=pan
    if (button === 0 && shiftKey) {
      if (mapping === THREE.MOUSE.PAN) return this.enableRotate ? 'rotate' : 'none';
      if (mapping === THREE.MOUSE.ROTATE) return this.enablePan ? 'pan' : 'none';
    }

    if (mapping === THREE.MOUSE.ROTATE) return this.enableRotate ? 'rotate' : 'none';
    if (mapping === THREE.MOUSE.PAN) return this.enablePan ? 'pan' : 'none';
    if (mapping === THREE.MOUSE.DOLLY) return this.enableZoom ? 'zoom' : 'none';

    return 'none';
  }

  private onPointerDown(event: PointerEvent): void {
    if (!this.enabled) return;

    if (this.pointers.length === 0) {
      this.domElement.setPointerCapture(event.pointerId);
      this.domElement.addEventListener('pointermove', this.boundOnPointerMove);
      this.domElement.addEventListener('pointerup', this.boundOnPointerUp);
      this.domElement.addEventListener('pointercancel', this.boundOnPointerUp);
    }

    this.pointers.push(event);
    this.pointerPositions.set(event.pointerId, new THREE.Vector2(event.clientX, event.clientY));

    if (event.pointerType === 'touch') {
      this.onTouchStart();
    } else {
      const action = this.getMouseAction(event.button, event.shiftKey);
      this.state = action;

      if (action === 'rotate') {
        this.rotateStart.copy(this.getPointerNDC(event));
      } else if (action === 'pan') {
        this.panStart.set(event.clientX, event.clientY);
      } else if (action === 'zoom') {
        this.dollyStart.set(event.clientX, event.clientY);
      }
    }

    if (this.state !== 'none') {
      this.dispatchEvent({ type: 'start' });
    }
  }

  private onPointerMove(event: PointerEvent): void {
    if (!this.enabled) return;

    // Update pointer position
    const pos = this.pointerPositions.get(event.pointerId);
    if (pos) pos.set(event.clientX, event.clientY);

    // Update the pointer in our array
    for (let i = 0; i < this.pointers.length; i++) {
      if (this.pointers[i].pointerId === event.pointerId) {
        this.pointers[i] = event;
        break;
      }
    }

    if (event.pointerType === 'touch') {
      this.onTouchMove(event);
      return;
    }

    if (this.state === 'rotate') {
      const endNDC = this.getPointerNDC(event);
      const deltaQuat = this.computeArcballRotation(this.rotateStart, endNDC);
      this.rotationDelta.multiply(deltaQuat);
      this.rotateStart.copy(endNDC);
    } else if (this.state === 'pan') {
      const deltaX = event.clientX - this.panStart.x;
      const deltaY = event.clientY - this.panStart.y;
      this.pan(deltaX, deltaY);
      this.panStart.set(event.clientX, event.clientY);
    } else if (this.state === 'zoom') {
      const deltaY = event.clientY - this.dollyStart.y;
      if (deltaY > 0) {
        this.zoomDelta += this.getZoomScale(deltaY) - 1;
      } else if (deltaY < 0) {
        this.zoomDelta -= this.getZoomScale(-deltaY) - 1;
      }
      this.dollyStart.set(event.clientX, event.clientY);
    }
  }

  private onPointerUp(event: PointerEvent): void {
    // Remove this pointer
    this.pointers = this.pointers.filter((p) => p.pointerId !== event.pointerId);
    this.pointerPositions.delete(event.pointerId);

    if (this.pointers.length === 0) {
      try {
        this.domElement.releasePointerCapture(event.pointerId);
      } catch {
        /* pointer capture may already be released on cancel */
      }
      this.domElement.removeEventListener('pointermove', this.boundOnPointerMove);
      this.domElement.removeEventListener('pointerup', this.boundOnPointerUp);
      this.domElement.removeEventListener('pointercancel', this.boundOnPointerUp);
    }

    this.state = 'none';
    this.dispatchEvent({ type: 'end' });
  }

  private onWheel(event: WheelEvent): void {
    if (!this.enabled || !this.enableZoom) return;
    event.preventDefault();

    const scale = this.getZoomScale(event.deltaY);
    if (event.deltaY < 0) {
      // Scroll up = zoom in
      this.zoomDelta += scale - 1;
    } else if (event.deltaY > 0) {
      // Scroll down = zoom out
      this.zoomDelta -= scale - 1;
    }

    // Wake up animation loop (zoomDelta is applied with damping in update())
    this.dispatchEvent({ type: 'change' });
  }

  // ---------------------------------------------------------------------------
  // Touch handling
  // ---------------------------------------------------------------------------

  private onTouchStart(): void {
    if (this.pointers.length === 1) {
      // Single finger: rotate (or pan if rotation disabled)
      if (this.enableRotate) {
        this.state = 'rotate';
        this.rotateStart.copy(this.getPointerNDC(this.pointers[0]));
      } else if (this.enablePan) {
        this.state = 'pan';
        this.panStart.set(this.pointers[0].clientX, this.pointers[0].clientY);
      }
    } else if (this.pointers.length === 2) {
      // Two fingers: dolly-pan
      this.state = 'zoom'; // Combined dolly + pan
      const dx = this.pointers[0].clientX - this.pointers[1].clientX;
      const dy = this.pointers[0].clientY - this.pointers[1].clientY;
      this.dollyStart.set(0, Math.sqrt(dx * dx + dy * dy));
      // Pan center
      this.panStart.set(
        (this.pointers[0].clientX + this.pointers[1].clientX) * 0.5,
        (this.pointers[0].clientY + this.pointers[1].clientY) * 0.5
      );
    }
  }

  private onTouchMove(_event: PointerEvent): void {
    if (this.pointers.length === 1 && this.state === 'rotate') {
      const endNDC = this.getPointerNDC(this.pointers[0]);
      const deltaQuat = this.computeArcballRotation(this.rotateStart, endNDC);
      this.rotationDelta.multiply(deltaQuat);
      this.rotateStart.copy(endNDC);
    } else if (this.pointers.length === 1 && this.state === 'pan') {
      const deltaX = this.pointers[0].clientX - this.panStart.x;
      const deltaY = this.pointers[0].clientY - this.panStart.y;
      this.pan(deltaX, deltaY);
      this.panStart.set(this.pointers[0].clientX, this.pointers[0].clientY);
    } else if (this.pointers.length >= 2) {
      // Two-finger dolly + pan
      const p0 = this.pointerPositions.get(this.pointers[0].pointerId);
      const p1 = this.pointerPositions.get(this.pointers[1].pointerId);
      if (!p0 || !p1) return;

      // Dolly (pinch)
      // Negate so pinch-out (fingers spread) = zoom in = negative zoomDelta,
      // consistent with scroll-up = zoom in = negative zoomDelta.
      const dx = p0.x - p1.x;
      const dy = p0.y - p1.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const dollyDelta = distance / this.dollyStart.y;
      if (dollyDelta > 0) {
        this.zoomDelta -= dollyDelta - 1;
      }
      this.dollyStart.set(0, distance);

      // Pan (two-finger drag)
      const centerX = (p0.x + p1.x) * 0.5;
      const centerY = (p0.y + p1.y) * 0.5;
      this.pan(centerX - this.panStart.x, centerY - this.panStart.y);
      this.panStart.set(centerX, centerY);
    }
  }

  // ---------------------------------------------------------------------------
  // Keyboard handling
  // ---------------------------------------------------------------------------

  /**
   * Enable keyboard controls (arrow keys for panning).
   * Call with the element that should receive key events (typically window or canvas).
   */
  public listenToKeyEvents(element: HTMLElement | Window): void {
    if (this.boundOnKeyDown) return; // Already listening

    this.boundOnKeyDown = (event: KeyboardEvent) => {
      if (!this.enabled || !this.enablePan) return;

      switch (event.code) {
        case 'ArrowUp':
          this.pan(0, this.keyPanSpeed);
          event.preventDefault();
          break;
        case 'ArrowDown':
          this.pan(0, -this.keyPanSpeed);
          event.preventDefault();
          break;
        case 'ArrowLeft':
          this.pan(this.keyPanSpeed, 0);
          event.preventDefault();
          break;
        case 'ArrowRight':
          this.pan(-this.keyPanSpeed, 0);
          event.preventDefault();
          break;
      }
    };

    element.addEventListener('keydown', this.boundOnKeyDown as EventListener);
    this.keyListenElement = element as HTMLElement;
  }

  /** Stop listening for keyboard events. */
  public stopListenToKeyEvents(): void {
    if (this.boundOnKeyDown && this.keyListenElement) {
      this.keyListenElement.removeEventListener('keydown', this.boundOnKeyDown as EventListener);
      this.boundOnKeyDown = null;
      this.keyListenElement = null;
    }
  }
}
