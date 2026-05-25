/**
 * Unit tests for LuxarOrbitControls — quaternion-based orbit with damping.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { LuxarOrbitControls } from '../../../controls/luxar-orbit-controls';
import { projectOnTrackball } from '../../../controls/luxar-orbit-controls/math/trackball';
import {
  mouseAction,
  type OrbitInputCtx,
} from '../../../controls/luxar-orbit-controls/input/pointer';

describe('LuxarOrbitControls', () => {
  let camera: THREE.PerspectiveCamera;
  let domElement: HTMLElement;
  let controls: LuxarOrbitControls;

  beforeEach(() => {
    camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    domElement = document.createElement('div');
    domElement.style.width = '800px';
    domElement.style.height = '600px';
    document.body.appendChild(domElement);
    // Mock getBoundingClientRect
    domElement.getBoundingClientRect = vi.fn(() => ({
      left: 0,
      top: 0,
      width: 800,
      height: 600,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
  });

  afterEach(() => {
    controls?.dispose();
    document.body.removeChild(domElement);
  });

  describe('initialization', () => {
    it('should initialize from camera position', () => {
      controls = new LuxarOrbitControls(camera, domElement);

      // Camera is at (0,0,5) looking at origin → distance = 5. Target is
      // initialised to exact origin (not derived from float arithmetic);
      // use toBe for exact equality so a regression that adds even a
      // sub-ULP perturbation is caught.
      expect(controls.target.x).toBe(0);
      expect(controls.target.y).toBe(0);
      expect(controls.target.z).toBe(0);
    });

    it('should apply default configuration', () => {
      controls = new LuxarOrbitControls(camera, domElement);

      expect(controls.enableDamping).toBe(true);
      // Default config values are stored verbatim — no float arithmetic.
      expect(controls.dampingFactor).toBe(0.25);
      expect(controls.enableRotate).toBe(true);
      expect(controls.enablePan).toBe(true);
      expect(controls.enableZoom).toBe(true);
      expect(controls.autoRotate).toBe(false);
    });

    it('should apply custom configuration', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        dampingFactor: 0.1,
        rotateSpeed: 2.0,
        enableRotate: false,
        minDistance: 1,
        maxDistance: 100,
      });

      // Custom config values are stored verbatim — no float arithmetic.
      expect(controls.dampingFactor).toBe(0.1);
      expect(controls.rotateSpeed).toBe(2.0);
      expect(controls.enableRotate).toBe(false);
      expect(controls.minDistance).toBe(1);
      expect(controls.maxDistance).toBe(100);
    });
  });

  describe('update', () => {
    it('should return false when idle (no movement)', () => {
      controls = new LuxarOrbitControls(camera, domElement);
      // First update applies initial state
      controls.update();
      // Second update: no changes
      const moved = controls.update();
      expect(moved).toBe(false);
    });

    it('should dispatch change event when camera moves', () => {
      controls = new LuxarOrbitControls(camera, domElement);
      controls.update(); // initial
      const handler = vi.fn();
      controls.addEventListener('change', handler);

      // Move the target to force camera movement
      controls.target.set(1, 0, 0);
      controls.update();

      expect(handler).toHaveBeenCalled();
    });
  });

  describe('quaternion rotation', () => {
    // [controls.md/O4][P4] Was a single example rotation; parametrize over
    // axis + angle to lock the distance-preservation invariant across the
    // unit sphere of rotations. This is a clean algebraic invariant
    // (orthonormal rotation preserves Euclidean distance to the target).
    it.each([
      { axis: [0, 1, 0] as const, angle: Math.PI / 4, label: 'yaw 45deg' },
      { axis: [1, 0, 0] as const, angle: Math.PI / 6, label: 'pitch 30deg' },
      { axis: [0, 0, 1] as const, angle: Math.PI / 3, label: 'roll 60deg' },
      { axis: [1, 1, 0] as const, angle: Math.PI / 2, label: 'diagonal 90deg' },
    ])('preserves camera distance to target after $label rotation', ({ axis, angle }) => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = camera.position.distanceTo(controls.target);

      const rotQuat = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(axis[0], axis[1], axis[2]).normalize(),
        angle,
      );
      (controls as any).rotationDelta.multiply(rotQuat);
      controls.update();

      const distAfter = camera.position.distanceTo(controls.target);
      expect(distAfter).toBeCloseTo(distBefore, 3);
    });

    it('should not have gimbal lock when looking straight down', () => {
      // Position camera straight above looking down
      camera.position.set(0, 5, 0.001); // Slightly off-axis to avoid degenerate lookAt
      camera.up.set(0, 0, -1);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();
      const posBefore = camera.position.clone();

      // Apply a rotation — should work smoothly without gimbal lock
      const rotQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.3);
      (controls as any).rotationDelta.multiply(rotQuat);
      controls.update();

      // Camera should have moved (not stuck at pole)
      const moved = camera.position.distanceTo(posBefore);
      expect(moved).toBeGreaterThan(0.01);
    });
  });

  describe('panning', () => {
    it('moves target by the deterministic pan formula (W9 strengthening)', () => {
      // W9 strengthening: was `target.x !== targetBefore.x`. Pan math is
      // pure and fully deterministic — assert the exact magnitude.
      // Formula (perspective, screenSpacePanning):
      //   panLeft.dist = (deltaX * height * panSpeed) / clientHeight
      //   height = 2 * distance * tan(fov/2 * π/180)
      // For our camera: distance=5, fov=60, clientHeight=600, panSpeed=1.
      // jsdom doesn't compute layout from CSS, so clientHeight reports 0;
      // override it to match the mocked getBoundingClientRect.
      Object.defineProperty(domElement, 'clientHeight', {
        configurable: true,
        get: () => 600,
      });
      Object.defineProperty(domElement, 'clientWidth', {
        configurable: true,
        get: () => 800,
      });
      const fovRad = (60 * Math.PI) / 180;
      const height = 2 * 5 * Math.tan(fovRad / 2);
      const expectedX = -(100 * height * 1) / 600; // panLeft direction = -X for identity orientation
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();
      const targetBefore = controls.target.clone();

      (controls as any).pan(100, 0); // Pan 100px right
      controls.update();

      // Target should have moved by expectedX in world X (camera matrix col 0
      // is world X axis when orientation = identity).
      expect(controls.target.x - targetBefore.x).toBeCloseTo(expectedX, 5);
      // Y and Z untouched.
      expect(controls.target.y).toBeCloseTo(targetBefore.y, 5);
      expect(controls.target.z).toBeCloseTo(targetBefore.z, 5);
    });
  });

  describe('zooming', () => {
    it('changes perspective distance by the deterministic zoom formula (W12)', () => {
      // W12 strengthening: was `distance < distBefore` only. The applyZoomScale
      // formula for perspective is `currentDistance * (1 + zoomDelta)`
      // (no damping). With zoomDelta=-0.5 and distance=5, expected=2.5.
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = (controls as any).distance;
      (controls as any).zoomDelta = -0.5;
      controls.update();
      // Expected: distance * (1 + zoomDelta) = 5 * 0.5 = 2.5.
      expect((controls as any).distance).toBeCloseTo(distBefore * 0.5, 5);
    });

    it('changes ortho zoom by the deterministic formula (W12)', () => {
      // W12 strengthening: applyZoomScale for ortho computes
      // `cam.zoom / scale` where scale = 1 + zoomDelta. zoomDelta=-0.5 →
      // scale=0.5 → new zoom = 1 / 0.5 = 2.0 (zoom in).
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.set(0, 0, 5);
      orthoCam.lookAt(0, 0, 0);
      orthoCam.updateMatrixWorld();

      controls = new LuxarOrbitControls(orthoCam, domElement, {
        enableDamping: false,
        minZoom: 0.001,
        maxZoom: 1000,
      });
      controls.update();

      const zoomBefore = orthoCam.zoom; // default = 1
      (controls as any).zoomDelta = -0.5;
      controls.update();
      // Expected: zoomBefore / (1 - 0.5) = 1 / 0.5 = 2.0
      expect(orthoCam.zoom).toBeCloseTo(zoomBefore / 0.5, 5);
    });

    it('should clamp distance to min/max', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        minDistance: 2,
        maxDistance: 10,
      });
      controls.update();

      // Try to zoom past min
      (controls as any).distance = 0.5;
      controls.update();
      expect((controls as any).distance).toBeGreaterThanOrEqual(2);

      // Try to zoom past max
      (controls as any).distance = 20;
      controls.update();
      expect((controls as any).distance).toBeLessThanOrEqual(10);
    });
  });

  describe('damping', () => {
    it('should decay rotation velocity over multiple frames', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: true,
        dampingFactor: 0.1,
      });
      controls.update();

      // Apply rotation
      const rotQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);
      (controls as any).rotationDelta.multiply(rotQuat);

      // After several updates, rotation delta should approach identity
      for (let i = 0; i < 50; i++) controls.update();

      const delta = (controls as any).rotationDelta as THREE.Quaternion;
      expect(delta.x).toBeCloseTo(0, 2);
      expect(delta.y).toBeCloseTo(0, 2);
      expect(delta.z).toBeCloseTo(0, 2);
      expect(delta.w).toBeCloseTo(1, 2);
    });

    it('should apply rotation immediately when damping disabled', () => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const rotQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);
      (controls as any).rotationDelta.multiply(rotQuat);
      controls.update();

      // Delta should be cleared (identity) after one update
      const delta = (controls as any).rotationDelta as THREE.Quaternion;
      expect(delta.x).toBeCloseTo(0, 5);
      expect(delta.y).toBeCloseTo(0, 5);
      expect(delta.z).toBeCloseTo(0, 5);
      expect(delta.w).toBeCloseTo(1, 5);
    });
  });

  describe('auto-rotation', () => {
    it('sweeps the camera by the deterministic per-frame angle (W11)', () => {
      // W11 strengthening: was `position.distanceTo(posBefore) > 0.01`.
      // The runUpdateStep formula is:
      //   angle_per_frame = (2π/60) * autoRotateSpeed * dt
      // We feed an explicit dt to update() so the swept angle is fully
      // deterministic. After N frames the camera should have rotated
      // by N * angle_per_frame around screen-up (the world Y axis when
      // orientation = identity at construction).
      const speed = 50;
      const dt = 1 / 60;
      const anglePerFrame = ((2 * Math.PI) / 60) * speed * dt;
      const frames = 10;
      const expectedAngle = anglePerFrame * frames;

      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        autoRotate: true,
        autoRotateSpeed: speed,
      });
      controls.update(dt); // initial

      const posBefore = camera.position.clone();
      for (let i = 0; i < frames; i++) controls.update(dt);

      // Initial: camera at (0,0,5), target=(0,0,0), screen-up = world-Y.
      // After rotating around Y by total angle θ, camera position:
      //   x = 5*sin(θ), z = 5*cos(θ), y=0
      // (positive sin → camera moved in +X for positive angle).
      // Distance from initial = 5 * sqrt(2 - 2*cos(θ)).
      const expectedDistance = 5 * Math.sqrt(2 - 2 * Math.cos(expectedAngle));
      const actualDistance = camera.position.distanceTo(posBefore);
      // 4-decimal tolerance allows for accumulated quaternion-normalization drift.
      expect(actualDistance).toBeCloseTo(expectedDistance, 3);
    });

    it('should not rotate when autoRotate is disabled', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        autoRotate: false,
      });
      controls.update();

      const posBefore = camera.position.clone();
      controls.update();

      // With damping disabled and autoRotate off, the second update is a
      // no-op — position must be EXACTLY preserved, not "approximately."
      expect(camera.position.x).toBe(posBefore.x);
      expect(camera.position.y).toBe(posBefore.y);
      expect(camera.position.z).toBe(posBefore.z);
    });
  });

  describe('mouse button mapping', () => {
    it('should default to left=pan, right=rotate', () => {
      controls = new LuxarOrbitControls(camera, domElement);

      expect(controls.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
      expect(controls.mouseButtons.MIDDLE).toBe(THREE.MOUSE.DOLLY);
      expect(controls.mouseButtons.RIGHT).toBe(THREE.MOUSE.ROTATE);
    });

    it('should support ortho-style mapping (left=pan, right=null)', () => {
      controls = new LuxarOrbitControls(camera, domElement);
      controls.mouseButtons.RIGHT = null;

      const ctx = {
        mouseButtons: controls.mouseButtons,
        enableRotate: controls.enableRotate,
        enablePan: controls.enablePan,
        enableZoom: controls.enableZoom,
      } as unknown as OrbitInputCtx;
      const action = mouseAction(2, false, ctx);
      expect(action).toBe('none');
    });

    it('should map Shift+left to rotate in default mode (inverts primary)', () => {
      controls = new LuxarOrbitControls(camera, domElement);

      const ctx = {
        mouseButtons: controls.mouseButtons,
        enableRotate: controls.enableRotate,
        enablePan: controls.enablePan,
        enableZoom: controls.enableZoom,
      } as unknown as OrbitInputCtx;
      const action = mouseAction(0, true, ctx);
      expect(action).toBe('rotate');
    });
  });

  describe('trackball projection', () => {
    // Default trackball radius matches LuxarOrbitControls' constructor default.
    const RADIUS = 1.0;

    it('should project center of screen to sphere cap', () => {
      const point = projectOnTrackball(0, 0, RADIUS);
      // At center: z should be maximum (top of sphere)
      expect(point.z).toBeGreaterThan(0.9);
      expect(point.x).toBeCloseTo(0);
      expect(point.y).toBeCloseTo(0);
    });

    it('should project edge of screen to hyperboloid', () => {
      const point = projectOnTrackball(0.9, 0, RADIUS);
      // At edge: z should be smaller than at center (grazing angle)
      const centerPoint = projectOnTrackball(0, 0, RADIUS);
      expect(point.z).toBeLessThan(centerPoint.z);
      expect(point.x).toBeGreaterThan(0.5);
    });

    it('should always return normalized vectors', () => {
      const testPoints = [
        [0, 0],
        [0.5, 0.5],
        [0.9, 0.1],
        [-0.3, 0.8],
        [1.0, 1.0],
      ];
      for (const [x, y] of testPoints) {
        const point = projectOnTrackball(x, y, RADIUS);
        expect(point.length()).toBeCloseTo(1.0, 5);
      }
    });
  });

  describe('save/reset', () => {
    it('should restore to saved state', () => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();
      controls.saveState();

      const savedPos = camera.position.clone();

      // Move camera by panning
      (controls as any).pan(200, 100);
      controls.update();
      expect(camera.position.x).not.toBeCloseTo(savedPos.x);

      // Reset — restoration is deterministic (saved state is copied verbatim),
      // so Float32 precision (~1e-5) is the appropriate tolerance, not the
      // one-decimal `1` precision that was previously used.
      controls.reset();
      expect(camera.position.x).toBeCloseTo(savedPos.x, 5);
      expect(camera.position.y).toBeCloseTo(savedPos.y, 5);
      expect(camera.position.z).toBeCloseTo(savedPos.z, 5);
    });

    it('reset() restores camera.zoom for perspective cameras too (HIGH-13)', () => {
      // PerspectiveCamera also has a `.zoom` field that affects the projection
      // matrix (telephoto-style zoom-in without FOV change). Previously
      // saveState() hard-coded zoom0 = 1 for non-ortho cameras, so reset()
      // silently dropped any user-modified perspective zoom. zoom0 must now
      // be saved unconditionally and reset() must restore + updateProjectionMatrix.
      camera.zoom = 1.0;
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.saveState(); // baseline zoom = 1.0

      // User zooms in.
      camera.zoom = 2.5;
      camera.updateProjectionMatrix();
      const projBefore = camera.projectionMatrix.elements[0];

      controls.reset();
      expect(camera.zoom).toBeCloseTo(1.0, 5);
      // updateProjectionMatrix() must run so the projection actually changes.
      expect(camera.projectionMatrix.elements[0]).not.toBeCloseTo(projBefore, 5);
    });

    it('reset() restores camera.zoom for orthographic cameras (regression)', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.set(0, 0, 5);
      orthoCam.lookAt(0, 0, 0);
      orthoCam.updateMatrixWorld();
      orthoCam.zoom = 1.0;

      const orthoControls = new LuxarOrbitControls(orthoCam, domElement, {
        enableDamping: false,
      });
      orthoControls.saveState();

      orthoCam.zoom = 3.0;
      orthoCam.updateProjectionMatrix();

      orthoControls.reset();
      expect(orthoCam.zoom).toBeCloseTo(1.0, 5);
      orthoControls.dispose();
    });
  });

  describe('dispose', () => {
    it('dispose() detaches pointer/wheel/contextmenu listeners on dom element [controls.md/W10][P2]', () => {
      // controls.md [W10][P2] strengthening: was `.not.toThrow()` only.
      // The contract is that dispose() calls removeEventListener for each
      // of the 6 event types it registered (pointerdown/pointermove/
      // pointerup/pointercancel/wheel/contextmenu). Spy on the dom
      // element's removeEventListener and confirm each is called. A
      // regression that forgot one removal would surface here.
      controls = new LuxarOrbitControls(camera, domElement);
      const removeSpy = vi.spyOn(domElement, 'removeEventListener');
      controls.dispose();
      const removedEvents = new Set(removeSpy.mock.calls.map((c) => c[0]));
      expect(removedEvents.has('pointerdown')).toBe(true);
      expect(removedEvents.has('pointermove')).toBe(true);
      expect(removedEvents.has('pointerup')).toBe(true);
      expect(removedEvents.has('pointercancel')).toBe(true);
      expect(removedEvents.has('wheel')).toBe(true);
      expect(removedEvents.has('contextmenu')).toBe(true);
      removeSpy.mockRestore();
    });

    it('dispose() removes the ortho view-axis wheel listener [controls.md/W10][P2]', () => {
      // controls.md [W10][P2] strengthening: was `.not.toThrow()` only.
      // After enabling ortho view-axis rotation and disposing, the
      // viewAxisRotationHandler private slot must be null so the wheel
      // path is detached (no leak on the dom element).
      controls = new LuxarOrbitControls(camera, domElement);
      controls.enableViewAxisRotation();
      expect((controls as unknown as { viewAxisRotationHandler: unknown }).viewAxisRotationHandler)
        .not.toBeNull();
      controls.dispose();
      // After dispose, the handler reference must be null/undefined so a
      // future wheel event can't fire through it.
      const handler = (controls as unknown as { viewAxisRotationHandler: unknown })
        .viewAxisRotationHandler;
      expect(handler === null || handler === undefined).toBe(true);
    });
  });

  describe('ortho view-axis rotation', () => {
    it('should enable view-axis rotation handler', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.set(0, 0, 5);
      orthoCam.lookAt(0, 0, 0);
      orthoCam.updateMatrixWorld();

      controls = new LuxarOrbitControls(orthoCam, domElement);
      controls.enableViewAxisRotation();

      // Handler should be set
      expect((controls as any).viewAxisRotationHandler).not.toBeNull();
    });
  });
});
