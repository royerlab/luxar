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
      // controls.md C6 fix: use the public `applyOrbitRotation(angle, axis)`
      // API instead of mutating private `rotationDelta`. applyOrbitRotation
      // premultiplies the orientation quaternion and calls applyToCamera()
      // directly — orthonormal rotation must preserve Euclidean distance
      // to the target regardless of the rotation pipeline used.
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = camera.position.distanceTo(controls.target);

      const rotAxis = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();
      controls.applyOrbitRotation(angle, rotAxis);

      const distAfter = camera.position.distanceTo(controls.target);
      expect(distAfter).toBeCloseTo(distBefore, 3);
    });

    it('should not have gimbal lock when looking straight down', () => {
      // controls.md C7 fix: drive rotation through the public
      // applyOrbitRotation API instead of mutating private rotationDelta.
      camera.position.set(0, 5, 0.001); // Slightly off-axis to avoid degenerate lookAt
      camera.up.set(0, 0, -1);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();
      const posBefore = camera.position.clone();

      controls.applyOrbitRotation(0.3, new THREE.Vector3(1, 0, 0));

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
    it('zooms in (distance decreases) on wheel scroll-up', () => {
      // controls.md C9 fix: drive the zoom via the public WheelEvent path on
      // `domElement` (the listener registered at `luxar-orbit-controls.ts:195`)
      // instead of writing `(controls as any).zoomDelta = -0.5`. The exact
      // `applyZoomScale` formula is unit-tested in
      // `luxar-orbit-controls/math/zoom.test.ts` and the gating + cancellation
      // is in `luxar-orbit-controls/update.test.ts`; here we only need to
      // verify that wheel-up is wired to the zoom-in branch end-to-end.
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = camera.position.distanceTo(controls.target);
      domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: -100 }));
      controls.update();

      const distAfter = camera.position.distanceTo(controls.target);
      expect(distAfter).toBeLessThan(distBefore);
    });

    it('zooms ortho camera in (camera.zoom increases) on wheel scroll-up', () => {
      // controls.md C9 fix (ortho variant): wheel scroll-up should increase
      // camera.zoom (ortho-style zoom-in). The exact formula is covered in
      // math/zoom.test.ts; we only verify the wiring here.
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

      const zoomBefore = orthoCam.zoom;
      domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: -100 }));
      controls.update();

      expect(orthoCam.zoom).toBeGreaterThan(zoomBefore);
    });

    it('should clamp distance to min/max', () => {
      // controls.md C10 fix: drive distance changes via the wheel path and
      // observe through the camera-position public observable, not by
      // mutating `(controls as any).distance` directly. Distance clamping is
      // separately covered formally in update.test.ts (step 6).
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        minDistance: 2,
        maxDistance: 10,
      });
      controls.update();

      // Scroll up many times to drive distance past min — clamp must hold.
      for (let i = 0; i < 200; i++) {
        domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: -500 }));
        controls.update();
      }
      expect(camera.position.distanceTo(controls.target)).toBeGreaterThanOrEqual(2 - 1e-5);

      // Scroll down many times to drive past max — clamp must hold.
      for (let i = 0; i < 200; i++) {
        domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: 500 }));
        controls.update();
      }
      expect(camera.position.distanceTo(controls.target)).toBeLessThanOrEqual(10 + 1e-5);
    });
  });

  describe('damping', () => {
    // controls.md C8: these two tests inject through private `rotationDelta`
    // because there is no public seam to enqueue a rotation that participates
    // in the damping pipeline — `applyOrbitRotation` commits orientation
    // directly and bypasses damping entirely. The proper test home is
    // `luxar-orbit-controls/update.test.ts`, which accesses `rotationDelta`
    // through the documented `OrbitUpdateCtx` seam. The orchestrator-level
    // tests are kept here as integration coverage for the wiring between the
    // private delta and update().
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
      expect(point.x).toBeCloseTo(0, 5);
      expect(point.y).toBeCloseTo(0, 5);
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
      // controls.md C11 fix: move the camera through the public
      // `applyOrbitRotation` API instead of calling the private `pan(dx, dy)`.
      // The test contract is reset()-restores-saved-state; the SPECIFIC action
      // that moved the camera is incidental.
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();
      controls.saveState();

      const savedPos = camera.position.clone();

      controls.applyOrbitRotation(Math.PI / 4, new THREE.Vector3(0, 1, 0));
      expect(camera.position.x).not.toBeCloseTo(savedPos.x, 5);

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

    it('dispose() removes the ortho view-axis wheel listener [controls.md/W10][P2][C12]', () => {
      // controls.md C12 fix: assert the BEHAVIORAL contract (no leak) rather
      // than the private `viewAxisRotationHandler` nulling. After dispose, a
      // shift+wheel event must NOT mutate orientation — the handler is
      // detached from `domElement`.
      controls = new LuxarOrbitControls(camera, domElement);
      controls.enableViewAxisRotation();
      controls.dispose();

      // Snapshot orientation before the post-dispose event.
      const orientationBefore = (controls as any).orientation.clone();

      // A shift+wheel event would normally trigger view-axis rotation. After
      // dispose, the listener must be detached — orientation stays put.
      domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: 100, shiftKey: true }));

      const orientationAfter = (controls as any).orientation as THREE.Quaternion;
      expect(orientationAfter.x).toBeCloseTo(orientationBefore.x, 10);
      expect(orientationAfter.y).toBeCloseTo(orientationBefore.y, 10);
      expect(orientationAfter.z).toBeCloseTo(orientationBefore.z, 10);
      expect(orientationAfter.w).toBeCloseTo(orientationBefore.w, 10);
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
