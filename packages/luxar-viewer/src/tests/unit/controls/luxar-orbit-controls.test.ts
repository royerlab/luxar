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

      // Camera is at (0,0,5) looking at origin → distance = 5
      expect(controls.target.x).toBeCloseTo(0);
      expect(controls.target.y).toBeCloseTo(0);
      expect(controls.target.z).toBeCloseTo(0);
    });

    it('should apply default configuration', () => {
      controls = new LuxarOrbitControls(camera, domElement);

      expect(controls.enableDamping).toBe(true);
      expect(controls.dampingFactor).toBeCloseTo(0.25);
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

      expect(controls.dampingFactor).toBeCloseTo(0.1);
      expect(controls.rotateSpeed).toBeCloseTo(2.0);
      expect(controls.enableRotate).toBe(false);
      expect(controls.minDistance).toBeCloseTo(1);
      expect(controls.maxDistance).toBeCloseTo(100);
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
    it('should maintain camera distance after rotation', () => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = camera.position.distanceTo(controls.target);

      // Simulate a rotation by accessing private method via any cast
      const rotQuat = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        Math.PI / 4
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
    it('should move target when panning', () => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const targetBefore = controls.target.clone();
      // Simulate pan
      (controls as any).pan(100, 0); // Pan 100px right
      controls.update();

      // Target should have moved
      expect(controls.target.x).not.toBeCloseTo(targetBefore.x);
    });
  });

  describe('zooming', () => {
    it('should change distance for perspective camera', () => {
      controls = new LuxarOrbitControls(camera, domElement, { enableDamping: false });
      controls.update();

      const distBefore = (controls as any).distance;
      // Zoom in
      (controls as any).zoomDelta = -0.5;
      controls.update();

      expect((controls as any).distance).toBeLessThan(distBefore);
    });

    it('should change camera.zoom for orthographic camera', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.set(0, 0, 5);
      orthoCam.lookAt(0, 0, 0);
      orthoCam.updateMatrixWorld();

      controls = new LuxarOrbitControls(orthoCam, domElement, { enableDamping: false });
      controls.update();

      const zoomBefore = orthoCam.zoom;
      // Zoom in (negative delta = zoom in for ortho)
      (controls as any).zoomDelta = -0.5;
      controls.update();

      expect(orthoCam.zoom).not.toBeCloseTo(zoomBefore);
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
    it('should rotate when autoRotate is enabled', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        autoRotate: true,
        autoRotateSpeed: 50, // Very fast for test visibility
      });
      controls.update(); // initial

      const posBefore = camera.position.clone();
      // Run multiple frames to accumulate rotation
      for (let i = 0; i < 10; i++) controls.update();

      const moved = camera.position.distanceTo(posBefore);
      expect(moved).toBeGreaterThan(0.01);
    });

    it('should not rotate when autoRotate is disabled', () => {
      controls = new LuxarOrbitControls(camera, domElement, {
        enableDamping: false,
        autoRotate: false,
      });
      controls.update();

      const posBefore = camera.position.clone();
      controls.update();

      expect(camera.position.x).toBeCloseTo(posBefore.x);
      expect(camera.position.y).toBeCloseTo(posBefore.y);
      expect(camera.position.z).toBeCloseTo(posBefore.z);
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

      // Reset
      controls.reset();
      expect(camera.position.x).toBeCloseTo(savedPos.x, 1);
      expect(camera.position.y).toBeCloseTo(savedPos.y, 1);
      expect(camera.position.z).toBeCloseTo(savedPos.z, 1);
    });
  });

  describe('dispose', () => {
    it('should not throw on dispose', () => {
      controls = new LuxarOrbitControls(camera, domElement);
      expect(() => controls.dispose()).not.toThrow();
    });

    it('should clean up ortho view-axis rotation handler', () => {
      controls = new LuxarOrbitControls(camera, domElement);
      controls.enableViewAxisRotation();
      expect(() => controls.dispose()).not.toThrow();
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
