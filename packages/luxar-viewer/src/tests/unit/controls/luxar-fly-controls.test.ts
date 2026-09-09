// @vitest-environment jsdom
/**
 * Unit tests for LuxarFlyControls
 *
 * Tests the custom fly control implementation including movement,
 * inertial physics, and keyboard/mouse input handling.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { LuxarFlyControls } from '../../../controls/luxar-fly-controls';
import { config } from '../../../config';
import { createTestCamera } from '../../test-config';

function makePointerEvent(
  type: string,
  init: PointerEventInit & { pointerId: number; pointerType: string }
): PointerEvent {
  return new PointerEvent(type, { cancelable: true, ...init });
}

describe('LuxarFlyControls', () => {
  const defaultWheelZoomSensitivity = config.controls.wheelZoomSensitivity;
  let camera: THREE.PerspectiveCamera;
  let domElement: HTMLElement;
  let controls: LuxarFlyControls;

  beforeEach(() => {
    // Create camera using test config
    camera = createTestCamera(1);
    camera.position.set(0, 0, 5); // Override for test
    camera.lookAt(0, 0, 0);

    // Create mock DOM element
    domElement = document.createElement('div');
    domElement.style.width = '800px';
    domElement.style.height = '600px';
    document.body.appendChild(domElement);

    // Create controls
    controls = new LuxarFlyControls(camera, domElement);
  });

  afterEach(() => {
    config.controls.wheelZoomSensitivity = defaultWheelZoomSensitivity;
    controls.dispose();
    document.body.removeChild(domElement);
  });

  describe('wheel input', () => {
    it('reads the live wheel sensitivity from config through the public event path', () => {
      const velocityFor = (wheelZoomSensitivity: number): number => {
        config.controls.wheelZoomSensitivity = wheelZoomSensitivity;
        (controls as any).velocity.set(0, 0, 0);
        domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: -100 }));
        return (controls as any).velocity.length();
      };

      const defaultVelocity = velocityFor(1);
      expect(velocityFor(0.25)).toBeCloseTo(defaultVelocity * 0.25, 10);
    });
  });

  describe('initialization', () => {
    it('should initialize with default configuration', () => {
      expect(controls.enabled).toBe(true);
      expect(controls.movementSpeed).toBe(config.controls.fly.movement.speed.default);
      expect(controls.lookSpeed).toBe(config.controls.fly.look.mouseSpeed.default);
      expect(controls.inertialMode).toBe(true); // Default is now true
      expect(controls.damping).toBe(config.controls.fly.movement.damping.default);
    });

    it('should accept custom configuration', () => {
      const customControls = new LuxarFlyControls(camera, domElement, {
        movementSpeed: 10,
        lookSpeed: 0.005,
        inertialMode: true,
        damping: 0.95,
      });

      expect(customControls.movementSpeed).toBe(10);
      expect(customControls.lookSpeed).toBe(0.005);
      expect(customControls.inertialMode).toBe(true);
      expect(customControls.damping).toBe(0.95);

      customControls.dispose();
    });

    it('initializes orientation quaternion exactly from current camera quaternion (W4)', () => {
      // W4 strengthening: was `forward.x > 0`. The contract is that
      // controls.orientation === camera.quaternion at construction time
      // (initializeFromCamera in luxar-fly-controls/camera-application.ts
      // does `orientation.copy(camera.quaternion)`). Assert the full
      // quaternion identity, not just a direction inequality.
      camera.position.set(0, 0, 5);
      camera.lookAt(1, 0, 0);
      camera.updateMatrixWorld();
      const expectedQuat = camera.quaternion.clone();

      const newControls = new LuxarFlyControls(camera, domElement);

      const actualQuat = (newControls as any).orientation as THREE.Quaternion;
      expect(actualQuat.x).toBeCloseTo(expectedQuat.x, 5);
      expect(actualQuat.y).toBeCloseTo(expectedQuat.y, 5);
      expect(actualQuat.z).toBeCloseTo(expectedQuat.z, 5);
      expect(actualQuat.w).toBeCloseTo(expectedQuat.w, 5);

      newControls.dispose();
    });
  });

  describe('keyboard input handling', () => {
    it('should handle WASD movement keys', () => {
      const event = new KeyboardEvent('keydown', { key: 'w' });
      controls.handleKeyDown(event);

      // Should register forward movement
      const moveState = (controls as any).moveState;
      expect(moveState.forward).toBe(1);

      const upEvent = new KeyboardEvent('keyup', { key: 'w' });
      controls.handleKeyUp(upEvent);
      expect(moveState.forward).toBe(0);
    });

    it('should handle Alt+W/S for vertical movement', () => {
      const event = new KeyboardEvent('keydown', { key: 'w', altKey: true });
      controls.handleKeyDown(event);

      const moveState = (controls as any).moveState;
      expect(moveState.up).toBe(1);
      expect(moveState.forward).toBe(0);
    });

    it('should handle arrow keys for camera look', () => {
      const event = new KeyboardEvent('keydown', { key: 'ArrowUp' });
      controls.handleKeyDown(event);

      const lookState = (controls as any).lookState;
      expect(lookState.vertical).toBe(-1); // Look up

      const upEvent = new KeyboardEvent('keyup', { key: 'ArrowUp' });
      controls.handleKeyUp(upEvent);
      expect(lookState.vertical).toBe(0);
    });

    it('should not respond when disabled', () => {
      controls.enabled = false;

      const event = new KeyboardEvent('keydown', { key: 'w' });
      controls.handleKeyDown(event);

      const moveState = (controls as any).moveState;
      expect(moveState.forward).toBe(0);
    });

    it('should dispatch change event on key press', () => {
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);

      const event = new KeyboardEvent('keydown', { key: 'w' });
      controls.handleKeyDown(event);

      expect(changeHandler).toHaveBeenCalled();
    });
  });

  describe('mouse input handling', () => {
    // controls.md C1-C4 fix: replace private `(controls as any).onMouseDown/Move/Up(e)`
    // calls with real DOM dispatch. luxar-fly-controls/listeners.ts wires
    // `mousedown` on `domElement` and `mousemove`/`mouseup` on `window`, so
    // dispatching to those targets exercises the public listener-attachment
    // path — a regression in `attachListeners` (e.g. wrong target, removed
    // listener) is now caught here instead of slipping through.
    it('should handle right-drag for camera rotation', () => {
      domElement.dispatchEvent(
        new MouseEvent('mousedown', { button: 2, clientX: 100, clientY: 100 })
      );

      expect((controls as any).activeMouseAction).toBe('rotate');

      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 120 }));

      // Should apply angular velocity for rotation
      expect((controls as any).angularVelocity.length()).toBeGreaterThan(0);

      window.dispatchEvent(new MouseEvent('mouseup', { button: 2 }));

      expect((controls as any).activeMouseAction).toBe('none');
    });

    it('right-drag mousemove produces angular velocity via the full pipeline', () => {
      // controls.md C1 fix: drive the full pipeline via real MouseEvents so
      // the public listener-attachment path is exercised. angularVelocity is
      // read via `as any` because it's the documented observable — the
      // alternative (calling update() and inspecting camera quaternion) is
      // covered separately.
      domElement.dispatchEvent(
        new MouseEvent('mousedown', { button: 2, clientX: 100, clientY: 100 })
      );
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 200, clientY: 200 }));

      expect((controls as any).angularVelocity.length()).toBeGreaterThan(0);
    });

    it('should handle left-drag for strafing', () => {
      domElement.dispatchEvent(
        new MouseEvent('mousedown', { button: 0, clientX: 100, clientY: 100 })
      );

      expect((controls as any).activeMouseAction).toBe('strafe');

      window.dispatchEvent(new MouseEvent('mouseup', { button: 0 }));

      expect((controls as any).activeMouseAction).toBe('none');
    });

    it('should dispatch events for mouse interaction', () => {
      const startHandler = vi.fn();
      const endHandler = vi.fn();
      const changeHandler = vi.fn();

      controls.addEventListener('start', startHandler);
      controls.addEventListener('end', endHandler);
      controls.addEventListener('change', changeHandler);

      // Right-drag dispatched through the real DOM listener wiring.
      domElement.dispatchEvent(
        new MouseEvent('mousedown', { button: 2, clientX: 100, clientY: 100 })
      );
      expect(startHandler).toHaveBeenCalled();

      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 150, clientY: 120 }));
      expect(changeHandler).toHaveBeenCalled();

      window.dispatchEvent(new MouseEvent('mouseup', { button: 2 }));
      expect(endHandler).toHaveBeenCalled();
    });
  });

  describe('touch input handling', () => {
    it('drives touch gestures through DOM listeners and finishes tracked pointers', () => {
      const initialPosition = camera.position.clone();
      const initialOrientation = camera.quaternion.clone();
      const startHandler = vi.fn();
      const endHandler = vi.fn();
      const changeHandler = vi.fn();
      controls.addEventListener('start', startHandler);
      controls.addEventListener('end', endHandler);
      controls.addEventListener('change', changeHandler);

      const down = makePointerEvent('pointerdown', {
        pointerId: 1,
        pointerType: 'touch',
        clientX: 100,
        clientY: 100,
      });
      domElement.dispatchEvent(down);
      window.dispatchEvent(
        makePointerEvent('pointermove', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 130,
          clientY: 120,
        })
      );
      domElement.dispatchEvent(
        makePointerEvent('pointerdown', {
          pointerId: 2,
          pointerType: 'touch',
          clientX: 200,
          clientY: 100,
        })
      );
      window.dispatchEvent(
        makePointerEvent('pointermove', {
          pointerId: 2,
          pointerType: 'mouse',
          clientX: 220,
          clientY: 110,
        })
      );
      controls.update(0.016);

      expect(down.defaultPrevented).toBe(true);
      expect(startHandler).toHaveBeenCalledTimes(1);
      expect(changeHandler).toHaveBeenCalled();
      expect(camera.position.distanceTo(initialPosition)).toBeGreaterThan(0);
      expect(camera.quaternion.angleTo(initialOrientation)).toBeGreaterThan(0);

      window.dispatchEvent(makePointerEvent('pointerup', { pointerId: 2, pointerType: 'mouse' }));
      window.dispatchEvent(makePointerEvent('pointerup', { pointerId: 1, pointerType: 'mouse' }));
      expect(endHandler).toHaveBeenCalledTimes(1);
    });

    it('settles residual touch-look motion after a drifting tap', () => {
      domElement.dispatchEvent(
        makePointerEvent('pointerdown', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 100,
          clientY: 100,
        })
      );
      window.dispatchEvent(
        makePointerEvent('pointermove', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 110,
          clientY: 100,
        })
      );
      window.dispatchEvent(
        makePointerEvent('pointerup', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 110,
          clientY: 100,
        })
      );

      expect((controls as any).angularVelocity.length()).toBeGreaterThan(0);
      controls.settleDamping();

      const settledOrientation = camera.quaternion.clone();
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);
      for (let frame = 0; frame < 60; frame++) controls.update(1 / 60);

      expect(changeHandler).not.toHaveBeenCalled();
      expect(camera.quaternion.angleTo(settledOrientation)).toBe(0);
    });

    it('settles angular damping without cancelling translational glide', () => {
      (controls as any).velocity.set(1, 2, 3);
      (controls as any).angularVelocity.set(0.1, 0.2, 0.3);

      controls.settleDamping();

      expect((controls as any).angularVelocity.length()).toBe(0);
      expect((controls as any).velocity.toArray()).toEqual([1, 2, 3]);
    });

    it('leaves mouse-typed pointerdown to the mouse handlers', () => {
      const startHandler = vi.fn();
      controls.addEventListener('start', startHandler);
      const down = makePointerEvent('pointerdown', {
        pointerId: 1,
        pointerType: 'mouse',
        clientX: 100,
        clientY: 100,
      });

      domElement.dispatchEvent(down);

      expect(down.defaultPrevented).toBe(false);
      expect(startHandler).not.toHaveBeenCalled();
      expect((controls as any).touchPointers.size).toBe(0);
    });

    it('ignores pointermove for an id that was never tracked', () => {
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);

      window.dispatchEvent(
        makePointerEvent('pointermove', {
          pointerId: 99,
          pointerType: 'touch',
          clientX: 300,
          clientY: 200,
        })
      );

      expect(changeHandler).not.toHaveBeenCalled();
      expect((controls as any).touchPointers.size).toBe(0);
    });

    it('makes pointermove inert after reset clears an active gesture', () => {
      domElement.dispatchEvent(
        makePointerEvent('pointerdown', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 100,
          clientY: 100,
        })
      );
      controls.reset();
      const resetOrientation = camera.quaternion.clone();
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);

      window.dispatchEvent(
        makePointerEvent('pointermove', {
          pointerId: 1,
          pointerType: 'touch',
          clientX: 300,
          clientY: 250,
        })
      );
      controls.update(0.016);

      expect(changeHandler).not.toHaveBeenCalled();
      expect(camera.quaternion.angleTo(resetOrientation)).toBe(0);
    });
  });

  describe('movement modes', () => {
    it('should handle direct movement mode', () => {
      controls.setInertialMode(false);

      // Set forward movement
      (controls as any).moveState.forward = 1;

      const initialZ = camera.position.z;
      controls.update(0.016); // ~60fps

      // Camera should move forward
      expect(camera.position.z).toBeLessThan(initialZ);

      // Stop movement
      (controls as any).moveState.forward = 0;
      const positionAfterStop = camera.position.clone();

      // Update a few times - with high damping it should stop quickly
      controls.update(0.016);
      controls.update(0.016);
      controls.update(0.016);

      // Should have stopped or nearly stopped (within threshold)
      const movement = camera.position.distanceTo(positionAfterStop);
      expect(movement).toBeLessThan(0.01); // Very small movement due to high damping
    });

    it('should handle inertial movement mode', () => {
      controls.setInertialMode(true);

      // Apply forward acceleration
      (controls as any).moveState.forward = 1;

      controls.update(0.016);

      // Should have velocity
      const velocityBeforeDamping = (controls as any).velocity.length();
      expect(velocityBeforeDamping).toBeGreaterThan(0);

      // Stop acceleration
      (controls as any).moveState.forward = 0;

      // Should continue moving due to inertia
      const positionAfterStop = camera.position.clone();
      controls.update(0.016);

      expect(camera.position.equals(positionAfterStop)).toBe(false);

      // Velocity should decrease due to damping
      const velocityAfterDamping = (controls as any).velocity.length();
      expect(velocityAfterDamping).toBeLessThan(velocityBeforeDamping);
    });

    it('should stop tiny movements below threshold', () => {
      controls.setInertialMode(true);

      // Set very small velocity
      (controls as any).velocity.set(1e-6, 0, 0);

      controls.update(0.016);

      // Should be zeroed out
      expect((controls as any).velocity.length()).toBe(0);
    });

    it('should switch between modes correctly', () => {
      controls.setInertialMode(true);

      // Build up some velocity
      (controls as any).velocity.set(1, 1, 1);

      // Switch to direct mode (high damping)
      controls.setInertialMode(false);

      // Velocity is not cleared immediately but will dampen quickly
      // The mode just changes the damping factor
      expect((controls as any).inertialMode).toBe(false);

      // After several updates with high damping (0.5), velocity should be near zero
      // High damping reduces velocity by ~50% each frame at 60fps
      for (let i = 0; i < 10; i++) {
        controls.update(0.016);
      }

      expect((controls as any).velocity.length()).toBeLessThan(0.01);
    });
  });

  describe('camera orientation', () => {
    it('right-arrow yaw rotates camera forward away from -X (W7 strengthening)', () => {
      // W7 strengthening: was `rotation.y !== initialRotation`. The integrator
      // negates lookState.horizontal when computing yaw torque around local Y,
      // so horizontal=1 (right arrow) produces a NEGATIVE yaw about world Y
      // (camera forward moves from -Z toward +X for an identity-orientation
      // camera). Assert the direction, not just inequality.
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0); // forward = -Z
      camera.updateMatrixWorld();
      const newControls = new LuxarFlyControls(camera, domElement);
      newControls.setInertialMode(false); // direct angular velocity, easier to reason about

      const initialForward = new THREE.Vector3();
      camera.getWorldDirection(initialForward);
      expect(initialForward.x).toBeCloseTo(0, 5);
      expect(initialForward.z).toBeCloseTo(-1, 5);

      (newControls as any).lookState.horizontal = 1; // right arrow
      // Multiple ticks to accumulate measurable rotation past the damping
      // threshold.
      for (let i = 0; i < 5; i++) newControls.update(0.05);

      const newForward = new THREE.Vector3();
      camera.getWorldDirection(newForward);
      // Right-arrow yaw moves the forward vector toward +X (when looking
      // along -Z initially), i.e. forward.x increases past 0.
      expect(newForward.x).toBeGreaterThan(0.01);

      newControls.dispose();
    });

    it('slerps half-way to target when smoothness=0.5 (W5 strengthening)', () => {
      // W5 strengthening: was `forward.x > 0`. lookAtSmooth slerps
      // orientation by `1 - smoothness` toward the lookAt quaternion.
      // With smoothness=0.5, the camera should be ~half-way between
      // initial and the look-at target. Compute the exact expected
      // quaternion (slerp of initial and target).
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0); // forward = -Z
      camera.updateMatrixWorld();
      const startQuat = camera.quaternion.clone();

      // Build the look-at target quaternion the same way the helper does.
      const target = new THREE.Vector3(10, 0, 5); // straight +X from camera
      const lookMat = new THREE.Matrix4().lookAt(
        camera.position,
        target,
        new THREE.Vector3(0, 1, 0)
      );
      const targetQuat = new THREE.Quaternion().setFromRotationMatrix(lookMat);

      // Fresh controls (so orientation = startQuat).
      const fresh = new LuxarFlyControls(camera, domElement);
      fresh.lookAtSmooth(target, 0.5); // → slerp(startQuat, targetQuat, 0.5)

      const expected = startQuat.clone().slerp(targetQuat, 0.5);
      const actual = camera.quaternion;
      // q and -q represent the same rotation; compare via abs(dot).
      const dot = Math.abs(actual.dot(expected));
      expect(dot).toBeCloseTo(1, 5);

      fresh.dispose();
    });

    it('should handle immediate look at target', () => {
      const target = new THREE.Vector3(0, 10, 0);

      controls.lookAtSmooth(target, 0); // No smoothing

      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);

      // Should be looking directly at target
      const toTarget = target.clone().sub(camera.position).normalize();
      expect(forward.x).toBeCloseTo(toTarget.x, 5);
      expect(forward.y).toBeCloseTo(toTarget.y, 5);
      expect(forward.z).toBeCloseTo(toTarget.z, 5);
    });
  });

  describe('external input management', () => {
    it('does not register window keyboard listeners when externalInputManagement is true', () => {
      const addSpy = vi.spyOn(window, 'addEventListener');
      const externalControls = new LuxarFlyControls(camera, domElement, {
        externalInputManagement: true,
      });

      const keyboardEvents = addSpy.mock.calls.filter(
        ([type]) => type === 'keydown' || type === 'keyup'
      );
      expect(keyboardEvents).toHaveLength(0);

      // Forwarded keys still drive movement state.
      externalControls.handleKeyDown(new KeyboardEvent('keydown', { key: 'w' }));
      expect((externalControls as any).moveState.forward).toBe(1);

      externalControls.dispose();
      addSpy.mockRestore();
    });

    it('registers window keyboard listeners when externalInputManagement is false (default)', () => {
      const addSpy = vi.spyOn(window, 'addEventListener');
      const internalControls = new LuxarFlyControls(camera, domElement);

      const keyboardEvents = addSpy.mock.calls.filter(
        ([type]) => type === 'keydown' || type === 'keyup'
      );
      expect(keyboardEvents.length).toBeGreaterThanOrEqual(2);

      internalControls.dispose();
      addSpy.mockRestore();
    });
  });

  describe('update loop', () => {
    it('should dispatch change event when moving', () => {
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);

      // Set movement
      (controls as any).moveState.forward = 1;

      controls.update(0.016);

      expect(changeHandler).toHaveBeenCalled();
    });

    it('should not dispatch change when stationary', () => {
      const changeHandler = vi.fn();
      controls.addEventListener('change', changeHandler);

      // No movement
      controls.update(0.016);

      expect(changeHandler).not.toHaveBeenCalled();
    });

    it('strafe-right moves the camera toward +X for identity-orientation (W6)', () => {
      // W6 strengthening: was `position.x !== initialX`. The integrator
      // moves along the camera's local +X (right axis from orientation).
      // With identity orientation, local +X == world +X.
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();
      const newControls = new LuxarFlyControls(camera, domElement, { movementSpeed: 10 });
      newControls.setInertialMode(false);
      (newControls as any).moveState.right = 1;

      const initialX = camera.position.x;
      // Drive multiple ticks to ensure measurable movement past damping.
      for (let i = 0; i < 5; i++) newControls.update(0.05);

      // Strictly greater than (positive direction).
      expect(camera.position.x).toBeGreaterThan(initialX + 0.001);
      newControls.dispose();
    });

    it('Alt+W vertical-up moves camera toward +Y (W6)', () => {
      // W6 strengthening: vertical strafe uses world-up (0,1,0) regardless
      // of orientation. Assert strictly positive Y delta.
      camera.position.set(0, 0, 5);
      const newControls = new LuxarFlyControls(camera, domElement, { movementSpeed: 10 });
      newControls.setInertialMode(false);
      (newControls as any).moveState.up = 1;

      const initialY = camera.position.y;
      for (let i = 0; i < 5; i++) newControls.update(0.05);

      expect(camera.position.y).toBeGreaterThan(initialY + 0.001);
      newControls.dispose();
    });
  });

  describe('reset functionality', () => {
    it('should reset velocity and movement state', () => {
      controls.setInertialMode(true);

      // Set some state
      (controls as any).velocity.set(1, 2, 3);
      (controls as any).angularVelocity.set(0.1, 0.2, 0.3);
      (controls as any).moveState.forward = 1;
      (controls as any).moveState.right = 1;
      (controls as any).lookState.horizontal = 1;
      (controls as any).lookState.vertical = 1;

      // Modify orientation from identity
      const testQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5, 0.5, 0));
      (controls as any).orientation.copy(testQuat);

      controls.reset();

      // Check velocities are zeroed
      expect((controls as any).velocity.length()).toBe(0);
      expect((controls as any).angularVelocity.length()).toBe(0);

      // Check movement states are zeroed
      expect((controls as any).moveState.forward).toBe(0);
      expect((controls as any).moveState.right).toBe(0);
      expect((controls as any).lookState.horizontal).toBe(0);
      expect((controls as any).lookState.vertical).toBe(0);

      // Check orientation is reset to identity
      const identity = new THREE.Quaternion(0, 0, 0, 1);
      expect((controls as any).orientation.equals(identity)).toBe(true);
    });
  });

  describe('roll controls', () => {
    it('Q and E roll in opposite directions around the viewing axis (W8 strengthening)', () => {
      // W8 strengthening: previously each test only asserted
      // `orientation.equals(initialOrientation) === false`. A sign-flip
      // mutation in physics.ts (`lookState.roll * rotationSpeed`) would
      // survive both tests. Here we exercise BOTH keys and verify that:
      //   1. After Q, the local-up vector tilts in one direction (around -Z).
      //   2. After E, it tilts in the OPPOSITE direction.
      // The two final up-vector x-components must have opposite signs.

      // Q controls (fresh).
      const qCam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
      qCam.position.set(0, 0, 5);
      qCam.lookAt(0, 0, 0);
      qCam.updateMatrixWorld();
      const qDom = document.createElement('div');
      document.body.appendChild(qDom);
      const qControls = new LuxarFlyControls(qCam, qDom);
      qControls.setInertialMode(false);
      (qControls as any).lookState.roll = -1; // Q
      for (let i = 0; i < 3; i++) qControls.update(0.05);
      const qUpAfter = new THREE.Vector3(0, 1, 0).applyQuaternion(qCam.quaternion);

      // E controls (fresh).
      const eCam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
      eCam.position.set(0, 0, 5);
      eCam.lookAt(0, 0, 0);
      eCam.updateMatrixWorld();
      const eDom = document.createElement('div');
      document.body.appendChild(eDom);
      const eControls = new LuxarFlyControls(eCam, eDom);
      eControls.setInertialMode(false);
      (eControls as any).lookState.roll = 1; // E
      for (let i = 0; i < 3; i++) eControls.update(0.05);
      const eUpAfter = new THREE.Vector3(0, 1, 0).applyQuaternion(eCam.quaternion);

      // For an initial -Z-facing camera with world-up = +Y, roll around
      // the forward axis tilts the up-vector left or right (along ±X).
      // Q and E must tilt to OPPOSITE sides — sign-flip mutant would
      // produce same-sign outputs and fail this.
      expect(Math.abs(qUpAfter.x)).toBeGreaterThan(0.01);
      expect(Math.abs(eUpAfter.x)).toBeGreaterThan(0.01);
      expect(Math.sign(qUpAfter.x)).not.toBe(Math.sign(eUpAfter.x));

      qControls.dispose();
      eControls.dispose();
      document.body.removeChild(qDom);
      document.body.removeChild(eDom);
    });

    it('should combine roll with other rotations', () => {
      // controls.md C5 fix: drive Q (roll) AND look state through the
      // orchestrator's public handleKeyDown/handleKeyUp instead of mixing
      // window.dispatchEvent + direct (controls as any).lookState mutation.
      // ArrowLeft/ArrowUp set lookState.horizontal=-1 and vertical=-1 via
      // the same code path the orchestrator wires for real keyboard input.
      //
      // controls.md O6 / Phase E34: wrap the post-assertion keyUp cleanup
      // in a try/finally. Without it, an `expect` failure mid-test left
      // the q/ArrowLeft/ArrowUp keys "held down" in the controls'
      // moveState/lookState and the next test running on the same
      // describe-scoped `controls` instance saw the leaked state. With
      // try/finally, cleanup runs whether the assertion passes or throws.
      controls.handleKeyDown(new KeyboardEvent('keydown', { key: 'q' }));
      controls.handleKeyDown(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
      controls.handleKeyDown(new KeyboardEvent('keydown', { key: 'ArrowUp' }));

      try {
        const initialOrientation = (controls as any).orientation.clone();

        controls.update(0.016);

        // Should apply both roll and look rotation
        const newOrientation = (controls as any).orientation;
        expect(newOrientation.equals(initialOrientation)).toBe(false);
      } finally {
        controls.handleKeyUp(new KeyboardEvent('keyup', { key: 'q' }));
        controls.handleKeyUp(new KeyboardEvent('keyup', { key: 'ArrowLeft' }));
        controls.handleKeyUp(new KeyboardEvent('keyup', { key: 'ArrowUp' }));
      }
    });
  });

  describe('frame-rate independence', () => {
    it('should produce consistent movement at different frame rates', () => {
      // Set up movement
      (controls as any).moveState.forward = 1;

      // Save initial position
      const startPos = camera.position.clone();

      // Simulate 60fps (one frame at 16.67ms)
      const controls60fps = new LuxarFlyControls(camera, domElement);
      camera.position.copy(startPos);
      (controls60fps as any).moveState.forward = 1;
      controls60fps.update(0.016);
      const pos60fps = camera.position.clone();
      controls60fps.dispose();

      // Simulate 30fps (one frame at 33.33ms)
      const controls30fps = new LuxarFlyControls(camera, domElement);
      camera.position.copy(startPos);
      (controls30fps as any).moveState.forward = 1;
      controls30fps.update(0.033);
      const pos30fps = camera.position.clone();
      controls30fps.dispose();

      // Positions should be very close (frame-rate independent physics)
      // Small difference acceptable due to discrete time steps
      const distance = pos60fps.distanceTo(pos30fps);
      expect(distance).toBeLessThan(0.01); // Within 1% tolerance
    });

    it('should handle variable delta times correctly', () => {
      (controls as any).moveState.forward = 1;

      const startPos = camera.position.clone();

      // Apply variable delta times (simulating frame drops)
      controls.update(0.016); // 60fps frame
      controls.update(0.05); // Frame drop
      controls.update(0.016); // Back to normal

      // Should still move forward (no NaN or infinity)
      expect(camera.position.z).toBeLessThan(startPos.z);
      expect(isFinite(camera.position.x)).toBe(true);
      expect(isFinite(camera.position.y)).toBe(true);
      expect(isFinite(camera.position.z)).toBe(true);
    });

    // [controls.md/O3][P9] Was a single-delta test mis-named "with different
    // delta times". Rename + parametrize to actually cover several deltas —
    // the damping formula `v * damping^(delta*60)` is differential in delta.
    it.each([
      { delta: 0.016, label: '60fps' },
      { delta: 0.032, label: '30fps' },
      { delta: 0.008, label: '120fps' },
    ])('damping applies frame-rate-independent decay $label (delta=$delta)', ({ delta }) => {
      const freshControls = new LuxarFlyControls(camera, domElement);
      freshControls.damping = 0.9;
      (freshControls as any).velocity.set(10, 0, 0);

      freshControls.update(delta);

      // v_new = v_old * damping^(delta*60)
      const expectedDecay = Math.pow(0.9, delta * 60);
      const actualVelocity = (freshControls as any).velocity.x;

      expect(actualVelocity).toBeCloseTo(10 * expectedDecay, 1);
      freshControls.dispose();
    });
  });

  describe('state management', () => {
    it('saveState + reset is a roundtrip (camera position + orientation restored)', () => {
      // Strengthened from "should save state / .not.toThrow()" which killed no
      // mutants. The saved fields are observable on the public surface via
      // reset(), so we exercise the full roundtrip.
      const startPos = camera.position.clone();
      const startQuat = camera.quaternion.clone();
      controls.saveState();
      // Mutate camera to force divergence.
      camera.position.set(startPos.x + 50, startPos.y + 50, startPos.z + 50);
      camera.quaternion.identity();
      controls.reset();
      expect(camera.position.x).toBeCloseTo(startPos.x, 5);
      expect(camera.position.y).toBeCloseTo(startPos.y, 5);
      expect(camera.position.z).toBeCloseTo(startPos.z, 5);
      // Quaternion roundtrip (component-wise, accounting for q == -q).
      const dot = Math.abs(camera.quaternion.dot(startQuat));
      expect(dot).toBeCloseTo(1, 5);
    });

    it('constructor saveState() captures camera state at construction time (HIGH-14)', () => {
      // Documents the "saves NOW" semantics: callers that mutate the camera
      // AFTER constructing controls must call saveState() again. Without that
      // explicit call, reset() restores the construction-time baseline (NOT
      // the post-mutation state). This test pins that contract so a future
      // refactor doesn't silently change reset() behaviour.
      camera.position.set(1, 2, 3);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      const c = new LuxarFlyControls(camera, domElement);

      // Caller mutates the camera after construction — they did NOT call
      // saveState() again, so reset() must restore (1, 2, 3), not the
      // post-mutation position.
      camera.position.set(99, 99, 99);
      c.reset();
      expect(camera.position.x).toBeCloseTo(1, 5);
      expect(camera.position.y).toBeCloseTo(2, 5);
      expect(camera.position.z).toBeCloseTo(3, 5);

      // Re-saving after a mutation makes that the new baseline.
      camera.position.set(7, 8, 9);
      c.saveState();
      camera.position.set(0, 0, 0);
      c.reset();
      expect(camera.position.x).toBeCloseTo(7, 5);
      expect(camera.position.y).toBeCloseTo(8, 5);
      expect(camera.position.z).toBeCloseTo(9, 5);

      c.dispose();
    });
  });

  describe('cleanup', () => {
    it('keydown event has no effect after dispose() (no stale listeners)', () => {
      // Strengthened from spy-on-removeEventListener (which only proved that
      // SOME removeEventListener call happened, with no behavioral check).
      // The contract is: after dispose, dispatching a keydown must not mutate
      // moveState. We exercise this by triggering a keydown and confirming
      // velocity is unchanged.
      const velocityBefore = (controls as any).velocity.x;
      controls.dispose();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'd', code: 'KeyD' }));
      controls.update(0.016);
      const velocityAfter = (controls as any).velocity.x;
      expect(velocityAfter).toBe(velocityBefore);
    });
  });
});
