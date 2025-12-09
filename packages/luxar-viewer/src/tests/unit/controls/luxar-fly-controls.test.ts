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

describe('LuxarFlyControls', () => {
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
    controls.dispose();
    document.body.removeChild(domElement);
  });

  describe('initialization', () => {
    it('should initialize with default configuration', () => {
      expect(controls.enabled).toBe(true);
      expect(controls.movementSpeed).toBe(config.controls.fly.movement.speed.default);
      expect(controls.lookSpeed).toBe(config.controls.fly.look.mouseSpeed.default);
      expect(controls.inertialMode).toBe(true); // Default is now true
      expect(controls.damping).toBe(config.controls.fly.movement.damping.default);
      expect(controls.acceleration).toBe(config.controls.fly.movement.acceleration.default);
    });

    it('should accept custom configuration', () => {
      const customControls = new LuxarFlyControls(camera, domElement, {
        movementSpeed: 10,
        lookSpeed: 0.005,
        inertialMode: true,
        damping: 0.95,
        acceleration: 1.0,
      });

      expect(customControls.movementSpeed).toBe(10);
      expect(customControls.lookSpeed).toBe(0.005);
      expect(customControls.inertialMode).toBe(true);
      expect(customControls.damping).toBe(0.95);
      expect(customControls.acceleration).toBe(1.0);

      customControls.dispose();
    });

    it('should initialize camera orientation from current camera state', () => {
      // Set camera to look in a specific direction
      camera.lookAt(1, 0, 0);

      const newControls = new LuxarFlyControls(camera, domElement);

      // Controls should maintain camera direction
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      expect(forward.x).toBeGreaterThan(0);

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
    it('should handle mouse drag for camera rotation', () => {
      const mouseDown = new MouseEvent('mousedown', {
        button: 0,
        clientX: 100,
        clientY: 100,
      });
      (controls as any).onMouseDown(mouseDown);

      expect((controls as any).isMouseDown).toBe(true);

      const mouseMove = new MouseEvent('mousemove', {
        clientX: 150,
        clientY: 120,
      });
      (controls as any).onMouseMove(mouseMove);

      // Should update look angles
      expect((controls as any).lon).not.toBe(0);
      expect((controls as any).lat).not.toBe(0);

      const mouseUp = new MouseEvent('mouseup', { button: 0 });
      (controls as any).onMouseUp(mouseUp);

      expect((controls as any).isMouseDown).toBe(false);
    });

    it('should apply angular velocity on mouse movement', () => {
      // Set up mouse drag state
      (controls as any).isMouseDown = true;
      (controls as any).mouseX = 100;
      (controls as any).mouseY = 100;

      // Move mouse to trigger angular velocity
      const mouseMove = new MouseEvent('mousemove', {
        clientX: 200,
        clientY: 200,
      });
      (controls as any).onMouseMove(mouseMove);

      // Check that angular velocity was applied (not zero)
      expect((controls as any).angularVelocity.length()).toBeGreaterThan(0);
    });

    it('should dispatch events for mouse interaction', () => {
      const startHandler = vi.fn();
      const endHandler = vi.fn();
      const changeHandler = vi.fn();

      controls.addEventListener('start', startHandler);
      controls.addEventListener('end', endHandler);
      controls.addEventListener('change', changeHandler);

      const mouseDown = new MouseEvent('mousedown', { button: 0, clientX: 100, clientY: 100 });
      (controls as any).onMouseDown(mouseDown);
      expect(startHandler).toHaveBeenCalled();

      const mouseMove = new MouseEvent('mousemove', { clientX: 150, clientY: 120 });
      (controls as any).onMouseMove(mouseMove);
      expect(changeHandler).toHaveBeenCalled();

      const mouseUp = new MouseEvent('mouseup', { button: 0 });
      (controls as any).onMouseUp(mouseUp);
      expect(endHandler).toHaveBeenCalled();
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
    it('should update camera look direction with arrow keys', () => {
      (controls as any).lookState.horizontal = 1; // Look right

      const initialRotation = camera.rotation.y;
      controls.update(0.016);

      // Camera should rotate
      expect(camera.rotation.y).not.toBe(initialRotation);
    });

    it('should handle smooth look at target', () => {
      const target = new THREE.Vector3(10, 0, 0);

      controls.lookAtSmooth(target, 0.5);

      // Should partially rotate toward target
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);

      // Should be looking more toward the target
      expect(forward.x).toBeGreaterThan(0);
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
    it('should support external input management', () => {
      controls.setExternalInputManagement(true);

      // Should remove internal event listeners
      expect((controls as any).externalInputManagement).toBe(true);

      // Can still handle events when called directly
      const event = new KeyboardEvent('keydown', { key: 'w' });
      controls.handleKeyDown(event);

      const moveState = (controls as any).moveState;
      expect(moveState.forward).toBe(1);
    });

    it('should restore internal management', () => {
      controls.setExternalInputManagement(true);
      controls.setExternalInputManagement(false);

      expect((controls as any).externalInputManagement).toBe(false);
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

    it('should handle strafe movement', () => {
      (controls as any).moveState.right = 1;

      const initialX = camera.position.x;
      controls.update(0.016);

      // Should move right relative to camera
      expect(camera.position.x).not.toBe(initialX);
    });

    it('should handle vertical movement', () => {
      (controls as any).moveState.up = 1;

      const initialY = camera.position.y;
      controls.update(0.016);

      // Should move up
      expect(camera.position.y).toBeGreaterThan(initialY);
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
    it('should roll left with Q key', () => {
      // Simulate Q key press
      const keyEvent = new KeyboardEvent('keydown', { key: 'q' });
      window.dispatchEvent(keyEvent);

      const initialOrientation = (controls as any).orientation.clone();

      // Update with time delta
      controls.update(0.1);

      // Orientation should have changed (roll applied)
      const newOrientation = (controls as any).orientation;
      expect(newOrientation.equals(initialOrientation)).toBe(false);

      // Clean up
      const keyUpEvent = new KeyboardEvent('keyup', { key: 'q' });
      window.dispatchEvent(keyUpEvent);
    });

    it('should roll right with E key', () => {
      // Simulate E key press
      const keyEvent = new KeyboardEvent('keydown', { key: 'e' });
      window.dispatchEvent(keyEvent);

      const initialOrientation = (controls as any).orientation.clone();

      // Update with time delta
      controls.update(0.1);

      // Orientation should have changed (roll applied)
      const newOrientation = (controls as any).orientation;
      expect(newOrientation.equals(initialOrientation)).toBe(false);

      // Clean up
      const keyUpEvent = new KeyboardEvent('keyup', { key: 'e' });
      window.dispatchEvent(keyUpEvent);
    });

    it('should combine roll with other rotations', () => {
      // Simulate Q (roll) + mouse look
      const qKey = new KeyboardEvent('keydown', { key: 'q' });
      window.dispatchEvent(qKey);

      // Add some angular velocity via lookState
      (controls as any).lookState.horizontal = 0.1;
      (controls as any).lookState.vertical = 0.05;

      const initialOrientation = (controls as any).orientation.clone();

      controls.update(0.016);

      // Should apply both roll and look rotation
      const newOrientation = (controls as any).orientation;
      expect(newOrientation.equals(initialOrientation)).toBe(false);

      // Clean up
      const qKeyUp = new KeyboardEvent('keyup', { key: 'q' });
      window.dispatchEvent(qKeyUp);
      (controls as any).lookState.horizontal = 0;
      (controls as any).lookState.vertical = 0;
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

    it('should apply damping correctly with different delta times', () => {
      controls.damping = 0.9; // 90% damping (direct property access)
      (controls as any).velocity.set(10, 0, 0); // Initial velocity

      // Update with delta time
      controls.update(0.016); // One frame at 60fps

      // Velocity should decay according to: v * damping^(delta*60)
      // Formula: v_new = v_old * (0.9)^(0.016*60) = v_old * (0.9)^0.96
      const expectedDecay = Math.pow(0.9, 0.016 * 60);
      const actualVelocity = (controls as any).velocity.x;

      expect(actualVelocity).toBeCloseTo(10 * expectedDecay, 1);
    });
  });

  describe('state management', () => {
    it('should save state', () => {
      // Should not throw
      expect(() => controls.saveState()).not.toThrow();
    });
  });

  describe('cleanup', () => {
    it('should remove event listeners on dispose', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      controls.dispose();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('keyup', expect.any(Function));
    });
  });
});
