/**
 * Unit tests for LuxarFlyControls
 * 
 * Tests the custom fly control implementation including movement,
 * inertial physics, and keyboard/mouse input handling.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { LuxarFlyControls } from '../controls/luxar-fly-controls';
import { CONTROL_CONFIG } from '../controls/control-config';

describe('LuxarFlyControls', () => {
  let camera: THREE.PerspectiveCamera;
  let domElement: HTMLElement;
  let controls: LuxarFlyControls;

  beforeEach(() => {
    // Create mock camera
    camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
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
      expect(controls.movementSpeed).toBe(CONTROL_CONFIG.fly.movement.speed.default);
      expect(controls.lookSpeed).toBe(CONTROL_CONFIG.fly.look.mouseSpeed.default);
      expect(controls.inertialMode).toBe(false);
      expect(controls.damping).toBe(CONTROL_CONFIG.fly.movement.damping.default);
      expect(controls.acceleration).toBe(CONTROL_CONFIG.fly.movement.acceleration.default);
    });

    it('should accept custom configuration', () => {
      const customControls = new LuxarFlyControls(camera, domElement, {
        movementSpeed: 10,
        lookSpeed: 0.005,
        inertialMode: true,
        damping: 0.95,
        acceleration: 1.0
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
        clientY: 100
      });
      (controls as any).onMouseDown(mouseDown);
      
      expect((controls as any).isMouseDown).toBe(true);
      
      const mouseMove = new MouseEvent('mousemove', {
        clientX: 150,
        clientY: 120
      });
      (controls as any).onMouseMove(mouseMove);
      
      // Should update look angles
      expect((controls as any).lon).not.toBe(0);
      expect((controls as any).lat).not.toBe(0);
      
      const mouseUp = new MouseEvent('mouseup', { button: 0 });
      (controls as any).onMouseUp(mouseUp);
      
      expect((controls as any).isMouseDown).toBe(false);
    });

    it('should clamp vertical rotation', () => {
      // Clamping happens in onMouseMove, not updateOrientation
      (controls as any).isMouseDown = true;
      (controls as any).mouseX = 100;
      (controls as any).mouseY = 100;
      
      // Move mouse to trigger lat clamping
      const mouseMove = new MouseEvent('mousemove', {
        clientX: 100,
        clientY: 2000 // Large Y movement
      });
      (controls as any).onMouseMove(mouseMove);
      
      expect((controls as any).lat).toBeLessThanOrEqual(85);
      expect((controls as any).lat).toBeGreaterThanOrEqual(-85);
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
      
      controls.update(0.016);
      
      // Should stop immediately
      expect(camera.position.equals(positionAfterStop)).toBe(true);
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
      
      // Switch to direct mode
      controls.setInertialMode(false);
      
      // Velocity should be cleared
      expect((controls as any).velocity.length()).toBe(0);
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
      (controls as any).moveState.forward = 1;
      (controls as any).moveState.right = 1;
      (controls as any).lat = 45;
      (controls as any).lon = 90;
      
      controls.reset();
      
      expect((controls as any).velocity.length()).toBe(0);
      expect((controls as any).moveState.forward).toBe(0);
      expect((controls as any).moveState.right).toBe(0);
      expect((controls as any).lat).toBe(0);
      expect((controls as any).lon).toBe(0);
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