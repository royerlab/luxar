/**
 * Unit tests for ControlsManager
 *
 * Tests the camera control switching system, state preservation,
 * and configuration management.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { ControlsManager } from '../controls/controls-manager';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { LuxarFlyControls } from '../controls/luxar-fly-controls';

describe('ControlsManager', () => {
  let camera: THREE.PerspectiveCamera;
  let domElement: HTMLElement;
  let controlsManager: ControlsManager;
  let scene: THREE.Scene;

  beforeEach(() => {
    // Create mock camera
    camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);

    // Create mock DOM element
    domElement = document.createElement('div');
    domElement.style.width = '800px';
    domElement.style.height = '600px';
    document.body.appendChild(domElement);

    // Create scene
    scene = new THREE.Scene();

    // Create controls manager
    controlsManager = new ControlsManager(camera, domElement, scene);
  });

  afterEach(() => {
    // Clean up
    controlsManager.dispose();
    document.body.removeChild(domElement);
  });

  describe('initialization', () => {
    it('should initialize with orbit controls by default', () => {
      expect(controlsManager.getControlType()).toBe('orbit');
      expect(controlsManager.getControls()).toBeInstanceOf(OrbitControls);
    });

    it('should have correct initial configuration', () => {
      const controls = controlsManager.getControls();
      expect(controls).toBeTruthy();
      expect(controls!.enabled).toBe(true);
    });
  });

  describe('control type switching', () => {
    it('should switch from orbit to fly controls', () => {
      controlsManager.setControlType('fly');
      expect(controlsManager.getControlType()).toBe('fly');
      expect(controlsManager.getControls()).toBeInstanceOf(LuxarFlyControls);
    });

    it('should switch from fly back to orbit controls', () => {
      controlsManager.setControlType('fly');
      controlsManager.setControlType('orbit');
      expect(controlsManager.getControlType()).toBe('orbit');
      expect(controlsManager.getControls()).toBeInstanceOf(OrbitControls);
    });

    it('should not recreate controls if already using the same type', () => {
      const initialControls = controlsManager.getControls();
      controlsManager.setControlType('orbit'); // Same type
      expect(controlsManager.getControls()).toBe(initialControls);
    });

    it('should dispatch change event when switching controls', () => {
      const changeHandler = vi.fn();
      controlsManager.addEventListener('change', changeHandler);

      controlsManager.setControlType('fly');

      expect(changeHandler).toHaveBeenCalledWith(expect.objectContaining({ controlType: 'fly' }));
    });

    it('should preserve camera state when switching', () => {
      // Set camera to a specific position
      camera.position.set(10, 20, 30);
      camera.rotation.set(0.1, 0.2, 0.3);

      // Switch controls
      controlsManager.setControlType('fly');

      // Camera position should be preserved
      expect(camera.position.x).toBeCloseTo(10);
      expect(camera.position.y).toBeCloseTo(20);
      expect(camera.position.z).toBeCloseTo(30);
    });
  });

  describe('orbit controls configuration', () => {
    it('should set auto-rotation', () => {
      controlsManager.setAutoRotate(true);
      expect(controlsManager.getAutoRotate()).toBe(true);

      controlsManager.setAutoRotate(false);
      expect(controlsManager.getAutoRotate()).toBe(false);
    });

    it('should set auto-rotation speed', () => {
      controlsManager.setAutoRotateSpeed(2.5);
      const controls = controlsManager.getControls() as OrbitControls;
      expect(controls.autoRotateSpeed).toBe(2.5);
    });

    it('should enable/disable zoom', () => {
      controlsManager.setEnableZoom(false);
      const controls = controlsManager.getControls() as OrbitControls;
      expect(controls.enableZoom).toBe(false);

      controlsManager.setEnableZoom(true);
      expect(controls.enableZoom).toBe(true);
    });
  });

  describe('fly controls configuration', () => {
    beforeEach(() => {
      controlsManager.setControlType('fly');
    });

    it('should set movement speed', () => {
      controlsManager.setFlyMovementSpeed(10);
      const controls = controlsManager.getControls() as LuxarFlyControls;
      expect(controls.movementSpeed).toBe(10);
    });

    it('should set inertial mode', () => {
      controlsManager.setFlyInertialMode(true);
      const controls = controlsManager.getControls() as LuxarFlyControls;
      expect(controls.inertialMode).toBe(true);

      controlsManager.setFlyInertialMode(false);
      expect(controls.inertialMode).toBe(false);
    });

    it('should set damping', () => {
      controlsManager.setFlyDamping(0.95);
      const controls = controlsManager.getControls() as LuxarFlyControls;
      expect(controls.damping).toBe(0.95);
    });

    it('should return fly controls when active', () => {
      const flyControls = controlsManager.getFlyControls();
      expect(flyControls).toBeInstanceOf(LuxarFlyControls);
    });

    it('should return null for fly controls when orbit is active', () => {
      controlsManager.setControlType('orbit');
      const flyControls = controlsManager.getFlyControls();
      expect(flyControls).toBeNull();
    });
  });

  describe('general control methods', () => {
    it('should enable/disable controls', () => {
      controlsManager.setEnabled(false);
      expect(controlsManager.getControls()!.enabled).toBe(false);

      controlsManager.setEnabled(true);
      expect(controlsManager.getControls()!.enabled).toBe(true);
    });

    it('should reset controls', () => {
      const resetSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.reset = resetSpy;

      controlsManager.reset();
      expect(resetSpy).toHaveBeenCalled();
    });

    it('should save control state', () => {
      const saveStateSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.saveState = saveStateSpy;

      controlsManager.saveState();
      expect(saveStateSpy).toHaveBeenCalled();
    });

    it('should handle lookAt for orbit controls', () => {
      const target = new THREE.Vector3(1, 2, 3);
      controlsManager.lookAt(target);

      const controls = controlsManager.getControls() as OrbitControls;
      expect(controls.target.x).toBe(1);
      expect(controls.target.y).toBe(2);
      expect(controls.target.z).toBe(3);
    });

    it('should handle lookAt for fly controls', () => {
      controlsManager.setControlType('fly');
      const lookAtSmoothSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.lookAtSmooth = lookAtSmoothSpy;

      const target = new THREE.Vector3(1, 2, 3);
      controlsManager.lookAt(target, true);

      expect(lookAtSmoothSpy).toHaveBeenCalledWith(target, 0.9);
    });
  });

  describe('update loop', () => {
    it('should update orbit controls', () => {
      const updateSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.update = updateSpy;

      controlsManager.update();
      expect(updateSpy).toHaveBeenCalled();
    });

    it('should update fly controls with delta time', () => {
      controlsManager.setControlType('fly');
      const updateSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.update = updateSpy;

      controlsManager.update();
      expect(updateSpy).toHaveBeenCalledWith(expect.any(Number));
    });

    it('should handle null controls gracefully', () => {
      // Force null controls
      (controlsManager as any).currentControls = null;

      // Should not throw
      expect(() => controlsManager.update()).not.toThrow();
    });
  });

  describe('event handling', () => {
    it('should forward change events from controls', () => {
      const changeHandler = vi.fn();
      controlsManager.addEventListener('change', changeHandler);

      const controls = controlsManager.getControls() as any;
      controls.dispatchEvent({ type: 'change' });

      expect(changeHandler).toHaveBeenCalled();
    });

    it('should forward start events from controls', () => {
      const startHandler = vi.fn();
      controlsManager.addEventListener('start', startHandler);

      const controls = controlsManager.getControls() as any;
      controls.dispatchEvent({ type: 'start' });

      expect(startHandler).toHaveBeenCalled();
    });

    it('should forward end events from controls', () => {
      const endHandler = vi.fn();
      controlsManager.addEventListener('end', endHandler);

      const controls = controlsManager.getControls() as any;
      controls.dispatchEvent({ type: 'end' });

      expect(endHandler).toHaveBeenCalled();
    });
  });

  describe('focus target', () => {
    it('should return orbit target for orbit controls', () => {
      const controls = controlsManager.getControls() as OrbitControls;
      controls.target.set(5, 10, 15);

      const target = controlsManager.getFocusTarget();
      expect(target.x).toBe(5);
      expect(target.y).toBe(10);
      expect(target.z).toBe(15);
    });

    it('should return point in front of camera for fly controls', () => {
      controlsManager.setControlType('fly');
      camera.lookAt(0, 0, -1); // Look along -Z axis

      const target = controlsManager.getFocusTarget();
      // Should be in front of camera
      expect(target.z).toBeLessThan(camera.position.z);
    });
  });

  describe('cleanup', () => {
    it('should dispose controls properly', () => {
      const disposeSpy = vi.fn();
      const controls = controlsManager.getControls() as any;
      controls.dispose = disposeSpy;

      controlsManager.dispose();
      expect(disposeSpy).toHaveBeenCalled();
    });

    it('should stop clock on dispose', () => {
      const clock = (controlsManager as any).clock;
      const stopSpy = vi.spyOn(clock, 'stop');

      controlsManager.dispose();
      expect(stopSpy).toHaveBeenCalled();
    });
  });
});
