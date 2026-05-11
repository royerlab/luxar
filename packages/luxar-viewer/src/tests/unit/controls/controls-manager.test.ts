/**
 * Unit tests for ControlsManager
 *
 * Tests the camera control switching system, state preservation,
 * and configuration management.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { ControlsManager } from '../../../controls/controls-manager';
import { LuxarOrbitControls } from '../../../controls/luxar-orbit-controls';
import { LuxarFlyControls } from '../../../controls/luxar-fly-controls';
import { createTestCamera } from '../../test-config';

describe('ControlsManager', () => {
  let camera: THREE.PerspectiveCamera;
  let domElement: HTMLElement;
  let controlsManager: ControlsManager;
  let scene: THREE.Scene;

  beforeEach(() => {
    // Create camera using test config
    camera = createTestCamera(1);
    camera.position.set(0, 0, 5); // Override position for specific test needs

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
      expect(controlsManager.getControls()).toBeInstanceOf(LuxarOrbitControls);
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
      expect(controlsManager.getControls()).toBeInstanceOf(LuxarOrbitControls);
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
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.autoRotateSpeed).toBe(2.5);
    });

    it('should enable/disable zoom', () => {
      controlsManager.setEnableZoom(false);
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.enableZoom).toBe(false);

      controlsManager.setEnableZoom(true);
      expect(controls.enableZoom).toBe(true);
    });

    describe('natural drag (LEFT ↔ RIGHT swap)', () => {
      it('swaps mouseButtons live when toggled on for an active orbit controls', () => {
        const controls = controlsManager.getControls() as LuxarOrbitControls;
        // Establish baseline: default mapping (LEFT=PAN, RIGHT=ROTATE).
        controlsManager.setNaturalDrag(false);
        expect(controls.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
        expect(controls.mouseButtons.RIGHT).toBe(THREE.MOUSE.ROTATE);

        controlsManager.setNaturalDrag(true);
        expect(controls.mouseButtons.LEFT).toBe(THREE.MOUSE.ROTATE);
        expect(controls.mouseButtons.MIDDLE).toBe(THREE.MOUSE.DOLLY);
        expect(controls.mouseButtons.RIGHT).toBe(THREE.MOUSE.PAN);
        expect(controlsManager.getNaturalDrag()).toBe(true);
      });

      it('reverts the swap when toggled off again', () => {
        controlsManager.setNaturalDrag(true);
        controlsManager.setNaturalDrag(false);
        const controls = controlsManager.getControls() as LuxarOrbitControls;
        expect(controls.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
        expect(controls.mouseButtons.RIGHT).toBe(THREE.MOUSE.ROTATE);
      });

      it('survives orbit→fly→orbit mode switch (stored value re-applied)', () => {
        controlsManager.setNaturalDrag(true);
        controlsManager.setControlType('fly');
        controlsManager.setControlType('orbit');
        const controls = controlsManager.getControls() as LuxarOrbitControls;
        expect(controls.mouseButtons.LEFT).toBe(THREE.MOUSE.ROTATE);
        expect(controls.mouseButtons.RIGHT).toBe(THREE.MOUSE.PAN);
      });

      it('does NOT mutate ortho mouseButtons (RIGHT stays null)', () => {
        const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
        orthoCam.position.copy(camera.position);
        controlsManager.setCamera(orthoCam);
        controlsManager.setControlType('ortho');

        controlsManager.setNaturalDrag(true);
        const controls = controlsManager.getControls() as LuxarOrbitControls;
        // Ortho's RIGHT=null mapping must not be touched.
        expect(controls.mouseButtons.RIGHT).toBeNull();
      });
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

      const controls = controlsManager.getControls() as LuxarOrbitControls;
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

    it('should detach forwarded listeners from disposed controls when switching modes', () => {
      const oldControls = controlsManager.getControls() as any;
      const changeHandler = vi.fn();
      controlsManager.addEventListener('change', changeHandler);

      controlsManager.setControlType('fly');
      changeHandler.mockClear();

      oldControls.dispatchEvent({ type: 'change' });

      expect(changeHandler).not.toHaveBeenCalled();
    });
  });

  describe('focus target', () => {
    it('should return orbit target for orbit controls', () => {
      const controls = controlsManager.getControls() as LuxarOrbitControls;
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

  describe('ortho controls', () => {
    it('should switch to ortho controls using LuxarOrbitControls', () => {
      // Ortho needs an OrthographicCamera — create one and set it
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.copy(camera.position);
      controlsManager.setCamera(orthoCam);
      controlsManager.setControlType('ortho');

      expect(controlsManager.getControlType()).toBe('ortho');
      expect(controlsManager.getControls()).toBeInstanceOf(LuxarOrbitControls);
    });

    it('should disable rotation in ortho mode', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.copy(camera.position);
      controlsManager.setCamera(orthoCam);
      controlsManager.setControlType('ortho');

      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.enableRotate).toBe(false);
      expect(controls.enableZoom).toBe(true);
      expect(controls.screenSpacePanning).toBe(true);
    });

    it('should preserve camera target when switching to ortho and back', () => {
      // Set a specific target in orbit mode
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      controls.target.set(1, 2, 3);

      // Switch to ortho
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.copy(camera.position);
      controlsManager.setCamera(orthoCam);
      controlsManager.setControlType('ortho');

      // Switch back to orbit
      controlsManager.setCamera(camera);
      controlsManager.setControlType('orbit');

      const orbitControls = controlsManager.getControls() as LuxarOrbitControls;
      expect(orbitControls.target.x).toBeCloseTo(1);
      expect(orbitControls.target.y).toBeCloseTo(2);
      expect(orbitControls.target.z).toBeCloseTo(3);
    });

    it('should update camera reference via setCamera', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      controlsManager.setCamera(orthoCam);

      // Internal camera should be updated
      expect((controlsManager as any).camera).toBe(orthoCam);
    });

    it('should dispatch change event when switching to ortho', () => {
      const changeHandler = vi.fn();
      controlsManager.addEventListener('change', changeHandler);

      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      controlsManager.setCamera(orthoCam);
      controlsManager.setControlType('ortho');

      expect(changeHandler).toHaveBeenCalledWith(expect.objectContaining({ controlType: 'ortho' }));
    });
  });
});
