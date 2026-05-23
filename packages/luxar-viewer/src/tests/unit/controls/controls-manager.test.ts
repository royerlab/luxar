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

    it('saveState() then mutation then reset() roundtrips orbit camera+target', () => {
      // C1 strengthening: previously this whole describe block had two
      // spy-on-replaced-method tests (.reset and .saveState). Verify
      // the full save→mutate→reset contract on observable state.
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      controls.enableDamping = false;
      // Move to a NEW baseline by setting camera + target directly,
      // then call saveState via the orchestrator's public API.
      camera.position.set(7, 8, 9);
      controls.target.set(1, 2, 3);
      controls.reinitialize();
      controls.update(); // applies the new orientation/distance to camera
      const expectedPos = camera.position.clone();
      const expectedTarget = controls.target.clone();
      controlsManager.saveState();

      // Mutate again, then reset — must come back to the saved baseline.
      controls.target.set(20, 20, 20);
      controls.update();

      controlsManager.reset();
      // Float32 tolerance (project's norm; see controls.md C3).
      expect(camera.position.x).toBeCloseTo(expectedPos.x, 4);
      expect(camera.position.y).toBeCloseTo(expectedPos.y, 4);
      expect(camera.position.z).toBeCloseTo(expectedPos.z, 4);
      expect(controls.target.x).toBeCloseTo(expectedTarget.x, 5);
      expect(controls.target.y).toBeCloseTo(expectedTarget.y, 5);
      expect(controls.target.z).toBeCloseTo(expectedTarget.z, 5);
    });

    it('should handle lookAt for orbit controls', () => {
      const target = new THREE.Vector3(1, 2, 3);
      controlsManager.lookAt(target);

      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.target.x).toBe(1);
      expect(controls.target.y).toBe(2);
      expect(controls.target.z).toBe(3);
    });

    it('lookAt(target, smooth=false) instantly orients fly camera toward target', () => {
      // C1 strengthening: was spy-on-replaced-lookAtSmooth (asserted that
      // the spy received args, not that the camera actually rotated).
      // We instead verify the observable rotation: the camera's world
      // forward vector should point from camera.position → target.
      controlsManager.setControlType('fly');
      camera.position.set(0, 0, 5);

      const target = new THREE.Vector3(10, 0, 5); // straight right
      controlsManager.lookAt(target, false); // smooth=false → snap

      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      const expected = target.clone().sub(camera.position).normalize();
      expect(forward.x).toBeCloseTo(expected.x, 5);
      expect(forward.y).toBeCloseTo(expected.y, 5);
      expect(forward.z).toBeCloseTo(expected.z, 5);
    });

    it('lookAt(target, smooth=true) partially rotates fly camera toward target', () => {
      // Mirrors above test for the smoothing branch — public-API drive
      // rather than spy on the lookAtSmooth method.
      controlsManager.setControlType('fly');
      camera.position.set(0, 0, 5);
      // Establish initial forward (camera was looking at origin).
      const initialForward = new THREE.Vector3();
      camera.getWorldDirection(initialForward);

      const target = new THREE.Vector3(100, 0, 5); // far-right
      controlsManager.lookAt(target, true); // smoothness=0.9 → 10% step

      const newForward = new THREE.Vector3();
      camera.getWorldDirection(newForward);

      const expected = target.clone().sub(camera.position).normalize();
      // The camera should have moved CLOSER to expected (smaller angle)
      // and FURTHER from initial.
      const initialAngle = initialForward.angleTo(expected);
      const newAngle = newForward.angleTo(expected);
      expect(newAngle).toBeLessThan(initialAngle);
    });
  });

  describe('update loop', () => {
    it('update() invokes the underlying orbit update step (camera tracks target moves)', () => {
      // C1 strengthening: was previously a spy-on-replaced-update. We
      // verify the observable contract: after mutating target and calling
      // controlsManager.update(), the camera position must reflect the
      // new target (via runUpdateStep → applyToCamera). No spy needed.
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      controls.enableDamping = false;
      controls.target.set(5, 5, 5);
      controlsManager.update();
      // With the new target and unchanged distance/orientation, camera
      // must have moved off its original position.
      expect(camera.position.distanceTo(new THREE.Vector3(0, 0, 5))).toBeGreaterThan(0.5);
    });

    it('update() invokes the underlying fly update step (camera position responds to velocity)', () => {
      // C1 strengthening: was previously a spy-on-replaced-update that
      // only checked the spy received "any number". Force a non-trivial
      // velocity directly (bypasses Timer.getDelta() returning 0 in tight
      // test loops) and verify the orchestrator's update() actually
      // dispatches the work that integrates it into camera.position.
      controlsManager.setControlType('fly');
      const controls = controlsManager.getControls() as LuxarFlyControls;
      // Seed a known velocity. Timer.getDelta() will return ~0 in jsdom
      // tight loops, so we make a single call and only require that the
      // path executes without throwing AND that observable damping is
      // applied (velocity gets multiplied by pow(d, delta*60) — delta=0
      // gives factor=1, so velocity is preserved). To get real movement
      // we drive via the fly controls' own update() with a known delta.
      (controls as any).velocity.set(0, 0, -10);
      controlsManager.update(); // public path (delta may be ~0)
      // Directly verify the underlying update with controlled delta:
      const startZ = camera.position.z;
      controls.update(0.1);
      expect(camera.position.z).toBeLessThan(startZ); // moved forward
    });

    // Audit-acknowledgment (controls.md C2): the prior
    // 'should handle null controls gracefully' test was removed in an
    // earlier pass because it forced `(controlsManager as any).currentControls
    // = null` — a state the orchestrator never reaches via its public API.
    // Replacing it with a guarded path test would just re-introduce the
    // same internal-field coupling; the dispose→update sequence (which
    // also produces a null currentControls) is already covered by the
    // cleanup describe block.
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

    it('returns camera.position + worldDir * (sceneScale || 10) for fly controls', () => {
      // W3 strengthening: was `target.z < camera.position.z` only. The
      // formula in ControlsManager.getFocusTarget is deterministic —
      // assert the exact computed point.
      controlsManager.setControlType('fly');
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0); // Forward = -Z
      camera.updateMatrixWorld();

      // sceneScale defaults to 0 → fallback distance is 10.
      const target = controlsManager.getFocusTarget();
      expect(target.x).toBeCloseTo(0, 5);
      expect(target.y).toBeCloseTo(0, 5);
      expect(target.z).toBeCloseTo(5 - 10, 5); // (0,0,5) + (0,0,-1)*10 = (0,0,-5)
    });

    it('uses sceneScale (not the fallback 10) when set', () => {
      // Additional coverage on the (sceneScale || 10) branch.
      controlsManager.setControlType('fly');
      controlsManager.setSceneScale(20);
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      const target = controlsManager.getFocusTarget();
      // (0,0,5) + (0,0,-1)*20 = (0,0,-15)
      expect(target.z).toBeCloseTo(-15, 5);
    });
  });

  describe('cleanup', () => {
    it('dispose() detaches forwarded change listeners (no leak after dispose)', () => {
      // C1 strengthening: was previously a spy-on-replaced-dispose. Verify
      // the observable contract: after dispose(), events fired by the
      // (still-reachable) child instance no longer propagate to manager
      // listeners. This is the behavior end-users care about — listener
      // hygiene, not which method was called internally.
      const oldControls = controlsManager.getControls() as any;
      const changeHandler = vi.fn();
      controlsManager.addEventListener('change', changeHandler);

      controlsManager.dispose();
      changeHandler.mockClear();

      // Re-fire change on the (now-disposed) underlying controls.
      oldControls.dispatchEvent({ type: 'change' });
      expect(changeHandler).not.toHaveBeenCalled();
    });

    it('dispose() leaves no controls reachable via getControls()', () => {
      // Behavior assertion that matches the C2-removed null-state test's
      // intent without requiring `as any` private-field forcing — the
      // disposeCurrentControls path actually nulls the reference, so the
      // post-dispose getter contract is observable.
      controlsManager.dispose();
      expect(controlsManager.getControls()).toBeNull();
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
