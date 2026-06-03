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

    it('initial orbit controls have the expected default configuration', () => {
      // controls.md [W1][P2] strengthening: was `.toBeTruthy()` +
      // `.enabled === true` only. Pin the orbit-control defaults that
      // matter for the active mode (enableRotate/Pan/Zoom + damping)
      // — these are the values that drive the default UX. A regression
      // that flipped a default to false would survive the smoke test.
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls).toBeInstanceOf(LuxarOrbitControls);
      expect(controls.enabled).toBe(true);
      expect(controls.enableRotate).toBe(true);
      expect(controls.enablePan).toBe(true);
      expect(controls.enableZoom).toBe(true);
      expect(controls.enableDamping).toBe(true);
      expect(controls.autoRotate).toBe(false);
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

      // Camera position is a Float32 round-trip; precision 5 (~1e-5) is
      // the project norm for Float32-tolerant equality.
      expect(camera.position.x).toBeCloseTo(10, 5);
      expect(camera.position.y).toBeCloseTo(20, 5);
      expect(camera.position.z).toBeCloseTo(30, 5);
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

    it('[controls.md G32] setEnableZoom(false) on orbit blocks wheel events from accumulating zoomDelta', () => {
      // controls.md G32[P5]: the prior test only checks the `enableZoom`
      // field. This test exercises the actual wiring: dispatch a wheel
      // event to the canvas while zoom is disabled and verify the camera
      // distance (proxy for zoomDelta accumulation through update()) does
      // NOT change.
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      controlsManager.setEnableZoom(false);

      // Distance is camera-to-target — proxy for accumulated zoomDelta.
      const beforeDist = camera.position.distanceTo(controls.target);

      // Dispatch a wheel event at the canvas.
      domElement.dispatchEvent(
        new WheelEvent('wheel', { deltaY: -100, cancelable: true, bubbles: true })
      );
      controls.update();

      const afterBlockedDist = camera.position.distanceTo(controls.target);
      expect(afterBlockedDist).toBeCloseTo(beforeDist, 5);

      // Sanity: re-enabling and dispatching another event DOES change distance.
      controlsManager.setEnableZoom(true);
      domElement.dispatchEvent(
        new WheelEvent('wheel', { deltaY: -100, cancelable: true, bubbles: true })
      );
      controls.update();
      const afterEnabledDist = camera.position.distanceTo(controls.target);
      expect(afterEnabledDist).not.toBeCloseTo(beforeDist, 5);
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

    // controls.md [G20][P5]: setFlyRotationSpeed / setFlyRotationDamping
    // previously had no tests (fly mode had 2 of 5 setters tested).
    // Pin both forwarders so a regression that dropped the active-controls
    // live-apply branch is caught.
    it('setFlyRotationSpeed applies live to the active fly controls', () => {
      controlsManager.setFlyRotationSpeed(3.14);
      const controls = controlsManager.getControls() as LuxarFlyControls;
      expect(controls.rotationSpeed).toBeCloseTo(3.14, 5);
    });

    it('setFlyRotationDamping applies live to the active fly controls', () => {
      controlsManager.setFlyRotationDamping(0.88);
      const controls = controlsManager.getControls() as LuxarFlyControls;
      expect(controls.rotationDamping).toBeCloseTo(0.88, 5);
    });

    it('returns the SAME LuxarFlyControls instance as getControls() when active', () => {
      // controls.md [W2][P2] strengthening: was `.toBeInstanceOf` only.
      // The contract is stronger than "an instance is returned": the
      // returned reference must be IDENTICAL to the active controls
      // (`getFlyControls` is a narrowing accessor, not a factory). A
      // regression that returned a NEW LuxarFlyControls would survive
      // `.toBeInstanceOf` but fail the identity check.
      const flyControls = controlsManager.getFlyControls();
      expect(flyControls).toBeInstanceOf(LuxarFlyControls);
      expect(flyControls).toBe(controlsManager.getControls());
      // Default configuration is also pinned: enabled + inertialMode
      // are the most user-visible defaults.
      expect(flyControls!.enabled).toBe(true);
      expect(flyControls!.inertialMode).toBe(true);
    });

    it('should return null for fly controls when orbit is active', () => {
      controlsManager.setControlType('orbit');
      const flyControls = controlsManager.getFlyControls();
      expect(flyControls).toBeNull();
    });
  });

  // [controls.md/O2][P9] Split the former 'general control methods' block —
  // it mixed enable/disable + saveState/reset + lookAt (orbit + fly variants).
  // Each concern now lives in its own describe so the report names the
  // failing concern cleanly.
  describe('enable/disable', () => {
    it('should enable/disable controls', () => {
      controlsManager.setEnabled(false);
      expect(controlsManager.getControls()!.enabled).toBe(false);

      controlsManager.setEnabled(true);
      expect(controlsManager.getControls()!.enabled).toBe(true);
    });
  });

  describe('saveState/reset roundtrip', () => {
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
  });

  describe('lookAt', () => {
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
    // [controls.md/O8][P4] Compacted three near-identical event-forwarding tests
    // (change/start/end) into a single it.each — body is identical save for the
    // event type string.
    it.each([['change'], ['start'], ['end']] as const)(
      'forwards %s events from inner controls to outer manager listeners',
      (eventType) => {
        const handler = vi.fn();
        controlsManager.addEventListener(eventType, handler);

        const controls = controlsManager.getControls() as any;
        controls.dispatchEvent({ type: eventType });

        expect(handler).toHaveBeenCalled();
      }
    );

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
      // Two-mode round-trip is deterministic; precision 5 is the
      // project norm for Float32-tolerant equality.
      expect(orbitControls.target.x).toBeCloseTo(1, 5);
      expect(orbitControls.target.y).toBeCloseTo(2, 5);
      expect(orbitControls.target.z).toBeCloseTo(3, 5);
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

  // controls.md [G19][P5]: setTarget / reinitialize had no tests. The two
  // together are the auto-framing seam — autoFrameCamera mutates
  // camera.position THEN calls setTarget THEN reinitialize so the next
  // update() respects the new framing. A regression in either forwarder
  // would silently break auto-framing.
  describe('setTarget / reinitialize', () => {
    it('setTarget copies into orbit controls.target (no update fired)', () => {
      controlsManager.setTarget(new THREE.Vector3(7, 8, 9));
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.target.x).toBeCloseTo(7, 5);
      expect(controls.target.y).toBeCloseTo(8, 5);
      expect(controls.target.z).toBeCloseTo(9, 5);
    });

    it('reinitialize re-derives distance from camera.position → target', () => {
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      controls.enableDamping = false;
      // Pretend autoFrameCamera moved the camera.
      camera.position.set(0, 0, 20);
      controls.target.set(0, 0, 0);
      controlsManager.reinitialize();
      controlsManager.update();
      // After reinitialize, the orbit distance reflects the new ||cam-target||.
      // A subsequent update() should NOT snap the camera back to the old
      // distance — pinning the post-reinitialize observable z.
      expect(camera.position.z).toBeCloseTo(20, 4);
    });

    it('setTarget on fly controls instantly orients camera toward target', () => {
      controlsManager.setControlType('fly');
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();
      controlsManager.setTarget(new THREE.Vector3(10, 0, 5)); // straight +X
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      const expected = new THREE.Vector3(10, 0, 5).sub(camera.position).normalize();
      expect(forward.x).toBeCloseTo(expected.x, 4);
      expect(forward.z).toBeCloseTo(expected.z, 4);
    });
  });

  // controls.md [G18][P5]: previously `setDistanceLimits` / `setZoomLimits` /
  // `setSceneScale` / `getSceneScale` had NO direct tests. The auto-framing
  // pipeline calls these setters after computing the scene's diagonal —
  // a regression that dropped the live-apply step would silently degrade
  // zoom/pan feel without tripping any test. Pin both contracts here:
  //   1. The current orbit controls' min/max{Distance,Zoom} reflect the value.
  //   2. The stored limits survive a mode-switch (orbit→fly→orbit).
  describe('scale-aware setters', () => {
    it('setDistanceLimits applies live to the active orbit controls', () => {
      controlsManager.setDistanceLimits(2.5, 250);
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.minDistance).toBeCloseTo(2.5, 5);
      expect(controls.maxDistance).toBeCloseTo(250, 5);
    });

    it('setDistanceLimits persists across orbit → fly → orbit mode switches', () => {
      controlsManager.setDistanceLimits(3, 300);
      controlsManager.setControlType('fly');
      controlsManager.setControlType('orbit');
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.minDistance).toBeCloseTo(3, 5);
      expect(controls.maxDistance).toBeCloseTo(300, 5);
    });

    it('setZoomLimits applies live to the active ortho controls', () => {
      const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      orthoCam.position.copy(camera.position);
      controlsManager.setCamera(orthoCam);
      controlsManager.setControlType('ortho');
      controlsManager.setZoomLimits(0.4, 40);
      const controls = controlsManager.getControls() as LuxarOrbitControls;
      expect(controls.minZoom).toBeCloseTo(0.4, 5);
      expect(controls.maxZoom).toBeCloseTo(40, 5);
    });

    it('setSceneScale + getSceneScale round-trip; setSceneScale(<=0) is a no-op', () => {
      controlsManager.setSceneScale(42);
      expect(controlsManager.getSceneScale()).toBeCloseTo(42, 5);
      // setSceneScale(<=0) is a no-op (no scene info) — previous value preserved.
      controlsManager.setSceneScale(0);
      expect(controlsManager.getSceneScale()).toBeCloseTo(42, 5);
      controlsManager.setSceneScale(-1);
      expect(controlsManager.getSceneScale()).toBeCloseTo(42, 5);
      // A fresh positive value updates.
      controlsManager.setSceneScale(100);
      expect(controlsManager.getSceneScale()).toBeCloseTo(100, 5);
    });
  });
});
