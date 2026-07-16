/**
 * Unit tests for controls-manager/factories.ts.
 *
 * Targets audit findings G1 (createOrbitControls / createFlyControls /
 * createOrthoControls / naturalDragButtonMap untested in isolation),
 * M5 (mutation-suspect: the three-way `??` fallback chain — mutating
 * any `??` to `||` would survive orchestrator-level tests).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  createOrbitControls,
  createFlyControls,
  createOrthoControls,
  naturalDragButtonMap,
  type ControlsCreationCtx,
} from '../../../../controls/controls-manager/factories';
import { LuxarOrbitControls } from '../../../../controls/luxar-orbit-controls';
import { LuxarFlyControls } from '../../../../controls/luxar-fly-controls';
import { config } from '../../../../config';

function makeCtx(overrides: Partial<ControlsCreationCtx> = {}): ControlsCreationCtx {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  const domElement = document.createElement('div');
  return {
    camera,
    domElement,
    config: {},
    sceneScale: 0,
    storedDistanceLimits: null,
    storedZoomLimits: null,
    ...overrides,
  };
}

describe('naturalDragButtonMap', () => {
  it('enabled=true → LEFT=ROTATE, RIGHT=PAN (touchpad-friendly)', () => {
    const m = naturalDragButtonMap(true);
    expect(m.LEFT).toBe(THREE.MOUSE.ROTATE);
    expect(m.MIDDLE).toBe(THREE.MOUSE.DOLLY);
    expect(m.RIGHT).toBe(THREE.MOUSE.PAN);
  });

  it('enabled=false → LEFT=PAN, RIGHT=ROTATE (CAD/Blender-style)', () => {
    const m = naturalDragButtonMap(false);
    expect(m.LEFT).toBe(THREE.MOUSE.PAN);
    expect(m.MIDDLE).toBe(THREE.MOUSE.DOLLY);
    expect(m.RIGHT).toBe(THREE.MOUSE.ROTATE);
  });

  it('MIDDLE is always DOLLY (mode-independent)', () => {
    expect(naturalDragButtonMap(true).MIDDLE).toBe(THREE.MOUSE.DOLLY);
    expect(naturalDragButtonMap(false).MIDDLE).toBe(THREE.MOUSE.DOLLY);
  });
});

describe('createOrbitControls — three-way distance fallback (M5)', () => {
  // The fallback chain is:
  //   minDist = storedDistanceLimits?.min ?? (sceneScale > 0 ? sceneScale * m.minDistanceFactor : config.controls.orbit.zoom.minDistance)
  //   (similar for maxDist)
  // M5 mutation suspect: mutating any `??` to `||` would silently change
  // semantics when stored is 0. We exercise all three branches.
  const m = config.controls.scaleMultipliers;
  const orbitConfig = config.controls.orbit.zoom;

  it('uses storedDistanceLimits when provided (branch 1)', () => {
    const ctx = makeCtx({
      storedDistanceLimits: { min: 1.5, max: 250 },
      sceneScale: 10, // would otherwise yield different values
    });
    const c = createOrbitControls(ctx);
    expect(c.minDistance).toBeCloseTo(1.5, 5);
    expect(c.maxDistance).toBeCloseTo(250, 5);
    c.dispose();
  });

  it('uses sceneScale-derived limits when no stored limits but sceneScale > 0 (branch 2)', () => {
    const ctx = makeCtx({ sceneScale: 100, storedDistanceLimits: null });
    const c = createOrbitControls(ctx);
    expect(c.minDistance).toBeCloseTo(100 * m.minDistanceFactor, 5);
    expect(c.maxDistance).toBeCloseTo(100 * m.maxDistanceFactor, 5);
    c.dispose();
  });

  it('uses hardcoded config defaults when no stored limits and sceneScale = 0 (branch 3, boundary)', () => {
    // P5 boundary: sceneScale=0 must NOT propagate as the scaled value (0).
    const ctx = makeCtx({ sceneScale: 0, storedDistanceLimits: null });
    const c = createOrbitControls(ctx);
    expect(c.minDistance).toBeCloseTo(orbitConfig.minDistance, 5);
    expect(c.maxDistance).toBeCloseTo(orbitConfig.maxDistance, 5);
    c.dispose();
  });

  it('stored zero (min=0) takes precedence — `??` is the operator, not `||`', () => {
    // The exact mutation M5 calls out: replacing `??` with `||` would skip
    // a stored value of 0. Verify we DO use 0 (and not the fallback).
    const ctx = makeCtx({
      storedDistanceLimits: { min: 0, max: 100 },
      sceneScale: 1000, // would yield very different fallback
    });
    const c = createOrbitControls(ctx);
    // minDistance must be 0, NOT scaleMultipliers.minDistanceFactor * 1000.
    expect(c.minDistance).toBeCloseTo(0, 5);
    c.dispose();
  });

  it('enables damping, screenSpacePanning, and view-axis rotation by default', () => {
    const c = createOrbitControls(makeCtx());
    expect(c.enableDamping).toBe(true);
    expect(c.screenSpacePanning).toBe(true);
    // viewAxisRotationHandler is private but exposed in tests via cast.
    expect((c as any).viewAxisRotationHandler).not.toBeNull();
    c.dispose();
  });

  it('applies naturalDrag mapping when config.naturalDrag = true', () => {
    const c = createOrbitControls(makeCtx({ config: { naturalDrag: true } }));
    expect(c.mouseButtons.LEFT).toBe(THREE.MOUSE.ROTATE);
    expect(c.mouseButtons.RIGHT).toBe(THREE.MOUSE.PAN);
    c.dispose();
  });

  it('keeps the default mapping when config.naturalDrag is unset', () => {
    const c = createOrbitControls(makeCtx({ config: {} }));
    expect(c.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
    expect(c.mouseButtons.RIGHT).toBe(THREE.MOUSE.ROTATE);
    c.dispose();
  });

  it('[controls.md G23] keeps the default mapping when config.naturalDrag === false (explicit, not just unset)', () => {
    // controls.md G23[P5]: prior test only covered `unset` (line 125-130).
    // Pin the explicit `false` path so a regression flipping the factory's
    // default (e.g. `if (!naturalDrag)` becoming `if (naturalDrag === undefined)`)
    // would be caught here.
    const c = createOrbitControls(makeCtx({ config: { naturalDrag: false } }));
    expect(c.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
    expect(c.mouseButtons.RIGHT).toBe(THREE.MOUSE.ROTATE);
    c.dispose();
  });

  it('honors config.autoRotate and config.autoRotateSpeed', () => {
    const c = createOrbitControls(makeCtx({ config: { autoRotate: true, autoRotateSpeed: 3.5 } }));
    expect(c.autoRotate).toBe(true);
    expect(c.autoRotateSpeed).toBeCloseTo(3.5, 5);
    c.dispose();
  });

  it('returns a LuxarOrbitControls instance', () => {
    const c = createOrbitControls(makeCtx());
    expect(c).toBeInstanceOf(LuxarOrbitControls);
    c.dispose();
  });
});

describe('createFlyControls', () => {
  it('returns a LuxarFlyControls instance', () => {
    const c = createFlyControls(makeCtx());
    expect(c).toBeInstanceOf(LuxarFlyControls);
    c.dispose();
  });

  it('plumbs movementSpeed / rotationSpeed / lookSpeed / damping through', () => {
    const c = createFlyControls(
      makeCtx({
        config: {
          flyMovementSpeed: 12,
          flyRotationSpeed: 2.3,
          flyLookSpeed: 0.007,
          flyDamping: 0.97,
          flyRotationDamping: 0.93,
          flyInertialMode: false,
        },
      })
    );
    expect(c.movementSpeed).toBeCloseTo(12, 5);
    expect(c.rotationSpeed).toBeCloseTo(2.3, 5);
    expect(c.lookSpeed).toBeCloseTo(0.007, 5);
    expect(c.damping).toBeCloseTo(0.97, 5);
    expect(c.rotationDamping).toBeCloseTo(0.93, 5);
    expect(c.inertialMode).toBe(false);
    c.dispose();
  });

  it('uses externalInputManagement: true (keyboard routed via InputContextManager)', () => {
    const c = createFlyControls(makeCtx());
    // externalInputManagement is private but reflects in NOT registering
    // window keyboard listeners. Indirectly: try a window keydown — fly
    // moveState must NOT change because the helper sets
    // externalInputManagement=true.
    const before = (c as any).moveState.forward;
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w' }));
    expect((c as any).moveState.forward).toBe(before);
    c.dispose();
  });
});

describe('createOrthoControls — zoom-limit fallback (M5)', () => {
  // Ortho fallback chain (zoom ~ 1/distance, so each zoom bound maps to the
  // OPPOSITE distance factor):
  //   minZoom = storedZoomLimits?.min ?? 1.0 / m.maxDistanceFactor
  //   maxZoom = storedZoomLimits?.max ?? 1.0 / m.minDistanceFactor
  const m = config.controls.scaleMultipliers;

  it('uses storedZoomLimits when provided', () => {
    const c = createOrthoControls(makeCtx({ storedZoomLimits: { min: 0.5, max: 50 } }));
    expect(c.minZoom).toBeCloseTo(0.5, 5);
    expect(c.maxZoom).toBeCloseTo(50, 5);
    c.dispose();
  });

  it('falls back to inverted scaleMultipliers when no stored zoom limits', () => {
    const c = createOrthoControls(makeCtx({ storedZoomLimits: null }));
    expect(c.minZoom).toBeCloseTo(1.0 / m.maxDistanceFactor, 9);
    expect(c.maxZoom).toBeCloseTo(1.0 / m.minDistanceFactor, 5);
    c.dispose();
  });

  it('disables rotation and sets RIGHT=null (Napari/Maps mapping)', () => {
    const c = createOrthoControls(makeCtx());
    expect(c.enableRotate).toBe(false);
    expect(c.mouseButtons.LEFT).toBe(THREE.MOUSE.PAN);
    expect(c.mouseButtons.RIGHT).toBeNull();
    c.dispose();
  });

  it('uses Infinity distance range (ortho zoom is independent of distance)', () => {
    const c = createOrthoControls(makeCtx());
    expect(c.minDistance).toBe(0);
    expect(c.maxDistance).toBe(Infinity);
    c.dispose();
  });
});
