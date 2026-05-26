/**
 * Unit tests for luxar-orbit-controls/math/zoom.ts.
 *
 * Targets audit findings G5 (boundary cases on computeZoomScale /
 * applyZoomScale), M1 (mutation suspect — only branch existence is
 * exercised by orchestrator-level tests), and H3 (round-trip /
 * monotonicity invariants).
 *
 * Pure math only; no DOM or three.js camera state mutation beyond the
 * caller-supplied camera.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  computeZoomScale,
  applyZoomScale,
} from '../../../../../controls/luxar-orbit-controls/math/zoom';

describe('computeZoomScale', () => {
  // Formula: Math.pow(0.95, zoomSpeed * Math.abs(delta * 0.01))
  it('returns 1.0 when delta is exactly 0 (boundary)', () => {
    // P5 boundary: delta=0 must not introduce any scale change.
    expect(computeZoomScale(0, 1.0)).toBe(1.0);
  });

  it('is symmetric in the sign of delta (|d| only)', () => {
    // The formula uses Math.abs(delta * 0.01), so +d and -d must produce
    // the same scale value.
    for (const d of [10, 50, 100, 250]) {
      expect(computeZoomScale(d, 1.0)).toBeCloseTo(computeZoomScale(-d, 1.0), 10);
    }
  });

  it('is monotonically decreasing in |delta| (larger |delta| → smaller scale)', () => {
    // Math.pow(0.95, x) with x > 0 is strictly decreasing in x.
    const scales = [10, 50, 100, 500, 1000].map((d) => computeZoomScale(d, 1.0));
    for (let i = 1; i < scales.length; i++) {
      expect(scales[i]).toBeLessThan(scales[i - 1]);
    }
  });

  it('is monotonically decreasing in zoomSpeed (faster speed → more zoom for same delta)', () => {
    // Math.pow(0.95, zoomSpeed * |d|/100) — larger zoomSpeed shrinks scale.
    const scales = [0.5, 1.0, 2.0, 4.0].map((s) => computeZoomScale(100, s));
    for (let i = 1; i < scales.length; i++) {
      expect(scales[i]).toBeLessThan(scales[i - 1]);
    }
  });

  it('matches the explicit closed-form value for representative inputs', () => {
    // delta=100, zoomSpeed=1 → 0.95^(1*1) = 0.95
    expect(computeZoomScale(100, 1.0)).toBeCloseTo(0.95, 10);
    // delta=200, zoomSpeed=1 → 0.95^2 = 0.9025
    expect(computeZoomScale(200, 1.0)).toBeCloseTo(0.95 * 0.95, 10);
    // delta=100, zoomSpeed=2 → 0.95^2 = 0.9025
    expect(computeZoomScale(100, 2.0)).toBeCloseTo(0.95 * 0.95, 10);
  });

  it('large delta produces a scale approaching 0 (asymptote)', () => {
    // Sanity — no overflow, no NaN.
    const s = computeZoomScale(1e6, 1.0);
    expect(s).toBeGreaterThan(0);
    expect(s).toBeLessThan(0.001);
    expect(Number.isFinite(s)).toBe(true);
  });

  it('[controls.md/G6] zero zoomSpeed produces scale = 1 regardless of delta', () => {
    // controls.md G6: zoomSpeed=0 → exponent = 0 → 0.95^0 = 1. A mutation
    // that swapped `*` for `+` in the exponent would survive the existing
    // monotonicity tests since they all use zoomSpeed > 0.
    for (const d of [-1000, -1, 0, 1, 1000]) {
      expect(computeZoomScale(d, 0)).toBe(1);
    }
  });

  it('[controls.md/G6] NaN delta propagates to NaN scale (current unguarded behaviour)', () => {
    // controls.md G6: pinning the CURRENT behaviour — Math.abs(NaN*0.01) is
    // NaN, Math.pow(0.95, NaN) is NaN. A future guard that returns 1.0 for
    // non-finite input would surface here and force an intentional contract
    // update.
    expect(Number.isNaN(computeZoomScale(Number.NaN, 1.0))).toBe(true);
  });

  it('[controls.md/G6] both delta and zoomSpeed zero produces scale = 1', () => {
    // Belt-and-braces for the degenerate-input pair.
    expect(computeZoomScale(0, 0)).toBe(1);
  });
});

describe('applyZoomScale — perspective camera (H3 round-trip)', () => {
  // For perspective: returns currentDistance * scale; doesn't touch the camera.
  it('returns currentDistance * scale unchanged for any scale', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    expect(applyZoomScale(cam, 10, 0.5, 0.01, 1000)).toBeCloseTo(5, 10);
    expect(applyZoomScale(cam, 10, 1.0, 0.01, 1000)).toBeCloseTo(10, 10);
    expect(applyZoomScale(cam, 10, 2.0, 0.01, 1000)).toBeCloseTo(20, 10);
  });

  it('does not mutate the camera for perspective', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const fovBefore = cam.fov;
    const zoomBefore = cam.zoom;
    applyZoomScale(cam, 10, 0.5, 0.01, 1000);
    expect(cam.fov).toBe(fovBefore);
    expect(cam.zoom).toBe(zoomBefore);
  });

  it('round-trips for perspective: scale then 1/scale recovers original distance (H3)', () => {
    // P12 / H3 property: zoom-in then zoom-out cancels.
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    for (const d of [1, 5, 10, 100]) {
      for (const s of [0.5, 0.95, 1.5, 2.0]) {
        const d1 = applyZoomScale(cam, d, s, 0.01, 1000);
        const d2 = applyZoomScale(cam, d1, 1 / s, 0.01, 1000);
        expect(d2).toBeCloseTo(d, 5);
      }
    }
  });
});

describe('applyZoomScale — orthographic camera', () => {
  it('mutates camera.zoom = zoom / scale (zoom in: scale < 1 → camera.zoom larger)', () => {
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 1.0;
    const dist = applyZoomScale(cam, 10, 0.5, 0.001, 1000);
    expect(cam.zoom).toBeCloseTo(2.0, 5);
    // Distance unchanged for ortho.
    expect(dist).toBeCloseTo(10, 10);
  });

  it('zoom out (scale > 1) reduces camera.zoom', () => {
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 1.0;
    applyZoomScale(cam, 10, 2.0, 0.001, 1000);
    expect(cam.zoom).toBeCloseTo(0.5, 5);
  });

  it('clamps camera.zoom at the lower bound (minZoom) — boundary', () => {
    // P5 boundary: try to zoom out past minZoom.
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 0.5;
    applyZoomScale(cam, 10, 100, 0.1, 100); // scale=100 → would give 0.005, clamps to 0.1
    expect(cam.zoom).toBeCloseTo(0.1, 5);
  });

  it('clamps camera.zoom at the upper bound (maxZoom) — boundary', () => {
    // P5 boundary: try to zoom in past maxZoom.
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 5.0;
    applyZoomScale(cam, 10, 0.01, 0.1, 10); // scale=0.01 → would give 500, clamps to 10
    expect(cam.zoom).toBeCloseTo(10, 5);
  });

  it('handles minZoom === maxZoom (degenerate clamp window) — boundary', () => {
    // P5 boundary: when the window collapses to a point, zoom must equal it.
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 2.0;
    applyZoomScale(cam, 10, 0.5, 1.0, 1.0);
    expect(cam.zoom).toBeCloseTo(1.0, 10);
  });

  it('[controls.md/G5] minZoom === maxZoom === 0 collapses zoom to 0 (degenerate)', () => {
    // controls.md G5: a caller that passes `0` defaults would collapse the
    // clamp window to {0}. The pure-math helper has no production guard;
    // pin the current behaviour so a future guard (e.g. "clamp negatives
    // up to a minimum positive epsilon") surfaces as an intentional change.
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 2.0;
    applyZoomScale(cam, 10, 0.5, 0, 0);
    expect(cam.zoom).toBe(0);
  });

  it('updateProjectionMatrix is called (ortho needs refresh after zoom change)', () => {
    // The projectionMatrix is recomputed on updateProjectionMatrix; we test
    // by checking that the projectionMatrix changes when zoom changes.
    const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    cam.zoom = 1.0;
    cam.updateProjectionMatrix();
    const projBefore = cam.projectionMatrix.clone();

    applyZoomScale(cam, 10, 0.5, 0.001, 1000);

    expect(cam.projectionMatrix.equals(projBefore)).toBe(false);
  });
});
