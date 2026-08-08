/**
 * Unit tests for the gallery harness's rock-axis classification.
 *
 * The capture side needs a browser, a dataset and a GPU; deciding which signed
 * world axis a camera up-vector points along does not. Sibling of
 * `gallery-frame-similarity.test.ts` and `gallery-exposure-policy.test.ts`,
 * which split the same way for the same reason.
 *
 * These pin the FIX for #1377, not its detector. The orbit hard-sets `cam.up`
 * from this axis on every frame, so an axis slip rolls the animation ~90° from
 * its own poster and a SIGN slip rolls it 180° — and the gallery harness is a
 * manual media run, not a CI job, so nothing else would notice until someone
 * re-captured the tiles and looked.
 */

import { describe, expect, it } from 'vitest';

import { dominantSignedAxis, FALLBACK_UP_AXIS } from '../screenshots/orbit-axis';

describe('dominantSignedAxis', () => {
  it('maps each signed unit axis to itself', () => {
    expect(dominantSignedAxis({ x: 1, y: 0, z: 0 })).toEqual({ axis: 'x', sign: 1 });
    expect(dominantSignedAxis({ x: -1, y: 0, z: 0 })).toEqual({ axis: 'x', sign: -1 });
    expect(dominantSignedAxis({ x: 0, y: 1, z: 0 })).toEqual({ axis: 'y', sign: 1 });
    expect(dominantSignedAxis({ x: 0, y: -1, z: 0 })).toEqual({ axis: 'y', sign: -1 });
    expect(dominantSignedAxis({ x: 0, y: 0, z: 1 })).toEqual({ axis: 'z', sign: 1 });
    expect(dominantSignedAxis({ x: 0, y: 0, z: -1 })).toEqual({ axis: 'z', sign: -1 });
  });

  it('keeps the sign of every up the manifest demos actually bake', () => {
    // The real inputs, from the demos' `CameraConfig(up=…)`: asteroids,
    // cosmicflows and the nexrad supercell bake +Z; the isosurface mesh bakes
    // +X; everything else bakes +Y. Deriving these is the whole point of the
    // change — before it they all rocked about world-Y.
    expect(dominantSignedAxis({ x: 0, y: 0, z: 1 }).axis).toBe('z');
    expect(dominantSignedAxis({ x: 1, y: 0, z: 0 }).axis).toBe('x');
    expect(dominantSignedAxis({ x: 0, y: 1, z: 0 })).toEqual(FALLBACK_UP_AXIS);
  });

  it('snaps a non-axis-aligned up to its dominant component, sign intact', () => {
    expect(dominantSignedAxis({ x: 0.1, y: 0.2, z: -0.95 })).toEqual({ axis: 'z', sign: -1 });
    expect(dominantSignedAxis({ x: -0.8, y: 0.3, z: 0.2 })).toEqual({ axis: 'x', sign: -1 });
    expect(dominantSignedAxis({ x: 0.05, y: -0.99, z: 0.02 })).toEqual({ axis: 'y', sign: -1 });
  });

  it('is invariant to the vector length', () => {
    // `cam.up` is normally unit, but nothing in three.js guarantees it, and a
    // magnitude must never change which axis wins.
    for (const k of [1e-6, 0.5, 1, 1000]) {
      expect(dominantSignedAxis({ x: 0, y: 0, z: -k })).toEqual({ axis: 'z', sign: -1 });
    }
  });

  it('falls back to +Y for a missing or degenerate up instead of guessing', () => {
    // A guess here would be an arbitrary axis dressed up as a measurement: the
    // `>=` chain answers 'x' for an all-zero vector.
    for (const bad of [
      null,
      undefined,
      { x: 0, y: 0, z: 0 },
      { x: NaN, y: NaN, z: NaN },
      { x: 0, y: NaN, z: 0 },
      { x: Infinity, y: 0, z: 0 },
      { x: 0, y: 0, z: -Infinity },
    ]) {
      expect(dominantSignedAxis(bad)).toEqual(FALLBACK_UP_AXIS);
    }
  });

  it('resolves ties x → z → y, deterministically', () => {
    // No real scene bakes these; pinning them keeps a refactor from silently
    // reordering the comparison chain.
    expect(dominantSignedAxis({ x: 1, y: 1, z: 0 }).axis).toBe('x');
    expect(dominantSignedAxis({ x: 0, y: 1, z: 1 }).axis).toBe('z');
    expect(dominantSignedAxis({ x: 1, y: 1, z: 1 }).axis).toBe('x');
  });

  it('never returns a sign outside {1, -1} or an unknown axis', () => {
    for (const v of [
      { x: 3, y: -4, z: 0 },
      { x: -0, y: -0, z: -2 },
      { x: 0.001, y: 0, z: 0 },
    ]) {
      const r = dominantSignedAxis(v);
      expect(['x', 'y', 'z']).toContain(r.axis);
      expect([1, -1]).toContain(r.sign);
    }
  });

  it('does not let a -0 component win the axis', () => {
    // `Math.abs(-0)` is 0, so -0 can only ever tie, never beat a real
    // component — a -0 must not steal the axis and roll the animation 90°.
    expect(dominantSignedAxis({ x: -0, y: 0, z: 1 })).toEqual({ axis: 'z', sign: 1 });
    expect(dominantSignedAxis({ x: -0, y: -0, z: -0 })).toEqual(FALLBACK_UP_AXIS);
  });
});
