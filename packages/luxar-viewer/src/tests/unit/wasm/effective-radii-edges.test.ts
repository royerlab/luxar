/**
 * Edge-case tests for WASM TypeScript-fallback effective-radii helper.
 *
 * Closes wasm.md gap cluster:
 *   - [wasm.md G16][P5] spatialExtendDims length=0 (every non-displayed dim
 *                       defaults to spatial via the `d < length ? ... : true`
 *                       fallback) and length > ndim (extra entries silently
 *                       ignored).
 *   - [wasm.md G17][P5] calculate_effective_radii: NaN in positions, Inf in
 *                       positions, radii[i] = 0 boundary, radii[i] < 0
 *                       (algorithm squares it, behaviour is well-defined).
 *
 * Pure math on typed arrays. Pins documented contracts so a future hardening
 * (throw on NaN, normalize radii to abs, require length === ndim) surfaces
 * as an intentional change.
 */

import { describe, it, expect } from 'vitest';
import { calculate_effective_radii } from '../../../wasm/typescript/effective-radii';

describe('calculate_effective_radii — spatialExtendDims length boundaries [wasm.md G16]', () => {
  it('[G16] length === 0: every non-displayed dim defaults to spatial (Pythagorean accumulation)', () => {
    // ndim=4, displayDims=[0,1,2] → only dim 3 is non-displayed.
    // spatialExtendDims is empty → fallback `true` → dim 3 is spatial.
    // Point at delta=0.6 in dim 3, radius=1 → distSq=0.36 < 1 → effective=sqrt(0.64) ≈ 0.8.
    const positions = new Float32Array([0, 0, 0, 0.6]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array(0); // empty → all default to spatial
    const output = new Float32Array(1);
    const n = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBeCloseTo(0.8, 5);
  });

  it('[G16] length === 0: a discrete-mismatch case becomes spatial (point at offset still visible if within R)', () => {
    // With explicit spatialExtendDims=[0] in dim 3, the point at delta=1.0
    // would FAIL the discrete check (|1.0| > 0.5) → output 0.
    // With length=0 (default spatial), distSq=1.0 = radius² → effective=0.
    // Both produce 0 here BUT for different reasons; use delta=0.6 instead
    // to discriminate.
    const positions = new Float32Array([0, 0, 0, 0.6]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);

    // Discrete (would fail tolerance check at |0.6| > 0.5):
    const sExplicit = new Uint8Array([1, 1, 1, 0]);
    const oExplicit = new Float32Array(1);
    calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      sExplicit,
      4,
      1,
      oExplicit
    );
    expect(oExplicit[0]).toBe(0); // hidden (discrete mismatch)

    // Default-spatial (sqrt(1 - 0.36) = 0.8):
    const sDefault = new Uint8Array(0);
    const oDefault = new Float32Array(1);
    calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      sDefault,
      4,
      1,
      oDefault
    );
    expect(oDefault[0]).toBeCloseTo(0.8, 5); // visible (spatial fallback)
  });

  it('[G16] length > ndim: extra entries beyond ndim are silently ignored', () => {
    // Pin: spatialExtendDims has 10 entries but ndim=2 → only first 2 read.
    const positions = new Float32Array([0, 0]); // 1 point × 2 dim
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0]);
    const slicePosition = new Float32Array([0, 0]);
    const spatialExtendDims = new Uint8Array(10); // all zeros, but only [0..1] read
    spatialExtendDims[1] = 1; // dim 1 spatial
    const output = new Float32Array(1);
    calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      2,
      1,
      output
    );
    // Dim 1 spatial, distSq=0 < 1 → effective = 1.
    expect(output[0]).toBe(1);
  });

  it('[G16] length === ndim: explicit per-dim control overrides the fallback', () => {
    // Canonical happy path: explicit array of correct length.
    const positions = new Float32Array([0, 0, 0, 1.0]); // delta=1 in dim 3
    const radii = new Float32Array([2.0]); // R=2, R²=4
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1, 1]); // length === ndim
    const output = new Float32Array(1);
    calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      output
    );
    // distSq = 1, sqrt(4 - 1) = sqrt(3) ≈ 1.732.
    expect(output[0]).toBeCloseTo(Math.sqrt(3), 5);
  });
});

describe('calculate_effective_radii — NaN / Inf / zero / negative radii [wasm.md G17]', () => {
  it('[G17] NaN in positions (non-displayed spatial dim) → distSq=NaN → distSq >= R² is false → effective = sqrt(NaN) = NaN', () => {
    // The branch `if (distanceSquared >= radiusSquared)` with NaN: NaN >= anything
    // is false → falls to ELSE: output = sqrt(radiusSquared - NaN) = sqrt(NaN) = NaN.
    // visibleCount IS incremented even though the output is NaN (documented quirk).
    const positions = new Float32Array([0, 0, 0, Number.NaN]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1, 1]);
    const output = new Float32Array(1);
    const n = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      output
    );
    expect(Number.isNaN(output[0])).toBe(true);
    // Documented quirk: visibleCount += 1 in the else branch even though output is NaN.
    expect(n).toBe(1);
  });

  it('[G17] Infinity in positions → distSq = Infinity → distSq >= R² is true → effective = 0', () => {
    const positions = new Float32Array([0, 0, 0, Number.POSITIVE_INFINITY]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1, 1]);
    const output = new Float32Array(1);
    const n = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      output
    );
    expect(output[0]).toBe(0);
    expect(n).toBe(0);
  });

  it('[G17] radii[i] === 0: R²=0; distSq=0 ≥ 0 is true → effective = 0 (hidden)', () => {
    // Boundary: point AT slice, zero radius. R²=0, distSq=0, 0 >= 0 → output=0.
    const positions = new Float32Array([0, 0, 0, 0]);
    const radii = new Float32Array([0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1, 1]);
    const output = new Float32Array(1).fill(99); // sentinel
    const n = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      output
    );
    expect(output[0]).toBe(0);
    expect(n).toBe(0);
  });

  it('[G17] radii[i] < 0: squared → positive R²; behaviour is well-defined (radius treated as |R|)', () => {
    // Pin documented contract: negative radius is squared, so the algorithm
    // treats it identically to its absolute value. A regression that errored
    // on negative radii would surface here.
    const positions = new Float32Array([0, 0, 0, 0.6]);
    const radiiNeg = new Float32Array([-1.0]);
    const radiiPos = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1, 1]);
    const outNeg = new Float32Array(1);
    const outPos = new Float32Array(1);
    calculate_effective_radii(
      positions,
      radiiNeg,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      outNeg
    );
    calculate_effective_radii(
      positions,
      radiiPos,
      displayDims,
      slicePosition,
      spatialExtendDims,
      4,
      1,
      outPos
    );
    expect(outNeg[0]).toBeCloseTo(outPos[0], 6);
    expect(outNeg[0]).toBeCloseTo(0.8, 5);
  });

  it('[G17] numPoints === 0 boundary: loop never executes, return 0, output untouched', () => {
    const output = new Float32Array(3).fill(99); // sentinel
    const n = calculate_effective_radii(
      new Float32Array(0),
      new Float32Array(0),
      new Uint32Array([0, 1, 2]),
      new Float32Array([0, 0, 0]),
      new Uint8Array([1, 1, 1]),
      3,
      0,
      output
    );
    expect(n).toBe(0);
    expect(Array.from(output)).toEqual([99, 99, 99]);
  });

  it('[G17] all-spatial-dim displayed: every point is fully visible with effective = original radius', () => {
    // ndim=3, displayDims=[0,1,2] — no non-displayed dim → distSq=0 always.
    // distSq >= R² is false (assuming R>0); effective = sqrt(R²) = R.
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const radii = new Float32Array([0.5, 0.75]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const spatialExtendDims = new Uint8Array([1, 1, 1]);
    const output = new Float32Array(2);
    const n = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePosition,
      spatialExtendDims,
      3,
      2,
      output
    );
    expect(n).toBe(2);
    expect(output[0]).toBeCloseTo(0.5, 5);
    expect(output[1]).toBeCloseTo(0.75, 5);
  });
});
