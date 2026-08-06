/**
 * Tests for `src/wasm/typescript/gsplats-processing.ts`
 * (Mahalanobis distance, marginal Cholesky, fused nD→3D projection).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D8): the 1714-line mega-file
 * is being split into per-source-module test files mirroring
 * `src/wasm/typescript/`.
 */

import { describe, it, expect } from 'vitest';
import {
  mahalanobis_distance,
  computeMarginalCholesky,
  project_gsplats_nd_to_3d,
} from '../../../../wasm/typescript';

// ============================================================================
// GSPLATS PROCESSING TESTS (Mahalanobis distance, Cholesky extraction)
// ============================================================================
describe('gsplats_processing: mahalanobis_distance', () => {
  it('should compute Euclidean distance with identity Cholesky', () => {
    // Identity Cholesky (L = I): Mahalanobis = Euclidean
    // 3D: packed = [1, 0, 1, 0, 0, 1]
    const packedL = new Float32Array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    const diff = new Float32Array([3.0, 4.0, 0.0]); // Distance should be 5.0

    const dist = mahalanobis_distance(diff, packedL, 3);
    expect(dist).toBeCloseTo(5.0, 5);
  });

  it('should scale distance with scaled Cholesky', () => {
    // Scaled Cholesky: L = diag(2, 2, 2)
    // packed = [2, 0, 2, 0, 0, 2]
    // Mahalanobis = ||L⁻¹ · diff|| = ||diff / 2||
    const packedL = new Float32Array([2.0, 0.0, 2.0, 0.0, 0.0, 2.0]);
    const diff = new Float32Array([4.0, 0.0, 0.0]); // Mahalanobis should be 4/2 = 2

    const dist = mahalanobis_distance(diff, packedL, 3);
    expect(dist).toBeCloseTo(2.0, 5);
  });

  it('should handle 2D case', () => {
    // 2D identity: packed = [1, 0, 1]
    const packedL = new Float32Array([1.0, 0.0, 1.0]);
    const diff = new Float32Array([3.0, 4.0]);

    const dist = mahalanobis_distance(diff, packedL, 2);
    expect(dist).toBeCloseTo(5.0, 5);
  });
});

// ============================================================================
// MARGINAL CHOLESKY TESTS (correlated covariance correctness)
// ============================================================================
describe('gsplats_processing: computeMarginalCholesky', () => {
  it('should match raw extraction for diagonal Cholesky', () => {
    // 4D diagonal L: diag(2, 3, 5, 7)
    // Packed: [2, 0,3, 0,0,5, 0,0,0,7]
    const packed = new Float32Array([2, 0, 3, 0, 0, 5, 0, 0, 0, 7]);
    const keepDims = new Uint32Array([0, 2]);
    const output = new Float32Array(3);

    computeMarginalCholesky(packed, 0, keepDims, 2, output, 0);

    // Σ = diag(4,9,25,49), marginal for [0,2] = diag(4,25)
    // Cholesky = diag(2, 5) → packed [2, 0, 5]
    expect(output[0]).toBeCloseTo(2.0, 4);
    expect(output[1]).toBeCloseTo(0.0, 4);
    expect(output[2]).toBeCloseTo(5.0, 4);
  });

  it('should produce correct marginal for correlated Cholesky', () => {
    // 3D L with correlations (matches Rust test):
    // L = [[2, 0, 0], [1, 3, 0], [0.5, 0.5, 4]]
    // Packed: [2, 1,3, 0.5,0.5,4]
    const packed = new Float32Array([2, 1, 3, 0.5, 0.5, 4]);
    const keepDims = new Uint32Array([0, 2]);
    const output = new Float32Array(3);

    computeMarginalCholesky(packed, 0, keepDims, 2, output, 0);

    // Σ_S = [[4, 1], [1, 16.5]]
    // L_S[0,0] = 2, L_S[1,0] = 0.5, L_S[1,1] = sqrt(16.25) ≈ 4.031
    expect(output[0]).toBeCloseTo(2.0, 4);
    expect(output[1]).toBeCloseTo(0.5, 4);
    expect(output[2]).toBeCloseTo(Math.sqrt(16.25), 3);
  });

  it('should give correct Mahalanobis distance with marginal Cholesky', () => {
    // Same 3D correlated L
    const packed = new Float32Array([2, 1, 3, 0.5, 0.5, 4]);
    const keepDims = new Uint32Array([0, 2]);

    const marginalL = new Float32Array(3);
    computeMarginalCholesky(packed, 0, keepDims, 2, marginalL, 0);

    const diff = new Float32Array([1.0, 0.0]);
    const dist = mahalanobis_distance(diff, marginalL, 2);

    // Forward substitution: y[0]=1/2=0.5, y[1]=(0-0.5*0.5)/4.031≈-0.0621
    // ||y|| ≈ sqrt(0.25 + 0.00386) ≈ 0.504
    expect(dist).toBeCloseTo(0.5, 1);
  });
});

/**
 * Visibility-gate contracts of the fused kernel. Migrated here from the
 * per-kernel attenuation tests that went with `compute_gsplats_attenuation`:
 * the gate is now inlined in `project_gsplats_nd_to_3d`, so it needs pinning
 * through the surviving entry point.
 */
describe('gsplats_processing: fused visibility gate', () => {
  // One splat, `ndim` dims, all-visible discrete gate; returns [count, amplitude].
  const project = (
    positions: number[],
    ndim: number,
    continuousHiddenDims: number[],
    amplitude: number,
    minAmplitude: number
  ): [number, number] => {
    const packedSize = (ndim * (ndim + 1)) / 2;
    const cholesky = new Float32Array(packedSize);
    for (let i = 0; i < ndim; i++) cholesky[(i * (i + 1)) / 2 + i] = 1.0; // identity L
    const outAmplitudes = new Float32Array(1);
    const count = project_gsplats_nd_to_3d(
      new Float32Array(positions),
      cholesky,
      new Float32Array([amplitude]),
      new Float32Array([1, 1, 1]),
      new Uint8Array([1]),
      new Float32Array(ndim), // slice at the origin
      new Uint32Array(continuousHiddenDims),
      new Uint32Array([0, 1, 2]),
      ndim,
      1,
      3,
      minAmplitude,
      3.0,
      new Float32Array(3),
      new Float32Array(6),
      outAmplitudes,
      new Float32Array(3)
    );
    return [count, outAmplitudes[0]];
  };

  it('keeps a splat whose attenuated amplitude sits EXACTLY on minAmplitude', () => {
    // No hidden dims → the `numContinuous === 0` short-circuit forces
    // attenuation = 1, so the emitted amplitude is the raw one and the gate is
    // tested at its boundary. The gate is `>= minAmplitude`; a mutant flipping
    // it to a strict `>` drops this splat.
    expect(project([0, 0, 0], 3, [], 0.25, 0.25)).toEqual([1, 0.25]);
    // One ulp of headroom below the threshold and it must go.
    expect(project([0, 0, 0], 3, [], 0.25, 0.2500001)[0]).toBe(0);
  });

  it('culls a splat whose hidden-dim position is NaN (WASM parity)', () => {
    // A NaN center makes the Mahalanobis distance NaN. Rust clamps with
    // `f32::max`, which ignores NaN and yields attenuation 0 → culled. This
    // backend must agree: emitting the splat instead would push a NaN amplitude
    // to the GPU (`NaN < minAmplitude` is false), the #725 corruption mode.
    expect(project([0, 0, 0, Number.NaN], 4, [3], 1.0, 1e-6)[0]).toBe(0);
  });
});

/**
 * Dense compaction of the fused kernel — the TS twin of
 * `gsplats_processing.rs::test_fused_compaction_stride`.
 *
 * Here and not only in `wasm-vs-typescript.test.ts` for the same reason as the
 * display-dims block below: those cases are `skipIf(!wasmFilesExist)` and vanish
 * without a built WASM artifact, while this backend is the production path for
 * ndim > 16. Without this, the discrete gate, the RGBA alpha stride and the
 * dense-slot bookkeeping would be unpinned on the TS side.
 */
describe('gsplats_processing: dense compaction stride (TS reference)', () => {
  // Same fixture as the Rust test: 4 splats, dim 3 the continuous hidden
  // slicing dim (slice at 0). splat1 is far off-slice in dim 3 (attenuated
  // below minAmplitude) and splat2 is discrete-gated, so the surviving pair
  // splat0/splat3 is NON-CONTIGUOUS — the only arrangement in which the dense
  // output slot diverges from the loop index.
  const ndim = 4;
  const splatCount = 4;
  // prettier-ignore
  const positions = new Float32Array([
    0, 0, 0, 0, // splat0 (on slice)
    1, 1, 1, 50, // splat1 (far in dim3 → attenuated out)
    2, 2, 2, 0, // splat2 (discrete-gated)
    3, 1, 2, 0.3, // splat3 (near slice → visible)
  ]);
  const one = [2, 1, 3, 0, 0, 2, 0.5, 0.5, 0, 4];
  const cholesky = new Float32Array([...one, ...one, ...one, ...one]);
  const amplitudes = new Float32Array([1.0, 0.5, 1.0, 0.8]);
  const discreteVisibility = new Uint8Array([1, 1, 0, 1]);

  // Attenuated amplitude of splat3: hidden dim 3 of the factor above
  // marginalizes to Σ₃₃ = 0.5² + 0.5² + 0² + 4² = 16.5, so the 1×1 factor is
  // √16.5 = 4.0620192 and D = 0.3 / 4.0620192 = 0.0738549. The shifted Gaussian
  // at truncate = 3 (shiftC = e^-4.5) is
  // (exp(-D²/2) - shiftC) / (1 - shiftC) = 0.9972458, so the raw 0.8 emerges as
  // 0.7977967. splat0 sits on the slice, so attenuation is exactly 1.
  const SPLAT3_AMPLITUDE = 0.7977967;

  const project = (colorComponents: number, colors: number[]) => {
    const centers = new Float32Array(splatCount * 3);
    const chol = new Float32Array(splatCount * 6);
    const outColors = new Float32Array(splatCount * colorComponents);
    const amps = new Float32Array(splatCount);
    const count = project_gsplats_nd_to_3d(
      positions,
      cholesky,
      amplitudes,
      new Float32Array(colors),
      discreteVisibility,
      new Float32Array(ndim), // slice at the origin
      new Uint32Array([3]), // dim 3 is the continuous hidden dim
      new Uint32Array([0, 1, 2]),
      ndim,
      splatCount,
      colorComponents,
      0.001,
      3.0,
      centers,
      chol,
      amps,
      outColors
    );
    return { count, centers, chol, amps, outColors };
  };

  it('compacts a non-contiguous visible pair at both RGB and RGBA stride', () => {
    // prettier-ignore
    const rgb = project(3, [
      0.1, 0.2, 0.3, // splat0
      0.4, 0.5, 0.6, // splat1
      0.7, 0.8, 0.9, // splat2
      0.15, 0.25, 0.35, // splat3
    ]);

    expect(rgb.count).toBe(2);
    // Centers in display order for splat0, splat3 — splat3 lands in slot 1,
    // not slot 3.
    expect(Array.from(rgb.centers.subarray(0, 6))).toEqual([0, 0, 0, 3, 1, 2]);
    // Display marginal over dims [0,1,2] of the shared factor: Σ_S =
    // [[4,2,0],[2,10,0],[0,0,4]] → packed [2,1,3,0,0,2]. Checked at dense slots
    // 0 AND 1 — slot 1 stays all-zero if the out index misses the compaction.
    // Colour stride cannot affect it, so the RGB run alone pins it.
    const expectedChol = [2, 1, 3, 0, 0, 2];
    for (let s = 0; s < rgb.count; s++) {
      for (let k = 0; k < expectedChol.length; k++) {
        expect(rgb.chol[s * 6 + k]).toBeCloseTo(expectedChol[k], 6);
      }
    }
    const expectedRgb = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35];
    for (let k = 0; k < expectedRgb.length; k++) {
      expect(rgb.outColors[k]).toBeCloseTo(expectedRgb[k], 6);
    }
    expect(rgb.amps[0]).toBe(1.0);
    // Same 1e-5-grade band as the Rust twin: tight enough to reject a wrong
    // exponent (exp(-D²) would give 0.7955994) and the raw-L33 shortcut that
    // skips the marginalization (4 → 0.7977279), loose enough for f32 rounding.
    expect(rgb.amps[1]).toBeCloseTo(SPLAT3_AMPLITUDE, 5);

    // ---- RGBA: alpha rides along in the same dense slot at stride 4 ----
    // prettier-ignore
    const rgba = project(4, [
      0.1, 0.2, 0.3, 0.9, // splat0 (alpha 0.9 ≠ r 0.1)
      0.4, 0.5, 0.6, 0.6, // splat1
      0.7, 0.8, 0.9, 0.3, // splat2
      0.15, 0.25, 0.35, 0.85, // splat3
    ]);

    expect(rgba.count).toBe(2);
    expect(Array.from(rgba.centers.subarray(0, 6))).toEqual([0, 0, 0, 3, 1, 2]);
    const expectedRgba = [0.1, 0.2, 0.3, 0.9, 0.15, 0.25, 0.35, 0.85];
    for (let k = 0; k < expectedRgba.length; k++) {
      expect(rgba.outColors[k]).toBeCloseTo(expectedRgba[k], 6);
    }
    expect(rgba.amps[0]).toBe(1.0);
    expect(rgba.amps[1]).toBeCloseTo(SPLAT3_AMPLITUDE, 5);
  });
});

/**
 * Fewer than 3 display dims (2D / 1D scenes).
 *
 * These live in the TS-only suite ON PURPOSE. The cross-language parity tests in
 * `wasm-vs-typescript.test.ts` are `skipIf(!wasmFilesExist)`, so on any machine or
 * CI leg without built WASM artifacts they vanish — and the TS reference is not
 * merely a fallback, it is the production backend for ndim > 16. Without these,
 * a TS-side regression in the display-marginal padding would ship silently.
 *
 * Contract (mirrors `gsplats_processing.rs::compute_display_cholesky_3d`): the n-D
 * marginal occupies the first n·(n+1)/2 packed slots; the remaining rows get zero
 * off-diagonals and a phantom diagonal equal to the geometric mean of the real
 * Cholesky pivots. The phantom is deliberately NOT an epsilon — an ε-thin splat is
 * invisible in sum/additive blending because the shader scales amplitude by the
 * Gaussian's extent along the view ray.
 */
describe('gsplats_processing: fewer than 3 display dims (TS reference)', () => {
  const SQRT_EPS = Math.sqrt(1e-10);

  // Run the fused kernel on a single all-visible splat (no discrete gate) and
  // return its 6-slot display Cholesky. Drives the phantom-padding contract
  // through the ONLY surviving entry point (extract_visible_cholesky_3d is gone).
  const cholFor = (
    cholesky: number[],
    ndim: number,
    displayDims: number[],
    continuousHiddenDims: number[] = []
  ): Float32Array => {
    const chol = new Float32Array(6);
    project_gsplats_nd_to_3d(
      new Float32Array(ndim), // positions: splat at origin (on slice → no attenuation)
      new Float32Array(cholesky),
      new Float32Array([1.0]),
      new Float32Array([1, 1, 1]),
      new Uint8Array([1]),
      new Float32Array(ndim),
      new Uint32Array(continuousHiddenDims),
      new Uint32Array(displayDims),
      ndim,
      1,
      3,
      1e-6,
      3.0,
      new Float32Array(3),
      chol,
      new Float32Array(1),
      new Float32Array(3)
    );
    return chol;
  };

  it('project_gsplats_nd_to_3d handles a 2D scene (no OOB read, z-padded centers)', () => {
    const one = [2.0, 0.5, 1.5];
    const splatCount = 2;
    const centers = new Float32Array(splatCount * 3);
    const chol = new Float32Array(splatCount * 6);
    const amps = new Float32Array(splatCount);
    const cols = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      new Float32Array([0, 0, 5, -3]),
      new Float32Array([...one, ...one]),
      new Float32Array([1.0, 0.8]),
      new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Uint32Array([]), // no hidden dims
      new Uint32Array([0, 1]), // 2D scene: two display dims
      2,
      splatCount,
      3,
      1e-6,
      3.0,
      centers,
      chol,
      amps,
      cols
    );

    expect(count).toBe(2);
    // Centers zero-padded in the missing z component.
    expect(Array.from(centers.subarray(0, 6))).toEqual([0, 0, 0, 5, -3, 0]);
    for (let s = 0; s < 2; s++) {
      const c = s * 6;
      expect(chol[c + 3]).toBe(0);
      expect(chol[c + 4]).toBe(0);
      expect(chol[c + 5]).toBeCloseTo(Math.sqrt(2.0 * 1.5), 5);
    }
  });

  it('pads BOTH missing rows for a single display dim', () => {
    // ndim=2, display=[0] only; dim 1 is a continuous hidden dim on the slice
    // (position 0) so the splat is not attenuated. The 1-D marginal over dim 0
    // is [sqrt(L00²)] = [2]; the phantom = 2 fills both padded rows.
    const chol = cholFor([2.0, 0.5, 1.5], 2, [0], [1]);
    expect(Array.from(chol)).toEqual([2, 0, 2, 0, 0, 2]);
  });

  it('keeps the phantom axis proportional to the splat (not a constant)', () => {
    const phantomFor = (scale: number): number =>
      cholFor([2.0 * scale, 0.5 * scale, 1.5 * scale], 2, [0, 1])[5];

    expect(phantomFor(2) / phantomFor(1)).toBeCloseTo(2.0, 4);
  });

  it('yields a finite positive phantom for a fully degenerate marginal', () => {
    // Crout floors the diagonals at sqrt(eps) before the mean is taken, so the
    // covariance stays non-singular instead of collapsing to zero.
    const chol = cholFor([0, 0, 0], 2, [0, 1]);
    expect(Number.isFinite(chol[5])).toBe(true);
    expect(chol[5]).toBeGreaterThan(0);
    expect(chol[5]).toBeCloseTo(SQRT_EPS, 10);
  });

  it('is rotation-invariant: an in-plane rotation does not change the phantom axis', () => {
    const [sx, sy] = [3.0, 0.75];
    const phantoms = [0, 17, 45, 73, 90].map((deg) => {
      const t = (deg * Math.PI) / 180;
      const [s, c] = [Math.sin(t), Math.cos(t)];
      const a = c * c * sx * sx + s * s * sy * sy;
      const b = c * s * (sx * sx - sy * sy);
      const d = s * s * sx * sx + c * c * sy * sy;
      const l00 = Math.sqrt(a);
      const l10 = b / l00;
      const l11 = Math.sqrt(d - l10 * l10);
      return cholFor([l00, l10, l11], 2, [0, 1])[5];
    });

    // Taken over the Cholesky pivots the mean is (det Sigma)^(1/4) = sqrt(sx*sy).
    for (const p of phantoms) {
      expect(p).toBeCloseTo(Math.sqrt(sx * sy), 4);
    }
  });
});

/**
 * Regression for issue #725: the TS reference is the UNCAPPED >16-dimension
 * backend (`pickBackend` routes ndim > MAX_SUPPORTED_DIMS here because the WASM
 * kernel panics above 16), yet its scratch buffers and the marginal-covariance
 * matrix stride were hardwired to MAX_SUPPORTED_DIMS (16). With more than 16
 * CONTINUOUS hidden dims the workspaces overflowed, the marginal Cholesky read
 * garbage, and amplitudes came back NaN — and since `NaN < minAmplitude` is
 * false EVERY splat was emitted as "visible" with a NaN amplitude (silent
 * corruption, no error).
 */
describe('gsplats_processing: >16 continuous hidden dims (issue #725)', () => {
  // Packed lower-triangular index, matching the module's packedIndex().
  const packedIdx = (row: number, col: number): number => (row * (row + 1)) / 2 + col;

  // Identity Cholesky (L = I) packed for `ndim` dims: unit diagonal, zero off-diag.
  const identityPackedCholesky = (ndim: number): Float32Array => {
    const packed = new Float32Array((ndim * (ndim + 1)) / 2);
    for (let i = 0; i < ndim; i++) packed[packedIdx(i, i)] = 1.0;
    return packed;
  };

  it('returns finite ~1.0 amplitudes for on-slice splats at ndim=20 (17 continuous hidden dims)', () => {
    const ndim = 20;
    const displayDims = new Uint32Array([0, 1, 2]);
    // Dims 3..19 (17 dims) are continuous hidden — one MORE than MAX_SUPPORTED_DIMS.
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));
    expect(continuousHiddenDims.length).toBe(17);

    const splatCount = 2;
    const fullPacked = identityPackedCholesky(ndim);
    // Both splats sit exactly on the slice (all coords 0) → hidden distance 0.
    const positions = new Float32Array(splatCount * ndim); // all zeros
    const cholesky = new Float32Array(splatCount * fullPacked.length);
    cholesky.set(fullPacked, 0);
    cholesky.set(fullPacked, fullPacked.length);
    const amplitudes = new Float32Array([1.0, 1.0]);
    const colors = new Float32Array(splatCount * 3).fill(1.0);
    const discreteVisibility = new Uint8Array([1, 1]);
    const slicePosition = new Float32Array(ndim); // all zeros

    const outCenters3d = new Float32Array(splatCount * 3);
    const outCholesky3d = new Float32Array(splatCount * 6);
    const outAmplitudes = new Float32Array(splatCount);
    const outColors = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      positions,
      cholesky,
      amplitudes,
      colors,
      discreteVisibility,
      slicePosition,
      continuousHiddenDims,
      displayDims,
      ndim,
      splatCount,
      3,
      1e-6,
      4.0,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    // On the buggy code the workspaces overflow, mahalDist is NaN, attenuation is
    // NaN, and `NaN < minAmplitude` is false → both splats emitted with NaN amps.
    // The NaN emission would still yield count===2, so the real assertion is that
    // the amplitudes are FINITE and ~1.0 (not NaN).
    expect(count).toBe(2);
    expect(Number.isFinite(outAmplitudes[0])).toBe(true);
    expect(Number.isFinite(outAmplitudes[1])).toBe(true);
    expect(outAmplitudes[0]).toBeCloseTo(1.0, 5);
    expect(outAmplitudes[1]).toBeCloseTo(1.0, 5);

    // GROWN-BUFFER REUSE: after the 17-D hidden marginal grows the workspaces,
    // the SAME call does a subNdim=3 display marginal (computeDisplayCholesky3D).
    // With an identity input the packed-3D display Cholesky must be exactly the
    // identity [1,0,1,0,0,1] — proving the grown workspaces are re-zeroed and no
    // stale 17-D data leaks into the later 3-D computation.
    for (let s = 0; s < 2; s++) {
      const c = s * 6;
      expect(Array.from(outCholesky3d.subarray(c, c + 6))).toEqual([1, 0, 1, 0, 0, 1]);
    }
  });

  it('attenuates a splat far off the slice in a hidden dim beyond index 16', () => {
    const ndim = 20;
    const displayDims = new Uint32Array([0, 1, 2]);
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));

    const splatCount = 3;
    const fullPacked = identityPackedCholesky(ndim);
    const cholesky = new Float32Array(splatCount * fullPacked.length);
    for (let s = 0; s < splatCount; s++) cholesky.set(fullPacked, s * fullPacked.length);

    const positions = new Float32Array(splatCount * ndim); // start all on-slice
    // Splat 2 sits far away in dim 19 — a CONTINUOUS hidden dim past index 16, so
    // its distance is only computed correctly if the >16-D marginal is right.
    positions[2 * ndim + 19] = 10.0;

    const amplitudes = new Float32Array([1.0, 1.0, 1.0]);
    const colors = new Float32Array(splatCount * 3).fill(1.0);
    const discreteVisibility = new Uint8Array([1, 1, 1]);
    const slicePosition = new Float32Array(ndim);

    const outCenters3d = new Float32Array(splatCount * 3);
    const outCholesky3d = new Float32Array(splatCount * 6);
    const outAmplitudes = new Float32Array(splatCount);
    const outColors = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      positions,
      cholesky,
      amplitudes,
      colors,
      discreteVisibility,
      slicePosition,
      continuousHiddenDims,
      displayDims,
      ndim,
      splatCount,
      3,
      1e-6,
      4.0,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    // Only the two on-slice splats survive. If the >16-D distance were silently
    // zero (as under the overflow), the far splat would pass with attenuation 1.0
    // and count would be 3.
    expect(count).toBe(2);
    expect(outAmplitudes[0]).toBeCloseTo(1.0, 5);
    expect(outAmplitudes[1]).toBeCloseTo(1.0, 5);
  });

  // A 20-D identity Cholesky with ONE off-diagonal in the hidden region:
  // L[19,18] = 1.5 (dims 18,19 are hidden indices 15,16). The hidden 17×17
  // marginal is identity except a 2×2 block on (15,16):
  //   Σ = [[1, 1.5], [1.5, 1 + 1.5²]] = [[1, 1.5], [1.5, 3.25]]  (det = 1).
  const correlatedNdim = 20;
  const correlatedOffDiag = 1.5;
  const correlatedPacked = (): Float32Array => {
    const packed = identityPackedCholesky(correlatedNdim);
    packed[packedIdx(19, 18)] = correlatedOffDiag; // L[19,18]
    return packed;
  };

  it('computes the CORRELATED >16-D marginal Cholesky (pins the stride, not just overflow)', () => {
    // Direct computeMarginalCholesky over the 17 hidden dims 3..19.
    const hidden = new Uint32Array(Array.from({ length: correlatedNdim - 3 }, (_, k) => k + 3));
    expect(hidden.length).toBe(17);
    const subPacked = (hidden.length * (hidden.length + 1)) / 2; // 153
    const out = new Float32Array(subPacked);

    computeMarginalCholesky(correlatedPacked(), 0, hidden, hidden.length, out, 0);

    // Cholesky of the 2×2 block: L_S[15,15]=1, L_S[16,15]=1.5, L_S[16,16]=√(3.25-2.25)=1.
    // (hidden index 15 = dim 18, hidden index 16 = dim 19.)
    expect(out[packedIdx(15, 15)]).toBeCloseTo(1.0, 5);
    expect(out[packedIdx(16, 15)]).toBeCloseTo(1.5, 5);
    expect(out[packedIdx(16, 16)]).toBeCloseTo(1.0, 5);
    // Every other diagonal is the untouched identity, and the coupling does NOT
    // bleed into neighbouring cells. A wrong stride scrambles these to finite
    // garbage, so this fails on a stride regression even though the NaN test won't.
    expect(out[packedIdx(14, 14)]).toBeCloseTo(1.0, 5);
    expect(out[packedIdx(16, 14)]).toBeCloseTo(0.0, 5);
    expect(out[packedIdx(15, 14)]).toBeCloseTo(0.0, 5);
  });

  it('applies the CORRELATED >16-D attenuation to a splat offset in a hidden dim past index 16', () => {
    const ndim = correlatedNdim;
    const displayDims = new Uint32Array([0, 1, 2]);
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));

    // Offset ONLY in dim 18 (hidden index 15), the correlated column. With the
    // marginal above, D² = δ²·(Σ⁻¹)[15,15] = δ²·(3.25/det) = 3.25·δ² — distinct
    // from the identity backend's δ². δ = 0.5 → D² = 0.8125.
    const delta = 0.5;
    const packed = correlatedPacked();
    const positions = new Float32Array(ndim);
    positions[18] = delta;

    const truncate = 4.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);
    const d2 = 3.25 * delta * delta;
    const expectedAtten = Math.max(0.0, invOneMinusC * (Math.exp(-0.5 * d2) - shiftC));
    // Sanity: the correlated answer is clearly separated from the identity one.
    const identityAtten = Math.max(0.0, invOneMinusC * (Math.exp(-0.5 * delta * delta) - shiftC));
    expect(Math.abs(expectedAtten - identityAtten)).toBeGreaterThan(0.1);

    const outCenters3d = new Float32Array(3);
    const outCholesky3d = new Float32Array(6);
    const outAmplitudes = new Float32Array(1);
    const outColors = new Float32Array(3);

    const count = project_gsplats_nd_to_3d(
      positions,
      packed,
      new Float32Array([1.0]),
      new Float32Array(3).fill(1.0),
      new Uint8Array([1]),
      new Float32Array(ndim),
      continuousHiddenDims,
      displayDims,
      ndim,
      1,
      3,
      1e-6,
      truncate,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    expect(count).toBe(1);
    expect(outAmplitudes[0]).toBeCloseTo(expectedAtten, 5);
  });

  it('INTERLEAVE: a small subNdim=2 marginal after a >16-D one is uncorrupted', () => {
    // First a >16-D correlated marginal — this GROWS the module workspaces.
    const hidden = new Uint32Array(Array.from({ length: correlatedNdim - 3 }, (_, k) => k + 3));
    const bigOut = new Float32Array((hidden.length * (hidden.length + 1)) / 2);
    computeMarginalCholesky(correlatedPacked(), 0, hidden, hidden.length, bigOut, 0);

    // Then a small subNdim=2 marginal on an UNRELATED 3-D correlated factor
    // (the exact case from the earlier `should produce correct marginal` test).
    // If grow-then-reuse leaked stale 17-D data, this small result would be wrong.
    const small = new Float32Array([2, 1, 3, 0.5, 0.5, 4]); // L=[[2,0,0],[1,3,0],[0.5,0.5,4]]
    const smallOut = new Float32Array(3);
    computeMarginalCholesky(small, 0, new Uint32Array([0, 2]), 2, smallOut, 0);

    // Σ_S = [[4,1],[1,16.5]] → L_S = [2, 0.5, √16.25].
    expect(smallOut[0]).toBeCloseTo(2.0, 4);
    expect(smallOut[1]).toBeCloseTo(0.5, 4);
    expect(smallOut[2]).toBeCloseTo(Math.sqrt(16.25), 3);
  });
});
