/**
 * GSplats Processing for nD → 3D Conversion
 *
 * TypeScript reference implementation matching gsplats_processing.rs
 *
 * High-performance implementations of GSplats processing operations:
 * - Mahalanobis distance computation (forward substitution)
 * - Marginal-covariance Cholesky factorization for a dimension subset
 *   (`computeMarginalCholesky`) + display-marginal padding for 1D/2D scenes
 * - Fused nD→3D projection (`project_gsplats_nd_to_3d`): attenuation,
 *   visibility, and compaction in a single pass
 */

import { MAX_SUPPORTED_DIMS } from '../../config/constants';

/** Maximum packed Cholesky size for MAX_SUPPORTED_DIMS. */
const MAX_PACKED_CHOLESKY_SIZE = (MAX_SUPPORTED_DIMS * (MAX_SUPPORTED_DIMS + 1)) / 2;

/** Epsilon for degenerate diagonal detection during Cholesky factorization. */
const CHOLESKY_EPSILON = 1e-10;

/**
 * RELATIVE floor for degenerate-variance detection, applied against the largest
 * diagonal of the covariance being factorized. Keeps the degeneracy test a
 * condition-number bound rather than a scene-scale one. MUST stay identical to
 * `CHOLESKY_RELATIVE_EPSILON` in `wasm/rust/src/common.rs`.
 */
const CHOLESKY_RELATIVE_EPSILON = 1e-12;

// Module-level workspace buffers — safe because JS is single-threaded.
// Avoids per-call allocation in hot loops. Sized for the common ndim <= 16 case
// so that path never reallocates; grown on demand only when the number of
// continuous hidden dims exceeds MAX_SUPPORTED_DIMS (the uncapped >16-D backend).
let _sigmaWorkspace = new Float32Array(MAX_SUPPORTED_DIMS * MAX_SUPPORTED_DIMS);
let _lSubWorkspace = new Float32Array(MAX_PACKED_CHOLESKY_SIZE);

/**
 * Return `buf` unchanged when it already holds at least `n` elements, otherwise a
 * freshly-allocated larger Float32Array. Keeps the ndim <= 16 hot path
 * allocation-free — the module-level buffers start at the 16-D sizes, so the
 * common case never reallocates.
 */
function ensureCapacity(buf: Float32Array<ArrayBuffer>, n: number): Float32Array<ArrayBuffer> {
  return buf.length >= n ? buf : new Float32Array(n);
}

/**
 * Compute the packed index for a Cholesky element L[row, col].
 * Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
 * Formula: row * (row + 1) / 2 + col (for col <= row)
 */
function packedIndex(row: number, col: number): number {
  return (row * (row + 1)) / 2 + col;
}

/**
 * Compute the correct marginal Cholesky factor for a subset of dimensions.
 *
 * For Σ = L·Lᵀ, the marginal covariance for dimensions S is:
 *   Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
 *
 * This function reconstructs Σ_S and then Cholesky-factorizes it.
 * Simply extracting the raw L row/column sub-matrix is INCORRECT when there are
 * cross-dimension correlations.
 *
 * @param fullPackedL - Full packed Cholesky factor array
 * @param fullPackedOffset - Offset into fullPackedL for this splat
 * @param keepDims - Ordered dimension indices to keep; their order defines the
 *   output-axis order
 * @param subNdim - Number of dimensions to keep
 * @param output - Output packed marginal Cholesky [subNdim*(subNdim+1)/2]
 * @param outputOffset - Start offset in output array
 */
export function computeMarginalCholesky(
  fullPackedL: Float32Array,
  fullPackedOffset: number,
  keepDims: Uint32Array | number[],
  subNdim: number,
  output: Float32Array,
  outputOffset: number
): void {
  // Step 1: Reconstruct marginal covariance Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
  // Uses module-level workspace buffer (zeroed below, safe in single-threaded JS).
  // The sigma matrix is subNdim x subNdim, so its row stride is subNdim — NOT the
  // fixed MAX_SUPPORTED_DIMS. Grow the workspace when subNdim > 16 (the >16-D
  // backend); the <= 16 case keeps the pre-allocated buffer (no reallocation).
  const stride = subNdim;
  const subPackedSize = (subNdim * (subNdim + 1)) / 2;
  _sigmaWorkspace = ensureCapacity(_sigmaWorkspace, stride * stride);
  _lSubWorkspace = ensureCapacity(_lSubWorkspace, subPackedSize);
  const sigma = _sigmaWorkspace;
  sigma.fill(0, 0, stride * stride);

  for (let i = 0; i < subNdim; i++) {
    const si = keepDims[i];
    for (let j = 0; j <= i; j++) {
      const sj = keepDims[j];
      const kMax = Math.min(si, sj);
      let sum = 0;
      for (let k = 0; k <= kMax; k++) {
        const lSiK = fullPackedL[fullPackedOffset + packedIndex(si, k)];
        const lSjK = fullPackedL[fullPackedOffset + packedIndex(sj, k)];
        sum += lSiK * lSjK;
      }
      sigma[i * stride + j] = sum;
      sigma[j * stride + i] = sum;
    }
  }

  // Step 2: Cholesky-Crout factorization of Σ_S → L_S
  const lSub = _lSubWorkspace;
  lSub.fill(0, 0, subPackedSize);

  // SCALE-RELATIVE degeneracy floor, mirroring
  // `gsplats_processing.rs::compute_marginal_cholesky` 1:1. A variance is in
  // world-units², so an ABSOLUTE floor conflates "this axis has no extent" with
  // "this scene uses small units": at a fixed 1e-10 a splat with σ = 1e-7
  // (nm-unit data) has variance 1e-14, trips the floor, and is inflated to
  // σ = 1e-5 — 100× larger than authored. Anchoring to the largest diagonal
  // turns the test into a pure condition-number check that behaves identically
  // at every scene scale; CHOLESKY_EPSILON stays the absolute backstop for an
  // all-zero Σ_S.
  let maxDiag = 0;
  for (let i = 0; i < subNdim; i++) {
    const d = sigma[i * stride + i];
    if (d > maxDiag) maxDiag = d;
  }
  // The absolute constant is a fallback for a SCALELESS (all-zero) Σ_S only —
  // as a general lower bound it would re-impose the scene-scale threshold this
  // replaces, since maxDiag * 1e-12 is below 1e-10 for any σ < ~1e-1.
  // MIN_VALUE keeps the floor non-zero if the relative product underflows.
  const degenerateFloor =
    maxDiag > 0
      ? Math.max(maxDiag * CHOLESKY_RELATIVE_EPSILON, Number.MIN_VALUE)
      : CHOLESKY_EPSILON;

  for (let i = 0; i < subNdim; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = sigma[i * stride + j];
      for (let k = 0; k < j; k++) {
        sum -= lSub[packedIndex(i, k)] * lSub[packedIndex(j, k)];
      }
      if (i === j) {
        lSub[packedIndex(i, i)] =
          sum > degenerateFloor ? Math.sqrt(sum) : Math.sqrt(degenerateFloor);
      } else {
        const diag = lSub[packedIndex(j, j)];
        lSub[packedIndex(i, j)] = diag > 0 ? sum / diag : 0;
      }
    }
  }

  // Copy to output
  for (let k = 0; k < subPackedSize; k++) {
    output[outputOffset + k] = lSub[k];
  }
}

/**
 * Marginal 3D Cholesky for the display dims, padded to the packed-3D layout
 * when fewer than 3 dims are displayed (1D/2D scenes).
 *
 * Mirrors `gsplats_processing.rs::compute_display_cholesky_3d` 1:1. For
 * `n = min(displayDims.length, 3)` the packed n-D marginal occupies the first
 * n·(n+1)/2 slots of the packed-3D layout verbatim. The renderer always consumes
 * a 3×3 covariance (Σ = L·Lᵀ), so the rows for display axes the data doesn't
 * have must still be filled: off-diagonals are 0 (the phantom axis is
 * uncorrelated with the real ones, so the in-plane profile is untouched) and the
 * diagonal is the GEOMETRIC MEAN of the real diagonals — the phantom axis gets
 * the splat's own in-plane scale, making a 2D splat a round blob rather than a
 * disk.
 *
 * The diagonal deliberately is NOT a small epsilon. In sum/additive projection
 * the shader scales amplitude by the Gaussian's extent along the view ray,
 * `sigmaRay = 1/√(rᵀΣ⁻¹r)` (`shader-glsl.ts`); a face-on ε-thin splat gets
 * `sigmaRay ≈ √ε`, i.e. amplitude × 1e-5, and the whole scene renders black.
 *
 * Writes 6 elements at `outputOffset`.
 */
function computeDisplayCholesky3D(
  fullPackedL: Float32Array,
  fullPackedOffset: number,
  displayDims: Uint32Array,
  output: Float32Array,
  outputOffset: number
): void {
  // `computeMarginalCholesky` reads only keepDims[0..n), so pass displayDims
  // whole — a `.subarray(0, n)` view would allocate once PER SPLAT here.
  const n = Math.min(displayDims.length, 3);
  computeMarginalCholesky(fullPackedL, fullPackedOffset, displayDims, n, output, outputOffset);
  if (n === 3) return;

  // Geometric mean of the real diagonals L[i,i], i < n. Falls back to the
  // degenerate-covariance regularizer when the marginal has no extent at all.
  //
  // The test is `> 0`, not `> CHOLESKY_EPSILON`: the Crout step above already
  // floors every diagonal to a strictly positive, SCALE-RELATIVE value, so an
  // absolute threshold here would drop legitimately tiny diagonals (σ < 1e-10)
  // from the mean — reintroducing the scene-scale dependence in miniature.
  let logSum = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const diag = output[outputOffset + packedIndex(i, i)];
    if (diag > 0) {
      logSum += Math.log(diag);
      counted++;
    }
  }
  const phantom = counted > 0 ? Math.exp(logSum / counted) : Math.sqrt(CHOLESKY_EPSILON);

  let idx = outputOffset + (n * (n + 1)) / 2;
  for (let row = n; row < 3; row++) {
    for (let col = 0; col < row; col++) {
      output[idx++] = 0;
    }
    output[idx++] = phantom;
  }
}

/**
 * Compute Mahalanobis distance for a single point using packed Cholesky factor.
 *
 * Given L (lower-triangular Cholesky of covariance),
 * Mahalanobis distance = ||L⁻¹ · (x - μ)||
 *
 * We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
 *
 * @param diff - Difference vector (x - μ) for the dimensions [ndim]
 * @param packedL - Packed Cholesky factor [packedSize]
 * @param ndim - Dimensionality of the Cholesky
 * @returns Mahalanobis distance
 */
export function mahalanobis_distance(
  diff: Float32Array,
  packedL: Float32Array,
  ndim: number
): number {
  // Forward substitution: solve L · y = diff
  const y = new Float32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[packedIndex(i, j)] * y[j];
    }
    const diag = packedL[packedIndex(i, i)];
    y[i] = diag > 1e-10 ? val / diag : 0;
  }

  // Compute ||y||
  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += y[i] * y[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Internal Mahalanobis distance (matches Rust internal function).
 * Accepts an optional pre-allocated buffer to avoid per-call allocation.
 */
function mahalanobisDistanceInternal(
  diff: Float32Array,
  packedL: Float32Array,
  ndim: number,
  yBuffer?: Float32Array
): number {
  const y = yBuffer ?? new Float32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[packedIndex(i, j)] * y[j];
    }
    const diag = packedL[packedIndex(i, i)];
    y[i] = diag > 1e-10 ? val / diag : 0;
  }

  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += y[i] * y[i];
  }
  return Math.sqrt(sumSq);
}

// Module-level forward-substitution + marginal-Cholesky scratch for the fused
// kernel (single-threaded JS — safe to reuse across the splat loop). Sized for
// ndim <= 16; grown on demand when numContinuous > 16 (the >16-D backend).
let _fusedDiff = new Float32Array(MAX_SUPPORTED_DIMS);
let _fusedY = new Float32Array(MAX_SUPPORTED_DIMS);
let _fusedHiddenCholesky = new Float32Array(MAX_PACKED_CHOLESKY_SIZE);

/**
 * Fused nD→3D GSplat projection — TypeScript reference mirroring
 * `gsplats_processing.rs::project_gsplats_nd_to_3d`.
 *
 * Single pass over the splats: discrete-visibility gate → continuous
 * attenuation (marginal Cholesky + shifted Gaussian) → visibility decision
 * (`amplitude * attenuation >= minAmplitude`) → write COMPACTED outputs
 * (visible centers3D, cholesky3D[6], attenuated amplitudes, colors). Bit-for-bit
 * equivalent to the legacy 6-call pipeline (reuses the same
 * `computeMarginalCholesky` / `mahalanobisDistanceInternal` helpers in the same
 * order). Replaces ~5 full passes and the repeated large-array copies.
 *
 * Colors are pre-normalized to f32 by the caller (white-filled when absent);
 * `colorComponents` is 3 (RGB) or 4 (RGBA — alpha is per-splat opacity and
 * compacts with its splat). Outputs are sized for the `splatCount` worst case;
 * the caller slices each to the returned visible count.
 *
 * @returns Number of visible splats written.
 */
export function project_gsplats_nd_to_3d(
  positions: Float32Array,
  cholesky: Float32Array,
  amplitudes: Float32Array,
  colors: Float32Array,
  discreteVisibility: Uint8Array,
  slicePosition: Float32Array,
  continuousHiddenDims: Uint32Array,
  displayDims: Uint32Array,
  ndim: number,
  splatCount: number,
  colorComponents: number,
  minAmplitude: number,
  truncate: number,
  outCenters3d: Float32Array,
  outCholesky3d: Float32Array,
  outAmplitudes: Float32Array,
  outColors: Float32Array
): number {
  if (colorComponents !== 3 && colorComponents !== 4) {
    throw new Error('project_gsplats_nd_to_3d: colorComponents must be 3 (RGB) or 4 (RGBA)');
  }
  const numContinuous = continuousHiddenDims.length;
  const numDisplay = Math.min(displayDims.length, 3);
  const fullPackedSize = (ndim * (ndim + 1)) / 2;

  const shiftC = Math.exp(-0.5 * truncate * truncate);
  const invOneMinusC = 1.0 / (1.0 - shiftC);

  // Grow the fused scratch when numContinuous > 16 (the uncapped >16-D backend);
  // the <= 16 case keeps the pre-allocated buffers (no reallocation). Re-bind the
  // locals AFTER growing so diff/hiddenCholesky/_fusedY point at the grown arrays.
  const continuousPackedSize = (numContinuous * (numContinuous + 1)) / 2;
  _fusedDiff = ensureCapacity(_fusedDiff, numContinuous);
  _fusedY = ensureCapacity(_fusedY, numContinuous);
  _fusedHiddenCholesky = ensureCapacity(_fusedHiddenCholesky, continuousPackedSize);
  const diff = _fusedDiff;
  const hiddenCholesky = _fusedHiddenCholesky;

  let out = 0;

  for (let i = 0; i < splatCount; i++) {
    // (1) Discrete gate first.
    if (discreteVisibility[i] === 0) continue;

    const centerOffset = i * ndim;
    const choleskyOffset = i * fullPackedSize;

    // (2) Continuous attenuation: marginal Cholesky over the hidden dims,
    //     Mahalanobis distance, then the shifted Gaussian.
    let attenuation: number;
    if (numContinuous === 0) {
      attenuation = 1.0;
    } else {
      for (let hIdx = 0; hIdx < numContinuous; hIdx++) {
        const d = continuousHiddenDims[hIdx];
        diff[hIdx] = slicePosition[d] - positions[centerOffset + d];
      }
      computeMarginalCholesky(
        cholesky,
        choleskyOffset,
        continuousHiddenDims,
        numContinuous,
        hiddenCholesky,
        0
      );
      // Pass `diff` whole — mahalanobisDistanceInternal reads only [0, ndim), so a
      // `.subarray(0, numContinuous)` view would allocate once PER SPLAT here.
      const mahalDist = mahalanobisDistanceInternal(diff, hiddenCholesky, numContinuous, _fusedY);
      const rawExp = Math.exp(-0.5 * mahalDist * mahalDist);
      attenuation = Math.max(0.0, invOneMinusC * (rawExp - shiftC));
    }

    // (3) Visibility decision.
    const attenuatedAmplitude = amplitudes[i] * attenuation;
    if (attenuatedAmplitude < minAmplitude) continue;

    // (4) Write compacted outputs at dense slot `out`.
    const cOff = out * 3;
    for (let j = 0; j < numDisplay; j++) {
      outCenters3d[cOff + j] = positions[centerOffset + displayDims[j]];
    }
    for (let j = numDisplay; j < 3; j++) {
      outCenters3d[cOff + j] = 0.0;
    }

    computeDisplayCholesky3D(cholesky, choleskyOffset, displayDims, outCholesky3d, out * 6);

    outAmplitudes[out] = attenuatedAmplitude;

    const colOff = out * colorComponents;
    const colSrc = i * colorComponents;
    for (let c = 0; c < colorComponents; c++) {
      outColors[colOff + c] = colors[colSrc + c];
    }

    out++;
  }

  return out;
}
