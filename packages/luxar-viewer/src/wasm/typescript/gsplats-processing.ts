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

import {
  GSPLAT_CHOLESKY_EPSILON as CHOLESKY_EPSILON,
  MAX_SUPPORTED_DIMS,
} from '../../config/constants';

/** Maximum packed Cholesky size for MAX_SUPPORTED_DIMS. */
const MAX_PACKED_CHOLESKY_SIZE = (MAX_SUPPORTED_DIMS * (MAX_SUPPORTED_DIMS + 1)) / 2;

// `CHOLESKY_EPSILON` (imported above as the local kernel name) is the epsilon for
// degenerate diagonal detection during Cholesky factorization. It is the ONLY thing
// that decides how wide a hidden dim with an all-zero covariance block still
// renders: the Crout step below floors such a pivot at `sqrt(CHOLESKY_EPSILON)`, so
// the splat keeps a σ of 1e-5 along that axis and attenuates to zero only at
// `truncation_radius × 1e-5`. The gsplats chunk-fetch epsilon has to cover that
// band, which is why the value lives in `config/constants.ts` (the
// mirrored-constant home the loaders read) and is imported here rather than
// duplicated. The Rust twin (`wasm/rust/src/common.rs`) is the backend that
// actually runs and keeps its own copy; the two are pinned equal by
// `tests/unit/data/loaders/spatial-query/tolerance-computer.test.ts`.

/**
 * RELATIVE floor for degenerate-variance detection, applied against the largest
 * diagonal of the covariance being factorized. Keeps the degeneracy test a
 * condition-number bound rather than a scene-scale one. MUST stay identical to
 * `CHOLESKY_RELATIVE_EPSILON` in `wasm/rust/src/common.rs`.
 *
 * Rust declares it `pub const CHOLESKY_RELATIVE_EPSILON: f32 = 1e-12`, so the
 * value that actually multiplies `maxDiag` there is f32(1e-12), NOT the f64
 * literal — hence the `Math.fround`. Same reasoning for every other constant in
 * this module that Rust types as `f32`.
 */
const CHOLESKY_RELATIVE_EPSILON = Math.fround(1e-12);

/**
 * `CHOLESKY_EPSILON` as the f32 Rust actually holds it. The shared export in
 * `config/constants.ts` is the mirrored SOURCE value (pinned by
 * `tests/unit/data/loaders/spatial-query/tolerance-computer.test.ts`) and must
 * stay an ordinary f64 literal there; the f32 rounding belongs at the point of
 * use, inside the kernel that has to agree with `common.rs`.
 *
 * This particular narrowing is UNOBSERVABLE and kept for consistency with the
 * three constants around it. Both of its uses take a square root, and
 * `fround(sqrt(fround(1e-10)))` and `fround(sqrt(1e-10))` are the same f32
 * (9.999999747378752e-6); the only other consumer is `sum > degenerateFloor`,
 * and this branch is reached only when `maxDiag === 0`, which forces every
 * diagonal `sum` to be ≤ 0. The all-zero-Σ_S parity case in
 * `tests/unit/wasm/wasm-vs-typescript.test.ts` pins the value the branch
 * produces; no fixture can distinguish the fround itself.
 */
const CHOLESKY_EPSILON_F32 = Math.fround(CHOLESKY_EPSILON);

/**
 * `f32::MIN_POSITIVE` — the smallest positive NORMAL f32, 2⁻¹²⁶.
 *
 * NOT `Number.MIN_VALUE` (≈5e-324, the smallest f64 SUBNORMAL), which is what
 * this file used to clamp with. The two agree only while `maxDiag × 1e-12`
 * stays above 2⁻¹²⁶; below that Rust returns 2⁻¹²⁶ and an f64 clamp returns the
 * product, so the regularized diagonal — and every splat that goes through it —
 * differs grossly rather than by an ulp.
 */
const F32_MIN_POSITIVE = 1.1754943508222875e-38;

/**
 * Degenerate-diagonal epsilon for the forward substitution, mirroring the
 * function-local `const EPSILON: f32 = 1e-10` in
 * `gsplats_processing.rs::mahalanobis_distance_internal`.
 */
const MAHALANOBIS_EPSILON_F32 = Math.fround(1e-10);

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
      // Rust accumulates this dot product in f32, rounding after every product
      // AND every partial sum. The store into `sigma` (a Float32Array) only
      // rounds the FINAL value, so an f64 accumulator here silently carries
      // extra precision through the whole reduction.
      let sum = 0;
      for (let k = 0; k <= kMax; k++) {
        const lSiK = fullPackedL[fullPackedOffset + packedIndex(si, k)];
        const lSjK = fullPackedL[fullPackedOffset + packedIndex(sj, k)];
        sum = Math.fround(sum + Math.fround(lSiK * lSjK));
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
  // f32::MIN_POSITIVE keeps the floor non-zero if the relative product
  // underflows — see F32_MIN_POSITIVE above for why it is not Number.MIN_VALUE.
  const degenerateFloor =
    maxDiag > 0
      ? Math.max(Math.fround(maxDiag * CHOLESKY_RELATIVE_EPSILON), F32_MIN_POSITIVE)
      : CHOLESKY_EPSILON_F32;

  for (let i = 0; i < subNdim; i++) {
    for (let j = 0; j <= i; j++) {
      // Crout: same f32-at-every-step rule as the Σ_S dot product above. This
      // reduction is a difference of like-magnitude terms, so the divergence an
      // f64 accumulator introduces is amplified by the cancellation rather than
      // damped — measured at up to ~3300 ulp against WASM before this fix.
      let sum = sigma[i * stride + j];
      for (let k = 0; k < j; k++) {
        sum = Math.fround(sum - Math.fround(lSub[packedIndex(i, k)] * lSub[packedIndex(j, k)]));
      }
      // The sqrt and the division need no explicit fround: `lSub` is a
      // Float32Array, so the store already rounds, and for √ and ÷ on operands
      // that are exactly f32 the f64-then-f32 double rounding is provably
      // benign (f64's 53 bits ≥ 2·24 + 2). It is only the ACCUMULATORS above
      // that have to be rounded by hand.
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
  //
  // Rust runs the whole reduction in f32: `log_sum += diag.ln()` then
  // `(log_sum / counted as f32).exp()`. Rounding each step matches all of that
  // EXCEPT the transcendentals themselves — see the note on `phantom` below.
  let logSum = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const diag = output[outputOffset + packedIndex(i, i)];
    if (diag > 0) {
      logSum = Math.fround(logSum + Math.fround(Math.log(diag)));
      counted++;
    }
  }
  // RESIDUAL, measured and accepted: `Math.fround(Math.log/exp(x))` is not
  // bit-identical to Rust's `f32::ln`/`f32::exp`, because V8's ieee754 kernels
  // and the wasm libm's `logf`/`expf` are different approximations of the same
  // function. That is a CHOICE, not a wall: porting musl's `logf`/`expf`
  // (what `f32::ln`/`f32::exp` lower to on wasm32) into frounded TS closes it
  // — a ~35-line `expf` port measured 0/200 000 mismatches against the real
  // WASM where `Math.fround(Math.exp(x))` measured 19 282 (9.64%). Deferred to
  // #1830 rather than done here; until then, treat the two transcendentals as
  // the only approximate steps in this file.
  //
  // Measured on the 20 000-splat 2D sweep in
  // `tests/unit/wasm/wasm-vs-typescript.test.ts` (mulberry32(4242) 2×2
  // factors, displayDims [0,1]): the phantom diagonal differs from WASM in
  // 3116/20000 cases, by AT MOST 2 ulp (1 from `ln`, 1 from `exp`); before
  // this fix it was 5344/20000, also at 2 ulp. The rounding's real win here is
  // the other five packed slots, which went 1294/100000 at this (unit) scale →
  // 0/100000. The rescaled variants of the same fixture, discussed just below,
  // went 1278 / 1216 / 1244 per 100000 → 0 at ×1e-4 / ×1e-7 / ×1e6.
  //
  // The 2-ulp figure is a property of that fixture's SCALE, not of the kernel:
  // `phantom = exp(mean(ln Lᵢᵢ))`, so the relative residual is ≈ |ln σ|·2⁻²⁴.
  // Same fixture with the factors rescaled: unit 2 ulp, ×1e-4 16 ulp, ×1e-7
  // 32 ulp, ×1e6 16 ulp. Parity tests over the phantom must be ULP-bounded
  // with a bound derived from the scale they use, never exact.
  //
  // Only the `ln` here is load-bearing. The other three roundings on this path
  // are provably inert and kept solely to mirror `common.rs` step for step:
  // `counted` can only be 1 or 2 (the n === 3 case returned above), division by
  // a power of two is exact, and `phantom` is stored into a Float32Array, which
  // rounds it anyway. Mutating any of them changes nothing in the sweep above;
  // mutating the `ln` moves it to 4484/20000.
  //
  // Two of those three are inert only in EACH OTHER'S PRESENCE, so do not read
  // the list above as three independently dead roundings. The accumulator
  // `fround(logSum + …)` is a no-op because rounding COMMUTES with the exact
  // power-of-two division that follows — it is redundant GIVEN
  // `fround(logSum / counted)`, not on its own. Measured over six scene scales
  // (120000 phantoms): dropping either alone changes 0, dropping BOTH changes
  // 55473. The `exp` rounding is independently inert (the Float32Array store).
  const phantom =
    counted > 0
      ? Math.fround(Math.exp(Math.fround(logSum / counted)))
      : Math.fround(Math.sqrt(CHOLESKY_EPSILON_F32));

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
  // Forward substitution: solve L · y = diff.
  // `y` is a Float32Array, so each SOLVED component is already rounded; the two
  // f64 accumulators (`val` and `sumSq` below) are what has to be frounded by
  // hand to reproduce Rust's f32 arithmetic step for step.
  const y = new Float32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val = Math.fround(val - Math.fround(packedL[packedIndex(i, j)] * y[j]));
    }
    const diag = packedL[packedIndex(i, i)];
    y[i] = diag > MAHALANOBIS_EPSILON_F32 ? val / diag : 0;
  }

  // Compute ||y||
  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq = Math.fround(sumSq + Math.fround(y[i] * y[i]));
  }
  // Rust returns an `f32`; wasm-bindgen hands that to JS as the f64 widening of
  // an f32. Round the result so the two backends return the SAME JS number
  // rather than one that merely compares close.
  return Math.fround(Math.sqrt(sumSq));
}

/**
 * Internal Mahalanobis distance (matches Rust internal function).
 * Accepts an optional pre-allocated buffer to avoid per-call allocation.
 *
 * Same f32 discipline as the exported twin: the accumulators are frounded and
 * the result is rounded to the f32 Rust returns, because the caller squares it
 * and feeds it to `exp` — an f64 residual there moves the attenuation and can
 * flip the `minAmplitude` gate.
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
      val = Math.fround(val - Math.fround(packedL[packedIndex(i, j)] * y[j]));
    }
    const diag = packedL[packedIndex(i, i)];
    y[i] = diag > MAHALANOBIS_EPSILON_F32 ? val / diag : 0;
  }

  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq = Math.fround(sumSq + Math.fround(y[i] * y[i]));
  }
  return Math.fround(Math.sqrt(sumSq));
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
 * `outSourceIndices` records, per emitted splat, the SOURCE index it came from
 * (issue #1423) — compaction destroys that mapping, and picking needs it to
 * translate a storage slot back into an on-disk element index. Pass an EMPTY
 * array to opt out; the recording is then skipped entirely.
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
  outColors: Float32Array,
  outSourceIndices: Uint32Array
): number {
  if (colorComponents !== 3 && colorComponents !== 4) {
    throw new Error('project_gsplats_nd_to_3d: colorComponents must be 3 (RGB) or 4 (RGBA)');
  }
  const numContinuous = continuousHiddenDims.length;
  const numDisplay = Math.min(displayDims.length, 3);
  const fullPackedSize = (ndim * (ndim + 1)) / 2;

  // `truncate` and `minAmplitude` are declared `f32` on the Rust side, so
  // wasm-bindgen narrows them at the boundary. This backend receives the raw
  // f64, so it has to narrow them itself or it thresholds against a value WASM
  // never sees.
  const truncateF32 = Math.fround(truncate);
  const minAmplitudeF32 = Math.fround(minAmplitude);

  // Rust: `(-0.5f32 * truncate * truncate).exp()`, left-associative, every step
  // f32; then `1.0 / (1.0 - shift_c)` — the reciprocal is rounded BEFORE it is
  // used as a multiplier below, so a single fused JS division would differ.
  //
  // `Math.fround(-0.5 * x)` is provably exact for any f32 `x` (halving only
  // decrements the exponent) and is kept for 1:1 symmetry with `common.rs`, not
  // because it can change an answer. The other roundings here DO matter, but
  // only at small `truncate`: `invOneMinusC = 1/(1 - shiftC)` is ~1.01 at
  // truncate 3, where an ulp of `shiftC` cannot move `1 - shiftC` at all, and
  // ~8.5 at truncate 0.5, where it moves the amplitude of every splat. Since
  // `truncate` is author-controlled (a per-dataset `truncation_radius`), the
  // parity fixtures deliberately sweep small values too.
  const shiftC = Math.fround(Math.exp(Math.fround(Math.fround(-0.5 * truncateF32) * truncateF32)));
  const invOneMinusC = Math.fround(1.0 / Math.fround(1.0 - shiftC));

  // Grow the fused scratch when numContinuous > 16 (the uncapped >16-D backend);
  // the <= 16 case keeps the pre-allocated buffers (no reallocation). Re-bind the
  // locals AFTER growing so diff/hiddenCholesky/_fusedY point at the grown arrays.
  const continuousPackedSize = (numContinuous * (numContinuous + 1)) / 2;
  _fusedDiff = ensureCapacity(_fusedDiff, numContinuous);
  _fusedY = ensureCapacity(_fusedY, numContinuous);
  _fusedHiddenCholesky = ensureCapacity(_fusedHiddenCholesky, continuousPackedSize);
  const diff = _fusedDiff;
  const hiddenCholesky = _fusedHiddenCholesky;

  // Empty array = "don't record the source indices" (the opt-out). Hoisted
  // out of the splat loop so the non-recording path pays nothing per splat.
  // Mirrors the Rust twin's `record_source_indices`.
  const recordSourceIndices = outSourceIndices.length > 0;

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
      // Rust: `(-0.5 * mahal_dist * mahal_dist).exp()` — left-associative f32.
      // RESIDUAL: `Math.fround(Math.exp(x))` is NOT bit-identical to Rust's
      // `f32::exp` (V8's ieee754 `exp` and the wasm libm's `expf` are different
      // approximations). That is a deliberate scope choice, not an impossibility
      // — porting musl's `expf` into frounded TS measured 0/200 000 mismatches
      // against the real WASM, and is tracked as #1830. Until then this one
      // operation stays approximate.
      // Measured over the 20 000-splat sweeps in
      // `tests/unit/wasm/wasm-vs-typescript.test.ts`: with `truncate` large
      // enough that the shift below underflows to 0 (so `exp` is the ONLY
      // approximate step), the emitted amplitudes differ in 1802/19943 cases by
      // AT MOST 2 ulp — that is the whole residual. At truncate = 3 everything
      // around it is exact and the same sweep went from 11907/19323 amplitudes
      // differing at up to 1367 ulp to 1734/19323 within 2⁻²³ absolute.
      // Parity tests crossing this path must be ULP- or absolute-bounded.
      // (As at the `shiftC` site above, the INNER `fround(-0.5 * x)` is exact
      // for any f32 `x` and is kept only to mirror `common.rs` step for step;
      // the outer product and the `exp` are the ones that can change an answer.)
      const rawExp = Math.fround(Math.exp(Math.fround(Math.fround(-0.5 * mahalDist) * mahalDist)));
      // Clamp at 0 with a comparison rather than Math.max: Rust's `f32::max`
      // IGNORES NaN and returns 0.0, while `Math.max(0, NaN)` is NaN. A NaN
      // anywhere in a splat's center/covariance would otherwise leave this
      // backend with a NaN attenuation where WASM had 0.0. `NaN > 0` is false,
      // so the two twins agree on every input. (The visibility gate below is
      // the second line of defence, for a NaN that arrives in `amplitudes`.)
      // `rawExp - shiftC` cancels catastrophically as a splat approaches the
      // truncation radius (that is the point of the shift — the attenuation
      // must reach 0 there), so the ≤2 ulp `exp` residual above is amplified
      // without bound near the cut: measured up to 134 ulp on the attenuated
      // amplitude at truncate = 3, and 475 ulp at truncate = 0.5. Inherent to
      // the formula, not to this rounding.
      //
      // Consequence worth stating plainly: the VISIBLE SET can still differ
      // between the two backends for splats sitting exactly on the shell. At
      // production defaults (`GSPLAT_DEFAULT_TRUNCATION_RADIUS` 2.75,
      // `MIN_AMPLITUDE` 1e-6), 200 000 splats with amplitudes uniform in
      // [0.2, 1.2] and their hidden coordinate in [2.7495, 2.7505] emit 94 550
      // from WASM and 94 540 from this backend — 10 splats apart. The f32
      // rounding makes that class much
      // rarer (it used to reach any splat whose amplitude sat within thousands
      // of ulps of the gate); it does not remove it, and it cannot while `exp`
      // differs at all. Do not write a test that asserts count equality as a
      // general property of the kernel — only on a fixture whose nearest
      // emitted amplitude clears the residual by a stated margin.
      const shifted = Math.fround(invOneMinusC * Math.fround(rawExp - shiftC));
      attenuation = shifted > 0.0 ? shifted : 0.0;
    }

    // (3) Visibility decision. The rejection is the NEGATION of the acceptance
    // rule, not `<`: a NaN amplitude (or `Infinity * 0` when a splat is fully
    // attenuated) is neither `<` nor `>=` the threshold, and plain `<` would let
    // it through to be emitted with a NaN amplitude — the #725 silent-corruption
    // mode. Mirrors the Rust twin.
    // The product is an f32 in Rust and the gate reads that rounded value, so
    // this is a DECISION, not just an output digit: an unrounded f64 product
    // sitting a fraction of an ulp under `minAmplitude` emits on one backend
    // and is culled on the other, changing the returned visible count.
    const attenuatedAmplitude = Math.fround(amplitudes[i] * attenuation);
    if (attenuatedAmplitude < minAmplitudeF32 || Number.isNaN(attenuatedAmplitude)) continue;

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

    if (recordSourceIndices) {
      outSourceIndices[out] = i;
    }

    out++;
  }

  return out;
}
