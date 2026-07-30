//! GSplats Processing for nD → 3D Conversion
//!
//! High-performance implementations of GSplats processing operations:
//! - Mahalanobis distance computation (forward substitution)
//! - Cholesky submatrix extraction
//! - Batch attenuation computation for visibility filtering
//!
//! # Dimension Limits
//!
//! **Maximum supported dimensions: 16**
//!
//! This module uses fixed-size arrays for performance. Datasets with more than
//! 16 dimensions will trigger a panic with a clear error message.

use wasm_bindgen::prelude::*;

use crate::common::{
    packed_index, validate_ndim, CHOLESKY_EPSILON, CHOLESKY_RELATIVE_EPSILON,
    MAX_PACKED_CHOLESKY_SIZE, MAX_SUPPORTED_DIMS,
};

/// Compute Mahalanobis distance for a single point using packed Cholesky factor.
///
/// Given L (lower-triangular Cholesky of covariance),
/// Mahalanobis distance = ||L⁻¹ · (x - μ)||
///
/// We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
///
/// # Arguments
/// * `diff` - Difference vector (x - μ) for the dimensions [ndim]
/// * `packed_l` - Packed Cholesky factor [packedSize]
/// * `ndim` - Dimensionality of the Cholesky (max 16)
///
/// # Returns
/// Mahalanobis distance
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
///
/// # Optimization Notes
/// - Direct division (bit-consistent with the TS reference)
/// - Hoisted epsilon constant
/// - Loop fusion for norm computation
#[wasm_bindgen]
pub fn mahalanobis_distance(diff: &[f32], packed_l: &[f32], ndim: usize) -> f32 {
    validate_ndim(ndim, "mahalanobis_distance");

    const EPSILON: f32 = 1e-10;

    // Forward substitution: solve L · y = diff
    // Using a small fixed-size array for common cases (up to 16 dims)
    let mut y = [0.0f32; MAX_SUPPORTED_DIMS];

    // Direct division (not reciprocal-multiply) keeps forward-substitution
    // bit-consistent with the TS reference; rounding from `val * (1/diag)`
    // would compound per dimension and silently drift at high ndim.
    for i in 0..ndim {
        let mut val = diff[i];
        for j in 0..i {
            val -= packed_l[packed_index(i, j)] * y[j];
        }
        let diag = packed_l[packed_index(i, i)];
        y[i] = if diag > EPSILON { val / diag } else { 0.0 };
    }

    // OPTIMIZATION: Loop fusion - compute sum of squares directly
    let mut sum_sq = 0.0f32;
    for i in 0..ndim {
        sum_sq += y[i] * y[i];
    }
    sum_sq.sqrt()
}

/// Compute the Cholesky factor of the marginal covariance for a subset of dimensions.
///
/// Given a full packed Cholesky factor L where Σ = L·Lᵀ, this computes
/// the Cholesky factor L_S of the marginal covariance Σ_S for the dimensions
/// specified by `keep_dims`.
///
/// **Why this is needed**: Simply extracting rows/columns from L does NOT give the
/// correct Cholesky of the marginal covariance when there are off-diagonal correlations
/// between the subset dimensions and other dimensions. The correct approach is:
/// 1. Reconstruct the marginal covariance: Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
/// 2. Cholesky-factorize Σ_S
///
/// # Arguments
/// * `full_packed_l` - Full packed Cholesky factor (may contain multiple splats)
/// * `full_packed_offset` - Offset into full_packed_l for this splat's data
/// * `keep_dims` - Indices of dimensions to keep [>= sub_ndim entries; only the
///   first `sub_ndim` are read]. Order is free: `Σ_S[i,j]` is the truncated dot
///   product of rows `keep_dims[i]`/`keep_dims[j]` of lower-triangular L, i.e.
///   exactly `Σ_full[s_i, s_j]`, for ANY ordering — so a permuted display order
///   (X=dim2, Y=dim0, …) yields the correctly permuted marginal. Hidden-dim
///   callers pass ascending indices, display-dim callers do not.
/// * `sub_ndim` - Number of dimensions to keep
/// * `output` - Output packed marginal Cholesky factor [subPackedSize]
///
/// # Performance
/// Cost: O(sub_ndim² · max_dim) for marginal covariance + O(sub_ndim³) for Cholesky.
/// For typical sub_ndim=1-3, this is negligible.
#[inline]
fn compute_marginal_cholesky(
    full_packed_l: &[f32],
    full_packed_offset: usize,
    keep_dims: &[u32],
    sub_ndim: usize,
    output: &mut [f32],
) {
    // Step 1: Reconstruct the dense marginal covariance matrix Σ_S
    // Σ_S[i,j] = Σ_{k=0}^{min(s_i,s_j)} L[s_i,k] · L[s_j,k]
    // Use fixed-size buffer on the stack (16×16 = 256 floats = 1KB)
    let mut sigma = [0.0f32; MAX_SUPPORTED_DIMS * MAX_SUPPORTED_DIMS];

    for i in 0..sub_ndim {
        let si = keep_dims[i] as usize;
        for j in 0..=i {
            let sj = keep_dims[j] as usize;
            // L is lower-triangular: L[row,k] = 0 for k > row
            // So we only sum up to min(si, sj)
            let k_max = si.min(sj);
            let mut sum = 0.0f32;
            for k in 0..=k_max {
                let l_si_k = full_packed_l[full_packed_offset + packed_index(si, k)];
                let l_sj_k = full_packed_l[full_packed_offset + packed_index(sj, k)];
                sum += l_si_k * l_sj_k;
            }
            sigma[i * MAX_SUPPORTED_DIMS + j] = sum;
            sigma[j * MAX_SUPPORTED_DIMS + i] = sum; // Symmetric
        }
    }

    // Step 2: Cholesky factorization of Σ_S (Cholesky-Crout algorithm)
    let mut l_sub = [0.0f32; MAX_PACKED_CHOLESKY_SIZE];

    // SCALE-RELATIVE degeneracy floor. The regularizer below has to distinguish
    // "this axis has no extent" from "this axis is small", and a variance is in
    // world-units², so an ABSOLUTE floor silently answers that question by scene
    // scale: with a fixed 1e-10, a splat with σ = 1e-7 (nm-unit data) has
    // variance 1e-14, trips the floor, and gets inflated to σ = 1e-5 — 100×
    // larger than authored (10⁴× at σ = 1e-9). A 3D scene displaying [0,1,2]
    // escapes via the standard-3D fast path, which copies the factor verbatim,
    // but a 2D scene — or any nD scene with hidden dims — always lands here.
    //
    // Anchoring the floor to the largest diagonal makes the test a pure
    // CONDITION-NUMBER check, identical in behavior at every scene scale. Same
    // reasoning as the shader's trace-normalized covariance inverse
    // (`shader-glsl.ts`): 1e-12 relative sits below f32's ~1e-7 relative
    // precision squared, so it only ever catches genuinely rank-deficient axes.
    // `CHOLESKY_EPSILON` remains the absolute backstop for an all-zero Σ_S.
    let mut max_diag = 0.0f32;
    for i in 0..sub_ndim {
        let d = sigma[i * MAX_SUPPORTED_DIMS + i];
        if d > max_diag {
            max_diag = d;
        }
    }
    // The absolute constant is a fallback for a SCALELESS (all-zero) Σ_S only —
    // using it as a general lower bound would re-impose the very scene-scale
    // threshold this replaces, since max_diag * 1e-12 is below 1e-10 for any
    // σ < ~1e-1. MIN_POSITIVE keeps the floor non-zero (hence the covariance
    // non-singular) if the relative product underflows.
    let degenerate_floor = if max_diag > 0.0 {
        (max_diag * CHOLESKY_RELATIVE_EPSILON).max(f32::MIN_POSITIVE)
    } else {
        CHOLESKY_EPSILON
    };

    for i in 0..sub_ndim {
        for j in 0..=i {
            let mut sum = sigma[i * MAX_SUPPORTED_DIMS + j];
            for k in 0..j {
                sum -= l_sub[packed_index(i, k)] * l_sub[packed_index(j, k)];
            }
            if i == j {
                // Diagonal: L[i,i] = sqrt(Σ[i,i] - Σ_{k<i} L[i,k]²)
                // Guard against numerical issues (negative due to floating point)
                l_sub[packed_index(i, i)] = if sum > degenerate_floor {
                    sum.sqrt()
                } else {
                    degenerate_floor.sqrt() // Regularize degenerate covariance
                };
            } else {
                // Off-diagonal: L[i,j] = (Σ[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
                let diag = l_sub[packed_index(j, j)];
                l_sub[packed_index(i, j)] = if diag > 0.0 { sum / diag } else { 0.0 };
            }
        }
    }

    // Copy result to output
    let sub_packed_size = (sub_ndim * (sub_ndim + 1)) / 2;
    output[..sub_packed_size].copy_from_slice(&l_sub[..sub_packed_size]);
}

/// Marginal 3D Cholesky for the display dims, padded to the packed-3D layout
/// when fewer than 3 dims are displayed (1D/2D scenes).
///
/// For `n = min(display_dims.len(), 3)` the packed n-D marginal occupies the
/// first n·(n+1)/2 slots of the packed-3D layout verbatim. The renderer always
/// consumes a 3×3 covariance (Σ = L·Lᵀ), so the rows for display axes the data
/// doesn't have must still be filled: off-diagonals are 0 (the phantom axis is
/// uncorrelated with the real ones, so the in-plane profile is untouched) and
/// the diagonal is the GEOMETRIC MEAN of the real diagonals — the phantom axis
/// gets the splat's own in-plane scale, making a 2D splat a round blob rather
/// than a disk.
///
/// The diagonal deliberately is NOT a small epsilon. In sum projection (additive,
/// luminous, volumetric) the shader scales amplitude by the Gaussian's extent
/// along the view ray, `sigmaRay = 1/√(rᵀΣ⁻¹r)` (`shader-glsl.ts`, and the same
/// math in the TSL twin); a face-on ε-thin splat gets `sigmaRay ≈ √ε`, i.e.
/// amplitude × 1e-5, which the fragment shader then discards outright — the whole
/// scene renders black. In volumetric the boosted amplitude also drives optical
/// depth, so absorption would vanish too. A scale-matched phantom axis keeps
/// `sigmaRay` proportional to the splat's own size — the same brightness
/// relationship a genuinely isotropic 3D splat has. `luxar.gsplats.lift` relies on
/// exactly this: it calibrates amplitude as `opacity / (rayIntegralFactor · σ)`,
/// which only holds for a 2D lift because `√(σ·σ) == σ`.
///
/// Two properties of the geometric mean worth knowing:
/// - Taken over the Cholesky PIVOTS it equals `(det Σ_S)^(1/2n)`, so it is
///   ROTATION-INVARIANT — a rotated 2D splat gets the same phantom axis. (The
///   geometric mean of the per-axis marginal sigmas would not be.)
/// - It is bounded by the largest real diagonal, so it cannot inflate
///   `maxLateralVar` / `maxRowNorm` — the coverage fade, extent clamp, and cull
///   bounds all stay exactly as tight as the real in-plane extent.
///
/// Known consequence: under PERSPECTIVE the projection Jacobian's third column
/// mixes the phantom variance into Σ_2D for off-axis splats, so a 2D splat far
/// from the optical axis gains a slight footprint shear that an ε axis would not
/// produce. That is the correct behavior for a splat with real z extent, and it
/// is identically zero under orthographic projection (`J[2] = 0`).
///
/// `output` must hold 6 elements.
#[inline]
fn compute_display_cholesky_3d(
    full_packed_l: &[f32],
    full_packed_offset: usize,
    display_dims: &[u32],
    output: &mut [f32],
) {
    // `compute_marginal_cholesky` reads only keep_dims[0..n), so pass
    // display_dims whole (mirrors the TS twin, where a sub-slice view would
    // allocate once per splat).
    let n = display_dims.len().min(3);
    compute_marginal_cholesky(full_packed_l, full_packed_offset, display_dims, n, output);
    if n == 3 {
        return;
    }

    // Geometric mean of the real diagonals L[i,i], i < n. Falls back to the
    // degenerate-covariance regularizer when the marginal has no extent at all.
    //
    // The test is `> 0`, not `> CHOLESKY_EPSILON`: the Crout step above already
    // floors every diagonal to a strictly positive, SCALE-RELATIVE value, so an
    // absolute threshold here would drop legitimately tiny diagonals (σ < 1e-10)
    // from the mean — reintroducing the scene-scale dependence in miniature. In
    // practice this leaves `counted == 0` reachable only for n == 0 or NaN input.
    let mut log_sum = 0.0f32;
    let mut counted = 0u32;
    for i in 0..n {
        let diag = output[packed_index(i, i)];
        if diag > 0.0 {
            log_sum += diag.ln();
            counted += 1;
        }
    }
    let phantom = if counted > 0 {
        (log_sum / counted as f32).exp()
    } else {
        CHOLESKY_EPSILON.sqrt()
    };

    let mut idx = n * (n + 1) / 2;
    for row in n..3 {
        for _ in 0..row {
            output[idx] = 0.0;
            idx += 1;
        }
        output[idx] = phantom;
        idx += 1;
    }
}

/// Extract raw elements from a packed Cholesky factor for specified dimensions.
///
/// **WARNING**: This extracts raw L elements, NOT the Cholesky factor of the
/// marginal covariance. For a full Cholesky L where Σ = L·Lᵀ, the Cholesky of
/// the marginal covariance Σ_S for dimensions S is generally NOT the submatrix
/// of L when there are cross-dimension correlations.
///
/// This function is only correct when the Cholesky factor is block-diagonal
/// (no correlations between the kept and removed dimensions). For the correct
/// marginal Cholesky, use `compute_gsplats_attenuation` or `extract_visible_cholesky_3d`
/// which handle this internally.
///
/// # Arguments
/// * `packed` - Full packed Cholesky [packedSize]
/// * `keep_dims` - Indices of dimensions to keep (must be sorted ascending) [subNdim]
/// * `sub_ndim` - Number of dimensions to keep
/// * `output` - Output packed submatrix [subPackedSize]
#[wasm_bindgen]
pub fn extract_cholesky_submatrix(
    packed: &[f32],
    keep_dims: &[u32],
    sub_ndim: usize,
    output: &mut [f32],
) {
    let mut out_idx = 0;

    for sub_row in 0..sub_ndim {
        let orig_row = keep_dims[sub_row] as usize;
        for sub_col in 0..=sub_row {
            let orig_col = keep_dims[sub_col] as usize;
            output[out_idx] = packed[packed_index(orig_row, orig_col)];
            out_idx += 1;
        }
    }
}

/// Compute attenuation factors for all GSplats based on hidden dimension distance.
///
/// For each splat, computes:
/// 1. Difference vector in hidden dimensions
/// 2. Mahalanobis distance using hidden Cholesky submatrix
/// 3. Attenuation = exp(-0.5 * mahal²) (standard Gaussian)
/// 4. Visibility = (amplitude * attenuation) >= threshold
///
/// # Arguments
/// * `positions` - Splat centers [splatCount * ndim]
/// * `cholesky` - Packed Cholesky factors [splatCount * packedSize]
/// * `amplitudes` - Splat amplitudes [splatCount]
/// * `slice_position` - Current slice position [ndim]
/// * `hidden_dims` - Indices of hidden dimensions (sorted) [numHidden]
/// * `ndim` - Total dimensionality (max 16)
/// * `splat_count` - Number of splats
/// * `min_amplitude` - Visibility threshold
/// * `output_visibility` - Output visibility mask [splatCount]
/// * `output_attenuation` - Output attenuation factors [splatCount]
///
/// # Returns
/// Number of visible splats
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn compute_gsplats_attenuation(
    positions: &[f32],
    cholesky: &[f32],
    amplitudes: &[f32],
    slice_position: &[f32],
    hidden_dims: &[u32],
    ndim: usize,
    splat_count: usize,
    min_amplitude: f32,
    truncate: f32,
    output_visibility: &mut [u8],
    output_attenuation: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "compute_gsplats_attenuation");

    debug_assert!(
        output_visibility.len() >= splat_count,
        "output_visibility too small: {} < {}",
        output_visibility.len(),
        splat_count
    );
    debug_assert!(
        output_attenuation.len() >= splat_count,
        "output_attenuation too small: {} < {}",
        output_attenuation.len(),
        splat_count
    );

    let num_hidden = hidden_dims.len();
    let full_packed_size = (ndim * (ndim + 1)) / 2;

    // Shifted Gaussian constants for C⁰ continuous truncation
    let shift_c = (-0.5f32 * truncate * truncate).exp();
    let inv_one_minus_c = 1.0 / (1.0 - shift_c);

    // Temporary buffers (use fixed-size arrays for performance)
    let mut diff = [0.0f32; MAX_SUPPORTED_DIMS];
    let mut hidden_cholesky = [0.0f32; MAX_PACKED_CHOLESKY_SIZE];

    let mut visible_count = 0u32;

    for i in 0..splat_count {
        let center_offset = i * ndim;
        let cholesky_offset = i * full_packed_size;

        let attenuation = if num_hidden == 0 {
            // No hidden dimensions, full visibility
            1.0
        } else {
            // Compute difference vector in hidden dimensions
            for (h_idx, &dim) in hidden_dims.iter().enumerate() {
                let d = dim as usize;
                diff[h_idx] = slice_position[d] - positions[center_offset + d];
            }

            // Compute correct marginal Cholesky for hidden dimensions
            compute_marginal_cholesky(
                cholesky,
                cholesky_offset,
                hidden_dims,
                num_hidden,
                &mut hidden_cholesky,
            );

            // Compute Mahalanobis distance
            let mahal_dist =
                mahalanobis_distance_internal(&diff[..num_hidden], &hidden_cholesky, num_hidden);

            // Shifted Gaussian attenuation: scale · max(0, exp(-0.5·D²) - C)
            let raw_exp = (-0.5 * mahal_dist * mahal_dist).exp();
            (inv_one_minus_c * (raw_exp - shift_c)).max(0.0)
        };

        output_attenuation[i] = attenuation;

        let attenuated_amplitude = amplitudes[i] * attenuation;
        let visible = attenuated_amplitude >= min_amplitude;
        output_visibility[i] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Internal Mahalanobis distance (no WASM binding, avoids allocation)
/// Note: ndim is already validated by caller, no need to validate again
///
/// # Optimization Notes
/// - Inlined for zero-overhead abstraction
/// - Direct division (bit-consistent with the TS reference, see `mahalanobis_distance`)
/// - Hoisted epsilon constant
#[inline]
fn mahalanobis_distance_internal(diff: &[f32], packed_l: &[f32], ndim: usize) -> f32 {
    const EPSILON: f32 = 1e-10;

    let mut y = [0.0f32; MAX_SUPPORTED_DIMS];

    for i in 0..ndim {
        let mut val = diff[i];
        for j in 0..i {
            val -= packed_l[packed_index(i, j)] * y[j];
        }
        let diag = packed_l[packed_index(i, i)];
        y[i] = if diag > EPSILON { val / diag } else { 0.0 };
    }

    let mut sum_sq = 0.0f32;
    for i in 0..ndim {
        sum_sq += y[i] * y[i];
    }
    sum_sq.sqrt()
}

/// Extract 3D Cholesky submatrices for visible splats.
///
/// # Arguments
/// * `cholesky` - Packed Cholesky factors [splatCount * packedSize]
/// * `visibility` - Visibility mask [splatCount]
/// * `display_dims` - Ordered display-axis dimension indices [1..=3]; their
///   order maps directly to output X/Y/Z, and missing rows are
///   scale-matched-padded for 1D/2D data (see `compute_display_cholesky_3d`)
/// * `ndim` - Total dimensionality (max 16)
/// * `splat_count` - Number of splats
/// * `output` - Output 3D Cholesky factors [visibleCount * 6]
///
/// # Returns
/// Number of visible splats processed
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn extract_visible_cholesky_3d(
    cholesky: &[f32],
    visibility: &[u8],
    display_dims: &[u32],
    ndim: usize,
    splat_count: usize,
    output: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "extract_visible_cholesky_3d");

    // output size depends on visible count which is unknown upfront;
    // assert minimum based on splat_count (upper bound for visible)
    debug_assert!(
        output.len() >= 6,
        "output must hold at least one 3D Cholesky (6 elements)"
    );

    let full_packed_size = (ndim * (ndim + 1)) / 2;
    let mut out_splat = 0u32;

    for i in 0..splat_count {
        if visibility[i] == 0 {
            continue;
        }

        let src_offset = i * full_packed_size;
        let dst_offset = (out_splat as usize) * 6;

        // Compute correct marginal Cholesky for display dimensions
        let mut temp_cholesky = [0.0f32; 6]; // 3D packed = 6 elements
        compute_display_cholesky_3d(cholesky, src_offset, display_dims, &mut temp_cholesky);
        output[dst_offset..dst_offset + 6].copy_from_slice(&temp_cholesky[..6]);

        out_splat += 1;
    }

    out_splat
}

/// Compact amplitudes by visibility mask, applying attenuation.
///
/// # Arguments
/// * `amplitudes` - Original amplitudes [splatCount]
/// * `attenuation` - Attenuation factors [splatCount]
/// * `visibility` - Visibility mask [splatCount]
/// * `splat_count` - Number of splats
/// * `output` - Output attenuated amplitudes [visibleCount]
///
/// # Returns
/// Number of visible splats
#[wasm_bindgen]
pub fn compact_attenuated_amplitudes(
    amplitudes: &[f32],
    attenuation: &[f32],
    visibility: &[u8],
    splat_count: usize,
    output: &mut [f32],
) -> u32 {
    let mut out_idx = 0u32;

    for i in 0..splat_count {
        if visibility[i] != 0 {
            output[out_idx as usize] = amplitudes[i] * attenuation[i];
            out_idx += 1;
        }
    }

    out_idx
}

/// Fused nD→3D GSplat projection in a SINGLE pass over the splats.
///
/// Collapses the previous 6-call worker pipeline
/// (`compute_gsplats_attenuation` + `extract_3d_positions` + `compact_by_mask`
/// for centers + `extract_visible_cholesky_3d` + `compact_attenuated_amplitudes`
/// + `compact_by_mask` for colors) into one kernel, eliminating ~5 full passes
/// over `splat_count` and the repeated copies of the large `positions` /
/// `cholesky` arrays across the wasm-bindgen boundary.
///
/// For each splat it: (1) applies the precomputed discrete-visibility gate,
/// (2) computes continuous attenuation (marginal Cholesky + shifted Gaussian,
/// identical math to `compute_gsplats_attenuation`), (3) decides visibility via
/// `amplitude * attenuation >= min_amplitude`, and (4) writes COMPACTED outputs
/// (visible centers3D, cholesky3D[6], attenuated amplitudes, colors) densely
/// from index 0. The visible set and all values are bit-identical to the
/// multi-call path (same helpers, same op order) — see the golden-equivalence
/// test `test_fused_matches_multicall`.
///
/// Colors are coerced to normalized f32 on the TS side (wasm-bindgen can't take
/// a typed-array union), so `colors` is `[splat_count * color_components]` f32
/// (white-filled when the dataset has no colors). `color_components` is 3 (RGB)
/// or 4 (RGBA — alpha is per-splat opacity and compacts with its splat).
///
/// Outputs must be sized for the `splat_count` worst case; the caller slices
/// each to the returned visible count.
///
/// # Arguments
/// * `positions` - Splat centers [splat_count * ndim]
/// * `cholesky` - Packed Cholesky factors [splat_count * packedSize]
/// * `amplitudes` - Splat amplitudes [splat_count]
/// * `colors` - Pre-normalized RGB(A) [splat_count * color_components]
/// * `color_components` - 3 (RGB) or 4 (RGBA)
/// * `discrete_visibility` - Precomputed discrete-dim gate [splat_count] (all 1 if none)
/// * `slice_position` - Current slice [ndim]
/// * `continuous_hidden_dims` - Sorted continuous hidden dims [num_continuous]
/// * `display_dims` - Display dims in requested order [2 or 3]
/// * `ndim`, `splat_count`, `min_amplitude`, `truncate`
/// * `out_centers3d` [splat_count * 3], `out_cholesky3d` [splat_count * 6],
///   `out_amplitudes` [splat_count], `out_colors` [splat_count * color_components]
///
/// # Returns
/// Number of visible splats written (dense prefix length / stride).
///
/// # Panics
/// Panics if `ndim > 16`. The worker routes >16D to the uncapped TS reference.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn project_gsplats_nd_to_3d(
    positions: &[f32],
    cholesky: &[f32],
    amplitudes: &[f32],
    colors: &[f32],
    discrete_visibility: &[u8],
    slice_position: &[f32],
    continuous_hidden_dims: &[u32],
    display_dims: &[u32],
    ndim: usize,
    splat_count: usize,
    color_components: usize,
    min_amplitude: f32,
    truncate: f32,
    out_centers3d: &mut [f32],
    out_cholesky3d: &mut [f32],
    out_amplitudes: &mut [f32],
    out_colors: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "project_gsplats_nd_to_3d");
    assert!(
        color_components == 3 || color_components == 4,
        "color_components must be 3 (RGB) or 4 (RGBA)"
    );

    debug_assert!(
        out_centers3d.len() >= splat_count * 3,
        "out_centers3d too small"
    );
    debug_assert!(
        out_cholesky3d.len() >= splat_count * 6,
        "out_cholesky3d too small"
    );
    debug_assert!(
        out_amplitudes.len() >= splat_count,
        "out_amplitudes too small"
    );
    debug_assert!(
        out_colors.len() >= splat_count * color_components,
        "out_colors too small"
    );

    let num_continuous = continuous_hidden_dims.len();
    let num_display = display_dims.len().min(3);
    let full_packed_size = (ndim * (ndim + 1)) / 2;

    // Shifted Gaussian constants (identical to compute_gsplats_attenuation).
    let shift_c = (-0.5f32 * truncate * truncate).exp();
    let inv_one_minus_c = 1.0 / (1.0 - shift_c);

    let mut diff = [0.0f32; MAX_SUPPORTED_DIMS];
    let mut hidden_cholesky = [0.0f32; MAX_PACKED_CHOLESKY_SIZE];

    let mut out = 0usize;

    for i in 0..splat_count {
        // (1) Discrete gate first — cheapest; lets us skip all Cholesky work.
        if discrete_visibility[i] == 0 {
            continue;
        }

        let center_offset = i * ndim;
        let cholesky_offset = i * full_packed_size;

        // (2) Continuous attenuation (identical to compute_gsplats_attenuation).
        let attenuation = if num_continuous == 0 {
            1.0
        } else {
            for (h_idx, &dim) in continuous_hidden_dims.iter().enumerate() {
                let d = dim as usize;
                diff[h_idx] = slice_position[d] - positions[center_offset + d];
            }
            compute_marginal_cholesky(
                cholesky,
                cholesky_offset,
                continuous_hidden_dims,
                num_continuous,
                &mut hidden_cholesky,
            );
            let mahal_dist = mahalanobis_distance_internal(
                &diff[..num_continuous],
                &hidden_cholesky,
                num_continuous,
            );
            let raw_exp = (-0.5 * mahal_dist * mahal_dist).exp();
            (inv_one_minus_c * (raw_exp - shift_c)).max(0.0)
        };

        // (3) Visibility decision (identical to compute_gsplats_attenuation).
        let attenuated_amplitude = amplitudes[i] * attenuation;
        if attenuated_amplitude < min_amplitude {
            continue;
        }

        // (4) Write compacted outputs at dense slot `out`.
        // Centers in display order (mirrors extract_3d_positions: min(len,3), zero-fill).
        let c_off = out * 3;
        for j in 0..num_display {
            out_centers3d[c_off + j] = positions[center_offset + display_dims[j] as usize];
        }
        for j in num_display..3 {
            out_centers3d[c_off + j] = 0.0;
        }

        // Marginal Cholesky for display dims (mirrors extract_visible_cholesky_3d),
        // scale-matched-padded when fewer than 3 dims are displayed (1D/2D scenes).
        let chol_off = out * 6;
        compute_display_cholesky_3d(
            cholesky,
            cholesky_offset,
            display_dims,
            &mut out_cholesky3d[chol_off..chol_off + 6],
        );

        out_amplitudes[out] = attenuated_amplitude;

        let col_off = out * color_components;
        let col_src = i * color_components;
        out_colors[col_off..col_off + color_components]
            .copy_from_slice(&colors[col_src..col_src + color_components]);

        out += 1;
    }

    out as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_index() {
        // [L00, L10, L11, L20, L21, L22, ...]
        assert_eq!(packed_index(0, 0), 0); // L00
        assert_eq!(packed_index(1, 0), 1); // L10
        assert_eq!(packed_index(1, 1), 2); // L11
        assert_eq!(packed_index(2, 0), 3); // L20
        assert_eq!(packed_index(2, 1), 4); // L21
        assert_eq!(packed_index(2, 2), 5); // L22
    }

    #[test]
    fn test_mahalanobis_distance_identity() {
        // Identity Cholesky (L = I): Mahalanobis = Euclidean
        // 3D: packed = [1, 0, 1, 0, 0, 1]
        let packed_l = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let diff = vec![3.0, 4.0, 0.0]; // Distance should be 5.0

        let dist = mahalanobis_distance(&diff, &packed_l, 3);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mahalanobis_distance_scaled() {
        // Scaled Cholesky: L = diag(2, 2, 2)
        // packed = [2, 0, 2, 0, 0, 2]
        // Mahalanobis = ||L⁻¹ · diff|| = ||diff / 2||
        let packed_l = vec![2.0, 0.0, 2.0, 0.0, 0.0, 2.0];
        let diff = vec![4.0, 0.0, 0.0]; // Mahalanobis should be 4/2 = 2

        let dist = mahalanobis_distance(&diff, &packed_l, 3);
        assert!((dist - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_cholesky_submatrix() {
        // 4D Cholesky: 10 elements
        // [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
        let packed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Extract dims [0, 2] (2D submatrix)
        let keep_dims = vec![0, 2];
        let mut output = vec![0.0f32; 3]; // 2D packed = 3 elements

        extract_cholesky_submatrix(&packed, &keep_dims, 2, &mut output);

        // Expected: [L00, L20, L22] = [1.0, 4.0, 6.0]
        assert_eq!(output[0], 1.0); // L[0,0]
        assert_eq!(output[1], 4.0); // L[2,0]
        assert_eq!(output[2], 6.0); // L[2,2]
    }

    #[test]
    fn test_compute_gsplats_attenuation_no_hidden() {
        // 3D splats with no hidden dimensions
        let positions = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 0
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 1
        ];
        let amplitudes = vec![1.0, 0.5];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let hidden_dims: Vec<u32> = vec![]; // No hidden dims

        let mut visibility = vec![0u8; 2];
        let mut attenuation = vec![0.0f32; 2];

        let count = compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &slice_pos,
            &hidden_dims,
            3,
            2,
            0.1,
            3.0, // truncation radius
            &mut visibility,
            &mut attenuation,
        );

        assert_eq!(count, 2); // Both visible
        assert_eq!(attenuation[0], 1.0); // No attenuation
        assert_eq!(attenuation[1], 1.0);
    }

    #[test]
    fn test_compute_gsplats_attenuation_with_hidden() {
        // 4D splats with dim 3 as hidden
        // Splat 0: at (0,0,0,0), visible
        // Splat 1: at (0,0,0,5), attenuated (far in hidden dim)
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, // Splat 0
            0.0, 0.0, 0.0, 5.0, // Splat 1
        ];
        // 4D Cholesky: 10 elements, identity
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 0
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 1
        ];
        let amplitudes = vec![1.0, 1.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        let hidden_dims = vec![3u32]; // Dim 3 is hidden

        let mut visibility = vec![0u8; 2];
        let mut attenuation = vec![0.0f32; 2];

        let count = compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &slice_pos,
            &hidden_dims,
            4,
            2,
            0.01, // Low threshold
            3.0,  // truncation radius
            &mut visibility,
            &mut attenuation,
        );

        // Splat 0: mahal = 0, attenuation = 1.0
        assert_eq!(visibility[0], 1);
        assert!((attenuation[0] - 1.0).abs() < 1e-5);

        // Splat 1: mahal = 5.0, beyond 3σ truncation → attenuation = 0
        assert_eq!(visibility[1], 0); // Below threshold
        assert!(attenuation[1] < 1e-6);

        assert_eq!(count, 1);
    }

    #[test]
    fn test_compute_marginal_cholesky_diagonal() {
        // For diagonal L, marginal Cholesky should equal raw extraction
        // 4D diagonal L: L = diag(2, 3, 5, 7)
        // Packed: [2, 0,3, 0,0,5, 0,0,0,7]
        let packed = vec![2.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        let keep_dims = vec![0u32, 2];
        let mut output = vec![0.0f32; 3]; // 2D packed

        compute_marginal_cholesky(&packed, 0, &keep_dims, 2, &mut output);

        // Σ = diag(4, 9, 25, 49), marginal for dims [0,2] = diag(4, 25)
        // Cholesky of diag(4, 25) = diag(2, 5) → packed [2, 0, 5]
        assert!(
            (output[0] - 2.0).abs() < 1e-5,
            "L_S[0,0] = 2.0, got {}",
            output[0]
        );
        assert!(output[1].abs() < 1e-5, "L_S[1,0] = 0.0, got {}", output[1]);
        assert!(
            (output[2] - 5.0).abs() < 1e-5,
            "L_S[1,1] = 5.0, got {}",
            output[2]
        );
    }

    #[test]
    fn test_compute_marginal_cholesky_correlated() {
        // 3D L with correlations:
        // L = [[2, 0, 0],
        //      [1, 3, 0],
        //      [0.5, 0.5, 4]]
        // Packed: [2, 1,3, 0.5,0.5,4]
        //
        // Σ = L·L^T:
        // Σ[0,0] = 2*2 = 4
        // Σ[1,0] = 1*2 = 2,  Σ[1,1] = 1*1 + 3*3 = 10
        // Σ[2,0] = 0.5*2 = 1,  Σ[2,1] = 0.5*1 + 0.5*3 = 2,  Σ[2,2] = 0.5*0.5 + 0.5*0.5 + 4*4 = 16.5
        //
        // Marginal for dims [0, 2]:
        // Σ_S = [[4, 1], [1, 16.5]]
        // Cholesky of Σ_S:
        //   L_S[0,0] = sqrt(4) = 2
        //   L_S[1,0] = 1/2 = 0.5
        //   L_S[1,1] = sqrt(16.5 - 0.25) = sqrt(16.25) ≈ 4.0311
        let packed = vec![2.0, 1.0, 3.0, 0.5, 0.5, 4.0];
        let keep_dims = vec![0u32, 2];
        let mut output = vec![0.0f32; 3];

        compute_marginal_cholesky(&packed, 0, &keep_dims, 2, &mut output);

        assert!(
            (output[0] - 2.0).abs() < 1e-4,
            "L_S[0,0] = 2.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 0.5).abs() < 1e-4,
            "L_S[1,0] = 0.5, got {}",
            output[1]
        );
        let expected_diag = (16.25_f32).sqrt(); // ≈ 4.0311
        assert!(
            (output[2] - expected_diag).abs() < 1e-4,
            "L_S[1,1] = {}, got {}",
            expected_diag,
            output[2]
        );
    }

    #[test]
    fn test_marginal_vs_raw_extraction_difference() {
        // Prove that raw extraction and marginal Cholesky differ for correlated L
        // Same L as above
        let packed = vec![2.0, 1.0, 3.0, 0.5, 0.5, 4.0];
        let keep_dims = vec![0u32, 2];

        let mut raw_output = vec![0.0f32; 3];
        extract_cholesky_submatrix(&packed, &keep_dims, 2, &mut raw_output);

        let mut marginal_output = vec![0.0f32; 3];
        compute_marginal_cholesky(&packed, 0, &keep_dims, 2, &mut marginal_output);

        // Raw extraction gives [L[0,0], L[2,0], L[2,2]] = [2.0, 0.5, 4.0]
        assert!((raw_output[0] - 2.0).abs() < 1e-5);
        assert!((raw_output[1] - 0.5).abs() < 1e-5);
        assert!((raw_output[2] - 4.0).abs() < 1e-5);

        // Marginal Cholesky gives different L[2,2]: sqrt(16.25) ≈ 4.031 ≠ 4.0
        assert!(
            (marginal_output[2] - raw_output[2]).abs() > 0.01,
            "Marginal and raw should differ for correlated L: marginal={}, raw={}",
            marginal_output[2],
            raw_output[2]
        );
    }

    #[test]
    fn test_mahalanobis_with_marginal_cholesky() {
        // Verify Mahalanobis distance is correct when using marginal Cholesky
        // 3D L with correlations, extracting marginal for dims [0, 2]
        let packed = vec![2.0, 1.0, 3.0, 0.5, 0.5, 4.0];
        let keep_dims = vec![0u32, 2];

        let mut marginal_l = vec![0.0f32; 3];
        compute_marginal_cholesky(&packed, 0, &keep_dims, 2, &mut marginal_l);

        // Mahalanobis distance with diff = [1, 0]
        let diff = vec![1.0f32, 0.0];
        let dist = mahalanobis_distance(&diff, &marginal_l, 2);

        // With Σ_S = [[4, 1], [1, 16.5]], Σ_S⁻¹ ≈ [[0.2538, -0.01538], [-0.01538, 0.06154]]
        // d^T Σ_S⁻¹ d = 0.2538 for diff = [1, 0]
        // Mahalanobis = sqrt(0.2538) ≈ 0.5038
        // Or via forward substitution: y[0] = 1/2 = 0.5, y[1] = (0 - 0.5*0.5) / 4.031 ≈ -0.0621
        // ||y|| = sqrt(0.25 + 0.00386) ≈ 0.5038
        assert!(
            (dist - 0.5).abs() < 0.005,
            "Mahalanobis distance should be ~0.5, got {}",
            dist
        );
    }

    #[test]
    fn test_attenuation_with_correlated_cholesky() {
        // End-to-end: 4D splat with correlated Cholesky, hidden dim = [3]
        // L = [[2, 0, 0, 0],
        //      [1, 3, 0, 0],
        //      [0, 0, 2, 0],
        //      [0.5, 0.5, 0, 4]]
        // Packed: [2, 1,3, 0,0,2, 0.5,0.5,0,4]
        let positions = vec![0.0, 0.0, 0.0, 0.0]; // at origin
        let cholesky = vec![2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        let amplitudes = vec![1.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 1.0]; // slice at dim3 = 1
        let hidden_dims = vec![3u32];

        let mut visibility = vec![0u8; 1];
        let mut attenuation = vec![0.0f32; 1];

        compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &slice_pos,
            &hidden_dims,
            4,
            1,
            0.001,
            3.0, // truncation radius
            &mut visibility,
            &mut attenuation,
        );

        // Hidden dim marginal for dim [3]:
        // Σ_33 = L[3,0]^2 + L[3,1]^2 + L[3,2]^2 + L[3,3]^2
        //      = 0.25 + 0.25 + 0 + 16 = 16.5
        // L_S = sqrt(16.5) ≈ 4.062
        // Mahalanobis distance of diff=1.0: 1.0 / 4.062 ≈ 0.2462
        // Attenuation = exp(-0.5 * 0.2462^2) ≈ exp(-0.0303) ≈ 0.970
        assert!(
            attenuation[0] > 0.9 && attenuation[0] < 1.0,
            "Expected attenuation ~0.97, got {}",
            attenuation[0]
        );
        assert_eq!(visibility[0], 1);
    }

    /// Golden equivalence: the fused single-pass kernel must produce a
    /// bit-identical visible set + outputs to the legacy 6-call pipeline
    /// (attenuation → combine discrete → extract_3d_positions + compact centers
    /// → extract_visible_cholesky_3d → compact_attenuated_amplitudes → compact
    /// colors). Exercises correlated covariance, a discrete-gated splat, an
    /// attenuated-out splat, and a fully-visible splat.
    #[test]
    fn test_fused_matches_multicall() {
        use crate::projection::{compact_by_mask, extract_3d_positions};

        let ndim = 4usize;
        let n = 4usize;
        let continuous_hidden = [3u32];
        let display = [0u32, 1, 2];
        let min_amp = 0.001f32;
        let truncate = 3.0f32;

        // 4 splats. Dim 3 is the hidden (continuous) slicing dim; slice at 0.
        // splat0: on slice (dim3=0) → visible; splat1: far in dim3 → attenuated
        // out; splat2: on slice but discrete-gated; splat3: near slice → visible.
        let positions: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // splat0
            1.0, 1.0, 1.0, 50.0, // splat1 (far in dim3)
            2.0, 2.0, 2.0, 0.0, // splat2 (discrete-gated)
            3.0, 1.0, 2.0, 0.3, // splat3
        ];
        // Correlated 4D Cholesky per splat (off-diagonals couple dim3 to others).
        let one: Vec<f32> = vec![2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        let cholesky: Vec<f32> = (0..n).flat_map(|_| one.clone()).collect();
        let amplitudes: Vec<f32> = vec![1.0, 0.5, 1.0, 0.8];
        let colors: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35,
        ];
        let slice = vec![0.0f32, 0.0, 0.0, 0.0];
        let discrete_visibility = [1u8, 1, 0, 1]; // splat2 gated

        // ---- Legacy multi-call pipeline ----
        let mut vis = vec![0u8; n];
        let mut atten = vec![0.0f32; n];
        compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &slice,
            &continuous_hidden,
            ndim,
            n,
            min_amp,
            truncate,
            &mut vis,
            &mut atten,
        );
        for i in 0..n {
            if discrete_visibility[i] == 0 {
                vis[i] = 0;
            }
        }
        let visible_count = vis.iter().filter(|&&v| v != 0).count();

        let mut all_centers = vec![0.0f32; n * 3];
        extract_3d_positions(&positions, &display, ndim, n, &mut all_centers);
        let mut exp_centers = vec![0.0f32; visible_count * 3];
        compact_by_mask(&all_centers, &vis, n, 3, &mut exp_centers);

        let mut exp_chol = vec![0.0f32; visible_count * 6];
        extract_visible_cholesky_3d(&cholesky, &vis, &display, ndim, n, &mut exp_chol);

        let mut exp_amps = vec![0.0f32; visible_count];
        compact_attenuated_amplitudes(&amplitudes, &atten, &vis, n, &mut exp_amps);

        let mut exp_colors = vec![0.0f32; visible_count * 3];
        compact_by_mask(&colors, &vis, n, 3, &mut exp_colors);

        // ---- Fused single-pass kernel ----
        let mut f_centers = vec![0.0f32; n * 3];
        let mut f_chol = vec![0.0f32; n * 6];
        let mut f_amps = vec![0.0f32; n];
        let mut f_colors = vec![0.0f32; n * 3];
        let f_count = project_gsplats_nd_to_3d(
            &positions,
            &cholesky,
            &amplitudes,
            &colors,
            &discrete_visibility,
            &slice,
            &continuous_hidden,
            &display,
            ndim,
            n,
            3,
            min_amp,
            truncate,
            &mut f_centers,
            &mut f_chol,
            &mut f_amps,
            &mut f_colors,
        ) as usize;

        // Must agree on the visible set and every output value (bit-identical).
        assert_eq!(f_count, visible_count, "visible count mismatch");
        assert!(visible_count >= 2, "test should keep ≥2 visible splats");
        assert_eq!(
            &f_centers[..f_count * 3],
            &exp_centers[..],
            "centers mismatch"
        );
        assert_eq!(&f_chol[..f_count * 6], &exp_chol[..], "cholesky mismatch");
        assert_eq!(&f_amps[..f_count], &exp_amps[..], "amplitudes mismatch");
        assert_eq!(&f_colors[..f_count * 3], &exp_colors[..], "colors mismatch");
    }

    /// RGBA parity: the fused kernel must compact a 4-channel (RGBA) color
    /// array bit-identically to the reference `compact_by_mask(.., 4, ..)`,
    /// including the alpha column. This is the one path the RGB-only golden
    /// test above cannot exercise; without it the Rust ↔ TypeScript twins
    /// could silently diverge in the 4th channel (CLAUDE.md's 1:1 parity rule).
    #[test]
    fn test_fused_matches_multicall_rgba() {
        use crate::projection::{compact_by_mask, extract_3d_positions};

        // Same geometry / visibility as `test_fused_matches_multicall`.
        let ndim = 4usize;
        let n = 4usize;
        let continuous_hidden = [3u32];
        let display = [0u32, 1, 2];
        let min_amp = 0.001f32;
        let truncate = 3.0f32;
        let positions: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 50.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.3,
        ];
        let one: Vec<f32> = vec![2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        let cholesky: Vec<f32> = (0..n).flat_map(|_| one.clone()).collect();
        let amplitudes: Vec<f32> = vec![1.0, 0.5, 1.0, 0.8];
        // 4×4 RGBA; alpha deliberately DISTINCT from RGB (alpha = 1 - r) so a
        // stride bug that copied RGB into alpha (or dropped it) would be caught.
        let colors: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.9, // splat0 (alpha 0.9 ≠ r 0.1)
            0.4, 0.5, 0.6, 0.6, // splat1
            0.7, 0.8, 0.9, 0.3, // splat2
            0.15, 0.25, 0.35, 0.85, // splat3
        ];
        let slice = vec![0.0f32, 0.0, 0.0, 0.0];
        let discrete_visibility = [1u8, 1, 0, 1];

        // Reference visible set (same pipeline as the RGB golden test).
        let mut vis = vec![0u8; n];
        let mut atten = vec![0.0f32; n];
        compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &slice,
            &continuous_hidden,
            ndim,
            n,
            min_amp,
            truncate,
            &mut vis,
            &mut atten,
        );
        for i in 0..n {
            if discrete_visibility[i] == 0 {
                vis[i] = 0;
            }
        }
        let visible_count = vis.iter().filter(|&&v| v != 0).count();
        assert!(visible_count >= 2, "test should keep ≥2 visible splats");

        let mut all_centers = vec![0.0f32; n * 3];
        extract_3d_positions(&positions, &display, ndim, n, &mut all_centers);
        let mut exp_colors = vec![0.0f32; visible_count * 4];
        compact_by_mask(&colors, &vis, n, 4, &mut exp_colors);

        let mut f_centers = vec![0.0f32; n * 3];
        let mut f_chol = vec![0.0f32; n * 6];
        let mut f_amps = vec![0.0f32; n];
        let mut f_colors = vec![0.0f32; n * 4]; // sized for color_components = 4
        let f_count = project_gsplats_nd_to_3d(
            &positions,
            &cholesky,
            &amplitudes,
            &colors,
            &discrete_visibility,
            &slice,
            &continuous_hidden,
            &display,
            ndim,
            n,
            4, // color_components = RGBA
            min_amp,
            truncate,
            &mut f_centers,
            &mut f_chol,
            &mut f_amps,
            &mut f_colors,
        ) as usize;

        assert_eq!(f_count, visible_count, "visible count mismatch");
        assert_eq!(
            &f_colors[..f_count * 4],
            &exp_colors[..],
            "RGBA colors (incl. alpha) mismatch"
        );
        // Alpha of the first visible splat must be its own alpha (0.9), not a
        // shifted/dropped channel.
        assert_eq!(
            f_colors[3], 0.9,
            "alpha of first visible splat not preserved"
        );
    }

    /// Regression: a 2D scene gives `display_dims.len() == 2`; the display
    /// marginal used to be computed with a hardcoded sub_ndim of 3, reading
    /// `display_dims[2]` out of bounds and panicking (wasm: `unreachable`).
    /// A 2D splat must project with its 2D marginal in the first three packed
    /// slots and a scale-matched phantom z row `[0, 0, √(L00·L11)]` — NOT an
    /// ε diagonal, which the sum-mode ray integral renders invisible.
    #[test]
    fn test_fused_2d_display_dims() {
        let ndim = 2usize;
        let n = 2usize;
        let one = [2.0f32, 0.5, 1.5]; // packed 2D Cholesky [L00, L10, L11]
        let cholesky = [one, one].concat();
        let positions = vec![0.0f32, 0.0, 5.0, -3.0];
        let amplitudes = vec![1.0f32, 0.8];
        let colors = vec![1.0f32; n * 3];
        let vis = vec![1u8; n];
        let slice_pos = vec![0.0f32; ndim];
        let hidden: Vec<u32> = vec![];
        let display = vec![0u32, 1];
        let mut out_c = vec![0.0f32; n * 3];
        let mut out_l = vec![0.0f32; n * 6];
        let mut out_a = vec![0.0f32; n];
        let mut out_col = vec![0.0f32; n * 3];

        let count = project_gsplats_nd_to_3d(
            &positions,
            &cholesky,
            &amplitudes,
            &colors,
            &vis,
            &slice_pos,
            &hidden,
            &display,
            ndim,
            n,
            3,
            1e-6,
            3.0,
            &mut out_c,
            &mut out_l,
            &mut out_a,
            &mut out_col,
        );

        assert_eq!(count, 2);
        // Centers: [x, y, 0] (z zero-filled).
        assert_eq!(&out_c[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&out_c[3..6], &[5.0, -3.0, 0.0]);
        // Cholesky: keeping ALL dims makes the marginal reproduce the input
        // factor. The 2D marginal here is [2.0, 0.5, √(1.5²+0.5²−0.5²)] — the
        // Crout factorization of Σ_S, so L11 = 1.5 exactly. The phantom z
        // diagonal is the geometric mean √(2.0·1.5).
        let expected_phantom = (2.0f32 * 1.5).sqrt();
        for s in 0..n {
            let l = &out_l[s * 6..s * 6 + 6];
            assert!((l[0] - 2.0).abs() < 1e-5, "L00");
            assert!((l[1] - 0.5).abs() < 1e-5, "L10");
            assert!((l[2] - 1.5).abs() < 1e-5, "L11");
            assert_eq!(l[3], 0.0, "L20");
            assert_eq!(l[4], 0.0, "L21");
            assert!(
                (l[5] - expected_phantom).abs() < 1e-5,
                "L22 must be the geometric mean of the real diagonals, got {}",
                l[5]
            );
        }
    }

    /// The phantom axis must scale WITH the splat: doubling the in-plane
    /// factor doubles the phantom diagonal. This is what keeps the sum-mode
    /// ray integral (∝ extent along the ray) proportional to splat size
    /// instead of collapsing to ~0 as an ε diagonal did.
    #[test]
    fn test_2d_phantom_axis_scales_with_splat() {
        let mut small = [0.0f32; 6];
        let mut large = [0.0f32; 6];
        compute_display_cholesky_3d(&[2.0, 0.5, 1.5], 0, &[0, 1], &mut small);
        compute_display_cholesky_3d(&[4.0, 1.0, 3.0], 0, &[0, 1], &mut large);

        assert!(small[5] > 0.1, "phantom diagonal must not be ~0");
        assert!(
            (large[5] / small[5] - 2.0).abs() < 1e-4,
            "phantom diagonal must scale linearly with the splat: {} vs {}",
            large[5],
            small[5]
        );
    }

    /// A 1D display (single display dim) pads BOTH missing rows.
    #[test]
    fn test_1d_display_dims_pads_two_rows() {
        let mut out = [0.0f32; 6];
        compute_display_cholesky_3d(&[2.0, 0.5, 1.5], 0, &[0], &mut out);

        assert!((out[0] - 2.0).abs() < 1e-5, "L00");
        assert_eq!(out[1], 0.0, "L10");
        assert!((out[2] - 2.0).abs() < 1e-5, "L11 = phantom = L00");
        assert_eq!(out[3], 0.0, "L20");
        assert_eq!(out[4], 0.0, "L21");
        assert!((out[5] - 2.0).abs() < 1e-5, "L22 = phantom = L00");
    }

    /// A fully degenerate marginal must still yield a finite, positive phantom
    /// axis rather than NaN or 0 (which would make the 3D covariance singular).
    ///
    /// Note this goes through the GEOMETRIC-MEAN path, not the `counted == 0`
    /// fallback: the Crout step already floors every marginal diagonal at √ε,
    /// which is above the ε threshold, so the mean of (√ε, √ε) lands on √ε.
    /// NaN input behaves the same way — Crout's `sum > CHOLESKY_EPSILON` is
    /// false for NaN, so the diagonal is regularized before it reaches here.
    #[test]
    fn test_2d_degenerate_marginal_yields_finite_positive_phantom() {
        let eps = CHOLESKY_EPSILON.sqrt();
        for (name, packed) in [
            ("all zeros", [0.0f32, 0.0, 0.0]),
            ("NaN diagonal", [f32::NAN, 0.0, f32::NAN]),
        ] {
            let mut out = [0.0f32; 6];
            compute_display_cholesky_3d(&packed, 0, &[0, 1], &mut out);

            assert!(out[5].is_finite(), "{name}: phantom must be finite");
            assert!(out[5] > 0.0, "{name}: phantom must be positive");
            assert!(
                (out[5] / eps - 1.0).abs() < 1e-5,
                "{name}: phantom lands on √ε, got {} (expected {})",
                out[5],
                eps
            );
        }
    }

    /// The marginal must preserve a splat's TRUE scale at any scene scale.
    ///
    /// Regression: the degeneracy floor used to be an absolute variance
    /// (1e-10), but a variance is world-units², so a splat with σ = 1e-7
    /// (nm-unit data) had variance 1e-14, tripped the floor, and was inflated to
    /// σ = 1e-5 — 100× too large (10⁴× at σ = 1e-9). A 3D scene displaying
    /// [0,1,2] escapes via the standard-3D fast path, but a 2D scene, or any nD
    /// scene with hidden dims, always goes through the marginal.
    #[test]
    fn test_marginal_preserves_scale_across_scene_scales() {
        for sigma in [1e-9f32, 1e-7, 1e-5, 1e-2, 1.0, 1e3, 1e6] {
            let mut out = [0.0f32; 6];
            compute_display_cholesky_3d(&[sigma, 0.0, sigma], 0, &[0, 1], &mut out);

            for (slot, label) in [(0usize, "L00"), (2, "L11"), (5, "phantom")] {
                let ratio = out[slot] / sigma;
                assert!(
                    (ratio - 1.0).abs() < 1e-3,
                    "sigma={sigma:e}: {label} = {} is {ratio:e}x the authored scale",
                    out[slot]
                );
            }
        }
    }

    /// A genuinely rank-deficient axis is still regularized — but RELATIVE to
    /// the splat's own scale, so the covariance stays non-singular without the
    /// regularizer's magnitude depending on the scene's units.
    #[test]
    fn test_rank_deficient_axis_regularized_relative_to_scale() {
        for scale in [1e-6f32, 1.0, 1e6] {
            let mut out = [0.0f32; 6];
            // Second axis has zero extent.
            compute_display_cholesky_3d(&[4.0 * scale, 0.0, 0.0], 0, &[0, 1], &mut out);

            assert!(
                out[2] > 0.0,
                "scale={scale:e}: degenerate axis must stay > 0"
            );
            let relative = out[2] / out[0];
            assert!(
                relative > 0.0 && relative < 1e-4,
                "scale={scale:e}: regularized axis should be a tiny FRACTION of the \
                 real one, got {relative:e}"
            );
        }
    }

    /// An ISOTROPIC 2D splat must lift to an exactly isotropic 3D covariance:
    /// `L = diag(σ, σ)` ⇒ phantom == σ, so Σ = σ²·I₃.
    ///
    /// This is a load-bearing contract, not a nicety. `luxar.gsplats.lift`
    /// (`lift_points_to_gsplats`, supported for d ∈ {2, 3, 4}) writes exactly this
    /// isotropic factor and calibrates amplitude as `opacity / (uRIF · σ)`, a match
    /// that holds only when the viewer's `sigmaRay` comes out to σ. An ε phantom
    /// made `sigmaRay ≈ 1e-5`, so 2D lifted point clouds rendered ~1e5× too dim.
    #[test]
    fn test_isotropic_2d_lift_yields_isotropic_3d() {
        for sigma in [0.05f32, 1.0, 7.5, 1200.0] {
            // Packed 2D isotropic factor, exactly as lift.py emits it.
            let mut out = [0.0f32; 6];
            compute_display_cholesky_3d(&[sigma, 0.0, sigma], 0, &[0, 1], &mut out);

            let expected = [sigma, 0.0, sigma, 0.0, 0.0, sigma];
            for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-5 * want.max(1.0),
                    "sigma={sigma}: slot {i} = {got}, expected {want} (isotropic)"
                );
            }
        }
    }

    /// Two display dims on a HIGHER-dimensional dataset — the general case, which
    /// every other 2D test here misses by using `ndim == 2`. Exercises hidden-dim
    /// attenuation and phantom padding in the same pass, including a permuted
    /// display order and a high-index dim mapped to X.
    #[test]
    fn test_nd_dataset_with_two_displayed_dims() {
        let ndim = 4usize;
        let n = 3usize;
        let one = [2.0f32, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0]; // correlated 4D
        let cholesky = [one, one, one].concat();
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, 0.3, //
            2.0, 2.0, 2.0, 50.0, // far in the hidden dim -> attenuated away
        ];

        let cases = [
            (vec![0u32, 1], vec![[0.0f32, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            (
                vec![3u32, 1],
                vec![[0.0f32, 0.0, 0.0], [0.3, 1.0, 0.0], [50.0, 2.0, 0.0]],
            ),
            (vec![2u32, 0], vec![[0.0f32, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ];

        for (display, expected_centers) in cases {
            let hidden = (0..ndim as u32)
                .filter(|dim| !display.contains(dim))
                .collect::<Vec<_>>();
            let mut out_c = vec![0.0f32; n * 3];
            let mut out_l = vec![0.0f32; n * 6];
            let mut out_a = vec![0.0f32; n];
            let mut out_col = vec![0.0f32; n * 3];

            let count = project_gsplats_nd_to_3d(
                &positions,
                &cholesky,
                &[1.0, 0.8, 0.5],
                &vec![1.0f32; n * 3],
                &vec![1u8; n],
                &vec![0.0f32; ndim],
                &hidden,
                &display,
                ndim,
                n,
                3,
                1e-6,
                3.0,
                &mut out_c,
                &mut out_l,
                &mut out_a,
                &mut out_col,
            ) as usize;

            assert_eq!(
                count,
                expected_centers.len(),
                "display={display:?}, hidden={hidden:?}: unexpected visible count"
            );
            for (s, expected_center) in expected_centers.iter().enumerate() {
                let l = &out_l[s * 6..s * 6 + 6];
                assert!(
                    l.iter().all(|v| v.is_finite()),
                    "display={display:?} splat {s}: non-finite Cholesky {l:?}"
                );
                // Phantom row: zero off-diagonals, positive diagonal (SPD).
                assert_eq!(l[3], 0.0, "display={display:?} splat {s}: L20");
                assert_eq!(l[4], 0.0, "display={display:?} splat {s}: L21");
                assert!(
                    l[5] > 0.0,
                    "display={display:?} splat {s}: L22 not positive"
                );
                for axis in 0..3 {
                    assert!(
                        (out_c[s * 3 + axis] - expected_center[axis]).abs() < 1e-5,
                        "display={display:?} splat {s} axis {axis}: got {}, expected {}",
                        out_c[s * 3 + axis],
                        expected_center[axis]
                    );
                }
            }
        }
    }

    /// The phantom axis is taken over the Cholesky PIVOTS, which makes it
    /// `(det Σ_S)^(1/2n)` and therefore ROTATION-INVARIANT: rotating a 2D splat
    /// in-plane must not change its phantom extent. (A geometric mean of the
    /// per-axis marginal sigmas would drift under rotation.)
    #[test]
    fn test_phantom_axis_is_rotation_invariant() {
        // Σ = R(θ) diag(sx², sy²) R(θ)ᵀ, factored to packed lower-triangular L.
        let (sx, sy) = (3.0f32, 0.75f32);
        let mut phantoms = Vec::new();
        for deg in [0.0f32, 17.0, 45.0, 73.0, 90.0] {
            let (s, c) = deg.to_radians().sin_cos();
            let (a, b, d) = (
                c * c * sx * sx + s * s * sy * sy,
                c * s * (sx * sx - sy * sy),
                s * s * sx * sx + c * c * sy * sy,
            );
            // Cholesky of the 2x2 [[a, b], [b, d]].
            let l00 = a.sqrt();
            let l10 = b / l00;
            let l11 = (d - l10 * l10).sqrt();

            let mut out = [0.0f32; 6];
            compute_display_cholesky_3d(&[l00, l10, l11], 0, &[0, 1], &mut out);
            phantoms.push(out[5]);
        }

        // det Σ = (sx·sy)², so the invariant value is √(sx·sy).
        let expected = (sx * sy).sqrt();
        for (i, &p) in phantoms.iter().enumerate() {
            assert!(
                (p / expected - 1.0).abs() < 1e-4,
                "rotation {i}: phantom {p} drifted from the invariant {expected}"
            );
        }
    }

    /// The `counted == 0` fallback is reachable only when there are NO real
    /// diagonals to average — i.e. an empty display-dims list. Nonsensical as a
    /// view, but it must not emit a singular (zero-diagonal) covariance.
    #[test]
    fn test_empty_display_dims_yields_regularized_identity() {
        let mut out = [0.0f32; 6];
        compute_display_cholesky_3d(&[2.0, 0.5, 1.5], 0, &[], &mut out);

        let eps = CHOLESKY_EPSILON.sqrt();
        for &d in &[0usize, 2, 5] {
            assert!(
                (out[d] / eps - 1.0).abs() < 1e-5,
                "diagonal slot {} must be √ε, got {}",
                d,
                out[d]
            );
        }
        for &o in &[1usize, 3, 4] {
            assert_eq!(out[o], 0.0, "off-diagonal slot {} must be 0", o);
        }
    }

    /// Same regression for the pre-fused kernel, kept for API completeness.
    #[test]
    fn test_extract_visible_cholesky_2d_display_dims() {
        let cholesky = [2.0f32, 0.5, 1.5];
        let vis = [1u8];
        let display = [0u32, 1];
        let mut out = [0.0f32; 6];

        let count = extract_visible_cholesky_3d(&cholesky, &vis, &display, 2, 1, &mut out);

        assert_eq!(count, 1);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 1.5).abs() < 1e-5);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 0.0);
        assert!((out[5] - (2.0f32 * 1.5).sqrt()).abs() < 1e-5);
    }
}
