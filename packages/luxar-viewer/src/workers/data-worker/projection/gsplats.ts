/**
 * Project GSplats from nD to 3D with Mahalanobis-based visibility filtering.
 *
 * Uses WASM batch functions for high performance:
 * 1. compute_gsplats_attenuation - computes visibility and attenuation
 * 2. extract_3d_positions - extracts 3D centers (via displayDims)
 * 3. extract_visible_cholesky_3d - extracts 3D Cholesky submatrices
 * 4. compact_attenuated_amplitudes - compacts amplitudes by visibility
 * 5. compact_by_mask - compacts other arrays by visibility
 *
 * WASM batch functions provide 3-5x speedup over per-splat TypeScript loops.
 */

import { transfer } from 'comlink';
import { pickBackend, type WasmCtx } from '../state';
import { validateProjectionInputs } from '../validation';
import { coerceColorsToFloat32, fillColorsWhite } from '../../color-utils';
import { classifyHiddenDims } from './hidden-dims';
import { MIN_AMPLITUDE, SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE } from './constants';
import type { ProjectionViewState } from '../types';

/**
 * Coerce input colors to a normalized RGB(A) Float32Array of length
 * `splatCount * components`, white-filling when no colors were provided
 * (opaque white — alpha 1 is the per-element-opacity identity). Shared
 * by the standard-3D fast path and the general fused path so the
 * `/255`, `/65535` normalization contract stays in one place
 * (`color-utils.coerceColorsToFloat32`).
 */
function coerceColorsOrWhite(
  colors: Float32Array | Uint8Array | Uint16Array | null,
  splatCount: number,
  components: number
): Float32Array {
  if (colors) return coerceColorsToFloat32(colors);
  const out = new Float32Array(splatCount * components);
  fillColorsWhite(out, splatCount, components);
  return out;
}

export async function projectGSplatsTo3D(
  ctx: WasmCtx,
  params: {
    positions: Float32Array;
    choleskyFactors: Float32Array;
    amplitudes: Float32Array;
    colors: Float32Array | Uint8Array | Uint16Array | null;
    sharpness: Float32Array | null;
    /**
     * View state for projection. Same `ProjectionViewState` shape every
     * `project*To3D` worker function accepts. GSplats consumes
     * `displayDims` + `slicePosition`; `tolerance` is required for API
     * parity but isn't consulted by the WASM kernel (per-axis hidden-
     * dim attenuation is computed from cholesky factors instead).
     */
    viewState: ProjectionViewState;
    ndim: number;
    splatCount: number;
    /** Indices of hidden dimensions that are discrete (binary visibility) */
    discreteDims?: readonly number[];
    /** Per-dimension step sizes for discrete dims (keyed by dim index) */
    discreteSteps?: Record<number, number>;
    /** Indices of dimensions to skip entirely (extend_to_all — always visible) */
    extendToAllDims?: readonly number[];
    /** Truncation radius in sigmas for shifted Gaussian attenuation (default 3.0) */
    truncate?: number;
    /** Components per color item: 3 (RGB, default) or 4 (RGBA — alpha = per-splat opacity) */
    colorComponents?: 3 | 4;
  }
): Promise<{
  centers3D: Float32Array;
  choleskyFactors3D: Float32Array;
  amplitudes: Float32Array;
  colors: Float32Array;
  visibleCount: number;
}> {
  const wasmModule = pickBackend(ctx, params.ndim); // >16D -> uncapped TS reference

  const { positions, choleskyFactors, amplitudes, colors, sharpness, viewState, ndim, splatCount } =
    params;
  const { displayDims, slicePosition } = viewState;
  const colorComponents = params.colorComponents ?? 3;

  validateProjectionInputs(
    'projectGSplatsTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    splatCount
  );
  // Cholesky factors: packed lower-triangular = ndim × (ndim + 1) / 2 per splat.
  const expectedCholesky = splatCount * ((ndim * (ndim + 1)) / 2);
  if (choleskyFactors.length < expectedCholesky) {
    throw new Error(
      'projectGSplatsTo3D: choleskyFactors too short ' +
        `(got ${choleskyFactors.length}, expected ≥ ${expectedCholesky})`
    );
  }
  if (amplitudes.length < splatCount) {
    throw new Error(
      `projectGSplatsTo3D: amplitudes too short (got ${amplitudes.length}, expected ≥ ${splatCount})`
    );
  }
  // RGB(A) tuple per splat — short colors reach the WASM kernel
  // unchecked and read past the buffer end.
  if (colors && colors.length < splatCount * colorComponents) {
    throw new Error(
      `projectGSplatsTo3D: colors too short (got ${colors.length}, expected ≥ ${splatCount * colorComponents})`
    );
  }
  // [workers OOS] Three-geometry symmetry: Points and Lines both validate
  // their per-element `sharpness` length. GSplats accepts `sharpness` for
  // call-signature parity but never reads it (gsplats have no sharpness
  // attribute; the result carries none). Even so, callers occasionally pass
  // it; validate the length so a caller bug — building a wrong-sized
  // sharpness array — surfaces with the same clear error the other
  // geometries produce rather than silently flowing through into the
  // discard.
  if (sharpness && sharpness.length < splatCount) {
    throw new Error(
      `projectGSplatsTo3D: sharpness too short (got ${sharpness.length}, expected ≥ ${splatCount})`
    );
  }

  // Standard-3D fast path: ndim === 3 with displayDims === [0, 1, 2]
  // has no hidden dimensions, so there is nothing to attenuate or
  // compact — every splat is visible. Preserves the semantics of the
  // former main-thread `projectGSplats3DOnly`: a straight copy with NO
  // amplitude filtering (the general fused path below culls amplitude <
  // MIN_AMPLITUDE; the standard-3D copy intentionally does not). centers
  // and Cholesky are already in 3D layout (3 and 6 elements per splat).
  if (
    ndim === 3 &&
    displayDims.length === 3 &&
    displayDims[0] === 0 &&
    displayDims[1] === 1 &&
    displayDims[2] === 2
  ) {
    if (splatCount === 0) {
      const emptyF32 = new Float32Array(0);
      return transfer(
        {
          centers3D: emptyF32,
          choleskyFactors3D: new Float32Array(0),
          amplitudes: new Float32Array(0),
          colors: new Float32Array(0),
          visibleCount: 0,
        },
        [emptyF32.buffer]
      );
    }
    const centers3D = positions.slice(0, splatCount * 3);
    const choleskyFactors3D = choleskyFactors.slice(0, splatCount * 6);
    const outAmplitudes = amplitudes.slice(0, splatCount);
    const outColors = coerceColorsOrWhite(colors, splatCount, colorComponents);
    return transfer(
      {
        centers3D,
        choleskyFactors3D,
        amplitudes: outAmplitudes,
        colors: outColors,
        visibleCount: splatCount,
      },
      [
        centers3D.buffer as ArrayBuffer,
        choleskyFactors3D.buffer as ArrayBuffer,
        outAmplitudes.buffer as ArrayBuffer,
        outColors.buffer as ArrayBuffer,
      ]
    );
  }

  // Partition the hidden (non-displayed) dims into discrete (binary
  // half-step visibility) and continuous (Gaussian attenuation),
  // skipping extend_to_all dims entirely. displayDims order is
  // preserved; hidden dims are sorted ascending for the WASM Mahalanobis
  // path. (Shared with the golden-equivalence tests via hidden-dims.ts.)
  const { orderedDisplayDims, continuousHiddenDims, discreteHiddenDims } = classifyHiddenDims(
    ndim,
    displayDims,
    new Set(params.discreteDims ?? []),
    new Set(params.extendToAllDims ?? [])
  );

  // Convert to WASM-compatible arrays
  const slicePosF32 = new Float32Array(slicePosition);
  const continuousHiddenDimsU32 = new Uint32Array(continuousHiddenDims);
  const displayDimsU32 = new Uint32Array(orderedDisplayDims);

  // Minimum attenuated-amplitude threshold for visibility.
  const minAmplitude = MIN_AMPLITUDE;

  // Step 0: Pre-filter discrete dimensions (TypeScript, before WASM).
  // Splats whose center is more than half a step away in any discrete dim are invisible.
  const discreteVisibility = new Uint8Array(splatCount);
  if (discreteHiddenDims.length > 0) {
    const discreteSteps = params.discreteSteps ?? {};
    for (let i = 0; i < splatCount; i++) {
      const centerOffset = i * ndim;
      let vis = true;
      for (let dIdx = 0; dIdx < discreteHiddenDims.length; dIdx++) {
        const dim = discreteHiddenDims[dIdx];
        const step = discreteSteps[dim] ?? 1.0;
        if (Math.abs(slicePosF32[dim] - positions[centerOffset + dim]) > step * 0.5) {
          vis = false;
          break;
        }
      }
      discreteVisibility[i] = vis ? 1 : 0;
    }
  } else {
    discreteVisibility.fill(1);
  }

  // Colors are coerced to normalized f32 once (or white-filled) on this side —
  // the fused kernel takes a single Float32Array (wasm-bindgen can't accept a
  // typed-array union), keeping the /255,/65535 contract centralized in
  // color-utils. White-fill covers the whole splatCount so the kernel can read
  // colors[i*colorComponents] for any visible splat.
  const coercedColors = coerceColorsOrWhite(colors, splatCount, colorComponents);

  const truncate = params.truncate ?? SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE;

  // FUSED single-call projection: discrete gate → continuous attenuation →
  // visibility → compacted outputs, all in one pass. Replaces the former
  // 6-call pipeline (compute_gsplats_attenuation + extract_3d_positions +
  // compact_by_mask×2 + extract_visible_cholesky_3d + compact_attenuated_
  // amplitudes), eliminating ~5 full passes and the repeated copies of the
  // large positions/cholesky arrays across the wasm boundary. Outputs are
  // worst-case sized to splatCount, then sliced to the returned visible count.
  const outCentersBuf = new Float32Array(splatCount * 3);
  const outCholBuf = new Float32Array(splatCount * 6);
  const outAmpsBuf = new Float32Array(splatCount);
  const outColorsBuf = new Float32Array(splatCount * colorComponents);

  const visibleCount = wasmModule.project_gsplats_nd_to_3d(
    positions,
    choleskyFactors,
    amplitudes,
    coercedColors,
    discreteVisibility,
    slicePosF32,
    continuousHiddenDimsU32,
    displayDimsU32,
    ndim,
    splatCount,
    colorComponents,
    minAmplitude,
    truncate,
    outCentersBuf,
    outCholBuf,
    outAmpsBuf,
    outColorsBuf
  );

  // Early exit if no visible splats
  if (visibleCount === 0) {
    const emptyF32 = new Float32Array(0);
    return transfer(
      {
        centers3D: emptyF32,
        choleskyFactors3D: new Float32Array(0),
        amplitudes: new Float32Array(0),
        colors: new Float32Array(0),
        visibleCount: 0,
      },
      [emptyF32.buffer]
    );
  }

  // Dense prefix views (zero-copy); transfer the parent worst-case buffers.
  // The renderer keys off `visibleCount`, and `.length` of each subarray is the
  // exact visible extent, so the unused tail is never read.
  const centers3D = outCentersBuf.subarray(0, visibleCount * 3);
  const choleskyFactors3D = outCholBuf.subarray(0, visibleCount * 6);
  const outAmplitudes = outAmpsBuf.subarray(0, visibleCount);
  const outColors = outColorsBuf.subarray(0, visibleCount * colorComponents);

  return transfer(
    {
      centers3D,
      choleskyFactors3D,
      amplitudes: outAmplitudes,
      colors: outColors,
      visibleCount,
    },
    [
      outCentersBuf.buffer as ArrayBuffer,
      outCholBuf.buffer as ArrayBuffer,
      outAmpsBuf.buffer as ArrayBuffer,
      outColorsBuf.buffer as ArrayBuffer,
    ]
  );
}
