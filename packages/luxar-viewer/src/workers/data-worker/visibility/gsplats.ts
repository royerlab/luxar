/**
 * nD visibility for GSplats — computes the ellipsoid extent from each
 * splat's packed Cholesky factors and checks intersection with the
 * current slice.
 *
 * Reuses the pooled `ctx.visibilityMaskBuffer` across calls.
 */

import { pickBackend, type WasmCtx } from '../state';
import { validateNDArrays } from '../validation';

export async function computeNDVisibilityGSplats(
  ctx: WasmCtx,
  params: {
    centers: Float32Array;
    choleskyFactors: Float32Array;
    slicePosition: Float32Array;
    tolerance: Float32Array;
    ndim: number;
    numSplats: number;
  }
): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  const wasmModule = pickBackend(ctx, params.ndim); // >16D -> uncapped TS reference

  const { centers, choleskyFactors, slicePosition, tolerance, ndim, numSplats } = params;
  validateNDArrays(
    'computeNDVisibilityGSplats',
    centers,
    slicePosition,
    tolerance,
    ndim,
    numSplats
  );
  // Cholesky factors: packed lower-triangular has ndim*(ndim+1)/2 entries per splat.
  const expectedCholesky = numSplats * ((ndim * (ndim + 1)) / 2);
  if (choleskyFactors.length < expectedCholesky) {
    throw new Error(
      'computeNDVisibilityGSplats: choleskyFactors too short ' +
        `(got ${choleskyFactors.length}, expected ≥ ${expectedCholesky})`
    );
  }

  // Ensure buffer capacity (pooled across calls via ctx.visibilityMaskBuffer)
  let buf = ctx.visibilityMaskBuffer;
  if (!buf || buf.length < numSplats) {
    buf = new Uint8Array(Math.ceil(numSplats * 1.5));
    ctx.visibilityMaskBuffer = buf;
  }

  // Call WASM (computes ellipsoid extent from Cholesky factors)
  const visibleCount = wasmModule.compute_nd_visibility_gsplats(
    centers,
    choleskyFactors,
    slicePosition,
    tolerance,
    ndim,
    numSplats,
    buf
  );

  return {
    visibilityMask: buf.subarray(0, numSplats),
    visibleCount,
  };
}
