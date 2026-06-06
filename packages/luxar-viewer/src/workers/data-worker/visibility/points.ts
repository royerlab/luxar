/**
 * nD visibility for Points — hypersphere intersection against the
 * current slice. Receives ALREADY DECODED data (ArrayDecoder runs on
 * the main thread).
 *
 * Reuses the pooled `ctx.visibilityMaskBuffer` across calls; grows it
 * to `numPoints` on demand and writes the new size back to ctx so the
 * next call can reuse it.
 */

import { pickBackend, type WasmCtx } from '../state';
import { validateNDArrays } from '../validation';

export async function computeNDVisibilityPoints(
  ctx: WasmCtx,
  params: {
    positions: Float32Array;
    radii: Float32Array;
    slicePosition: Float32Array;
    tolerance: Float32Array;
    ndim: number;
    numPoints: number;
  }
): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  const wasmModule = pickBackend(ctx, params.ndim); // >16D -> uncapped TS reference

  const { positions, radii, slicePosition, tolerance, ndim, numPoints } = params;
  validateNDArrays(
    'computeNDVisibilityPoints',
    positions,
    slicePosition,
    tolerance,
    ndim,
    numPoints,
    ndim,
    radii
  );

  // Ensure buffer capacity (pooled across calls via ctx.visibilityMaskBuffer)
  let buf = ctx.visibilityMaskBuffer;
  if (!buf || buf.length < numPoints) {
    buf = new Uint8Array(Math.ceil(numPoints * 1.5));
    ctx.visibilityMaskBuffer = buf;
  }

  // Call WASM (or TypeScript fallback)
  const visibleCount = wasmModule.compute_nd_visibility_points(
    positions,
    radii,
    slicePosition,
    tolerance,
    ndim,
    numPoints,
    buf
  );

  return {
    visibilityMask: buf.subarray(0, numPoints),
    visibleCount,
  };
}
