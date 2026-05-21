/**
 * nD visibility for Lines — checks whether either endpoint of each
 * segment is visible in the current slice.
 *
 * Reuses the pooled `ctx.visibilityMaskBuffer` across calls.
 */

import { requireWasm, type WasmCtx } from '../state';
import { MAX_WASM_DIMS, validateLineSegmentReferences } from '../validation';

export async function computeNDVisibilityLines(
  ctx: WasmCtx,
  params: {
    vertices: Float32Array;
    segments: Uint32Array;
    widths: Float32Array;
    slicePosition: Float32Array;
    tolerance: Float32Array;
    ndim: number;
    numSegments: number;
  }
): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  const wasmModule = requireWasm(ctx);

  const { vertices, segments, widths, slicePosition, tolerance, ndim, numSegments } = params;
  if (!Number.isInteger(ndim) || ndim < 1 || ndim > MAX_WASM_DIMS) {
    throw new Error(`computeNDVisibilityLines: ndim=${ndim} out of range [1, ${MAX_WASM_DIMS}]`);
  }
  if (slicePosition.length < ndim || tolerance.length < ndim) {
    throw new Error(`computeNDVisibilityLines: slicePosition/tolerance too short for ndim=${ndim}`);
  }
  // Validates segments[i] < vertex-count, plus widths length, against the
  // max referenced vertex (a stronger check than `>= numSegments`, which
  // earlier code did).
  validateLineSegmentReferences('computeNDVisibilityLines', segments, numSegments, vertices, ndim, {
    widths,
  });

  // Ensure buffer capacity (pooled across calls via ctx.visibilityMaskBuffer)
  let buf = ctx.visibilityMaskBuffer;
  if (!buf || buf.length < numSegments) {
    buf = new Uint8Array(Math.ceil(numSegments * 1.5));
    ctx.visibilityMaskBuffer = buf;
  }

  // Call WASM (checks if EITHER endpoint is visible)
  const visibleCount = wasmModule.compute_nd_visibility_lines(
    vertices,
    segments,
    widths,
    slicePosition,
    tolerance,
    ndim,
    numSegments,
    buf
  );

  return {
    visibilityMask: buf.subarray(0, numSegments),
    visibleCount,
  };
}
