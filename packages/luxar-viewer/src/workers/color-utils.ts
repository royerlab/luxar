/**
 * Color helpers extracted from `workers/data-worker.ts` (Phase 18 W2).
 *
 * Both functions live here so the main-thread color paths (Lines and
 * GSplats) and the worker thread share one normalization contract
 * instead of inlining the same `1/255` / `1/65535` math in two
 * places — see `data/lines/projection.ts:buildInstanceBuffers` and
 * `data/gsplats/gsplats-processor.ts:processGSplats3DOnly` for the
 * main-thread call sites.
 *
 * Workers are a cross-cutting layer per `.dependency-cruiser.cjs`,
 * so importing from this file is fine from any layer.
 */

/**
 * Coerce a (possibly quantized) color buffer to Float32 in [0, 1].
 * F32 inputs are assumed pre-normalized and pass through unchanged;
 * Uint8 inputs are scaled by 1/255; Uint16 inputs are scaled by
 * 1/65535.
 *
 * The downstream WASM color helpers (`interpolate_colors_batch`,
 * `compact_by_mask`) and the shaders that consume their output
 * interpret values as `[0, 1]`. Phase 15.2 fixed the worker path to
 * match the main-thread normalization; Phase 18 W2 unifies them
 * here so future paths can't drift.
 */
export function coerceColorsToFloat32(
  colors: Float32Array | Uint8Array | Uint16Array
): Float32Array {
  if (colors instanceof Float32Array) return colors;
  const norm = colors instanceof Uint8Array ? 1 / 255 : 1 / 65535;
  const out = new Float32Array(colors.length);
  for (let i = 0; i < colors.length; i++) {
    out[i] = colors[i] * norm;
  }
  return out;
}

/**
 * Fill an RGB-triplet color array with white (1.0, 1.0, 1.0) for the
 * first `count` triplets. Used as the no-color default by Lines and
 * GSplats projections — the per-vertex color attribute defaults to
 * white when the node has no `colors` array.
 */
export function fillColorsWhite(out: Float32Array, count: number): void {
  for (let i = 0; i < count; i++) {
    out[i * 3] = 1.0;
    out[i * 3 + 1] = 1.0;
    out[i * 3 + 2] = 1.0;
  }
}
