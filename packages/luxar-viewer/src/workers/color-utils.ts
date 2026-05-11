/**
 * Color helpers shared between the main thread and the data worker.
 *
 * Sharing matrix:
 *
 * - **Lines** uses this helper on **both** threads — the main-thread
 *   `data/lines/projection.ts:buildInstanceBuffers` and both WASM
 *   call paths import `coerceColorsToFloat32` from here.
 * - **GSplats** uses this helper on the **worker thread only**
 *   (`workers/data-worker.ts:projectGSplatsTo3D`). The main-thread
 *   path `data/gsplats/gsplats-processor.ts:processGSplats3DOnly`
 *   intentionally inlines the same `1/255` / `1/65535` math: it
 *   already has a pre-allocated output Float32Array and writes
 *   directly with no extra allocation. Per-frame splat counts are
 *   high enough that the extra alloc would be measurable; readability
 *   wins from sharing a 5-line helper aren't worth it. See commit
 *   `c93a9c20` for the trade-off rationale.
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
 * interpret values as `[0, 1]`. Worker and main-thread paths share
 * this helper so the normalization can't drift between them.
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

/**
 * Coerce a per-vertex scalar buffer (`ScalarArray`) to Float32 so the
 * WASM `interpolate_scalars_batch` kernel can consume it.
 *
 * - `Float32Array` passes through unchanged (zero copy).
 * - `Float16Array` is expanded element-wise (Float16Array's element
 *   accessor returns a JS `number`, so a simple copy through a new
 *   Float32Array is correct).
 * - `Uint8Array` is normalized by `1/255` to match the colormap
 *   shader's `[0, 1]` scalar contract (consistent with how Points
 *   handles its uint8 scalar attribute via `radiusScale = 1/255`).
 *
 * Float16 is *not* normalized: it's already a real-valued scalar in
 * whatever range the dataset author chose. The shader's
 * `uScalarMin`/`uScalarScale` uniforms perform the LUT remapping.
 */
export function coerceScalarsToFloat32(
  scalars: Float32Array | Float16Array | Uint8Array
): Float32Array {
  if (scalars instanceof Float32Array) return scalars;
  const isUint8 = scalars instanceof Uint8Array;
  const norm = isUint8 ? 1 / 255 : 1;
  const out = new Float32Array(scalars.length);
  for (let i = 0; i < scalars.length; i++) {
    out[i] = (scalars[i] as number) * norm;
  }
  return out;
}
