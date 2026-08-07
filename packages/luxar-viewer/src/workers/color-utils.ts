/**
 * Color helpers shared between the main thread and the data worker.
 *
 * Sharing matrix:
 *
 * - Both **Lines** and **GSplats** projection now lives solely in the
 *   worker dispatchers (`workers/data-worker/projection/{lines,gsplats}.ts`),
 *   run either on a worker or on the main thread via
 *   `workers/data-worker/projection/in-process.ts`. Both import
 *   `coerceColorsToFloat32` from here, so the `1/255` / `1/65535`
 *   normalization contract has a single home. (The deleted main-thread
 *   projection copies used to inline the same math; W4 removed them.)
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
 * The downstream WASM color helpers (e.g. `interpolate_colors_batch`) and the
 * shaders that consume their output interpret values as `[0, 1]`. Worker and
 * main-thread paths share this helper so the normalization can't drift between
 * them.
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
 * Fill a color array with opaque white for the first `count` items.
 * Used as the no-color default by Lines and GSplats projections — the
 * per-vertex color attribute defaults to white when the node has no
 * `colors` array. `components` is 3 (RGB) or 4 (RGBA); the alpha fill
 * of 1.0 is "fully opaque", the per-element-opacity identity.
 */
export function fillColorsWhite(out: Float32Array, count: number, components = 3): void {
  const n = count * components;
  for (let i = 0; i < n; i++) {
    out[i] = 1.0;
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
