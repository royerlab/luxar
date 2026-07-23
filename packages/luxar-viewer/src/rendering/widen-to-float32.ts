/**
 * Dtype widening for the texel writers.
 *
 * The sole survivor of the interleaved-attributes era: the fused texel
 * writers (`point-geometry.ts::writePointTexels`,
 * `line-geometry.ts::writeLineTexels`, the gsplat packer) consume
 * ALREADY-WIDENED Float32 arrays, and this helper is how the call sites
 * (`gpu-buffer-pool/points-adapter.ts`,
 * `node-factory/create-points-node.ts`) widen compact integer sources
 * with the exact same normalization the per-attribute `normalized: true`
 * flag used to provide — so texel values are bit-identical to what the
 * interleaved-attribute uploads produced.
 *
 * @module rendering/widen-to-float32
 */

/**
 * Widen a `Uint8Array` / `Uint16Array` / `Float16Array` source to a
 * `Float32Array`.
 *
 * When `divisor` is supplied (e.g. `255` for normalized uint8), the
 * widened floats are divided by that divisor — preserves the
 * shader-facing `[0, 1]` range previously achieved via per-attribute
 * `normalized: true`.
 */
export function widenToFloat32(src: ArrayLike<number>, divisor?: number): Float32Array {
  if (src instanceof Float32Array && divisor === undefined) {
    // No conversion needed — caller can keep the reference. Cloning
    // would just waste memory.
    return src;
  }
  const out = new Float32Array(src.length);
  if (divisor !== undefined) {
    const inv = 1.0 / divisor;
    for (let i = 0; i < src.length; i++) {
      out[i] = src[i] * inv;
    }
  } else {
    for (let i = 0; i < src.length; i++) {
      out[i] = src[i];
    }
  }
  return out;
}
