/**
 * Pack per-instance vertex attributes into a single
 * `InstancedInterleavedBuffer` with `InterleavedBufferAttribute`
 * views per attribute. Three.js's WebGPU backend (in
 * `three.webgpu.js::createShaderVertexBuffers`) collapses multiple
 * views over one buffer into a single vertex-buffer entry in the
 * pipeline descriptor — so a geometry with 12 attributes uses 1
 * vertex-buffer slot instead of 12.
 *
 * This matters because Chrome's compat-mode WebGPU adapter
 * hard-caps `maxVertexBuffers=8`; line materials with 6+ attributes
 * won't build their pipeline otherwise. Interleaving also gives
 * better vertex-throughput (one cache line per instance vs N
 * parallel fetches) and fewer driver calls per `setVertexBuffer`.
 *
 * Symmetry: Points / Lines flow through this helper so their
 * geometry construction stays parallel. GSplats left this path in
 * depth-sorting Phase 1 — their per-splat data lives in an RGBA32F
 * splat texture (`gsplat-geometry.ts::attachSplatStorage`), with a
 * single `aSortedIndex` instanced attribute (symmetry restored
 * when/if Points/Lines migrate — spec §8).
 *
 * **Single-dtype today.** Every attribute is packed as Float32 into
 * one shared buffer. The first attempt at narrowing
 * (BOUNDED_SCALAR / COLOR / POSITIVE_SCALAR / CHOLESKY → Float16)
 * was reverted in `dd7c4478` after we discovered that Three.js
 * r184 ignores `gpuType = HalfFloatType` on
 * `InterleavedBufferAttribute` — its WebGL path keys off
 * `InterleavedBuffer.isFloat16BufferAttribute` and its WebGPU
 * path keys off `array.constructor`. Both backends silently fell
 * through to integer vertex formats, the parity spec compared
 * two consistently-wrong backends, and the bug shipped briefly.
 *
 * **Re-introducing narrowing.** Resurrecting the narrowing path is
 * a single conceptual change, not a series of incremental flips.
 * The next attempt needs to land all of these together:
 *
 *   1. A `GpuDtype` discriminator on the spec (Float32 / Float16 /
 *      Uint8-norm) plus matching `dtypeByteWidth` and alignment
 *      logic.
 *   2. A real conversion path from the source typed-array to the
 *      target dtype (native `Float16Array` or `convertFloat32ToFloat16`
 *      shim), wired in {@link packInterleavedAttributes}.
 *   3. A per-attribute `Float16BufferAttribute` (or equivalent
 *      `InstancedInterleavedBuffer` view whose `array.constructor`
 *      is Float16Array, so the WebGL/WebGPU paths route correctly).
 *   4. An updated TSL/GLSL parity harness — the parity spec must
 *      run against a Float16-backed mesh, not just a Float32 mesh
 *      reinterpreted as Float16.
 *
 * Driving the change purely from a "default dtype per semantic"
 * map without (2)+(3)+(4) is what shipped the regression last
 * time. Keeping (1) absent from this file forces the next change
 * to land them as one PR.
 *
 * @module rendering/interleaved-attributes
 */

import * as THREE from 'three';

/**
 * Semantic tag for a per-instance attribute. Documents intent (and
 * mirrors the Python-side `SemanticType` enum in
 * `packages/luxar/src/luxar/encoding/semantic_types.py`) so a future
 * narrowing pass can route on it. Today every semantic packs as
 * Float32 — see the module header for the narrowing-redesign
 * contract.
 */
export type SemanticType =
  | 'coordinate'
  | 'color'
  | 'positive_scalar'
  | 'bounded_scalar'
  | 'cholesky'
  | 'index';

/**
 * One attribute's contribution to the interleaved buffer.
 *
 * `data` is the packed source array — `instanceCount * itemSize`
 * elements laid out as one record per instance. The helper copies
 * this into the strided position inside the shared Float32 buffer
 * at pack time.
 */
export interface InterleavedAttributeSpec {
  /** Shader-facing attribute name (e.g. `'aStartPos'`). */
  readonly name: string;
  /**
   * Per-instance packed source array. May be `Uint8Array`,
   * `Uint16Array`, or `Float32Array`; the packer widens to Float32
   * via {@link widenToFloat32}.
   */
  readonly data: Float32Array | Uint16Array | Uint8Array;
  /** Components per instance (1 = scalar, 3 = vec3, etc.). */
  readonly itemSize: 1 | 2 | 3 | 4;
  /**
   * Three.js per-view normalization flag (default `false`).
   */
  readonly normalized?: boolean;
  /**
   * Semantic tag — documents the attribute's role and is the routing
   * key for a future narrowing redesign. Today the field is
   * advisory: every spec packs as Float32 regardless. See module
   * header for the redesign contract.
   */
  readonly semantic?: SemanticType;
}

/**
 * Result of packing. Callers register `views[name]` via
 * `geometry.setAttribute(name, views[name])`; the shader-facing
 * surface is identical to the previous per-attribute layout.
 */
export interface InterleavedAttributesResult {
  /** Shared `InstancedInterleavedBuffer` holding every attribute. */
  readonly buffer: THREE.InstancedInterleavedBuffer;
  /**
   * All `InstancedInterleavedBuffer`s produced. Length 1 today;
   * preserved as an array so the narrowing redesign can grow it
   * to one buffer per dtype group without breaking callers.
   */
  readonly buffers: readonly THREE.InstancedInterleavedBuffer[];
  /** View per attribute name; pass each to `geometry.setAttribute`. */
  readonly views: Readonly<Record<string, THREE.InterleavedBufferAttribute>>;
  /** Components-per-instance in the shared buffer (sum of itemSize). */
  readonly stride: number;
  /**
   * Component-offset for each attribute name within the shared
   * buffer. Useful for {@link writeInterleavedAttribute}.
   */
  readonly offsets: Readonly<Record<string, number>>;
}

/**
 * Build the interleaved buffer and per-attribute views.
 *
 * Every attribute is packed as Float32 today. The narrowing
 * redesign (see module header) will introduce a `dtype` field on
 * the spec and a grouping step that builds one buffer per dtype
 * group; until then the API stays single-buffer for simplicity.
 *
 * @throws if `instanceCount` is negative, the spec list is empty,
 *   or any spec's `data.length !== instanceCount * itemSize`.
 */
export function packInterleavedAttributes(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number
): InterleavedAttributesResult {
  if (instanceCount < 0) {
    throw new Error(`packInterleavedAttributes: instanceCount must be >= 0, got ${instanceCount}`);
  }
  if (specs.length === 0) {
    throw new Error('packInterleavedAttributes: at least one attribute spec is required');
  }
  for (const spec of specs) {
    if (spec.data.length !== instanceCount * spec.itemSize) {
      throw new Error(
        `packInterleavedAttributes: spec '${spec.name}' has data.length=` +
          `${spec.data.length}, expected ${instanceCount * spec.itemSize} ` +
          `(instanceCount=${instanceCount} * itemSize=${spec.itemSize})`
      );
    }
  }

  // Compute stride and per-attribute float-offsets in declaration order.
  let stride = 0;
  const offsets: Record<string, number> = {};
  for (const spec of specs) {
    offsets[spec.name] = stride;
    stride += spec.itemSize;
  }

  // Allocate the shared buffer + interleave the source arrays into it.
  const packed = new Float32Array(stride * instanceCount);
  for (const spec of specs) {
    const offset = offsets[spec.name];
    const src = spec.data as ArrayLike<number>;
    const itemSize = spec.itemSize;
    for (let i = 0; i < instanceCount; i++) {
      const srcStart = i * itemSize;
      const dstStart = i * stride + offset;
      // Manual loop unroll for the common itemSize=1/2/3/4 paths beats
      // `set(src.subarray(...))` because src.subarray allocates a new
      // view per instance (millions of allocs on large datasets).
      switch (itemSize) {
        case 1:
          packed[dstStart] = src[srcStart];
          break;
        case 2:
          packed[dstStart] = src[srcStart];
          packed[dstStart + 1] = src[srcStart + 1];
          break;
        case 3:
          packed[dstStart] = src[srcStart];
          packed[dstStart + 1] = src[srcStart + 1];
          packed[dstStart + 2] = src[srcStart + 2];
          break;
        case 4:
          packed[dstStart] = src[srcStart];
          packed[dstStart + 1] = src[srcStart + 1];
          packed[dstStart + 2] = src[srcStart + 2];
          packed[dstStart + 3] = src[srcStart + 3];
          break;
      }
    }
  }

  const buffer = new THREE.InstancedInterleavedBuffer(packed, stride, 1);
  const views: Record<string, THREE.InterleavedBufferAttribute> = {};
  for (const spec of specs) {
    views[spec.name] = new THREE.InterleavedBufferAttribute(
      buffer,
      spec.itemSize,
      offsets[spec.name],
      spec.normalized ?? false
    );
  }
  return { buffer, buffers: [buffer], views, stride, offsets };
}

/**
 * Write a single packed attribute's column back into an existing
 * interleaved buffer, then mark the buffer dirty so Three.js
 * re-uploads on the next render.
 *
 * Used by in-place update paths: the loader supplies updated
 * attribute arrays one at a time (e.g., new positions arrived
 * without the colors changing); this helper writes them into
 * the strided slot inside the shared buffer.
 *
 * `opts.fromInstance` (append fast path, depth-sorting Phase 4
 * Stage 2): skip the first `fromInstance` instances — the caller
 * vouches the buffer already holds byte-identical data there — and
 * write + dirty only the `[fromInstance, instanceCount)` suffix.
 * `src` stays FULL-LENGTH (the projection always produces the whole
 * array; slicing it would just copy). Mirrors the gsplat sibling
 * `writeSplatTexels(..., { fromSplat })`.
 *
 * @throws if `src.length !== instanceCount * itemSize` or if the
 *   target range overflows the buffer.
 */
export function writeInterleavedAttribute(
  buffer: THREE.InstancedInterleavedBuffer,
  offset: number,
  itemSize: number,
  src: Float32Array,
  instanceCount: number,
  opts?: { fromInstance?: number }
): void {
  if (src.length !== instanceCount * itemSize) {
    throw new Error(
      `writeInterleavedAttribute: src.length=${src.length}, expected ` +
        `${instanceCount * itemSize} (instanceCount=${instanceCount} * itemSize=${itemSize})`
    );
  }
  const stride = buffer.stride;
  if (offset + itemSize > stride) {
    throw new Error(
      `writeInterleavedAttribute: offset+itemSize=${offset + itemSize} ` +
        `exceeds buffer stride=${stride}`
    );
  }
  if (instanceCount * stride > buffer.array.length) {
    throw new Error(
      `writeInterleavedAttribute: instanceCount=${instanceCount} * stride=${stride} ` +
        `exceeds buffer.array.length=${buffer.array.length}`
    );
  }

  // Clamp so a stale/overshooting caller degrades to a no-op write
  // rather than a negative loop bound or out-of-range dirty start.
  const from = Math.max(0, Math.min(opts?.fromInstance ?? 0, instanceCount));

  // The underlying array is a Float32Array (we always allocate one
  // in `packInterleavedAttributes`). InstancedInterleavedBuffer
  // types `.array` as the broader BufferAttribute's TypedArray
  // union; narrow here for the indexed write.
  const dst = buffer.array as Float32Array;
  for (let i = from; i < instanceCount; i++) {
    const srcStart = i * itemSize;
    const dstStart = i * stride + offset;
    switch (itemSize) {
      case 1:
        dst[dstStart] = src[srcStart];
        break;
      case 2:
        dst[dstStart] = src[srcStart];
        dst[dstStart + 1] = src[srcStart + 1];
        break;
      case 3:
        dst[dstStart] = src[srcStart];
        dst[dstStart + 1] = src[srcStart + 1];
        dst[dstStart + 2] = src[srcStart + 2];
        break;
      case 4:
        dst[dstStart] = src[srcStart];
        dst[dstStart + 1] = src[srcStart + 1];
        dst[dstStart + 2] = src[srcStart + 2];
        dst[dstStart + 3] = src[srcStart + 3];
        break;
      default:
        for (let k = 0; k < itemSize; k++) {
          dst[dstStart + k] = src[srcStart + k];
        }
        break;
    }
  }
  // Ranged upload: only the written span ([from × stride,
  // instanceCount × stride)) goes to the GPU. Without a range, THREE's
  // `bufferSubData(…, 0, array)` uploads the ENTIRE backing array —
  // including the pool's 1.5×-growth / bucket-capacity tail — on every
  // commit. All backends honor ranges on interleaved buffers, but ONLY
  // the classic WebGLRenderer merges duplicates at flush time — both
  // WebGPU backends (native and WebGL2-fallback) replay `updateRanges`
  // verbatim, and ranges also accumulate across commits while the mesh
  // is not drawn (nothing clears them until a flush). Multiple
  // per-attribute writes per commit — or thousands while a hidden layer
  // scrubs k timepoints — must therefore collapse to ONE range here,
  // not at flush time. (The gsplat commit that motivated this now
  // writes a texture, but points/lines commits still take this path
  // per attribute.)
  // Union as a single covering interval (min start / max end): the
  // per-attribute writes of one commit all share the same span, and a
  // full write is a [0, end) prefix, so the cover is exact in practice.
  let rangeStart = from * stride;
  let rangeEnd = instanceCount * stride;
  for (const range of buffer.updateRanges) {
    if (range.start < rangeStart) rangeStart = range.start;
    const end = range.start + range.count;
    if (end > rangeEnd) rangeEnd = end;
  }
  buffer.clearUpdateRanges();
  buffer.addUpdateRange(rangeStart, rangeEnd - rangeStart);
  buffer.needsUpdate = true;
}

/**
 * Widen a `Uint8Array` or `Uint16Array` source to a `Float32Array`
 * for interleaving. Useful for Points color / radius / sharpness
 * attributes that arrive as compact integer arrays.
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
