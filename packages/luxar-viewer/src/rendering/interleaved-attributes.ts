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
 * hard-caps `maxVertexBuffers=8`; line / gsplat materials with 6+
 * attributes won't build their pipeline otherwise. Interleaving
 * also gives better vertex-throughput (one cache line per instance
 * vs N parallel fetches) and fewer driver calls per `setVertexBuffer`.
 *
 * Symmetry: Points / Lines / GSplats all flow through the same
 * helper so the geometry construction shape stays parallel across
 * the three node types (project policy: three-geometry symmetry).
 *
 * @module rendering/interleaved-attributes
 */

import * as THREE from 'three';

/**
 * One attribute's contribution to the interleaved buffer.
 *
 * `data` is the packed source array — `instanceCount * itemSize`
 * floats laid out as one record per instance. The helper copies
 * this into the strided position inside the shared buffer at
 * pack time.
 */
export interface InterleavedAttributeSpec {
  /** Shader-facing attribute name (e.g. `'aStartPos'`). */
  readonly name: string;
  /** Per-instance packed source array. */
  readonly data: Float32Array;
  /** Components per instance (1 = scalar, 3 = vec3, etc.). */
  readonly itemSize: 1 | 2 | 3 | 4;
  /**
   * Three.js per-view normalization flag (default `false`). Kept
   * for parity with the legacy `InstancedBufferAttribute` ctor —
   * Luxar geometries widen Uint8 → Float32 at pack time, so this
   * is `false` everywhere today; the field is here for future
   * Uint8 / Uint16 attributes that legitimately need normalization.
   */
  readonly normalized?: boolean;
}

/**
 * Result of packing. Callers register `views[name]` via
 * `geometry.setAttribute(name, views[name])`; the shader-facing
 * surface is identical to the previous per-attribute layout.
 */
export interface InterleavedAttributesResult {
  /**
   * Shared `InstancedInterleavedBuffer` holding all attributes
   * for every instance. One per geometry.
   */
  readonly buffer: THREE.InstancedInterleavedBuffer;
  /** View per attribute name; pass each to `geometry.setAttribute`. */
  readonly views: Readonly<Record<string, THREE.InterleavedBufferAttribute>>;
  /** Floats per instance — sum of every spec's `itemSize`. */
  readonly stride: number;
  /**
   * Float-offset for each attribute name within the stride. Useful
   * when an update path needs to write a single attribute's column
   * back into the interleaved storage via {@link writeInterleavedAttribute}.
   */
  readonly offsets: Readonly<Record<string, number>>;
}

/**
 * Build a single `InstancedInterleavedBuffer` from a declarative
 * list of per-instance attribute specs.
 *
 * Layout: attributes appear in declaration order; the stride equals
 * the sum of every `itemSize`. (No padding for alignment — Three.js
 * formats interleaved attributes element-wise, not vec4-bucketed.
 * The GPU happily reads any aligned float offset.)
 *
 * @throws if `instanceCount` is negative or any spec's
 *   `data.length !== instanceCount * itemSize`.
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

  // Compute stride and per-attribute offsets in declaration order.
  let stride = 0;
  const offsets: Record<string, number> = {};
  for (const spec of specs) {
    if (spec.data.length !== instanceCount * spec.itemSize) {
      throw new Error(
        `packInterleavedAttributes: spec '${spec.name}' has data.length=` +
          `${spec.data.length}, expected ${instanceCount * spec.itemSize} ` +
          `(instanceCount=${instanceCount} * itemSize=${spec.itemSize})`
      );
    }
    offsets[spec.name] = stride;
    stride += spec.itemSize;
  }

  // Allocate the shared buffer + interleave the source arrays into it.
  const packed = new Float32Array(stride * instanceCount);
  for (const spec of specs) {
    const offset = offsets[spec.name];
    const src = spec.data;
    const itemSize = spec.itemSize;
    for (let i = 0; i < instanceCount; i++) {
      const srcStart = i * itemSize;
      const dstStart = i * stride + offset;
      // Manual loop unroll for the common itemSize=1/2/3 paths beats
      // a generic `set(src.subarray(...))` because src.subarray
      // allocates a new view per instance (millions of allocs on
      // large datasets).
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

  // Construct the buffer + views. The buffer is per-instance
  // (meshPerAttribute = 1 on InstancedInterleavedBuffer ctor).
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

  return { buffer, views, stride, offsets };
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
 * @throws if `src.length !== instanceCount * itemSize` or if the
 *   target range overflows the buffer.
 */
export function writeInterleavedAttribute(
  buffer: THREE.InstancedInterleavedBuffer,
  offset: number,
  itemSize: number,
  src: Float32Array,
  instanceCount: number
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

  // The underlying array is a Float32Array (we always allocate one
  // in `packInterleavedAttributes`). InstancedInterleavedBuffer
  // types `.array` as the broader BufferAttribute's TypedArray
  // union; narrow here for the indexed write.
  const dst = buffer.array as Float32Array;
  for (let i = 0; i < instanceCount; i++) {
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
  buffer.needsUpdate = true;
}

/**
 * Widen a `Uint8Array` or `Uint16Array` source to a `Float32Array`
 * for interleaving. Useful for Points' historically-Uint8 color /
 * radius / sharpness attributes.
 *
 * When `divisor` is supplied (e.g. `255` for normalized uint8), the
 * widened floats are divided by that divisor — preserves the
 * shader-facing `[0, 1]` range previously achieved via per-attribute
 * `normalized: true`.
 */
export function widenToFloat32(
  src: ArrayLike<number>,
  divisor?: number
): Float32Array {
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
