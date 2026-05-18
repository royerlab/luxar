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
 * Layouts:
 * - `'mixed'` (default, Points + GSplats): one `InstancedInterleavedBuffer`
 *   holding every attribute. Per-attribute typed views over the shared
 *   `ArrayBuffer` give heterogeneous-dtype support; the GPU sees one
 *   vertex-buffer slot. Best cache locality.
 * - `'split'` (Lines): one `InstancedInterleavedBuffer` per dtype group
 *   (Float32 / Float16 / Uint8-norm). 3 GPU vertex-buffer slots in the
 *   worst case — still well under the WebGPU max-vertex-buffer limit (8).
 *   Avoids per-attribute manual stride alignment when 11+ attributes
 *   span heterogeneous dtypes.
 *
 * Phase 4 (C-ts-1) introduces the foundation: per-attribute
 * `semantic` and `dtype` annotations, layout dispatch, conversion
 * helpers. Behavior is unchanged today — every spec resolves to
 * `'float32'` via {@link defaultGpuDtypeForSemantic}, so both
 * layouts collapse to a single Float32 buffer.
 *
 * A first attempt at narrowing (C-ts-2..5) set
 * `view.gpuType = THREE.HalfFloatType` on each `InterleavedBufferAttribute`,
 * but Three.js r184 ignores that field on interleaved attributes —
 * its WebGL path keys off `InterleavedBuffer.isFloat16BufferAttribute`
 * and its WebGPU path keys off the attribute's `constructor` /
 * `array.constructor`. Both backends fell through to integer
 * vertex formats, the parity spec compared two consistently-wrong
 * backends, and the bug shipped briefly. Reverted in `dd7c4478`.
 * A proper redesign (native `Float16Array` storage or non-interleaved
 * `Float16BufferAttribute` per narrowed attr) is a follow-up PR.
 *
 * @module rendering/interleaved-attributes
 */

import * as THREE from 'three';

/**
 * Semantic tag for a per-instance attribute. Drives the
 * default GPU dtype via {@link defaultGpuDtypeForSemantic}.
 *
 * Names parallel the Python-side `SemanticType` enum in
 * `packages/luxar/src/luxar/encoding/semantic_types.py`.
 */
export type SemanticType =
  | 'coordinate'
  | 'color'
  | 'positive_scalar'
  | 'bounded_scalar'
  | 'cholesky'
  | 'index';

/**
 * GPU-side storage dtype for an attribute. The disk dtype (whatever
 * the decoder returned) is converted to this at pack time.
 *
 * - `'float32'`: 4 bytes/component, full range, lossless.
 * - `'float16'`: 2 bytes/component, 11-bit mantissa. Bound to
 *   `THREE.HalfFloatType` on the buffer attribute.
 * - `'uint8-norm'`: 1 byte/component, `[0,255]` source mapped to
 *   `[0,1]` on the GPU via `normalize=true`. A multiplier uniform
 *   recovers the original range in the shader.
 */
export type GpuDtype = 'float32' | 'float16' | 'uint8-norm';

/**
 * Buffer layout strategy.
 *
 * - `'mixed'`: one buffer with per-attribute typed views (default).
 * - `'split'`: one buffer per dtype group.
 */
export type InterleavedLayout = 'mixed' | 'split';

/**
 * One attribute's contribution to the interleaved buffer.
 *
 * `data` is the packed source array — `instanceCount * itemSize`
 * elements laid out as one record per instance. The helper copies
 * (and optionally narrows) this into the strided position inside the
 * shared buffer at pack time.
 */
export interface InterleavedAttributeSpec {
  /** Shader-facing attribute name (e.g. `'aStartPos'`). */
  readonly name: string;
  /**
   * Per-instance packed source array. May be `Uint8Array` or
   * `Uint16Array` once the loader stops widening at decode time;
   * the packer converts to the target {@link GpuDtype}.
   */
  readonly data: Float32Array | Uint16Array | Uint8Array;
  /** Components per instance (1 = scalar, 3 = vec3, etc.). */
  readonly itemSize: 1 | 2 | 3 | 4;
  /**
   * Three.js per-view normalization flag (default `false`).
   * For `uint8-norm` dtype this is forced to `true` automatically
   * so the shader sees `[0,1]` floats.
   */
  readonly normalized?: boolean;
  /**
   * Semantic tag — drives the default GPU dtype. Optional; when
   * omitted, defaults to `'float32'`.
   */
  readonly semantic?: SemanticType;
  /**
   * Explicit GPU dtype override. Takes precedence over
   * `semantic`'s default. Use for per-geometry tuning (e.g.,
   * keep GSplats cholesky at `'float32'` if Float16 hurts
   * covariance-inversion precision).
   */
  readonly dtype?: GpuDtype;
}

/**
 * Result of packing. Callers register `views[name]` via
 * `geometry.setAttribute(name, views[name])`; the shader-facing
 * surface is identical to the previous per-attribute layout.
 */
export interface InterleavedAttributesResult {
  /**
   * Primary `InstancedInterleavedBuffer`. For `'mixed'` layout this
   * holds every attribute. For `'split'` layout this points to the
   * first dtype group's buffer; consumers should use {@link views}
   * to access individual attributes.
   */
  readonly buffer: THREE.InstancedInterleavedBuffer;
  /**
   * All `InstancedInterleavedBuffer`s produced. Length 1 for
   * `'mixed'` layout, ≥1 for `'split'` layout (one per dtype group).
   */
  readonly buffers: readonly THREE.InstancedInterleavedBuffer[];
  /** View per attribute name; pass each to `geometry.setAttribute`. */
  readonly views: Readonly<Record<string, THREE.InterleavedBufferAttribute>>;
  /**
   * Components-per-instance for the primary buffer. For `'mixed'`
   * layout: sum of every spec's `itemSize`. For `'split'` layout:
   * the primary (first) group's stride only.
   */
  readonly stride: number;
  /**
   * Component-offset for each attribute name within its buffer. For
   * `'mixed'` layout, all offsets share the same buffer's stride;
   * for `'split'` layout each offset is into its own group buffer.
   * Useful for {@link writeInterleavedAttribute}.
   */
  readonly offsets: Readonly<Record<string, number>>;
}

/**
 * Resolve the effective GPU dtype for a spec: explicit `dtype`
 * wins, then semantic-default, then `'float32'`.
 */
export function effectiveDtype(spec: InterleavedAttributeSpec): GpuDtype {
  return spec.dtype ?? defaultGpuDtypeForSemantic(spec.semantic);
}

/**
 * Default GPU dtype for each {@link SemanticType}. The map is the
 * single source of truth for "what does this kind of value want to
 * become on the GPU" once a proper narrowing path lands.
 *
 * Currently every entry is `'float32'` — see the module docstring
 * for why the first narrowing attempt was reverted in `dd7c4478`.
 * A follow-up PR will set individual entries (e.g., `color: 'float16'`
 * or `bounded_scalar: 'uint8-norm'`) once the backend-attribute API
 * is wired correctly for the Three.js r184 vertex-format path.
 */
const DEFAULT_GPU_DTYPE_BY_SEMANTIC: Record<SemanticType, GpuDtype> = {
  coordinate: 'float32',
  color: 'float32',
  positive_scalar: 'float32',
  bounded_scalar: 'float32',
  cholesky: 'float32',
  // INDEX attributes don't flow through this packer (they're handled
  // by the geometry's element-array buffer); the value is here for
  // type-coverage only.
  index: 'float32',
};

/**
 * Return the GPU dtype to use for an attribute whose `semantic` is
 * known but whose explicit `dtype` is not set.
 */
export function defaultGpuDtypeForSemantic(
  semantic: SemanticType | undefined
): GpuDtype {
  if (semantic === undefined) return 'float32';
  return DEFAULT_GPU_DTYPE_BY_SEMANTIC[semantic];
}

/** Byte width of one component of the given GPU dtype. */
export function dtypeByteWidth(dtype: GpuDtype): 1 | 2 | 4 {
  switch (dtype) {
    case 'float32':
      return 4;
    case 'float16':
      return 2;
    case 'uint8-norm':
      return 1;
  }
}

/**
 * Round a byte offset up to the next 4-byte boundary. WebGPU's
 * vertex-fetch unit requires Float32 attribute offsets at 4-byte
 * alignment; Float16 needs 2-byte; Uint8 needs 1-byte. Ordering
 * attributes Float32 → Float16 → Uint8 within a mixed stride keeps
 * all of them aligned without manual padding except the trailing
 * stride round-up.
 */
export function alignTo4(byteOffset: number): number {
  return (byteOffset + 3) & ~3;
}

/**
 * Convert a source typed array to the target GPU dtype. Used by the
 * packer when narrowing a Float32 disk-decoded array to Float16 or
 * Uint8-norm for GPU upload.
 *
 * For C-ts-1 only the `'float32'` target is implemented — narrowing
 * targets activate in C-ts-2..4 as the corresponding semantic flips.
 */
export function convertToGpuDtype(
  src: ArrayLike<number>,
  target: GpuDtype,
  divisorIfNormalized?: number
): Float32Array | Uint16Array | Uint8Array {
  if (target === 'float32') {
    return widenToFloat32(src, divisorIfNormalized);
  }
  throw new Error(
    `convertToGpuDtype: target '${target}' is not yet implemented ` +
      `(activates in phase-4 C-ts-${target === 'float16' ? '2/3/4' : '3'})`
  );
}

/**
 * Build the interleaved buffer(s) and per-attribute views.
 *
 * Dispatches to the requested layout — `'mixed'` (default) packs
 * every spec into one buffer; `'split'` groups by dtype.
 *
 * At C-ts-1 every spec must resolve to `'float32'` (the only
 * dtype with a packing implementation). Other dtypes throw via
 * {@link convertToGpuDtype}.
 *
 * @throws if `instanceCount` is negative or any spec's
 *   `data.length !== instanceCount * itemSize`.
 */
export function packInterleavedAttributes(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number,
  layout: InterleavedLayout = 'mixed'
): InterleavedAttributesResult {
  if (instanceCount < 0) {
    throw new Error(
      `packInterleavedAttributes: instanceCount must be >= 0, got ${instanceCount}`
    );
  }
  if (specs.length === 0) {
    throw new Error(
      'packInterleavedAttributes: at least one attribute spec is required'
    );
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

  return layout === 'split'
    ? packSplitLayout(specs, instanceCount)
    : packMixedLayout(specs, instanceCount);
}

/**
 * Pack every spec into one shared `InstancedInterleavedBuffer`.
 * For all-Float32 specs (current state at C-ts-1) the buffer is a
 * single `Float32Array`. Heterogeneous-dtype support lands in C-ts-2+.
 */
function packMixedLayout(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number
): InterleavedAttributesResult {
  const allFloat32 = specs.every((s) => effectiveDtype(s) === 'float32');
  if (!allFloat32) {
    // Future commits (C-ts-2..4) will implement the mixed-dtype path
    // here. At C-ts-1 every caller still passes `dtype: 'float32'`.
    throw new Error(
      'packMixedLayout: mixed-dtype packing not yet implemented (C-ts-1 supports float32 only)'
    );
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
 * Pack specs by grouping on their effective dtype. Each group gets
 * its own `InstancedInterleavedBuffer`. At C-ts-1 the only dtype is
 * `'float32'`, so this collapses to one group whose output matches
 * {@link packMixedLayout} byte-for-byte.
 *
 * Result invariant: the primary buffer (`.buffer` field) is the
 * *first* group's buffer in attribute declaration order, so a
 * Lines geometry that starts with `aStartPos` (Float32) keeps
 * `result.buffer` pointing at the Float32 group.
 */
function packSplitLayout(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number
): InterleavedAttributesResult {
  // Group specs by dtype preserving first-occurrence order so the
  // primary buffer is deterministic.
  const dtypeOrder: GpuDtype[] = [];
  const groups = new Map<GpuDtype, InterleavedAttributeSpec[]>();
  for (const spec of specs) {
    const dtype = effectiveDtype(spec);
    if (!groups.has(dtype)) {
      dtypeOrder.push(dtype);
      groups.set(dtype, []);
    }
    groups.get(dtype)!.push(spec);
  }

  const buffers: THREE.InstancedInterleavedBuffer[] = [];
  const views: Record<string, THREE.InterleavedBufferAttribute> = {};
  const offsets: Record<string, number> = {};
  let primaryStride = 0;

  for (const dtype of dtypeOrder) {
    const groupSpecs = groups.get(dtype)!;
    if (dtype !== 'float32') {
      throw new Error(
        `packSplitLayout: group dtype '${dtype}' not yet implemented ` +
          '(C-ts-1 supports float32 only)'
      );
    }

    // Float32 group — same packing as mixed layout, scoped to this group.
    let groupStride = 0;
    const groupOffsets: Record<string, number> = {};
    for (const spec of groupSpecs) {
      groupOffsets[spec.name] = groupStride;
      offsets[spec.name] = groupStride;
      groupStride += spec.itemSize;
    }

    const packed = new Float32Array(groupStride * instanceCount);
    for (const spec of groupSpecs) {
      const offset = groupOffsets[spec.name];
      const src = spec.data as ArrayLike<number>;
      const itemSize = spec.itemSize;
      for (let i = 0; i < instanceCount; i++) {
        const srcStart = i * itemSize;
        const dstStart = i * groupStride + offset;
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

    const buffer = new THREE.InstancedInterleavedBuffer(packed, groupStride, 1);
    buffers.push(buffer);
    if (primaryStride === 0) {
      primaryStride = groupStride;
    }

    for (const spec of groupSpecs) {
      views[spec.name] = new THREE.InterleavedBufferAttribute(
        buffer,
        spec.itemSize,
        groupOffsets[spec.name],
        spec.normalized ?? false
      );
    }
  }

  return {
    buffer: buffers[0],
    buffers,
    views,
    stride: primaryStride,
    offsets,
  };
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
