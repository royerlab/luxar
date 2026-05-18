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
 * helpers. Behavior is unchanged at this commit — every caller still
 * passes `dtype: 'float32'` explicitly, so both layouts collapse to a
 * single Float32 buffer (current behavior). Narrowing activates in
 * C-ts-2..4 by dropping the explicit `dtype` override on individual
 * attributes so the semantic-default kicks in.
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
 * become on the GPU". Phase-4 commits flip individual entries:
 *
 * - C-ts-2: `color: 'float16'`
 * - C-ts-3: `positive_scalar: 'float16'`, `bounded_scalar: 'uint8-norm'`
 * - C-ts-4: `cholesky: 'float16'`
 *
 * At C-ts-1 every entry is `'float32'` → behavior unchanged.
 */
const DEFAULT_GPU_DTYPE_BY_SEMANTIC: Record<SemanticType, GpuDtype> = {
  coordinate: 'float32',
  // C-ts-2: COLOR narrowed to Float16 on GPU. 11-bit mantissa gives
  // ~5e-4 absolute error on SDR colours in [0,1] — well below the
  // parity-spec mean-abs-diff threshold on 0-255 scale.
  color: 'float16',
  // C-ts-3: POSITIVE_SCALAR (radii, widths, amplitudes) narrowed to
  // Float16 on GPU. These hold scene-unit magnitudes; Float16 covers
  // up to ±65504 with ~5e-4 relative precision — overkill for scene
  // distances and far more headroom than needed for typical scenes.
  // No shader change: TSL/GLSL still read `float`. GPU widens at
  // attribute fetch.
  positive_scalar: 'float16',
  // C-ts-3b (future): BOUNDED_SCALAR (sharpness, clipped flags)
  // will narrow to Uint8-norm. Requires threading the encoder's
  // bounds metadata (V_max) to the shader's `sharpnessScale` /
  // `radiusScale` uniforms — non-trivial because bounds vary per
  // dataset. Stays Float32 for now.
  bounded_scalar: 'float32',
  // C-ts-4: CHOLESKY (GSplats covariance) narrowed to Float16 on
  // GPU. Float16 mantissa (~5e-4 relative precision) is adequate
  // for typical splat aspect ratios up to ~1000:1. The shader
  // reads cholesky as `float` and runs inversion in single
  // precision, so the quantization is bounded to the storage step.
  // Per-attribute override (`dtype: 'float32'` on specific
  // gsplat specs) is available if parity regresses on extreme
  // anisotropy.
  cholesky: 'float16',
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
 * For `float16` target: returns a `Uint16Array` carrying half-float
 * bits per `THREE.DataUtils.toHalfFloat` — Three.js's WebGPU/WebGL
 * backends treat a `Uint16Array` attribute with `gpuType =
 * THREE.HalfFloatType` as a half-precision float.
 *
 * For `uint8-norm`: not yet implemented (lands in C-ts-3).
 */
export function convertToGpuDtype(
  src: ArrayLike<number>,
  target: GpuDtype,
  divisorIfNormalized?: number
): Float32Array | Uint16Array | Uint8Array {
  if (target === 'float32') {
    return widenToFloat32(src, divisorIfNormalized);
  }
  if (target === 'float16') {
    const out = new Uint16Array(src.length);
    if (divisorIfNormalized !== undefined) {
      const inv = 1.0 / divisorIfNormalized;
      for (let i = 0; i < src.length; i++) {
        out[i] = THREE.DataUtils.toHalfFloat(src[i] * inv);
      }
    } else {
      for (let i = 0; i < src.length; i++) {
        out[i] = THREE.DataUtils.toHalfFloat(src[i]);
      }
    }
    return out;
  }
  throw new Error(
    `convertToGpuDtype: target '${target}' is not yet implemented ` +
      '(activates in phase-4 C-ts-3)'
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
 * Pack every spec into one or more `InstancedInterleavedBuffer`s,
 * one per dtype group. `'mixed'` and `'split'` both group by dtype —
 * the distinction is documentation-only since Three.js's
 * `InterleavedBuffer.array` is a single TypedArray. The two layouts
 * diverge if/when WebGPU vertex-pulling makes "one ArrayBuffer with
 * mixed-dtype views" viable; today they share an implementation.
 *
 * For all-Float32 specs (most current geometries when no semantic
 * has flipped yet) the result is one Float32 buffer — byte-for-byte
 * identical to pre-phase-4 behavior.
 */
function packMixedLayout(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number
): InterleavedAttributesResult {
  return packByDtypeGroups(specs, instanceCount);
}

/**
 * Split layout — currently identical to mixed (see {@link packMixedLayout}).
 * Kept as a separate entry point so per-geometry tuning can diverge
 * later without churning every caller.
 */
function packSplitLayout(
  specs: readonly InterleavedAttributeSpec[],
  instanceCount: number
): InterleavedAttributesResult {
  return packByDtypeGroups(specs, instanceCount);
}

/**
 * Shared dtype-grouped packer. Groups specs by their effective GPU
 * dtype preserving first-occurrence order, then for each group:
 * - Allocates a TypedArray of the right type.
 * - Interleaves the source arrays into it (narrowing via
 *   {@link convertToGpuDtype} where needed).
 * - Builds an `InstancedInterleavedBuffer` + per-attribute views.
 *
 * The primary buffer (returned in `.buffer`) is the first group's
 * buffer in attribute declaration order.
 */
function packByDtypeGroups(
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

    let groupStride = 0;
    const groupOffsets: Record<string, number> = {};
    for (const spec of groupSpecs) {
      groupOffsets[spec.name] = groupStride;
      offsets[spec.name] = groupStride;
      groupStride += spec.itemSize;
    }

    // Allocate the typed buffer + pack
    const packed = packGroup(dtype, groupSpecs, groupStride, instanceCount, groupOffsets);
    const buffer = new THREE.InstancedInterleavedBuffer(packed, groupStride, 1);
    buffers.push(buffer);
    if (primaryStride === 0) {
      primaryStride = groupStride;
    }

    for (const spec of groupSpecs) {
      const view = new THREE.InterleavedBufferAttribute(
        buffer,
        spec.itemSize,
        groupOffsets[spec.name],
        // For uint8-norm dtype force normalize=true so the shader
        // sees [0,1]. Activated in C-ts-3.
        dtype === 'uint8-norm' ? true : spec.normalized ?? false
      );
      // Three.js needs gpuType=HalfFloatType to interpret a Uint16
      // attribute as half-float (rather than as a Uint16 vertex
      // input). InterleavedBufferAttribute inherits gpuType from
      // BufferAttribute in r152+.
      if (dtype === 'float16') {
        (view as unknown as { gpuType: number }).gpuType = THREE.HalfFloatType;
      }
      views[spec.name] = view;
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
 * Allocate the typed array for one dtype group and interleave the
 * source data into it, narrowing per-attribute via
 * {@link convertToGpuDtype} when the source is not already at the
 * target dtype.
 *
 * For the Float32 group on Float32 sources this is a tight memcpy
 * loop with the itemSize=1/2/3/4 unrolls. For the Float16 group on
 * Float32 sources, each value goes through `toHalfFloat` — slightly
 * slower per element but the buffer is half the size on the GPU.
 */
function packGroup(
  dtype: GpuDtype,
  groupSpecs: readonly InterleavedAttributeSpec[],
  stride: number,
  instanceCount: number,
  groupOffsets: Record<string, number>
): Float32Array | Uint16Array | Uint8Array {
  const totalElements = stride * instanceCount;
  let packed: Float32Array | Uint16Array | Uint8Array;
  switch (dtype) {
    case 'float32':
      packed = new Float32Array(totalElements);
      break;
    case 'float16':
      packed = new Uint16Array(totalElements);
      break;
    case 'uint8-norm':
      packed = new Uint8Array(totalElements);
      break;
  }

  // Float32 fast path — direct memcpy, no narrowing.
  if (dtype === 'float32') {
    for (const spec of groupSpecs) {
      const offset = groupOffsets[spec.name];
      const src = spec.data as ArrayLike<number>;
      const itemSize = spec.itemSize;
      for (let i = 0; i < instanceCount; i++) {
        const srcStart = i * itemSize;
        const dstStart = i * stride + offset;
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
    return packed;
  }

  // Float16 / Uint8-norm: narrow via `convertToGpuDtype` (per spec).
  // For each spec we allocate a narrowed view of its source, then
  // interleave that into the group buffer. The narrowed view is a
  // single allocation per spec, not per instance — millions of
  // instances do not multiply the alloc count.
  for (const spec of groupSpecs) {
    const offset = groupOffsets[spec.name];
    const narrowed = convertToGpuDtype(spec.data, dtype) as
      | Uint16Array
      | Uint8Array;
    const itemSize = spec.itemSize;
    for (let i = 0; i < instanceCount; i++) {
      const srcStart = i * itemSize;
      const dstStart = i * stride + offset;
      switch (itemSize) {
        case 1:
          packed[dstStart] = narrowed[srcStart];
          break;
        case 2:
          packed[dstStart] = narrowed[srcStart];
          packed[dstStart + 1] = narrowed[srcStart + 1];
          break;
        case 3:
          packed[dstStart] = narrowed[srcStart];
          packed[dstStart + 1] = narrowed[srcStart + 1];
          packed[dstStart + 2] = narrowed[srcStart + 2];
          break;
        case 4:
          packed[dstStart] = narrowed[srcStart];
          packed[dstStart + 1] = narrowed[srcStart + 1];
          packed[dstStart + 2] = narrowed[srcStart + 2];
          packed[dstStart + 3] = narrowed[srcStart + 3];
          break;
      }
    }
  }
  return packed;
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
 * `src` must already match the buffer's underlying TypedArray dtype.
 * For narrowing writes (Float32 source → Float16 destination), call
 * {@link convertToGpuDtype} first to produce the right typed array.
 *
 * @throws if `src.length !== instanceCount * itemSize`, the target
 *   range overflows the buffer, or the src and buffer TypedArrays
 *   don't match.
 */
export function writeInterleavedAttribute(
  buffer: THREE.InstancedInterleavedBuffer,
  offset: number,
  itemSize: number,
  src: Float32Array | Uint16Array | Uint8Array,
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
  if (src.constructor !== buffer.array.constructor) {
    throw new Error(
      `writeInterleavedAttribute: src is ${src.constructor.name} but buffer.array ` +
        `is ${buffer.array.constructor.name} — narrow via convertToGpuDtype first`
    );
  }

  // After the constructor check above, `dst` shares the indexable
  // number-array shape with `src`. Cast both to a common indexable
  // type so the unrolled writes stay tight.
  const dst = buffer.array as unknown as { [k: number]: number; length: number };
  const s = src as unknown as { [k: number]: number };
  for (let i = 0; i < instanceCount; i++) {
    const srcStart = i * itemSize;
    const dstStart = i * stride + offset;
    switch (itemSize) {
      case 1:
        dst[dstStart] = s[srcStart];
        break;
      case 2:
        dst[dstStart] = s[srcStart];
        dst[dstStart + 1] = s[srcStart + 1];
        break;
      case 3:
        dst[dstStart] = s[srcStart];
        dst[dstStart + 1] = s[srcStart + 1];
        dst[dstStart + 2] = s[srcStart + 2];
        break;
      case 4:
        dst[dstStart] = s[srcStart];
        dst[dstStart + 1] = s[srcStart + 1];
        dst[dstStart + 2] = s[srcStart + 2];
        dst[dstStart + 3] = s[srcStart + 3];
        break;
      default:
        for (let k = 0; k < itemSize; k++) {
          dst[dstStart + k] = s[srcStart + k];
        }
        break;
    }
  }
  buffer.needsUpdate = true;
}

/**
 * Write an attribute's data column from a spec into the geometry's
 * existing interleaved storage, narrowing if needed. Resolves the
 * spec's view + buffer + offset by name from the geometry.
 *
 * Used by in-place size-unchanged updates that re-emit ALL specs:
 * the spec list may span multiple buffers after dtype groups split
 * (e.g., COLOR moves to its own Float16 group in C-ts-2). This
 * helper resolves each attribute's storage independently rather
 * than assuming a single shared buffer.
 *
 * @throws if `geometry` doesn't have an attribute named `spec.name`
 *   bound as an `InterleavedBufferAttribute`.
 */
export function writeInterleavedAttributeFromSpec(
  geometry: THREE.InstancedBufferGeometry,
  spec: InterleavedAttributeSpec,
  instanceCount: number
): void {
  const view = geometry.getAttribute(spec.name) as
    | THREE.InterleavedBufferAttribute
    | undefined;
  if (!view || !(view as { isInterleavedBufferAttribute?: boolean }).isInterleavedBufferAttribute) {
    throw new Error(
      `writeInterleavedAttributeFromSpec: '${spec.name}' is not bound as an ` +
        'InterleavedBufferAttribute on this geometry'
    );
  }
  const buffer = view.data as THREE.InstancedInterleavedBuffer;
  const dtype = effectiveDtype(spec);
  // Narrow only if the buffer's TypedArray doesn't already match the
  // source's. For all-Float32 (most paths today) this is a no-op
  // alias when the source is already Float32.
  const src =
    dtype === 'float32'
      ? widenToFloat32(spec.data as ArrayLike<number>)
      : convertToGpuDtype(spec.data as ArrayLike<number>, dtype);
  writeInterleavedAttribute(buffer, view.offset, spec.itemSize, src, instanceCount);
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
