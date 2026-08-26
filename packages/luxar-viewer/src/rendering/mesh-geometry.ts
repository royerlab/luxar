/**
 * Build and update the `THREE.BufferGeometry` for a Mesh node.
 *
 * Mesh is the only geometry type that draws a plain indexed
 * `BufferGeometry` + `THREE.Mesh`. The other three build an
 * `InstancedBufferGeometry` of quads and push per-element attributes through an
 * RGBA32F element texture, because they render *per-element sprites* whose size
 * and orientation are computed in the shader. A triangle is already geometry —
 * there is nothing to instance and no per-element extent to encode — so the
 * instanced-quad/element-texture stack is not a gap here, it is the wrong shape.
 *
 * A direct consequence: mesh does **not** use the GPU buffer pool
 * (`gpu-buffer-pool/pool-stats.ts` must keep listing exactly the three instanced
 * types), and its `color`/`aScalar` data goes to **vertex attributes** rather than
 * a texture. Mesh is the first type to do that, which is why the dtype rules below
 * exist and why nothing in the shipped tree had hit them before.
 *
 * @module rendering/mesh-geometry
 */

import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import type { MeshColorArray, MeshProjectionBounds } from '../types/mesh';

/**
 * Everything {@link createMeshGeometry} and {@link updateMeshGeometry} need to lay out
 * the buffers.
 *
 * `*Config` after the sibling geometry modules' `InstancedLinesMeshConfig` /
 * `InstancedGSplatsMeshConfig` — same role, minus the `Instanced` qualifier those two
 * carry because a mesh is not an instanced quad.
 */
export interface MeshGeometryConfig {
  /** Display-space positions (`vertexCount * 3`), from `projectMeshTo3D`. */
  position: Float32Array;
  /** Index buffer of visible triangles (`visibleFaceCount * 3`). */
  indices: Uint32Array;
  /**
   * Whether `position` holds newly extracted values this epoch.
   *
   * Load-bearing, and not derivable here. The projection REUSES one
   * loader-owned position buffer across epochs (`LoadedMeshData.projection`), so
   * array identity can no longer answer "did the displayed axes change": the
   * identity is stable while the contents change on a `displayDims` change, and
   * unchanged on a pure slice move. Gating on identity therefore either stops
   * re-uploading after an axis permutation (reused buffer) or re-uploads the whole
   * vertex buffer on every slice move (freshly allocated buffer) — #1245.
   */
  positionChanged: boolean;
  /**
   * The projection's `displayDims.join()`. Compared against the key the geometry last
   * uploaded (`geometry.userData.meshUploadedKey`); a difference forces the position
   * re-upload even when `positionChanged` is false, which repairs a commit that was
   * superseded after its projection advanced the loader's key but before it uploaded.
   * Optional: callers that omit it (the placeholder factory, unit tests) keep the
   * pure-`positionChanged` behavior.
   */
  positionKey?: string;
  /** Per-vertex colors in their native dtype, or `null` for the white default. */
  colors: MeshColorArray | null;
  /** Channels per color entry when `colors` is present. */
  colorComponents?: 3 | 4;
  /**
   * Per-vertex normals (`vertexCount * 3`), or `null` when the node has none.
   *
   * Whether the `normal` attribute exists at all is decided ONCE, at creation, from
   * this being non-null — see {@link createMeshGeometry}. The placeholder factory
   * passes a 1-vertex stub when the node's metadata says `has_normals`, so a
   * normal-bearing mesh binds the attribute before any data arrives and the first
   * commit only replaces its contents.
   *
   * Always bound when the node HAS normals, even during epochs whose `displayDims`
   * make them meaningless (§3.4). That is a deliberate trade: the alternative —
   * unbinding on a frame change — would mutate a live geometry's attribute SET,
   * which the WebGPU backend bakes into its render pipeline. The unused buffer costs
   * `V * 12` bytes of VRAM; the shader variant is what actually stops reading it.
   */
  normals?: Float32Array | null;
  /**
   * Per-vertex scalars for the colormap path (`vertexCount`), or `null`/absent when
   * the node has none. Bound as `aScalar`. Same creation-time set rule as `normals`.
   *
   * Already `float32` by the time it reaches here (the loader decodes to
   * `Float32Array`), which is also the only itemSize-1 vertex format three r184 can
   * bind on both backends: it has no format for a `Uint8Array` or a native
   * `Float16Array` at itemSize 1 (§6.1.1).
   */
  scalars?: Float32Array | null;
  /**
   * Per-vertex texture coordinates (`vertexCount * 2`), or `null`/absent when the
   * node has no texture. Bound as `uv`. Same creation-time set rule as `normals`.
   *
   * Named `uv` on the GPU rather than `aUv`, unlike `aScalar`: three.js's vertex
   * prefix declares `position`/`normal`/`uv` for every program, so the standard
   * name is already there to be filled and a custom one would need its own
   * declaration in both shader backends for no gain.
   */
  uvs?: Float32Array | null;
  /** Vertices in the buffers (NOT the visible count — nothing is compacted). */
  vertexCount: number;
  /**
   * Total faces in the node, which sizes the index buffer's CAPACITY.
   *
   * Not `indices.length / 3`: that is the currently-visible count, which changes
   * every slice move. The buffer is allocated once at the node's full face count and
   * the visible prefix is drawn via `drawRange` — see {@link applyMeshIndices}.
   */
  faceCount: number;
  /**
   * Projected AABB over the indexed vertices, or `null` when nothing is drawn.
   *
   * Mirrors `LineTexelSource.bounds` and the gsplat mesh config's `bounds`: the
   * projection precomputes the box and {@link computeMeshBounds} sets the geometry's
   * bounding box and sphere from it, instead of a `computeBoundingBox()` scan that
   * would also span vertices the cull removed.
   *
   * Optional so the placeholder factory can build a geometry without a projection;
   * `undefined` falls back to the scan, `null` means nothing is drawn.
   */
  bounds?: MeshProjectionBounds | null;
  /**
   * Whether this node's vertex count legitimately GROWS between commits.
   *
   * True for a reveal-ladder node, whose every level adds vertices, and false for
   * every other mesh — a whole-node mesh is fetched once, so a changed vertex count
   * there means the buffers and the metadata disagree and is worth a warning. The
   * flag exists only to keep that warning meaningful; the rebind itself is identical
   * either way.
   */
  vertexCountGrows?: boolean;
  /**
   * The node's LIFETIME vertex/face totals, which every buffer here is sized and
   * dtype-chosen from — as distinct from {@link vertexCount} / {@link faceCount},
   * which describe the data being committed NOW.
   *
   * They differ for exactly one thing: a reveal ladder, whose committed prefix
   * grows a level at a time while the node's total does not. That distinction is
   * the whole point. Sizing from the prefix would re-run `setAttribute` /
   * `setIndex` on every level, and three frees a replaced attribute's GL buffer
   * from nowhere — not on replacement, and not on dispose either (only the
   * attributes still bound at that moment are freed). So each level would orphan
   * the previous level's buffers for the session: ~150 MB for a 4-level 2M-vertex
   * surface with normals and colours, near a gigabyte at 10M (#1521). Sizing from
   * the total restores the allocate-once invariant `applyMeshIndices` documents,
   * and the growing prefix is written INTO the same buffers.
   *
   * Absent (or equal to the live counts) for every unladdered mesh, which is why
   * this change is a no-op there — including the zero-copy colour path.
   */
  capacityVertexCount?: number;
  capacityFaceCount?: number;
}

/**
 * Fit `source` into a buffer of `capacity` rows, copying only when it must.
 *
 * Returns the source array UNCHANGED when the capacity is exactly the live count,
 * which is every mesh that is not a reveal ladder — so the zero-copy paths below
 * stay zero-copy and their buffers stay byte-identical to before #1521.
 */
function atCapacity<A extends { length: number; set(a: ArrayLike<number>, o?: number): void }>(
  source: A,
  perItem: number,
  count: number,
  capacity: number
): A {
  if (capacity <= count) return source;
  const ctor = (source as unknown as { constructor: new (n: number) => A }).constructor;
  const out = new ctor(capacity * perItem);
  out.set(source as unknown as ArrayLike<number>, 0);
  return out;
}

/**
 * Whether `geometry.userData` already stamps `source`/`count` under `key` — the
 * shared currency check `refreshMeshColors` and `replaceVertexAttribute` both need.
 *
 * Once an attribute is bound at the node's CAPACITY (#1521), the bound buffer is a
 * copy `atCapacity` made once at the rebind and is never identity-equal to the
 * caller's array again — not even on the very level that produced it. (The
 * RGB→RGBA pad path is the other way to lose identity: it allocates a fresh
 * `padded` array on every write.) So currency has to be tracked by the SOURCE
 * array's own identity, stamped separately on `geometry.userData`, rather than by
 * comparing it against the bound buffer.
 */
function isAttributeCurrent(
  geometry: THREE.BufferGeometry,
  key: string,
  source: unknown,
  count: number
): boolean {
  return geometry.userData[`${key}Source`] === source && geometry.userData[`${key}Count`] === count;
}

/** Stamp the `{key}Source`/`{key}Count` currency pair {@link isAttributeCurrent} reads. */
function stampAttributeCurrency(
  geometry: THREE.BufferGeometry,
  key: string,
  source: unknown,
  count: number
): void {
  geometry.userData[`${key}Source`] = source;
  geometry.userData[`${key}Count`] = count;
}

/**
 * Pad `colors`' RGB triples into an already-sized RGBA `dst`, one vertex at a
 * time, with a fully opaque pad alpha (`255` for `uint8`, `65535` for `uint16` —
 * both normalize to `1.0`, the same *"1.0 for RGB data"* contract the gsplat and
 * line shaders document).
 *
 * Shared by {@link createMeshColorAttribute} (padding into a fresh buffer) and
 * `refreshMeshColors` (padding a committed prefix into an already-bound one), so
 * the `uint8`/`uint16` opaque constant cannot diverge between them.
 *
 * Only the first `vertexCount` vertices of `dst` are touched — on a reveal
 * ladder the tail past it belongs to a not-yet-revealed level and stays
 * whatever `dst` already held there (zero-filled for a fresh buffer).
 */
function padRgbIntoRgba(
  dst: Uint8Array | Uint16Array,
  colors: Uint8Array | Uint16Array,
  vertexCount: number
): void {
  const opaque = colors instanceof Uint8Array ? 255 : 65535;
  for (let v = 0; v < vertexCount; v++) {
    const src = v * 3;
    const dstOff = v * 4;
    dst[dstOff] = colors[src];
    dst[dstOff + 1] = colors[src + 1];
    dst[dstOff + 2] = colors[src + 2];
    dst[dstOff + 3] = opaque;
  }
}

/**
 * Bind `colors` as a 4-component `color` attribute, padding RGB→RGBA if needed.
 *
 * ## Why always 4 components for the 8/16-bit family
 *
 * The TSL materials run on a real `WebGPURenderer`, and three r184's WebGPU
 * backend exposes **no 3-component 8/16-bit vertex format** — its
 * `GPUVertexFormat` table lists only `unorm8x2`/`unorm8x4` and
 * `unorm16x2`/`unorm16x4`. WebGPU also requires `arrayStride` to be a multiple of
 * 4, and three uploads a tightly-packed size-3 attribute with
 * `arrayStride = itemSize · BYTES_PER_ELEMENT` — a 3-byte stride for `uint8`, 6
 * for `uint16`. Both fail `createRenderPipeline` validation, so an RGB
 * `uint8`/`uint16` mesh renders **nothing** on the WebGPU backend while looking
 * fine on WebGL.
 *
 * Padding to `unorm8x4` (4-byte stride) / `unorm16x4` (8-byte stride) fixes both
 * the format and the stride, and the memory win over widening to float32 survives:
 * 4 bytes/vertex for `uint8` RGBA against 12 for `f32×3`.
 *
 * `float32` colors are left alone — `float32x3` is a valid WebGPU format with a
 * 12-byte (4-multiple) stride, so they bind at their native 3 or 4 components.
 *
 * The pad alpha is fully opaque (`255` for `uint8`, `65535` for `uint16`), both of
 * which normalize to `1.0`. That is the same *"1.0 for RGB data"* contract the
 * gsplat and line shaders document; mesh just supplies it CPU-side for the formats
 * WebGPU refuses at size 3, and leaves it to the size-3 `w = 1.0` attribute default
 * for the ones it accepts.
 */
export function createMeshColorAttribute(
  colors: MeshColorArray,
  colorComponents: 3 | 4,
  vertexCount: number,
  capacityVertexCount: number = vertexCount
): THREE.BufferAttribute {
  if (colors instanceof Float32Array) {
    // Valid WebGPU format at either width; no pad, no widen.
    return new THREE.BufferAttribute(
      atCapacity(colors, colorComponents, vertexCount, capacityVertexCount),
      colorComponents,
      false
    );
  }

  if (colorComponents === 4) {
    return new THREE.BufferAttribute(
      atCapacity(colors, 4, vertexCount, capacityVertexCount),
      4,
      true
    );
  }

  const padded =
    colors instanceof Uint8Array
      ? new Uint8Array(capacityVertexCount * 4)
      : new Uint16Array(capacityVertexCount * 4);
  padRgbIntoRgba(padded, colors, vertexCount);
  // `normalized: true` — the GPU maps [0, 255] / [0, 65535] to [0, 1] for free.
  return new THREE.BufferAttribute(padded, 4, true);
}

/**
 * The opaque-white fallback bound when a node has no `colors` array.
 *
 * Bound rather than omitted, and that is load-bearing: an *unbound* `color`
 * attribute reads the GL default `(0, 0, 0, 1)`, so a bare
 * `add_mesh(vertices, faces)` surface would render solid **black** the moment any
 * multiplicative shade term is applied. `float32x3` is valid on both backends, so
 * this can stay size-3 and pick up `w = 1.0` from the attribute default. Mirrors
 * `create-points-node.ts`'s white fill.
 */
export function createMeshDefaultColorAttribute(
  vertexCount: number,
  capacityVertexCount: number = vertexCount
): THREE.BufferAttribute {
  const white = new Float32Array(Math.max(vertexCount, capacityVertexCount) * 3);
  white.fill(1.0);
  return new THREE.BufferAttribute(white, 3, false);
}

/**
 * The node's lifetime totals, defaulted to the committed counts.
 *
 * `Math.max` rather than a plain `??`: the totals come from the store's declared
 * attrs, and a store whose parent count is SMALLER than what its levels actually
 * hold would otherwise size the buffers below the data and truncate the surface.
 * Trusting the larger of the two makes a wrong attr cost memory, not geometry.
 */
function resolveCapacity(input: MeshGeometryConfig): { capVertices: number; capFaces: number } {
  return {
    capVertices: Math.max(input.capacityVertexCount ?? input.vertexCount, input.vertexCount),
    capFaces: Math.max(input.capacityFaceCount ?? input.faceCount, input.faceCount),
  };
}

/**
 * Choose the index-buffer dtype.
 *
 * `Uint16Array` below 65536 vertices, `Uint32Array` at or above. Not a micro-
 * optimization: WebGL1-era `OES_element_index_uint` aside, the real reason is that
 * three sets `geometry.index.array` verbatim, and a `Uint32Array` index forces the
 * 32-bit path for every draw. Meshes in this domain are usually well under 65k
 * vertices per node.
 *
 * That rationale only holds on the WebGL backends: the WebGPU backend widens a
 * narrow index attribute to Uint32Array in place at first upload, so the narrow
 * choice buys nothing there — see {@link applyMeshIndices} for the detail. It is
 * still the right allocation, because WebGL never gets that widening for free.
 *
 * The threshold is on `vertexCount`, not on the max index present: `vertexCount` is
 * fixed for the node, whereas the largest index actually drawn changes with the slice.
 * Keying off the observed maximum would let the dtype differ between epochs, which
 * would defeat the buffer reuse in {@link applyMeshIndices} — a dtype flip forces a
 * fresh `setIndex` call, orphaning the old attribute's GPU buffer, which is exactly
 * the leak {@link applyMeshIndices} exists to avoid.
 */
export function createMeshIndexAttribute(
  indices: Uint32Array,
  vertexCount: number,
  faceCount: number
): THREE.BufferAttribute {
  // Capacity is the node's FULL face count, not the visible one, so the buffer is
  // allocated once for the node's lifetime — see `applyMeshIndices` for why. On a
  // reveal ladder the caller passes the LADDER's totals here, not the committed
  // prefix's, so "once for the node's lifetime" survives the growth (#1521): both
  // the size and the dtype are then fixed from the first commit, and the dtype
  // cannot flip Uint16 → Uint32 mid-reveal on an already-drawn geometry.
  const capacity = Math.max(faceCount * 3, indices.length);
  const array = vertexCount < 65536 ? new Uint16Array(capacity) : new Uint32Array(capacity);
  array.set(indices);
  return new THREE.BufferAttribute(array, 1, false);
}

/**
 * Point `geometry`'s index at `indices`, reusing the existing buffer when it can.
 *
 * ## Why not simply `setIndex(createMeshIndexAttribute(...))` every epoch
 *
 * Because that leaks GPU memory on every slice move. Three caches attribute buffers
 * in a `WeakMap` keyed by the attribute object and only ever calls `gl.deleteBuffer`
 * from `WebGLAttributes.remove()` — which runs on geometry disposal (for whichever
 * index is current at that moment) and, notably, when the *wireframe* attribute is
 * replaced. Nothing calls it when `geometry.index` itself is replaced: the old
 * attribute becomes unreachable, its `WeakMap` entry is collected, and the GPU buffer
 * it owned is never freed. Mesh is the only geometry type that rewrites its index per
 * epoch — the other three update pooled attributes in place — so nothing in the tree
 * had hit this before.
 *
 * So the index buffer is allocated once at the node's full face-count capacity and
 * the visible prefix is drawn with `setDrawRange`. The tail past the range keeps
 * stale indices, which is safe precisely because `drawRange` bounds the draw; three
 * clamps it to `index.count`.
 *
 * Reuse also means the index attribute OBJECT is stable after the first commit, so a
 * slice move rebinds nothing. It does not participate in the `attributesRebuilt`
 * eviction contract either way: the index contributes only its PRESENCE to three's
 * geometry cache key, and the WebGPU backend re-derives `indexFormat` from
 * `index.array` on every draw, so even the dtype-widening fall-through below needs no
 * `RenderObject` eviction — its only cost is the orphaned buffer.
 *
 * A consequence worth knowing when reading counts elsewhere: `index.count` is now the
 * CAPACITY, not what is drawn. `drawRange.count` is the drawn quantity — which is why
 * `camera-framing.ts` reads that instead.
 */
export function applyMeshIndices(
  geometry: THREE.BufferGeometry,
  indices: Uint32Array,
  vertexCount: number,
  faceCount: number
): void {
  const existing = geometry.index;
  const wantUint16 = vertexCount < 65536;

  // The question this predicate has to answer is not "is this the dtype
  // `createMeshIndexAttribute` would allocate for this node", but "can this buffer
  // carry the indices we are about to write." The distinction is not academic: the
  // WebGPU backend rewrites a non-normalized Uint16Array (or Uint8Array) index
  // attribute to Uint32Array IN PLACE the first time it uploads it
  // (`WebGPUAttributeUtils.createAttribute` assigns straight into
  // `bufferAttribute.array`, mutating the very attribute this predicate is about to
  // interrogate). So after the first WebGPU upload of any sub-65536-vertex mesh,
  // `existing.array` is already a Uint32Array even though `createMeshIndexAttribute`
  // allocated it as a Uint16Array — asking "does this match what I would allocate"
  // therefore always failed on WebGPU, forcing a `setIndex` (and the leak this
  // function exists to prevent) on every epoch after the first.
  //
  // So: a Uint32Array can carry the indices of ANY node; a Uint16Array only of one
  // under 65,536 vertices. That threshold is strict `<` for two separate reasons.
  // Above 65,536 vertices indices genuinely truncate on `.set()`. AT exactly 65,536
  // nothing truncates — 65,535 is still exactly representable in 16 bits — but 65,535
  // is reserved: three's own `arrayNeedsUint32` refuses a 16-bit index array
  // containing it (citing `PRIMITIVE_RESTART_FIXED_INDEX`), because WebGL2's
  // permanently-enabled fixed-index restart silently DROPS the triangle referencing
  // vertex 65,535, and three's WebGPU path rewrites that entry to `0xffffffff` — on a
  // triangle-list not a restart but an out-of-range fetch. Corrupt on either backend.
  // Below the threshold that same rewrite is a non-issue for us: the largest possible
  // index is then 65,534, so 65,535 never appears in our data, and the stale tail past
  // `drawRange` only ever holds indices we ourselves wrote.
  const dtypeCanCarry =
    existing?.array instanceof Uint32Array ||
    (wantUint16 && existing?.array instanceof Uint16Array);

  // Reuse requires the buffer to already be at the node's FULL capacity, not merely
  // big enough for this epoch. The placeholder is born with a zero-length index
  // (`faceCount: 0`), so a first epoch that happens to be fully culled would otherwise
  // "fit" in it, leave it at zero, and force a `setIndex` on the next epoch that
  // reveals a triangle — reintroducing exactly the per-move reallocation this avoids.
  const capacity = Math.max(faceCount * 3, indices.length);
  if (existing && dtypeCanCarry && existing.array.length >= capacity) {
    // Bound the upload to the prefix actually rewritten. Without this the whole
    // capacity buffer is re-uploaded every slice move, which for a large mesh with a
    // small visible set is far more bandwidth than the old reallocating path spent —
    // i.e. it would trade the leak for a per-move bandwidth regression. All three
    // backends honour `BufferAttribute` update ranges — the classic `WebGLRenderer`,
    // and both of `WebGPURenderer`'s backends (native WebGPU and the WebGL2
    // fallback) — so bounding the range is a real bandwidth win everywhere, not just
    // on classic WebGL.
    //
    // A fully-culled epoch rewrites NOTHING, so it must not set `needsUpdate` at
    // all: on the WebGL backend an EMPTY update-range list means "upload the whole
    // attribute", so flagging an update here would re-upload the entire
    // capacity-sized buffer once per empty slice — the exact regression the ranges
    // exist to prevent. Only `drawRange` (below) has to change to draw nothing.
    if (indices.length > 0) {
      (existing.array as Uint16Array | Uint32Array).set(indices);
      existing.clearUpdateRanges();
      existing.addUpdateRange(0, indices.length);
      existing.needsUpdate = true;
    }
  } else {
    geometry.setIndex(createMeshIndexAttribute(indices, vertexCount, faceCount));
  }

  // The tail past this range keeps stale indices, which is safe precisely because the
  // draw range bounds the draw — three clamps it to `index.count`.
  geometry.setDrawRange(0, indices.length);
}

/**
 * Set a mesh geometry's bounding box and sphere from the projection's precomputed
 * bounds.
 *
 * The Mesh counterpart of `computeLineBounds` / the gsplat bounds pass, and the same
 * shape: the projection already touched every vertex, so the commit sets the box from
 * its result rather than paying a second O(N) `computeBoundingBox()` scan. No extent
 * expansion, because a mesh has no per-element footprint to expand by — lines add
 * `maxWidth` and gsplats `maxRowNorm` precisely because their elements are sprites
 * larger than their centers.
 *
 * These bounds cover only the vertices the current index references, which is TIGHTER
 * than `computeBoundingBox()` over the whole position buffer — and correct for both
 * consumers: frustum culling and the raycast broad phase are exact over what is
 * actually drawn, and camera framing stops covering culled geometry (#1252).
 *
 * `null` means nothing is drawn, which yields an empty box — and three's own
 * `Box3.getBoundingSphere()` guards that case, calling `Sphere.makeEmpty()` for a
 * radius of -1 that the frustum test correctly rejects. So there is ONE code path
 * here, as in `computeLineBounds`. (The NaN hazard documented on
 * `createEmptyMeshNode` is a different function: `BufferGeometry.computeBoundingSphere()`
 * over a ZERO-vertex position buffer, which is why that placeholder carries one vertex.)
 */
export function computeMeshBounds(
  geometry: THREE.BufferGeometry,
  bounds: MeshProjectionBounds | null
): void {
  const box = bounds
    ? new THREE.Box3(
        new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]),
        new THREE.Vector3(bounds.max[0], bounds.max[1], bounds.max[2])
      )
    : new THREE.Box3();
  geometry.boundingBox = box;
  geometry.boundingSphere = new THREE.Sphere();
  box.getBoundingSphere(geometry.boundingSphere);
}

/**
 * Build a fresh geometry for a mesh node.
 *
 * The attribute SET is decided here, once, and never changes afterwards —
 * {@link updateMeshGeometry} only replaces contents. That is deliberate: changing a
 * drawn geometry's attribute set is silently broken on the WebGPU backend, where
 * attribute identity is baked into the render pipeline.
 *
 * `normal` and `aScalar` join the set here — never later — which is why the
 * placeholder factory passes 1-vertex stubs for them whenever the node's METADATA
 * says the arrays exist. Their presence is a per-node constant (`has_normals` /
 * `has_scalars`), so the set stays fixed for the geometry's whole life even though
 * whether the shader *reads* `normal` varies per `displayDims` epoch.
 */
export function createMeshGeometry(input: MeshGeometryConfig): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const { capVertices, capFaces } = resolveCapacity(input);
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      atCapacity(input.position, 3, input.vertexCount, capVertices),
      3,
      false
    )
  );
  geometry.setAttribute(
    'color',
    input.colors
      ? createMeshColorAttribute(
          input.colors,
          input.colorComponents ?? 3,
          input.vertexCount,
          capVertices
        )
      : createMeshDefaultColorAttribute(input.vertexCount, capVertices)
  );
  // Stamped so `updateMeshGeometry` can tell authored colors from the placeholder
  // WITHOUT comparing counts — see its color guard for why counts alone fail.
  // `meshColorsSource`/`meshColorsCount` are the CURRENCY markers `refreshMeshColors`
  // reads: stamping them here means the very first `updateMeshGeometry` call for
  // this node already sees a matching source and does nothing, instead of treating
  // the create-time write as stale and re-writing it immediately.
  if (input.colors) {
    geometry.userData.meshColorsInstalled = true;
    stampAttributeCurrency(geometry, 'meshColors', input.colors, input.vertexCount);
  }
  if (input.normals) {
    geometry.setAttribute(
      'normal',
      new THREE.BufferAttribute(
        atCapacity(input.normals, 3, input.vertexCount, capVertices),
        3,
        false
      )
    );
  }
  if (input.uvs) {
    // `uv` joins the set HERE, like `normal` and `aScalar`, because its existence is
    // a per-node constant (`has_uvs`) — and unlike them it is bound whenever the
    // node HAS uvs, not whenever the shader reads them, which is the same rule for
    // the same reason: WebGPU bakes the attribute set into the pipeline at first
    // draw, so an attribute that appears later renders the node black.
    geometry.setAttribute(
      'uv',
      new THREE.BufferAttribute(atCapacity(input.uvs, 2, input.vertexCount, capVertices), 2, false)
    );
    geometry.userData.hasUVs = true;
  }
  if (input.scalars) {
    geometry.setAttribute(
      'aScalar',
      new THREE.BufferAttribute(
        atCapacity(input.scalars, 1, input.vertexCount, capVertices),
        1,
        false
      )
    );
    // The scalar-presence stamp every geometry type uses, and the signal
    // `supportsScalarColormap('mesh', …)` fails closed on. Deliberately the same
    // mechanism as points/lines even though mesh binds a real attribute it could
    // probe for: one rule means one way to be wrong.
    geometry.userData.hasScalars = true;
  }
  applyMeshIndices(geometry, input.indices, capVertices, capFaces);
  if (input.bounds !== undefined) {
    computeMeshBounds(geometry, input.bounds);
  } else {
    // No projected bounds supplied — the placeholder node. Scan the (1-vertex) buffer.
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }
  return geometry;
}

/**
 * Bring one already-bound `float32` vertex attribute up to date, in place where
 * possible.
 *
 * Shared by `normal`, `aScalar` and `uv`, which have identical lifecycles: all are
 * per-node-constant in EXISTENCE (decided at creation from the metadata) and
 * uploaded-once in CONTENT PER LEVEL — a slice move rebuilds only the index, but
 * a reveal ladder's later levels genuinely grow the committed prefix.
 *
 * Four cases, and only the last touches the GPU (the rebind below is a fifth):
 * - the attribute is not bound → do nothing. The node has no such array; adding one
 *   now would grow a live geometry's attribute set.
 * - no data this epoch → do nothing. Keeps whatever is bound (the 1-vertex
 *   placeholder stub, or the last real upload).
 * - **already current** → do nothing, which is the STEADY STATE and the whole
 *   reason this case is called out. Currency is tracked by the SOURCE array's own
 *   identity plus the committed count (`isAttributeCurrent`), not by comparing it
 *   against the bound buffer: once the buffer is bound at the node's CAPACITY
 *   (#1521), a ladder's later levels never see `existing.array === data` again —
 *   the buffer is a copy `atCapacity` made once at the rebind, not the caller's
 *   array. The whole-node loader serves one cached `LoadedMeshData` for the
 *   node's lifetime and the commit passes `data.normals` / `data.scalars` on
 *   every epoch, so after the first write at a given level the SOURCE is always
 *   equal too. Flagging `needsUpdate` there would re-upload the entire normal
 *   (`V·12` bytes) and scalar (`V·4` bytes) buffers on EVERY slice move — real
 *   bandwidth during a scrub at the 2^27-vertex cap, and the exact bug this
 *   mirrors for `color` (#1522): a reveal ladder's later levels silently never
 *   reaching the buffer because currency looked permanently stale. Nothing
 *   mutates these arrays in place (normals are never re-projected, §3.4; scalars
 *   are view-independent), so current means the GPU copy already matches.
 * - not current, buffer already fits → copy the prefix + flag. Either a genuine
 *   re-fetch (dispose/reload handing over fresh buffers at an unchanged count) or
 *   a new ladder level (a longer prefix at the same capacity) — both need the
 *   upload, bounded to the prefix actually written.
 *
 * Buffer too small → rebind, the expected first commit (placeholder stub → real
 * buffer), reported so the caller evicts three's cached WebGPU `RenderObject`.
 *
 * @returns `true` when a `setAttribute` rebind happened.
 */
function replaceVertexAttribute(
  geometry: THREE.BufferGeometry,
  name: 'normal' | 'aScalar' | 'uv',
  data: Float32Array | null | undefined,
  itemSize: 1 | 2 | 3,
  vertexCount: number,
  capacityVertexCount: number = vertexCount
): boolean {
  const existing = geometry.getAttribute(name) as THREE.BufferAttribute | undefined;
  if (!existing || !data) return false;
  // A DISTINCT currency key per attribute. Sharing one would make an epoch that
  // uploaded normals look like it had uploaded uvs too, so the uv write would be
  // skipped as already-current and the mesh would sample the 1-vertex stub.
  const key = name === 'normal' ? 'meshNormal' : name === 'uv' ? 'meshUv' : 'meshAScalar';
  // A ladder's buffer is CAPACITY-sized while `data` is the committed prefix, so
  // the in-place copy is the steady state there too — the length test compares the
  // bound buffer against the capacity, and the copy writes only the prefix.
  if (existing.count === capacityVertexCount && existing.array.length >= data.length) {
    // `existing.array !== data` alone still catches the zero-copy unladdered case
    // (the bound buffer IS the caller's array, so nothing needs a stamp to know
    // it); `isAttributeCurrent` catches the capacity-bound case identity can no
    // longer see. Either being false is enough to skip the write.
    if (existing.array !== data && !isAttributeCurrent(geometry, key, data, vertexCount)) {
      (existing.array as Float32Array).set(data);
      // Bound the upload to the prefix actually written — mirrors
      // `applyMeshIndices`. An empty update-range list means "upload the
      // whole capacity-sized buffer" on the classic WebGL path, so never
      // flag an update without adding a range.
      existing.clearUpdateRanges();
      existing.addUpdateRange(0, data.length);
      existing.needsUpdate = true;
    }
    stampAttributeCurrency(geometry, key, data, vertexCount);
    return false;
  }
  geometry.setAttribute(
    name,
    new THREE.BufferAttribute(
      atCapacity(data, itemSize, vertexCount, capacityVertexCount),
      itemSize,
      false
    )
  );
  stampAttributeCurrency(geometry, key, data, vertexCount);
  return true;
}

/**
 * Write `colors`' committed prefix into an already-bound, capacity-sized `color`
 * attribute, returning `false` only when the bound attribute's FORMAT cannot hold
 * `colors` — the caller then falls back to `createMeshColorAttribute` and rebinds.
 *
 * ## Why a format check, not a count check
 *
 * Once `color` is bound at the node's capacity, every level of a reveal ladder
 * binds at the SAME count (`capVertices`), so count can no longer tell "already
 * installed" apart from "installed, but still holding an earlier level's prefix"
 * (#1522). The bound attribute's LAYOUT can, and it has to mirror
 * `createMeshColorAttribute` exactly: `float32` colors bind at their native
 * `colorComponents`, `uint8`/`uint16` always at itemSize 4 (RGB padded, opaque
 * alpha) — see that function for why.
 *
 * ## Why currency is tracked by the SOURCE array's identity
 *
 * Not the bound buffer's, because the RGB→RGBA pad path allocates a fresh
 * `padded` array every time it runs — the bound array is therefore never
 * identity-equal to `colors`, even in the unladdered steady state. Tracking
 * currency there would re-pad and re-upload the whole buffer on every slice
 * move, exactly the bandwidth regression `replaceVertexAttribute` documents for
 * `normal`/`aScalar`. `isAttributeCurrent`/`stampAttributeCurrency` read and
 * write `geometry.userData.meshColorsSource` / `meshColorsCount` everywhere
 * AUTHORED `color` is written (here, in `createMeshGeometry`, and on the rebind
 * path below) so the steady state is reached from the first commit. The
 * null-colors steady state (the guard in `updateMeshGeometry`) never calls this
 * function at all — it assumes the default-white attribute is already
 * installed, which the null-colors rebind now guarantees: it clears the
 * authored-colour currency stamps, so a later authored commit can never be
 * mistaken for already-current against a source array that installed the
 * white default, not a colour.
 *
 * @returns `true` when the existing attribute now holds `colors`' prefix
 * (written in place or already current); `false` when the caller must rebind.
 */
function refreshMeshColors(
  geometry: THREE.BufferGeometry,
  existing: THREE.BufferAttribute,
  colors: MeshColorArray,
  colorComponents: 3 | 4,
  vertexCount: number
): boolean {
  const padsRgb = !(colors instanceof Float32Array) && colorComponents === 3;
  const expectedItemSize = padsRgb ? 4 : colorComponents;
  if (
    existing.itemSize !== expectedItemSize ||
    // Unlike the index dtype check `applyMeshIndices` has to special-case, this
    // "does this match what I would allocate" comparison is safe: every narrow
    // colour path binds `normalized: true`, which WebGPU's in-place substitution
    // never touches, and the float path is always `Float32Array`.
    (existing.array as object).constructor !== colors.constructor ||
    existing.array.length < vertexCount * existing.itemSize
  ) {
    return false;
  }

  // Already current — the steady state for an unladdered mesh from its second
  // commit on, and for a ladder level committed twice in a row (a slice move
  // between two reveals at the same level).
  if (isAttributeCurrent(geometry, 'meshColors', colors, vertexCount)) {
    return true;
  }

  if (padsRgb) {
    // The same pad loop `createMeshColorAttribute` runs, writing into the
    // already-bound buffer instead of a fresh one. Only the committed prefix —
    // the tail past `vertexCount` is unreachable until a later level reveals
    // it, and `drawRange`/the index bound the draw until then.
    padRgbIntoRgba(
      existing.array as Uint8Array | Uint16Array,
      colors as Uint8Array | Uint16Array,
      vertexCount
    );
  } else if (existing.array !== colors) {
    // Skipped when the bound array IS the source — the zero-copy wrap an
    // unladdered non-padded mesh binds directly.
    (existing.array as Float32Array | Uint8Array | Uint16Array).set(colors as ArrayLike<number>);
  }
  // Bound the upload to the prefix actually written — mirrors `applyMeshIndices`.
  // An empty update-range list means "upload the whole capacity-sized buffer" on
  // the classic WebGL path, so never flag an update without adding a range.
  existing.clearUpdateRanges();
  existing.addUpdateRange(0, vertexCount * existing.itemSize);
  existing.needsUpdate = true;
  stampAttributeCurrency(geometry, 'meshColors', colors, vertexCount);
  return true;
}

/**
 * Update an existing mesh geometry in place for a new projection epoch.
 *
 * On a pure slice move only the index buffer changes, which is the whole point of
 * the no-compaction design: the vertex attribute buffers stay uploaded and
 * untouched, and `drawElements` simply stops referencing the culled vertices.
 *
 * `position` is re-uploaded on any of three conditions: `input.positionChanged`
 * is set (a `displayDims` change, where the display-space projection is
 * re-extracted); the geometry's last upload used a different `displayDims` key
 * than this commit (`keyChanged`); or it covered a different vertex count than
 * this commit (`countChanged` — the reveal-ladder case, where a later level's
 * committed prefix grows). **Bounds are recomputed whenever position is replaced**, because
 * re-uploading the buffer does *not* invalidate Three.js's cached
 * `boundingBox`/`boundingSphere`, which `frustumCulled` and the raycaster broad
 * phase both consult — and the display-space AABB genuinely changes under an axis
 * permutation. Skipping that recompute makes a permuted mesh vanish from the
 * frustum test while still being "loaded", which is a confusing failure to debug.
 *
 * The `color` attribute is **installed on the first commit, then kept current with
 * an in-place write on every commit that carries authored colors**: the node is
 * created with a 1-vertex placeholder color (see `createEmptyMeshNode`), so the
 * authored colors have to be bound here or they would never reach the shader —
 * and on a reveal ladder every later level's prefix has to reach the buffer too,
 * or the vertices it reveals keep the zero-filled slot they were born with
 * (#1522). See the guard below for the install-vs-refresh split.
 *
 * @returns `attributesRebuilt` — `true` when a VERTEX attribute was rebound via
 * `setAttribute`: the first-commit position grow-rebind, the first-commit color
 * install, a color rebind forced by a format mismatch (`refreshMeshColors`
 * returning `false`), or a `normal`/`aScalar` rebind (`replaceVertexAttribute`'s
 * buffer-too-small case) — `false` otherwise, including a color REFRESH, which
 * writes into the already-bound buffer rather than rebinding it. The commit uses this to evict three's stale WebGPU
 * `RenderObject` cache after a vertex-attribute rebind (its cached `vertexBuffers`
 * keeps pointing at the OLD GPU buffer/pipeline otherwise) — the same contract the
 * points/lines/gsplats commits follow via `invalidateRenderObjectFor`. A pure slice
 * move rebinds nothing at all — the index is written into its existing buffer
 * ({@link applyMeshIndices}) — so it returns `false` and the commit skips the
 * eviction. The geometry is mutated in place, so no caller
 * needs it returned.
 */
export function updateMeshGeometry(
  geometry: THREE.BufferGeometry,
  input: MeshGeometryConfig
): boolean {
  let attributesRebuilt = false;
  let positionRebound = false;
  const { capVertices, capFaces } = resolveCapacity(input);
  const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
  // `input.positionChanged`, NOT array identity — the projection reuses one buffer, so
  // identity is stable across a `displayDims` change and would suppress the re-upload.
  //
  // The upload ALSO fires when the geometry has not yet uploaded this `displayDims` key
  // (`keyChanged`): a commit can be superseded after its projection advanced the
  // loader's `displayDimsKey` but before it uploaded, and the next same-key commit then
  // reads `positionChanged === false` even though the geometry still holds the old
  // frame. A projection-time flag cannot see that gap; the geometry-stamped key can.
  // The key is stamped only where the upload happens, so an aborted commit leaves it
  // untouched and the next real commit refreshes.
  const storedKey = geometry.userData.meshUploadedKey as string | undefined;
  const keyChanged = input.positionKey !== undefined && input.positionKey !== storedKey;
  // The upload ALSO fires when the geometry's last upload covered a vertex count
  // that differs from this commit's (`countChanged`) — the reveal-ladder counterpart of
  // `keyChanged`. A ladder level can be superseded (its commit aborted) after
  // `projectMeshTo3D` already extracted and stamped the loader-owned scratch's
  // `displayDimsKey`, so the NEXT sweep re-projects the same memoized level with
  // `positionChanged === false` and an unchanged `positionKey` — neither flag sees
  // the gap, because both are keyed on `displayDims`, not on how many vertices are
  // committed. Without this, the index/drawRange advance to the new level while
  // `position` keeps the previous level's zero-filled tail, so the newly revealed
  // triangles collapse to the display-space origin (#1522). Like the key, stamped
  // only where the upload actually runs, so an aborted commit leaves it lagging.
  const storedVertexCount = geometry.userData.meshUploadedVertexCount as number | undefined;
  const countChanged = storedVertexCount !== input.vertexCount;
  if (positionAttr && (input.positionChanged || keyChanged || countChanged)) {
    // `>=`, not `===`: on a ladder the bound buffer is sized to the node's TOTAL
    // while `input.position` is the committed prefix, so every level after the
    // first copies INTO the same buffer instead of rebinding (#1521). For an
    // unladdered mesh the two are equal and this is the original test.
    if (positionAttr.array.length >= input.position.length && positionAttr.count > 1) {
      // Already the same buffer when the geometry bound the projection scratch
      // directly (the steady state after the first commit), in which case the copy
      // is a self-copy and only the upload flag matters.
      if (positionAttr.array !== input.position) {
        (positionAttr.array as Float32Array).set(input.position);
      }
      // Bound the upload to the prefix actually written — mirrors
      // `applyMeshIndices`. An empty update-range list means "upload the
      // whole capacity-sized buffer" on the classic WebGL path, so never
      // flag an update without adding a range.
      positionAttr.clearUpdateRanges();
      positionAttr.addUpdateRange(0, input.position.length);
      positionAttr.needsUpdate = true;
    } else {
      // A length change on a LIVE node means the vertex count changed, which a
      // whole-node loader never does — warn and rebind rather than silently
      // truncate. Two expected cases stay quiet: the 1-vertex placeholder growing
      // to the real buffer (the first commit of every mesh), and a reveal ladder,
      // whose every level genuinely adds vertices (`vertexCountGrows`).
      if (positionAttr.count !== 1 && input.vertexCountGrows !== true) {
        log.warning(
          Modules.SCENE_LOADER,
          `Mesh position length changed (${positionAttr.array.length} -> ${input.position.length}); rebinding`
        );
      }
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(
          atCapacity(input.position, 3, input.vertexCount, capVertices),
          3,
          false
        )
      );
      // A new attribute object → three's cached WebGPU RenderObject is now stale.
      attributesRebuilt = true;
    }
    positionRebound = true;
    // Stamp the key the geometry now holds, ONLY where the upload actually ran — so an
    // aborted commit (which never reaches here) leaves the last-uploaded key lagging and
    // the next real commit's `keyChanged` still fires. Only when a key was provided, so
    // the placeholder factory / unit-test callers stay byte-identical to before.
    if (input.positionKey !== undefined) geometry.userData.meshUploadedKey = input.positionKey;
    // Same idea for the count: stamped here, not before, so a superseded commit that
    // never reaches this line leaves `countChanged` true for the next real one.
    geometry.userData.meshUploadedVertexCount = input.vertexCount;
  }

  // Install the `color` attribute on the first commit, then REFRESH it in place on
  // every later commit that carries authored colors. The node is born with the
  // 1-vertex placeholder color from `createEmptyMeshNode` (`colors: null,
  // vertexCount: 1`), so without the install the authored per-vertex colors would
  // never bind and an indexed draw would fetch `color` out of bounds (black under
  // WebGL2 robust access; a pipeline-validation failure on WebGPU). Without the
  // refresh, a reveal ladder's later levels would keep whatever the buffer held at
  // install time: `color` is bound at the node's full `capVertices` from the first
  // commit (#1521), so `count` is capVertices on every level and can no longer
  // distinguish "installed" from "installed, but still holding an earlier level's
  // prefix" — the vertices a later level reveals would keep the zero-filled slot
  // they were born with, which for `uint8`/`uint16` RGB or ANY RGBA mesh zeroes
  // `vAlpha` (the mesh's entire coverage term) and makes the revealed surface
  // invisible (#1522).
  //
  // Install is still keyed off `vertexCount` PLUS an explicit marker: the
  // placeholder is 1-vertex (`colorAttr.count === 1`) and a real drawable mesh has
  // N vertices, so `count !== capVertices` catches the first commit of every mesh
  // with more than one vertex — but a 1-VERTEX mesh with authored colors matches
  // the placeholder's count and would keep the placeholder white forever on the
  // count test alone. `userData.meshColorsInstalled` (stamped here, and by
  // `createMeshGeometry` when authored colors are bound) closes that hole. Note
  // `projectMeshTo3D` reuses one loader `position` buffer, so color must NOT be
  // tied to position identity — that would re-upload color whenever position does.
  //
  // Format stability across the guard: the no-colors default-white path stays
  // `float32x3` at count N — a format-preserving rebind exactly like `position`'s
  // first-commit grow, so it never changes attribute format. Only authored,
  // non-float32-RGB colors change format ONCE on first install (placeholder
  // `float32x3` → e.g. `unorm8x4`) and never again, so there is no per-rebuild dtype
  // flip — the WebGPU attribute-identity hazard `createMeshGeometry` documents for the
  // attribute SET. (The index's own dtype stability is a separate, cheaper concern:
  // see `createMeshIndexAttribute`, where a flip costs a buffer, not a pipeline.)
  // `refreshMeshColors` re-validates that
  // same format on every refresh and reports `false` (forcing a rebind) if it ever
  // disagrees, rather than assuming it silently still holds.
  //
  // Only the install rebinds a vertex attribute, so only it sets
  // `attributesRebuilt` to drive the commit's WebGPU RenderObject eviction (see
  // the @returns note) — a refresh writes into the SAME buffer.
  const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute | undefined;
  const authoredColorsPending =
    input.colors !== null && geometry.userData.meshColorsInstalled !== true;
  if (colorAttr && colorAttr.count === capVertices && !authoredColorsPending) {
    // The steady state. A `null`-colors epoch needs nothing: the default-white
    // fill already spans the whole capacity (`createMeshDefaultColorAttribute`).
    // An authored epoch refreshes its committed prefix in place, falling back to
    // a rebind only if the bound attribute's format cannot hold it.
    if (
      input.colors !== null &&
      !refreshMeshColors(
        geometry,
        colorAttr,
        input.colors,
        input.colorComponents ?? 3,
        input.vertexCount
      )
    ) {
      geometry.setAttribute(
        'color',
        createMeshColorAttribute(
          input.colors,
          input.colorComponents ?? 3,
          input.vertexCount,
          capVertices
        )
      );
      geometry.userData.meshColorsInstalled = true;
      stampAttributeCurrency(geometry, 'meshColors', input.colors, input.vertexCount);
      attributesRebuilt = true;
    }
  } else {
    geometry.setAttribute(
      'color',
      input.colors
        ? createMeshColorAttribute(
            input.colors,
            input.colorComponents ?? 3,
            input.vertexCount,
            capVertices
          )
        : createMeshDefaultColorAttribute(input.vertexCount, capVertices)
    );
    if (input.colors) {
      geometry.userData.meshColorsInstalled = true;
      stampAttributeCurrency(geometry, 'meshColors', input.colors, input.vertexCount);
    } else {
      // Belt-and-braces for a state the loaders currently forbid — colour
      // presence is a per-node constant and `concatenateMeshData` refuses a
      // ladder whose levels disagree, so this rebind is never reached with a
      // stale AUTHORED stamp today. Clearing it anyway means a hypothetical
      // later authored commit can never be mistaken for "already current"
      // against a source array that installed the white default's rebind, not
      // a colour.
      geometry.userData.meshColorsInstalled = false;
      delete geometry.userData.meshColorsSource;
      delete geometry.userData.meshColorsCount;
    }
    attributesRebuilt = true;
  }

  // `normal` and `aScalar` follow the same install-once rule as `color`, with one
  // difference that matters: they are only ever REPLACED, never added or removed.
  // Whether each exists was fixed at creation from the node's metadata
  // (`has_normals` / `has_scalars`), so a commit that finds the attribute absent
  // must leave it absent — adding one here would grow a live geometry's attribute
  // set, which the WebGPU backend bakes into its cached vertex layout at first draw
  // and never rebuilds. Conversely a frame change that makes stored normals
  // meaningless must NOT unbind them: the shader variant stops reading the
  // attribute instead (§3.4, and `MeshGeometryConfig.normals`).
  if (
    replaceVertexAttribute(geometry, 'normal', input.normals, 3, input.vertexCount, capVertices)
  ) {
    attributesRebuilt = true;
  }
  if (
    replaceVertexAttribute(geometry, 'aScalar', input.scalars, 1, input.vertexCount, capVertices)
  ) {
    attributesRebuilt = true;
  }
  if (replaceVertexAttribute(geometry, 'uv', input.uvs, 2, input.vertexCount, capVertices)) {
    attributesRebuilt = true;
  }

  applyMeshIndices(geometry, input.indices, capVertices, capFaces);

  // Bounds track the VISIBLE set, so they refresh on EVERY epoch — a slice move
  // changes which vertices are indexed even when `position` is untouched. Cheap: the
  // projection already computed the AABB, so this only builds the box and sphere.
  if (input.bounds !== undefined) {
    computeMeshBounds(geometry, input.bounds);
  } else if (positionRebound) {
    // Legacy path for a caller that supplies no projected bounds.
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  return attributesRebuilt;
}
