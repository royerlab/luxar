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
  vertexCount: number
): THREE.BufferAttribute {
  if (colors instanceof Float32Array) {
    // Valid WebGPU format at either width; no pad, no widen.
    return new THREE.BufferAttribute(colors, colorComponents, false);
  }

  if (colorComponents === 4) {
    return new THREE.BufferAttribute(colors, 4, true);
  }

  const opaque = colors instanceof Uint8Array ? 255 : 65535;
  const padded =
    colors instanceof Uint8Array
      ? new Uint8Array(vertexCount * 4)
      : new Uint16Array(vertexCount * 4);
  for (let v = 0; v < vertexCount; v++) {
    const src = v * 3;
    const dst = v * 4;
    padded[dst] = colors[src];
    padded[dst + 1] = colors[src + 1];
    padded[dst + 2] = colors[src + 2];
    padded[dst + 3] = opaque;
  }
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
export function createMeshDefaultColorAttribute(vertexCount: number): THREE.BufferAttribute {
  const white = new Float32Array(vertexCount * 3);
  white.fill(1.0);
  return new THREE.BufferAttribute(white, 3, false);
}

/**
 * Choose the index-buffer dtype.
 *
 * `Uint16Array` below 65536 vertices, `Uint32Array` above. Not a micro-
 * optimization: WebGL1-era `OES_element_index_uint` aside, the real reason is that
 * three sets `geometry.index.array` verbatim, and a `Uint32Array` index forces the
 * 32-bit path for every draw. Meshes in this domain are usually well under 65k
 * vertices per node.
 *
 * The threshold is on `vertexCount`, not on the max index present: `vertexCount` is
 * fixed for the node, whereas the largest index actually drawn changes with the slice.
 * Keying off the observed maximum would let the dtype differ between epochs, which
 * would defeat the buffer reuse in {@link applyMeshIndices} — and re-binding a drawn
 * geometry's index with a different dtype is exactly the kind of attribute-identity
 * change the WebGPU backend does not tolerate.
 */
export function createMeshIndexAttribute(
  indices: Uint32Array,
  vertexCount: number,
  faceCount: number
): THREE.BufferAttribute {
  // Capacity is the node's FULL face count, not the visible one, so the buffer is
  // allocated once for the node's lifetime — see `applyMeshIndices` for why.
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
 * Reuse also means the index attribute OBJECT is stable after the first commit, so —
 * unlike a per-epoch `setIndex` — a slice move leaves three's cached `RenderObject`
 * untouched. That is why this does not participate in the `attributesRebuilt`
 * eviction contract: it never rebinds anything after the build.
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
  const dtypeMatches = wantUint16
    ? existing?.array instanceof Uint16Array
    : existing?.array instanceof Uint32Array;

  // Reuse requires the buffer to already be at the node's FULL capacity, not merely
  // big enough for this epoch. The placeholder is born with a zero-length index
  // (`faceCount: 0`), so a first epoch that happens to be fully culled would otherwise
  // "fit" in it, leave it at zero, and force a `setIndex` on the next epoch that
  // reveals a triangle — reintroducing exactly the per-move reallocation this avoids.
  const capacity = Math.max(faceCount * 3, indices.length);
  if (existing && dtypeMatches && existing.array.length >= capacity) {
    (existing.array as Uint16Array | Uint32Array).set(indices);
    // Bound the upload to the prefix actually rewritten. Without this the whole
    // capacity buffer is re-uploaded every slice move, which for a large mesh with a
    // small visible set is far more bandwidth than the old reallocating path spent —
    // i.e. it would trade the leak for a per-move bandwidth regression. The classic
    // WebGL backend honours update ranges; the WebGPU backends ignore them and
    // re-upload in full, so this is an improvement there and neutral here.
    existing.clearUpdateRanges();
    if (indices.length > 0) existing.addUpdateRange(0, indices.length);
    existing.needsUpdate = true;
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
 * `normal` and `aScalar` are **not** bound in this phase. Nothing reads them yet —
 * the shading model and the scalar colormap path arrive with the material pair — and
 * binding an attribute with no consumer would mean deciding its WebGPU-safe dtype
 * before there is a shader to validate the choice against. They join the set at
 * creation time in that later phase, which is a new build rather than a runtime
 * mutation of a live geometry.
 */
export function createMeshGeometry(input: MeshGeometryConfig): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(input.position, 3, false));
  geometry.setAttribute(
    'color',
    input.colors
      ? createMeshColorAttribute(input.colors, input.colorComponents ?? 3, input.vertexCount)
      : createMeshDefaultColorAttribute(input.vertexCount)
  );
  applyMeshIndices(geometry, input.indices, input.vertexCount, input.faceCount);
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
 * Update an existing mesh geometry in place for a new projection epoch.
 *
 * On a pure slice move only the index buffer changes, which is the whole point of
 * the no-compaction design: the vertex attribute buffers stay uploaded and
 * untouched, and `drawElements` simply stops referencing the culled vertices.
 *
 * `position` is re-uploaded only when it actually differs in identity — that
 * happens on a `displayDims` change, where the display-space projection is
 * re-extracted. **Bounds are recomputed whenever position is replaced**, because
 * re-uploading the buffer does *not* invalidate Three.js's cached
 * `boundingBox`/`boundingSphere`, which `frustumCulled` and the raycaster broad
 * phase both consult — and the display-space AABB genuinely changes under an axis
 * permutation. Skipping that recompute makes a permuted mesh vanish from the
 * frustum test while still being "loaded", which is a confusing failure to debug.
 *
 * The `color` attribute is **installed on the first commit and then left alone**:
 * the node is created with a 1-vertex placeholder color (see
 * `createEmptyMeshNode`), so the authored colors have to be bound here or they
 * would never reach the shader. It is a first-commit-only install, not a
 * per-slice-move update — see the guard below.
 *
 * @returns `attributesRebuilt` — `true` when a VERTEX attribute was rebound via
 * `setAttribute` (the first-commit position grow-rebind or the first-commit color
 * install), `false` otherwise. The commit uses this to evict three's stale WebGPU
 * `RenderObject` cache after a vertex-attribute rebind (its cached `vertexBuffers`
 * keeps pointing at the OLD GPU buffer/pipeline otherwise) — the same contract the
 * points/lines/gsplats commits follow via `invalidateRenderObjectFor`. A pure slice
 * move rebinds no vertex attribute (only `setIndex` runs), so it returns `false` and
 * the commit skips the eviction. The geometry is mutated in place, so no caller
 * needs it returned.
 */
export function updateMeshGeometry(
  geometry: THREE.BufferGeometry,
  input: MeshGeometryConfig
): boolean {
  let attributesRebuilt = false;
  let positionRebound = false;
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
  if (positionAttr && (input.positionChanged || keyChanged)) {
    if (positionAttr.array.length === input.position.length) {
      // Already the same buffer when the geometry bound the projection scratch
      // directly (the steady state after the first commit), in which case the copy
      // is a self-copy and only the upload flag matters.
      if (positionAttr.array !== input.position) {
        (positionAttr.array as Float32Array).set(input.position);
      }
      positionAttr.needsUpdate = true;
    } else {
      // A length change on a LIVE node means the vertex count changed, which the
      // whole-node loader never does — warn and rebind rather than silently
      // truncate. The 1-vertex placeholder growing to the real buffer is the
      // expected first commit of every mesh, not an anomaly, so it stays quiet.
      if (positionAttr.count !== 1) {
        log.warning(
          Modules.SCENE_LOADER,
          `Mesh position length changed (${positionAttr.array.length} -> ${input.position.length}); rebinding`
        );
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(input.position, 3, false));
      // A new attribute object → three's cached WebGPU RenderObject is now stale.
      attributesRebuilt = true;
    }
    positionRebound = true;
    // Stamp the key the geometry now holds, ONLY where the upload actually ran — so an
    // aborted commit (which never reaches here) leaves the last-uploaded key lagging and
    // the next real commit's `keyChanged` still fires. Only when a key was provided, so
    // the placeholder factory / unit-test callers stay byte-identical to before.
    if (input.positionKey !== undefined) geometry.userData.meshUploadedKey = input.positionKey;
  }

  // Install the `color` attribute exactly once, on the first commit. The node is
  // born with the 1-vertex placeholder color from `createEmptyMeshNode`
  // (`colors: null, vertexCount: 1`), so without this the authored per-vertex
  // colors would never bind and an indexed draw would fetch `color` out of bounds
  // (black under WebGL2 robust access; a pipeline-validation failure on WebGPU).
  //
  // Keyed off `vertexCount` for the SAME reason `createMeshIndexAttribute` keys the
  // index dtype off it: the placeholder is 1-vertex (`colorAttr.count === 1`) and a
  // real drawable mesh has N vertices, so `count !== vertexCount` is true exactly
  // on the first commit and false on every subsequent slice move. That keeps the
  // color buffer uploaded-once (the no-compaction doctrine — only the index rebuilds
  // on a slice move) instead of re-uploading it on every scrub. Note `projectMeshTo3D`
  // reallocates `position` on every call, so color must NOT be tied to position
  // identity — that would re-upload color on every slice move.
  //
  // Format stability across the guard: the no-colors default-white path stays
  // `float32x3` at count N — a format-preserving rebind exactly like `position`'s
  // first-commit grow, so it never changes attribute format. Only authored,
  // non-float32-RGB colors change format ONCE on first install (placeholder
  // `float32x3` → e.g. `unorm8x4`) and never again, so there is no per-rebuild dtype
  // flip — the WebGPU attribute-identity hazard the surrounding code and
  // `createMeshIndexAttribute` guard against.
  //
  // This install rebinds a vertex attribute, so it sets `attributesRebuilt` to
  // drive the commit's WebGPU RenderObject eviction (see the @returns note).
  const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute | undefined;
  if (!colorAttr || colorAttr.count !== input.vertexCount) {
    geometry.setAttribute(
      'color',
      input.colors
        ? createMeshColorAttribute(input.colors, input.colorComponents ?? 3, input.vertexCount)
        : createMeshDefaultColorAttribute(input.vertexCount)
    );
    attributesRebuilt = true;
  }

  applyMeshIndices(geometry, input.indices, input.vertexCount, input.faceCount);

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
