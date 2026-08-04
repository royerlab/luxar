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
import type { MeshColorArray } from '../types/mesh';

/** What {@link buildMeshGeometry} needs to lay out the buffers. */
export interface MeshGeometryInput {
  /** Display-space positions (`vertexCount * 3`), from `projectMesh`. */
  position: Float32Array;
  /** Index buffer of visible triangles (`visibleFaceCount * 3`). */
  indices: Uint32Array;
  /** Per-vertex colors in their native dtype, or `null` for the white default. */
  colors: MeshColorArray | null;
  /** Channels per color entry when `colors` is present. */
  colorComponents?: 3 | 4;
  /** Vertices in the buffers (NOT the visible count — nothing is compacted). */
  vertexCount: number;
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
export function buildColorAttribute(
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
export function buildDefaultColorAttribute(vertexCount: number): THREE.BufferAttribute {
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
 * The threshold is on `vertexCount`, not on the max index present: the index buffer
 * is rebuilt on every slice change while `vertexCount` is fixed for the node, so
 * keying off the observed maximum would let the dtype flip between rebuilds — and
 * changing a drawn geometry's index dtype is exactly the kind of attribute-identity
 * change the WebGPU backend does not tolerate.
 */
export function buildIndexAttribute(
  indices: Uint32Array,
  vertexCount: number
): THREE.BufferAttribute {
  if (vertexCount < 65536) {
    return new THREE.BufferAttribute(new Uint16Array(indices), 1, false);
  }
  return new THREE.BufferAttribute(indices, 1, false);
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
export function buildMeshGeometry(input: MeshGeometryInput): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(input.position, 3, false));
  geometry.setAttribute(
    'color',
    input.colors
      ? buildColorAttribute(input.colors, input.colorComponents ?? 3, input.vertexCount)
      : buildDefaultColorAttribute(input.vertexCount)
  );
  geometry.setIndex(buildIndexAttribute(input.indices, input.vertexCount));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
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
 * @returns the geometry, for call-site chaining.
 */
export function updateMeshGeometry(
  geometry: THREE.BufferGeometry,
  input: MeshGeometryInput
): THREE.BufferGeometry {
  const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
  if (positionAttr && positionAttr.array !== input.position) {
    if (positionAttr.array.length === input.position.length) {
      (positionAttr.array as Float32Array).set(input.position);
      positionAttr.needsUpdate = true;
    } else {
      // A length change means the vertex count changed, which the whole-node
      // loader never does for a live node. Rebind rather than silently truncate.
      log.warning(
        Modules.SCENE_LOADER,
        `Mesh position length changed (${positionAttr.array.length} -> ${input.position.length}); rebinding`
      );
      geometry.setAttribute('position', new THREE.BufferAttribute(input.position, 3, false));
    }
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  geometry.setIndex(buildIndexAttribute(input.indices, input.vertexCount));
  return geometry;
}
