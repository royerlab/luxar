/**
 * Force Three's `WebGPURenderer` to discard the cached `RenderObject`
 * associated with a given mesh + its material, so the next render
 * builds a fresh one with a clean `vertexBuffers` cache.
 *
 * Why this exists: when a commit swaps a mesh's `InstancedBufferGeometry`
 * (pool grow / best-fit adoption / non-pool rebuild — the per-instance
 * buffers, e.g. `aSortedIndex`, are new objects),
 * Three's `RenderObject` (cached in `RenderObjects.chainMap` keyed
 * by `[mesh, material, …]`) still holds:
 *
 *   - `attributes` / `attributesId` — reset to `null` by
 *     `setGeometry` when Three's `needsGeometryUpdate` notices the
 *     attribute ids changed.
 *   - **`vertexBuffers`** — NOT reset by `setGeometry`. Stays as the
 *     OLD `Set<InterleavedBuffer>` until `getAttributes()` rebuilds
 *     it.
 *
 * `getAttributes()` is called from `Geometries.updateForRender`,
 * which `Renderer._renderObjectDirect` (r184 line 3535-3546) only
 * runs when `_nodes.needsRefresh(renderObject)` is `true`. A
 * pure-geometry / buffer-resize change does NOT mark `needsRefresh`,
 * so on the next draw `WebGPUBackend.draw` calls
 * `renderObject.getVertexBuffers()` which returns the stale
 * `vertexBuffers` array (skips `getAttributes()` because
 * `vertexBuffers !== null`). The renderer then binds the OLD GPU
 * buffer at slot 0 and validation fails with "Instance range …
 * requires a larger buffer than the bound buffer size (oldSize)".
 *
 * The cleanest in-userland fix is to dispatch a `dispose` event on
 * the mesh's material. `RenderObject.onMaterialDispose` (line 307-310
 * in `common/RenderObject.js`) calls `renderObject.dispose()`, whose
 * `onDispose` callback (set in `RenderObjects.createRenderObject`,
 * line 201-209) deletes the entry from `chainMap`. The next
 * `RenderObjects.get(...)` call returns `undefined` from the map and
 * builds a fresh `RenderObject` whose `vertexBuffers` cache starts
 * `null` — forcing `getVertexBuffers()` to call `getAttributes()` and
 * pick up the current `InterleavedBuffer`.
 *
 * Dispatching `dispose` on the material does NOT actually free the
 * material's GPU resources or recompile its shader — Three.js only
 * frees those when `material.dispose()` is called. The event fires
 * regardless of `dispose()` being invoked, but Luxar's MaterialManager
 * also listens for `'dispose'` to unregister the material from its
 * global-update set / caches. To prevent that side effect while still
 * triggering Three's `RenderObject` eviction, we set a transient
 * symbol-keyed flag on the material before dispatching;
 * MaterialManager's listener checks for the flag and treats the
 * event as a soft cache-invalidation rather than a real dispose.
 * See `SOFT_DISPOSE_FLAG` in `rendering/material-manager.ts`.
 *
 * @module data/scene-loader/commit/invalidate-render-object
 */

import type * as THREE from 'three';

import { SOFT_DISPOSE_FLAG } from '../../../rendering/material-manager';

/**
 * Discard Three's cached `RenderObject` for the given mesh's material
 * combination. Safe to call every commit cycle when the GPU buffer
 * pool reports a rebuild; cheap (just fires an event); no-op when
 * the renderer holds no cached state for the material yet.
 *
 * The mesh's `material` is the receiver; if the mesh uses an array
 * of materials (Luxar doesn't today, but be defensive), every entry
 * is signaled.
 */
export function invalidateRenderObjectFor(mesh: THREE.Mesh): void {
  // Geometry identity changed (grow-swap / pool swap / fresh alloc):
  // eagerly re-point the paired pick mesh at the new geometry. The
  // picking system also re-syncs lazily at pick time
  // (picking-system.ts renderPickBuffer), but between the swap and the
  // next pick the pick mesh would otherwise keep the OLD geometry
  // alive — and under the WebGPU renderer's strong Info.memoryMap,
  // its attribute views with it. Pick meshes share the main mesh's
  // geometry by design (they never own one), so this is a pure
  // reference update.
  const pickNode = (mesh.userData as { pickNode?: THREE.Mesh } | undefined)?.pickNode;
  if (pickNode?.isMesh) {
    pickNode.geometry = mesh.geometry;
    // The pick mesh has its OWN cached RenderObject (chainMap keyed by
    // [pickMesh, pickMaterial, …]) with the same stale `vertexBuffers`
    // problem: the pick pass binds whatever that cache holds, so after a
    // pool grow/swap a pick render on the WebGPU backend would bind the
    // old (smaller/disposed) GPU buffer — "Instance range requires a
    // larger buffer" validation errors or silent mis-picks. Evict it via
    // the same soft-dispose mechanism as the main material below.
    softDisposeAll(pickNode.material);
  }

  softDisposeAll(mesh.material);
}

/** Soft-dispose a mesh's material, handling the `Material[]` case. */
function softDisposeAll(material: THREE.Material | THREE.Material[]): void {
  if (Array.isArray(material)) {
    for (const m of material) {
      dispatchSoftDispose(m);
    }
  } else if (material) {
    dispatchSoftDispose(material);
  }
}

function dispatchSoftDispose(material: THREE.Material): void {
  // Tag the material so MaterialManager's `'dispose'` listener
  // recognises this as a soft eviction (RenderObject cache flush
  // only) and skips its own registry / cache cleanup. The tag is
  // cleared in a `finally` so an exception in `dispatchEvent` can't
  // leave the material in a poisoned state.
  const tagged = material as unknown as Record<symbol, boolean> & {
    dispatchEvent: (e: { type: string }) => void;
  };
  tagged[SOFT_DISPOSE_FLAG] = true;
  try {
    tagged.dispatchEvent({ type: 'dispose' });
  } finally {
    delete tagged[SOFT_DISPOSE_FLAG];
  }
}
