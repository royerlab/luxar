# Geometry commit

Synchronous GPU-commit step of the scene-loader pipeline. One module
per first-class geometry kind (Points, Lines, GSplats) writes the
already-projected buffers into the matching `THREE.Mesh` inside the
root scene group, plus a shared renderer-cache eviction helper used
by all three.

Commit is the **atomic, synchronous tail** of an `updateView` cycle.
By the time these functions run, the async work (nD slicing, worker
projection, Cholesky factoring, segment clipping) has already
produced staged data; this folder's job is the GPU-buffer write
itself and the bookkeeping that goes with it. No async operations
are allowed inside commit — by contract.

## Files

| File                          | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `commit-points-geometry.ts`   | Synchronous Points commit. GPU-buffer-pool path (zero-alloc on reuse) → fused texel write into the point data texture → dispose-and-recreate via `NodeFactory.createPointsGeometry`. Propagates the dtype-aware `radiusScale` onto `geometry.userData` and calls `syncPointMaterialWithGeometry` to push it into the material uniforms. (Sharpness needs no scale — it is authored natively in `[0, 1]`.)                                                                                                                                                                                                                                                                          |
| `commit-lines-geometry.ts`    | Synchronous Lines commit. Pool path via `acquireLinesGeometry` / `updateLinesGeometry` (the acquire declares `hasScalars`, deciding the scalar spec set up front — no lazy promotion remains, so the acquire's rebuild flag is the complete signal), fallback via `updateInstancedLinesMesh`. Updates `visibleSegmentCount`.                                                                                                                                                                                                                                                                                                                                                        |
| `commit-gsplats-geometry.ts`  | Synchronous GSplats commit. Pool path passes the live `uTruncate` uniform into `updateGSplatsGeometry` so frustum-cull sizing matches the shader; fallback via `updateInstancedGSplatsMesh`. Updates `visibleSplatCount`. Reads `uTruncate` defensively with a `3.0` default.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `invalidate-render-object.ts` | Shared helper for geometry-identity changes. (1) Eagerly re-points `userData.pickNode.geometry` at the new mesh geometry (pick meshes share, never own, the geometry; without this a grow-swap leaves the pick mesh pinning the OLD geometry until the next pick). (2) Dispatches a tagged `'dispose'` event on the mesh's material so Three's `WebGPURenderer` evicts the cached `RenderObject` and rebuilds its `vertexBuffers` set against the geometry returned by the acquire (growth = release + reacquire, so a grow always returns a different geometry/buffer). The `SOFT_DISPOSE_FLAG` symbol tells `MaterialManager` to treat this as a cache-flush, not a real dispose. |
| `noop-commit.ts`              | Shared "unchanged data" fast path: `StagedNoopCommit` + `isAlreadyCommitted` (reference-identity check against `mesh.userData.committedData`). When a loader returns the exact object the GPU already holds (progressive loaders memoize their LOD concatenation), the handler skips the process/upload steps and flows a stamp-only commit through the atomic stage — geometry untouched, `loadedViewVersion` still refreshed.                                                                                                                                                                                                                                                     |
| `stamp-view-version.ts`       | Shared commit-time stamps: `stampLoadedViewVersion` (which view version the committed geometry reflects — LOD freshness) and `stampLadderComplete` (the committed ladder state, read off `userData.loader` at commit time: `committedLadderComplete` from `hasMoreLODs`, plus `committedEnergyFraction` — the committed prefix's e(k) energy fraction from the quality stamps; 1 for non-progressive leaves, REMOVED on unstamped datasets — both feeding the never-downgrade display gate). All written by every commit, INCLUDING the stamp-only no-op branches.                                                                                                                  |

## Invariants

- **Synchronous only.** No `await`, no microtask hops. Commit runs
  inside the orchestrator's atomic stage so the visible scene
  transitions in one frame. Async projection lives in the sibling
  `process/data-processor-{lines,gsplats}.ts` modules.
- **Three-geometry symmetry.** Each commit module exports a single
  `commit{Points|Lines|GSplats}Geometry` function with the same
  shape: early-return on missing root or mismatched node type, log
  on empty-data frames, branch on `gpuBufferPool` presence,
  invalidate the renderer cache when the pool reports a buffer
  rebuild, write the visible-count back onto `userData`. Filename
  matches the single export.
- **Why no `data-processor-points.ts`?** Points's facade folds
  nD → 3D projection into `loadPoints()` itself, so the orchestrator
  only commits — there is no async processing step. Lines (segment
  clipping) and GSplats (Cholesky-factored projection) need per-frame
  worker dispatches and so do have matching
  `process/data-processor-{lines,gsplats}.ts` modules upstream of
  commit.
- **Pool-rebuild → render-object invalidation.** Whenever
  `gpuBufferPool.didLastAcquireRebuildAttributes()` returns `true`
  (grow-swap, pool swap, fresh allocation, or a lines scalar
  spec-set change), the
  commit module calls `invalidateRenderObjectFor(mesh)`. Without
  this, `WebGPURenderer` keeps the stale `vertexBuffers` set cached
  on its `RenderObject` and the next draw binds the OLD GPU buffer,
  failing validation with "Instance range … requires a larger
  buffer than the bound buffer size". See the file-level docstring
  in `invalidate-render-object.ts` for the full r184 trace.
- **Soft dispose, not real dispose.** The `'dispose'` event we
  dispatch is tagged via the `SOFT_DISPOSE_FLAG` symbol on the
  material so Luxar's `MaterialManager` skips its registry-cleanup
  branch. The tag is cleared in a `finally` so an exception in
  `dispatchEvent` cannot leave the material poisoned.
- **Bounds come from loader metadata, never `computeBoundingBox`.**
  After interleaving, the `'position'` attribute holds the unit-quad
  template, not per-instance positions; computing bounds from it
  would yield `[-1, 1]²`. The commit modules clone
  `data.metadata.bounds` (Points) or rely on the pool/factory path
  to set bounds for Lines/GSplats.
- **Dtype-aware scale propagation is Points-only.** Lines and GSplats
  don't carry a Uint8-normalised radius attribute, so they don't need
  the `radiusScale` round-trip; only `commit-points-geometry.ts` calls
  `syncPointMaterialWithGeometry`. (Sharpness carries no scale on any
  geometry — it is authored natively in `[0, 1]`.)
- **Commit-time stamps are the display-side truth.** The count,
  `loadedViewVersion`, `committedLadderComplete`, and
  `committedEnergyFraction` stamps are written
  in the same synchronous call as the buffer write (and refreshed in
  the stamp-only no-op branches), so readers — the LOD registry's
  freshness fallback and the never-downgrade display gate
  (`scene/lod-display-gate.ts`) — can never observe "complete"/"fresh"
  paired with a stale count. Loaders' live `hasMoreLODs` getters flip
  at fetch-resolve, frames earlier; display decisions must read the
  stamps, never the live getters.

## See also

- `../process/data-processor-lines.ts`,
  `../process/data-processor-gsplats.ts` — the async processing
  step upstream of commit; defines `StagedLinesCommit` and
  `StagedGSplatsCommit` consumed here.
- `../../../rendering/gpu-buffer-pool.ts` — `GPUBufferPool`'s
  `acquire*` / `update*` / `didLastAcquireRebuildAttributes`
  contract.
- `../../../rendering/node-factory.ts` — `createPointsGeometry` used
  by the points dispose-and-recreate fallback.
- `../../../rendering/line-geometry.ts`,
  `../../../rendering/gsplat-geometry.ts` —
  `updateInstancedLinesMesh` / `updateInstancedGSplatsMesh`
  pool-disabled fallbacks.
- `../../../rendering/material-sync-helpers.ts` —
  `syncPointMaterialWithGeometry` (re-exported from
  `commit-points-geometry.ts`).
- `../../../rendering/material-manager.ts` — `SOFT_DISPOSE_FLAG`
  symbol and the soft-dispose listener that pairs with this
  folder's `invalidateRenderObjectFor`.
- `../../../rendering/widen-to-float32.ts` —
  `widenToFloat32` used by the points commit paths.
