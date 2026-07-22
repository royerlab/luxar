/**
 * Points geometry-commit concern extracted from `scene-loader.ts`.
 *
 * Single function (`commitPointsGeometry`) matching the inline
 * `updatePointsGeometry` method on the SceneLoader class. The inline
 * version interleaved three concerns:
 *
 *   1. Find the THREE.Points by name in the root group.
 *   2. Update `userData.visiblePointCount` and log on empty data.
 *   3. Write the new attribute data into GPU buffers — either via the
 *      GPU buffer pool (zero allocations on reuse) or, with the pool
 *      disabled, dispose+recreate via NodeFactory.
 *
 * **Why is there no `data-processor-points.ts`?** Lines and gsplats
 * each have their own `data-processor-{lines,gsplats}.ts` running a
 * `process*Data` step (worker-driven nD → 3D projection + clipping)
 * before commit. Points has no equivalent module because the points
 * facade folds the equivalent projection into `loadPoints()` itself
 * — the data returned by the facade is already 3D-projected and
 * ready for GPU upload, so the orchestrator only needs to commit it.
 * That asymmetry is a real architectural split, not drift: points'
 * projection is single-pass and fits in the loader; lines clipping
 * and gsplats Cholesky-projection are per-frame transforms that
 * the orchestrator needs to schedule on a worker.
 *
 * Filename matches the single export. If a future per-frame points
 * processing step appears (e.g. nD intensity attenuation), this
 * file should grow into `data-processor-points.ts` matching the
 * other two.
 *
 * @module data/scene-loader/commit/commit-points-geometry
 */

import * as THREE from 'three';
import type { LoadedPointsData } from '../../data-loader-types';
import { noteDepthSortCommit } from '../../../rendering/depth-sort-coordinator';
import { isPointsUserData } from '../../../types/points';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import { isAlreadyCommitted } from './noop-commit';
import {
  getCommittedData,
  hasCommittedData,
  setCommittedData,
} from '../../../types/committed-data';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';
import { clampPointCapacity } from '../../../rendering/element-texture-layout';
import { log, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { NodeFactory } from '../../../rendering/node-factory';
import { syncPointMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { invalidateRenderObjectFor } from './invalidate-render-object';

// Re-export so callers can import this name from the commit module while
// the implementation lives in the rendering layer.
export { syncPointMaterialWithGeometry };

/**
 * Synchronous GPU commit step for a points node. Same behavior as the
 * inline `updatePointsGeometry`:
 *
 *   - early return if the rootGroup or the named THREE.Points is gone,
 *   - log + zero `visiblePointCount` on an empty-data frame,
 *   - GPU-buffer-pool path (zero alloc on reuse) when the pool is set,
 *   - dispose + recreate via `nodeFactory.createPointsGeometry`
 *     otherwise (the factory owns all dtype/bounds logic).
 */
export function commitPointsGeometry(
  path: string,
  data: LoadedPointsData,
  rootGroup: THREE.Group | null,
  gpuBufferPool: GPUBufferPool | null,
  nodeFactory: NodeFactory,
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  if (!rootGroup) return;

  const points = rootGroup.getObjectByName(path) as THREE.Mesh;
  // Parity with lines/gsplats commit helpers — verify the named
  // object actually IS a Points node (not e.g. a stray Group with the
  // same name). Guards against bugs where a placeholder of the wrong
  // type is attached at this path.
  if (!points || !isPointsUserData(points.userData)) return;

  // No-op fast path: the data reference matches what the GPU already holds
  // (memoized progressive concat — see noop-commit.ts). Points has no
  // separate process step, so unlike lines/gsplats the check lives here in
  // the commit helper, covering the atomic-commit, refinement, retry, and
  // lazy paths uniformly. Refresh only the LOD freshness stamp.
  if (isAlreadyCommitted(points, data)) {
    stampLoadedViewVersion(points.userData, loadedViewVersion);
    stampLadderComplete(points.userData);
    return;
  }

  if (data.pointCount === 0) {
    log.info(
      Modules.SCENE_LOADER,
      `Clearing points for ${path} (no visible points at current slice)`
    );
  }

  // SEMANTIC clamp at the commit choke point (mirrors
  // commit-gsplats-geometry.ts): the GPU writers below clamp the WRITTEN
  // points to the per-node texture bound (element-texture-layout), so
  // every count this commit records — visiblePointCount, the append-gate
  // comparisons — must be the clamped one. Otherwise a later sorted
  // ordering over the recorded count would carry aSortedIndex slot values
  // ≥ the texture capacity, and those entries would fetch out-of-bounds
  // texels.
  const pointCount = clampPointCapacity(data.pointCount);

  // Pre-commit state for the append predicate below. Captured BEFORE the
  // pool branch reassigns `points.geometry` and the success-only stamps
  // below overwrite `visiblePointCount`. (The freshness stamps —
  // visiblePointCount / loadedViewVersion / ladderComplete — are written
  // ONLY after a successful GPU write, mirroring the gsplats twin: a
  // throwing write must not leave the mesh stamped fresh-for-this-view
  // with a count that never landed, or the LOD freshness registry would
  // trust it until the next user interaction.)
  const prevGeometry = points.geometry;
  const hadCommittedData = hasCommittedData(points);
  const prevCount = points.userData.visiblePointCount;

  // World-space radius footprint, shared by every commit path so the
  // boundingBox carries the rendered disc extent (the three-geometry
  // invariant — see create-points-node.ts). Uint8 radii normalize to
  // [0, max_radius]; Float32 radii are already world units, so max_radius
  // is the correct world-space max for both. No radii → 0.5 fill default.
  const attrs = points.userData.attrs;
  const maxRadius = (attrs?.max_radius as number | undefined) ?? 1.0;
  const footprintRadius = data.radii ? maxRadius : 0.5;

  // Lazy projected-centers provider for the depth-sort coordinator
  // (invoked only when the node actually registers — order-dependent
  // effective mode, non-empty, still the latest generation — so the
  // common additive path never pays the copy). MUST allocate fresh:
  // the coordinator TRANSFERS the returned buffer to the SortWorker,
  // and `data.positions` is the committed/lineage reference the noop
  // and append gates key on — a `subarray` view would detach it. The
  // elementwise copy also widens Float16 positions to the Float32 the
  // sort kernel expects.
  const sortCenters3 = (): Float32Array => {
    const src = data.positions;
    const out = new Float32Array(pointCount * 3);
    for (let i = 0; i < out.length; i++) out[i] = src[i];
    return out;
  };

  const bufferSession = session?.begin('Update Buffers');
  try {
    if (gpuBufferPool) {
      // Acquire geometry from pool (capacity-aware: the fixed 3-texel
      // layout means any pooled points geometry fits any points node).
      const geometry = gpuBufferPool.acquirePointsGeometry(path, pointCount);
      // Pool rebuilt the geometry's storage (grow, pool swap, or fresh
      // allocation). The mesh's cached RenderObject in Three's
      // WebGPURenderer still references the old buffers; the helper
      // dispatches a `dispose` event on the material to evict that
      // cache. No-op under WebGL2 / pre-init / no cached entry.
      const attributesRebuilt = gpuBufferPool.didLastAcquireRebuildAttributes();
      // Keep the previous depth-sort permutation on a same-node same-count
      // in-place recommit (timepoint scrub): a permutation of [0,count) is
      // a strictly-no-worse prior than storage order for the ≥1 frame
      // until the re-sort dispatched by noteDepthSortCommit below lands.
      // Guards mirror the gsplats twin (commit-gsplats-geometry.ts) — the
      // full rationale lives there.
      const preserveOrdering =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        prevCount === pointCount;
      // Append fast path (depth-sorting Phase 4 Stage 2): when this commit
      // merely EXTENDS the prefix already on the GPU, write & upload only the
      // new `[prevCount, pointCount)` suffix. Correctness rests on the
      // buffer's prefix being byte-identical to what a full write would
      // produce, which every conjunct establishes (see the gsplats twin in
      // commit-gsplats-geometry.ts for the full rationale):
      // - !attributesRebuilt && geometry === prevGeometry: the pool reused
      //   THIS node's buffer in place (grow/best-fit/fresh acquire sets
      //   attributesRebuilt and may hand back another node's data). Also
      //   bounds pointCount ≤ the existing capacity.
      // - gpuPrefixIntact: a WebGL context restore zeroed the GPU buffers;
      //   the restore hook clears this so the next commit does a full rewrite.
      // - pointCount > prevCount: a genuine append (equal → the no-op
      //   identity path above; shrink/first-commit → full write).
      // - prefix lineage === committedData: the new concat result forward-
      //   chains to the exact object last committed here — proving same
      //   generation (view unchanged), a genuine extension, and that the GPU
      //   still holds that parent's projection. This conjunct also covers
      //   dtype: the concat's arrays carry one dtype per field, so the
      //   prefix widens to bit-identical floats on both commits.
      // - optional-field presence must MATCH the committed parent: the concat
      //   is all-or-nothing per field (concatOptionalField), so a new level
      //   WITHOUT e.g. Float32 colors drops the merged field entirely and the
      //   adapter's constant fill would differ from the prefix's committed
      //   values. (The fixed texel layout means the pool no longer rebuilds
      //   on dtype/scalar-presence changes — these presence conjuncts are
      //   now the SOLE guard for every optional field, scalars included.)
      const committed = getCommittedData(points) as LoadedPointsData | undefined;
      const canAppend =
        hadCommittedData &&
        !attributesRebuilt &&
        geometry === prevGeometry &&
        points.userData.gpuPrefixIntact === true &&
        pointCount > (prevCount ?? 0) &&
        committed !== undefined &&
        getPrefixParent(data) !== undefined &&
        getPrefixParent(data) === committed &&
        !!data.colors === !!committed.colors &&
        !!data.radii === !!committed.radii &&
        !!data.sharpness === !!committed.sharpness &&
        !!data.scalars === !!committed.scalars;
      // Consume-and-clear (see prefix-lineage.ts retention contract): the
      // lineage entry existed solely for the gate check above — clearing it
      // unpins the parent concat's CPU arrays. A retry after a throwing
      // write below reads `undefined` and full-rewrites, the safe direction.
      setPrefixParent(data, null);
      // Propagate the dtype-aware radius scale onto geometry userData
      // BEFORE the write: the finally's material sync runs even on a
      // throwing write (the geometry was handed off regardless), and it
      // must push THIS commit's scale — not a best-fit-adopted previous
      // tenant's (Uint8 maxRadius vs Float32 1.0 is a 50× radius skew).
      if (!geometry.userData) {
        geometry.userData = {};
      }
      geometry.userData.radiusScale = data.radii instanceof Uint8Array ? maxRadius : 1.0;
      try {
        gpuBufferPool.updatePointsGeometry(geometry, data, pointCount, {
          preserveOrdering,
          fromInstance: canAppend ? (prevCount ?? 0) : 0,
        });

        if (data.metadata.bounds) {
          geometry.boundingBox = data.metadata.bounds.clone();
          if (footprintRadius > 0) geometry.boundingBox.expandByScalar(footprintRadius);
          geometry.boundingSphere = new THREE.Sphere();
          geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);
        } else if (geometry.boundingBox && footprintRadius > 0) {
          // metadata.bounds is typed required, so this fallback is
          // near-dead — but if it ever fires, the adapter's position-scan
          // box still needs the disc-footprint expansion or edge sprites
          // frustum-clip while visible (the gsplats adapter always
          // footprint-expands; keep points equivalent).
          geometry.boundingBox.expandByScalar(footprintRadius);
          geometry.boundingSphere = new THREE.Sphere();
          geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);
        }
      } catch (err) {
        // Defense-in-depth: a throwing write leaves the buffer content
        // unproven — clear the append-safety flag so the next commit
        // full-rewrites regardless of lineage (the lineage consume-and-
        // clear already forces this today; the flag makes the gate robust
        // against future loader/cache changes that re-stamp lineage).
        points.userData.gpuPrefixIntact = false;
        throw err;
      } finally {
        // Ownership handoff must happen even if the update throws: the
        // acquire may have RELEASED the mesh's current geometry into the
        // free pool (grow / attribute-spec-mismatch path), so bailing out
        // before this assignment would leave the mesh rendering a
        // free-pooled geometry that the evictor can dispose — or another
        // node adopt — mid-render. On a throw the mesh shows one frame of
        // partially-written data instead; the skipped committedData stamp
        // below guarantees the next update re-uploads in full.
        points.geometry = geometry;
        syncPointMaterialWithGeometry(points);
        if (attributesRebuilt) invalidateRenderObjectFor(points);
        // Dispose a replaced NON-pool geometry (the creation-time
        // placeholder from createPointsNode). Pool-owned geometries are
        // released to the free list by acquire and must never be disposed
        // here (`luxarPooled` marker); the placeholder has no other owner,
        // and its element texture (~65 KB minimum-row alloc) would
        // otherwise leak per node per dataset switch. Runs AFTER
        // invalidateRenderObjectFor so the pick node no longer references
        // it.
        if (prevGeometry !== geometry && !prevGeometry.userData?.luxarPooled) {
          prevGeometry.dispose();
        }
      }
      // SUCCESS-ONLY stamps (a throw above propagates past this point,
      // mirroring the gsplats twin's ordering): freshness first —
      points.userData.visiblePointCount = pointCount;
      // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
      stampLoadedViewVersion(points.userData, loadedViewVersion);
      // Ladder-completeness stamp for the never-downgrade display gate.
      stampLadderComplete(points.userData);
      // Record the committed data reference — a later update returning the
      // SAME reference (memoized progressive concat) takes the stamp-only
      // no-op path above instead of re-uploading.
      setCommittedData(points, data);
      // Append-fast-path bookkeeping: the buffer now holds this commit's
      // data in full (whether written fully or by suffix-extension), so the
      // next commit may append. A context restore clears this flag.
      points.userData.gpuPrefixIntact = true;
      // Depth-sorting (points integration): every non-noop commit bumps
      // the node's sort generation; order-dependent (effective `normal`)
      // nodes additionally register their centers with the SortWorker and
      // get one sort from the current camera pose. Success-only (a
      // throwing write propagates before this line, mirroring the gsplats
      // twin) with the CLAMPED count, so permutation values stay inside
      // [0, textureCapacity). The lazy provider defers the O(N) positions
      // copy to the sorted path.
      noteDepthSortCommit(points, sortCenters3, pointCount);
      return;
    }

    // Pool disabled: recreate unconditionally. Recreation handles all
    // the dtype logic (divisor-based widenToFloat32 for Uint8/Uint16,
    // Float16 widening, bounds/footprint, radiusScale userData) via
    // NodeFactory, which builds the same texture-backed storage as the
    // pool (attachPointStorage + writePointTexels) sized exactly.
    // (A historical same-count in-place branch assumed a different
    // layout and threw against factory-built geometry; correctness over
    // reuse on this non-default fallback.)
    // Create-then-swap-then-dispose: building first keeps the mesh on its
    // old (valid) geometry if the factory throws on malformed data —
    // dispose-first would strand the mesh on a disposed geometry whose
    // element texture is already freed.
    const oldGeometry = points.geometry;
    // Pass max_radius so the rebuilt geometry bakes the correct
    // footprint into boundingBox (and the right dtype scale); omitting
    // it would default maxRadius=1.0 and clip large radii.
    points.geometry = nodeFactory.createPointsGeometry(data, maxRadius);
    if (oldGeometry) {
      oldGeometry.dispose();
    }
    // dispose+recreate path picks up new dtype-aware scales from
    // the freshly built geometry's userData.
    syncPointMaterialWithGeometry(points);
    // Fresh GPU buffers replaced the geometry: evict Three's cached
    // RenderObject (stale `vertexBuffers` on the WebGPU backend) —
    // same contract as the pool path's attributesRebuilt branch.
    invalidateRenderObjectFor(points);
    // SUCCESS-ONLY stamps (see the pool path).
    points.userData.visiblePointCount = pointCount;
    stampLoadedViewVersion(points.userData, loadedViewVersion);
    stampLadderComplete(points.userData);
    // Record the committed data reference (see the pool path above).
    setCommittedData(points, data);
    // The freshly built geometry holds this commit's data in full. The flag
    // is only consulted on the pool path, but keeping the stamp uniform
    // across paths mirrors the gsplats commit.
    points.userData.gpuPrefixIntact = true;
    // Depth-sorting registration — see the pool path above.
    noteDepthSortCommit(points, sortCenters3, pointCount);
  } finally {
    bufferSession?.end();
  }
}
