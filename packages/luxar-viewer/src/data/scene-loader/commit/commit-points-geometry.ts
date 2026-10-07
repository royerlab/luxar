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
 * **Where the no-op fast path lives, vs. `data-processor-points.ts`.**
 * `data-processor-points.ts` exists as the *synchronous staging half*:
 * the points loader folds nD → 3D projection into `loadPoints()`
 * itself, so the data is already 3D-projected and ready for GPU upload
 * and there is no async worker projection step to run. Lines (segment
 * clipping) and gsplats (Cholesky-factored projection) instead run
 * per-frame worker dispatches upstream of commit. Because points has no
 * projection to skip, its no-op reference-identity fast path lives HERE
 * in `commitPointsGeometry` — unlike lines/gsplats, which must
 * short-circuit *before* their worker projection.
 *
 * Filename matches the single export.
 *
 * @module data/scene-loader/commit/commit-points-geometry
 */

import { findObjectByName } from '../../../utils/scene-graph-index';
import * as THREE from 'three';
import type { LoadedPointsData } from '../../data-loader-types';
import type { GeometryCommitHost } from './commit-host';
import { isPointsUserData } from '../../../types/points';
import { stampLadderComplete, stampLoadedViewVersion } from './stamp-view-version';
import { markFirstCommit } from '../../../profiling/load-timeline';
import { isAlreadyCommitted } from './noop-commit';
import {
  getCommittedData,
  hasCommittedData,
  setCommittedData,
  setElementIdMap,
} from '../../../types/committed-data';
import { getPrefixParent, setPrefixParent } from '../../../types/prefix-lineage';
import { clampPointCapacity } from '../../../rendering/element-texture-layout';
import { log, LogEmoji, Modules } from '../../../utils/log';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { NodeFactory } from '../../../rendering/node-factory';
import { syncPointMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { invalidateRenderObjectFor } from './invalidate-render-object';
import { planInstancedOrdering } from './plan-instanced-ordering';
import { DEFAULT_POINT_RADIUS } from '../../../config/constants';

/** Per-part detail line (hot on time-partitioned stores): `?verboseLog` only. */
function logClearedPoints(path: string): void {
  log.verbose(
    LogEmoji.INFO,
    Modules.SCENE_LOADER,
    `Clearing points for ${path} (no visible points at current slice)`
  );
}

/**
 * True when every optional field a points append would leave un-rewritten in
 * the prefix agrees with the committed parent (see the append gate).
 */
function sameOptionalPointFields(data: LoadedPointsData, committed: LoadedPointsData): boolean {
  return (
    !!data.colors === !!committed.colors &&
    (data.colorComponents ?? 3) === (committed.colorComponents ?? 3) &&
    !!data.radii === !!committed.radii &&
    !!data.sharpness === !!committed.sharpness &&
    !!data.scalars === !!committed.scalars
  );
}

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
  host: GeometryCommitHost & { nodeFactory: NodeFactory },
  session: UpdateSession | undefined,
  loadedViewVersion: number
): void {
  const { rootGroup, gpuBufferPool, nodeFactory, depthSort } = host;
  if (!rootGroup) return;

  const points = findObjectByName(rootGroup, path) as THREE.Mesh;
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
    markFirstCommit('points');
    return;
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
  // A pool grow releases the old geometry without ending its held draw.
  // Preserve the count that was actually on screen.
  const prevDrawnCount = (prevGeometry as THREE.InstancedBufferGeometry).instanceCount;
  const hadCommittedData = hasCommittedData(points);
  const prevCount = points.userData.visiblePointCount;

  // World-space radius footprint, shared by every commit path so the
  // boundingBox carries the rendered disc extent (the three-geometry
  // invariant — see create-points-node.ts). Uint8 radii normalize to
  // [0, max_radius]; Float32 radii are already world units, so max_radius
  // is the correct world-space max for both. No radii → the DEFAULT_POINT_RADIUS
  // fill default.
  const attrs = points.userData.attrs;
  const maxRadius = (attrs?.max_radius as number | undefined) ?? 1.0;
  const footprintRadius = data.radii ? maxRadius : DEFAULT_POINT_RADIUS;

  // Lazy projected-centers provider for the depth-sort coordinator
  // (invoked only when the node actually registers — order-dependent
  // effective mode, non-empty, still the latest generation — so the
  // common additive path never pays the copy). MUST allocate fresh:
  // the coordinator TRANSFERS the returned buffer to the SortWorker,
  // and `data.positions` is the committed/lineage reference the noop
  // and append gates key on — passing a `subarray` VIEW to the transfer
  // would detach it. Reading through a subarray into `out.set` is safe
  // (set copies; only `out.buffer` is later transferred).
  const sortCenters3 = (): Float32Array => {
    const src = data.positions;
    const out = new Float32Array(pointCount * 3);
    if (src instanceof Float32Array) {
      out.set(src.subarray(0, out.length)); // memcpy fast path
    } else {
      // Float16 positions: elementwise widen to the Float32 the sort
      // kernel expects.
      for (let i = 0; i < out.length; i++) out[i] = src[i];
    }
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
      // Ordering + append gate, shared with lines/gsplats
      // (plan-instanced-ordering.ts, which states the prior/lineage rules).
      // The suffix-only upload additionally needs the GPU prefix intact (a
      // WebGL context restore clears the flag) and every optional field's
      // PRESENCE to match the committed parent. Radii, sharpness and scalars
      // concat all-or-nothing (concatOptionalField), so a new level WITHOUT
      // one drops the merged field and the adapter's constant fill would
      // differ from the prefix's committed values. Colours white-fill a
      // colourless rung instead (concatColorsWhiteFilled), but the first
      // coloured level joining a colourless prefix still flips presence ON,
      // rewriting the prefix's colours from the default to white. The colour
      // LAYOUT (RGB vs RGBA) must match too: an append writes only the suffix,
      // so a flip would leave the prefix's texel2.y alphas stale. (The fixed
      // texel layout means the pool no longer rebuilds on dtype or presence
      // changes, so these conjuncts are the SOLE guard for every optional
      // field.) Dtype needs no conjunct: the concat's arrays carry one dtype
      // per field, so a proven lineage widens the prefix to identical floats.
      const committed = getCommittedData(points) as LoadedPointsData | undefined;
      const orderingOptions = planInstancedOrdering({
        geometry,
        prevGeometry,
        prevDrawnCount,
        hadCommittedData,
        prevCount,
        count: pointCount,
        attributesRebuilt,
        prefixParent: getPrefixParent(data),
        committedData: committed,
        prefixReusable:
          points.userData.gpuPrefixIntact === true &&
          committed !== undefined &&
          sameOptionalPointFields(data, committed),
      });
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
        gpuBufferPool.updatePointsGeometry(geometry, data, pointCount, orderingOptions);

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
    } else {
      // Consume-and-clear on this path too (prefix-lineage.ts retention
      // contract): no append gate reads the lineage here, and leaving it set
      // would keep the parent concat's CPU arrays pinned for the payload's life.
      setPrefixParent(data, null);
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
    }

    // SUCCESS-ONLY tail, shared by both branches (a throwing write above
    // propagates past this point, mirroring the gsplats twin's single
    // post-if/else tail): freshness first —
    points.userData.visiblePointCount = pointCount;
    points.userData.requestedElementCount = data.pointCount;
    points.userData.droppedElementCount = data.pointCount - pointCount;
    // Append-fast-path bookkeeping: the buffer now holds this commit's
    // data in full (whether written fully or by suffix-extension), so the
    // next commit may append. Only consulted on the pool path, but the
    // stamp stays uniform across paths (gsplats parity). A context
    // restore clears this flag.
    points.userData.gpuPrefixIntact = true;
    // Slice-aware LOD freshness stamp (see commit-gsplats-geometry.ts).
    stampLoadedViewVersion(points.userData, loadedViewVersion);
    // Ladder-completeness stamp for the never-downgrade display gate.
    stampLadderComplete(points.userData);
    markFirstCommit('points');
    // Record the committed data reference — a later update returning the
    // SAME reference (memoized progressive concat) takes the stamp-only
    // no-op path above instead of re-uploading.
    setCommittedData(points, data);
    // Slot → on-disk element index map for picking, in lockstep with the
    // stamp above so it always describes the buffers just uploaded. Points'
    // loader legitimately produces the map (its projection is folded into
    // `loadPoints`, upstream of the SliceCache measure), but the PICKER reads
    // it off the mesh — same mechanism as gsplats. A payload without a map
    // clears any previous commit's.
    setElementIdMap(points, data.elementIds);

    if (pointCount === 0) logClearedPoints(path);

    // Depth-sorting (points integration): every non-noop commit bumps
    // the node's sort generation; order-dependent (effective `normal`)
    // nodes additionally register their centers with the SortWorker and
    // get one sort from the current camera pose. Runs LAST like the
    // gsplats twin, success-only, with the CLAMPED count so permutation
    // values stay inside [0, textureCapacity). The lazy provider defers
    // the O(N) positions copy to the sorted path.
    depthSort?.noteCommit(points, sortCenters3, pointCount);
  } finally {
    bufferSession?.end();
  }
}
