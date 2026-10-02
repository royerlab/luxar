/**
 * The depth-sort coordinator's APPLY side: pushing a geometry's active
 * ordering slot to its visual and pick materials, the post-render draw
 * acknowledgement hook, the commit-time synchronous first sort, and the
 * small helpers the resolve path shares (pose record, ordering-length check,
 * monitor byte label, fresh model-view).
 *
 * @module rendering/depth-sort-coordinator/ordering-apply
 */

import * as THREE from 'three';
import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  getActiveSortedIndexAttribute,
  writeSortedIndexOrderingLive,
} from '../element-storage';
// The TypeScript reference kernel, not the WASM one: the WASM module is
// instantiated inside the SortWorker, and the whole point of the synchronous
// path is to answer without touching the worker. Exact-parity with the Rust
// kernel (see its module doc), so the two agree on the permutation.
import { sort_splats_by_depth } from '../../wasm/typescript/depth-sort';
import { acknowledgeTriangleOrderingDraw, writeSortedTriangleOrdering } from './triangle-ordering';
import { log, Modules } from '../../utils/log';
import { syncSortElementLimit, type CoordinatorState, type NodeSortState } from './state';

/**
 * Push the geometry's active ordering slot onto a material's
 * `uSortedIndexSlot` uniform. Covers both backends: the GLSL materials
 * expose a plain `IUniform`, and the TSL materials expose a
 * `proxyIUniform` that writes straight through to the node — neither
 * rebuilds or recompiles on a value change.
 *
 * `slot` is `0 | 1`, not `number`, and that is load-bearing: this is the
 * ONLY path by which a value reaches `uSortedIndexSlot`, and the two
 * backends do not agree outside that domain. GLSL selects with
 * `slot == 1 ? back : front`, so anything else reads the FRONT buffer;
 * the TSL twin is branchless (`a·(1-slot) + b·slot`, forced by
 * `.select()` being a statement — see `sortedIndexNode`), so a slot of 2
 * would evaluate to `2b - a`: garbage indices, not a fallback. Rather
 * than clamp on every vertex for a state nothing can produce, the type
 * keeps it unrepresentable at the one entry point. Widening this
 * signature — or adding a third ordering buffer — means giving the two
 * shaders a shared, tested selection rule first.
 */
function applySortedIndexSlotToMaterial(
  material: THREE.Material | THREE.Material[] | undefined,
  slot: 0 | 1
): boolean {
  if (!material) return false;
  // Scalar and array handled without a temporary wrapper array: this runs
  // for every tracked node on every frame (twice with a pick material), so
  // it must stay allocation-free — the per-frame scratch invariant below.
  if (!Array.isArray(material)) return setSortedIndexSlotUniform(material, slot);
  let changed = false;
  for (const m of material) changed = setSortedIndexSlotUniform(m, slot) || changed;
  return changed;
}

/** Write one material's `uSortedIndexSlot`, if it has one; true when it changed. */
function setSortedIndexSlotUniform(material: THREE.Material, slot: 0 | 1): boolean {
  const uniform = (material as THREE.ShaderMaterial | undefined)?.uniforms?.uSortedIndexSlot;
  if (!uniform) return false;
  const changed = uniform.value !== slot;
  uniform.value = slot;
  return changed;
}

/**
 * Point a node's shaders at whichever ordering buffer is currently
 * complete. The PICK material must move with the visual one: it shares
 * the geometry and emits `vElementId` from the same index, so a pick
 * pass reading the other buffer would resolve hovers against a stale
 * permutation.
 *
 * Records a changed uniform on `c.drawnStateChanged`, which tells the
 * render-on-change loop that this frame draws something new.
 */
export function syncSortedIndexSlot(c: CoordinatorState, mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (!geometry) return;
  const slot = activeSortedIndexSlot(geometry);
  if (applySortedIndexSlotToMaterial(mesh.material, slot)) c.drawnStateChanged = true;
  const pickNode = (mesh.userData as { pickNode?: THREE.Mesh } | undefined)?.pickNode;
  if (pickNode && applySortedIndexSlotToMaterial(pickNode.material, slot)) {
    c.drawnStateChanged = true;
  }
}

/** Installed post-render acknowledgement hook for each sortable mesh. */
const drawAcknowledgementHooks = new WeakMap<THREE.Mesh, THREE.Mesh['onAfterRender']>();

/**
 * Chain a mesh-local post-render hook that acknowledges the selected ordering
 * only after THREE has consumed its pending attribute ranges and drawn it.
 * The hook uses the render callback's geometry argument (rather than
 * `mesh.geometry`) so a pool swap inside another callback cannot acknowledge
 * the wrong buffer. Existing user callbacks are preserved.
 */
export function ensureDrawAcknowledgementHook(mesh: THREE.Mesh): void {
  const installed = drawAcknowledgementHooks.get(mesh);
  if (installed && mesh.onAfterRender === installed) return;

  const previous = mesh.onAfterRender;
  const hook: THREE.Mesh['onAfterRender'] = (...args) => {
    const geometry = args[3];
    acknowledgeSortedIndexOrderingDraw(geometry as THREE.InstancedBufferGeometry);
    // The indexed (mesh) apply acknowledges at the same point and for the
    // same reason. Both are keyed by geometry and both no-op when the
    // geometry has nothing pending, so calling each unconditionally is
    // cheaper than discriminating the node type on every rendered frame.
    acknowledgeTriangleOrderingDraw(geometry);
    previous.apply(mesh, args);
  };
  drawAcknowledgementHooks.set(mesh, hook);
  mesh.onAfterRender = hook;
}

/**
 * Whether a resolved ordering covers exactly the population its generation
 * committed. A mismatch means the worker sorted a clamped registration (its
 * centers under-delivered): the ordering is not a permutation of the drawn
 * population, so the resolve stages nothing and a held append draw is
 * released instead of waiting for a length that will never land.
 */
export function orderingCoversCommit(
  ordering: Uint32Array,
  state: NodeSortState,
  nodeId: string
): boolean {
  if (ordering.length === state.count) return true;
  if (state.warnedMismatchGeneration !== state.generation) {
    state.warnedMismatchGeneration = state.generation;
    log.warning(
      Modules.WORKER_POOL,
      `Depth-sort ordering for ${nodeId} has ${ordering.length} elements but the commit has ` +
        `${state.count} — dropped`
    );
  }
  return false;
}

/**
 * Record the pose a sort was dispatched from (the model-view z-row that
 * fully determines the resulting permutation — see NodeSortState). The
 * per-frame scheduler compares live poses against this.
 */
export function recordSortPose(state: NodeSortState, modelView: THREE.Matrix4): void {
  const e = modelView.elements;
  if (!state.lastSortAxis) state.lastSortAxis = new THREE.Vector3();
  state.lastSortAxis.set(e[2], e[6], e[10]);
  const len = state.lastSortAxis.length();
  if (len > 0) {
    state.lastSortAxis.divideScalar(len);
    state.lastSortOffset = e[14] / len;
  } else {
    state.lastSortOffset = e[14];
  }
}

/**
 * Format an ordering-upload byte count for the 'Depth Sort' monitor line.
 * `uploaded` distinguishes bytes that have reached the GPU (`up`, after the
 * selected mesh completes a render) from bytes merely STAGED for upload
 * (`sched`, at worker resolve) — issue #713. The panel shows this as the
 * pass's `info` tag.
 */
export function formatOrderingBytes(bytes: number, uploaded: boolean): string {
  const suffix = uploaded ? 'up' : 'sched';
  return bytes >= 1_000_000
    ? `${(bytes / 1_000_000).toFixed(1)} MB ${suffix}`
    : `${Math.round(bytes / 1000)} KB ${suffix}`;
}

/**
 * Result of {@link computeModelView}; every caller copies out of it
 * synchronously. Created on first use, like the per-frame scratch.
 */
let modelViewScratch: THREE.Matrix4 | null = null;

/**
 * The model-view matrix to sort against, in a shared scratch (copy it before
 * the next call).
 *
 * Both matrices are normally renderer-maintained (updated during render), but
 * a commit can fire BEFORE the next frame — the first commit of a load, or
 * while the on-demand loop is idle-paused — and would otherwise read a
 * stale/identity pose, so both are refreshed here. The view is the plain
 * `inverse(camera.matrixWorld)`, the same one the per-frame trigger and the
 * frame's view snapshot use (three's `matrixWorldInverse`, which
 * `updateMatrixWorld` also refreshes, is built with the scale removed).
 */
export function computeModelView(mesh: THREE.Mesh, camera: THREE.Camera): THREE.Matrix4 {
  mesh.updateWorldMatrix(true, false);
  camera.updateMatrixWorld();
  modelViewScratch ??= new THREE.Matrix4();
  return modelViewScratch.copy(camera.matrixWorld).invert().multiply(mesh.matrixWorld);
}

/**
 * Sort a just-committed node's ordering RIGHT NOW, on the main thread, so the
 * very first frame after the commit is already ordered.
 *
 * Returns the resolved centers buffer when the sort landed (so the caller can
 * hand the SAME copy to the worker registration instead of paying the lazy
 * provider twice), or `undefined` when the synchronous path was declined — in
 * which case the caller must behave exactly as it did before.
 *
 * WHY this exists. Every commit of an order-dependent node writes a fallback
 * ordering and then waits for the worker's answer, which is a Comlink round
 * trip through `ensureWorker` + `registerNode` + `sort` — measured at ~5 ms,
 * so at least one frame renders on the fallback. On a one-off load that frame
 * is invisible. During nD playback it is not, because a commit lands at EVERY
 * timepoint: on the `cloud` demo (4D Points, `volumetric` blending) the
 * fallback was reached 14 times a second, and it composited only 61.7% of
 * sampled element pairs in correct back-to-front order against 100% for a real
 * sort. That is the flash reported in #2290.
 *
 * WHY it is bounded. The kernel is a counting sort — two O(n) passes and a
 * 65,536-bucket histogram — so its cost is linear and measurable rather than
 * data-dependent: 0.8 ms at 34k elements, 2.5 ms at 250k, 16.2 ms at 1M.
 * `config.depthSort.syncSortMaxElements` is both the per-node ceiling and the
 * shared element budget for every commit between frame evaluations; once spent,
 * the async path stays the only sane answer. `repairSortedIndexForCount` (the
 * commit paths' fallback) keeps that one frame far closer to sorted than storage
 * order was.
 *
 * The result is written LIVE rather than staged, because a permutation
 * computed in one shot has no partial state to hide — see
 * {@link writeSortedIndexOrderingLive}. A MESH (`triangleSource` present)
 * sorts its face centroids the same way and writes the permuted triples into
 * its index buffer (`writeSortedTriangleOrdering`, atomic by construction):
 * its commit rewrites the index in canonical order on every slice move, so
 * without this each timepoint drew one storage-order frame.
 *
 * The async pipeline behind this is deliberately left ALONE: the node is still
 * registered and still dispatches its usual first sort. Suppressing that would
 * save a redundant sort, but the kernels are exact-parity so the worker's
 * answer is the SAME permutation, landing as a no-op overwrite — not worth
 * changing the coordinator's dispatch contract for. The whole of this
 * function's job is to make the FIRST frame correct; the pipeline's job, of
 * keeping the ordering current as the camera moves, is unchanged.
 */
export function trySynchronousFirstSort(
  c: CoordinatorState,
  mesh: THREE.Mesh,
  centers3: Float32Array | (() => Float32Array),
  count: number,
  triangleSource: Uint32Array | undefined
): Float32Array | undefined {
  const limit = syncSortElementLimit();
  if (limit <= 0 || count > limit || count > c.syncSortElementsRemaining) return undefined;
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  // A mesh applies through its index buffer instead of `aSortedIndex`.
  if (triangleSource ? !geometry.index : !getActiveSortedIndexAttribute(geometry)) {
    return undefined;
  }
  const camera = c.getCamera?.();
  if (!camera) return undefined;

  let buffer: Float32Array;
  try {
    buffer = typeof centers3 === 'function' ? centers3() : centers3;
  } catch {
    // A throwing lazy provider must land where it always did: the async
    // path's outer `.catch`, which owns the once-per-episode report. Decline
    // and let it be called again there.
    return undefined;
  }
  // A provider that under-delivers (or a detached buffer) is not sortable —
  // decline rather than sort garbage. The async path cannot decline up front
  // (the worker clamps to the centers it got); its resolve rejects an ordering
  // whose length is not the committed count instead (orderingCoversCommit).
  if (buffer.length < count * 3) return undefined;

  const modelView = computeModelView(mesh, camera);
  const ordering = new Uint32Array(count);
  sort_splats_by_depth(buffer, new Float32Array(modelView.elements), ordering, count);
  const written = triangleSource
    ? writeSortedTriangleOrdering(geometry, triangleSource, ordering, count) === count * 3
    : writeSortedIndexOrderingLive(geometry, ordering, count) === count;
  if (!written) return undefined;
  c.syncSortElementsRemaining -= count;
  c.requestRender?.();
  return buffer;
}
