/**
 * SortWorker task bodies: node registration, depth sorting, release.
 *
 * The generation contract (spec §5): `generation` is a per-node monotonic
 * counter bumped by the main thread on every NON-noop commit. A sort
 * request carries the generation it was issued for; if the node has been
 * re-registered since (newer generation), the request is stale and
 * returns `null` — a shorter stale permutation applied to a grown buffer
 * would be corrupt, not just outdated. Stamp-only noop commits leave the
 * generation unchanged, so in-flight sorts stay valid across them.
 */

import { transfer } from 'comlink';
import { requireWasm, type SortWorkerCtx } from './state';
import { log, Modules } from '../../utils/log';

export interface RegisterNodeParams {
  /** Node identity (mesh UUID on the main thread). */
  nodeId: string;
  /** Per-node monotonic non-noop commit counter. */
  generation: number;
  /** Projected 3D centers [count * 3] — TRANSFERRED, not cloned. */
  centers3: Float32Array;
  /** Number of splats. */
  count: number;
}

export interface SortParams {
  nodeId: string;
  /** Generation this request was issued for (stale-drop guard). */
  generation: number;
  /** Column-major 4x4 model-view matrix [16]. */
  modelView: Float32Array;
  /**
   * Contiguous equal-population ranges of the ordering to report local-space
   * AABBs for — the node's shard count for cross-node depth ordering
   * (`docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md` §3.3). Omitted or 0
   * means "don't", which is the unsharded default and costs nothing.
   */
  shardCount?: number;
}

export interface SortResult {
  /** Generation the ordering was computed for (re-checked on the main thread). */
  generation: number;
  /** Back-to-front permutation [count] — TRANSFERRED to the main thread. */
  ordering: Uint32Array;
  /**
   * Time spent inside the backend `sort_splats_by_depth` call, in ms.
   * For compiled WASM this INCLUDES the wasm-bindgen boundary copies
   * (centers copy-in, ordering copy-in/out, mallocs) — the shim performs
   * them inside the exported function; for the TS fallback it is the pure
   * kernel. Measured with `performance.now()` on the worker's clock.
   */
  kernelMs: number;
  /**
   * Whole `sortNode` body duration in ms (registry lookup + output
   * allocation + kernel). `workerMs - kernelMs` is the worker-side
   * overhead around the backend call. Same clock as `kernelMs`, so the
   * difference is meaningful; both are durations, safe to compare with
   * main-thread-measured durations.
   */
  workerMs: number;
  /**
   * Number of shard AABBs reported in `shardBoundsMin` / `shardBoundsMax`.
   *
   * **`0` means the bounds carry no depth meaning and the node must be merged
   * as ONE whole-node interval** — either none were requested, or the kernel
   * took its identity-ordering fallback (degenerate depth range: a single depth
   * plane, ≤1 in-front element, everything behind the camera, or NaN centers).
   * Collapsing those two cases is deliberate: both mean "do not treat this
   * node's ranges as depth intervals", which is exactly today's behaviour.
   */
  shardCount: number;
  /**
   * Per-shard local-space AABB minima `[shardCount * 3]`, TRANSFERRED. Absent
   * when `shardCount === 0`. A shard with no finite element on an axis carries
   * the empty sentinel (`min = +Infinity`, `max = -Infinity`) on that axis, so
   * `min > max` is the caller's "no usable bounds" test.
   */
  shardBoundsMin?: Float32Array;
  /** Per-shard local-space AABB maxima `[shardCount * 3]`, TRANSFERRED. */
  shardBoundsMax?: Float32Array;
  /**
   * Per-shard VIEW-space z interval `[shardCount]` each, at THIS sort's pose —
   * the cross-node merge key. TRANSFERRED; absent when `shardCount === 0`.
   *
   * Deliberately a sort-pose quantity rather than something the main thread
   * re-derives per frame: a shard is a slab perpendicular to the view axis at
   * this pose, so the GROUPING is pose-dependent and a freshly re-projected key
   * over a stale grouping is incoherent. The key is held exactly as long as the
   * grouping it describes. See the Rust kernel's `write_shard_bounds`, including
   * its note on what this did and did not fix.
   */
  shardViewZMin?: Float32Array;
  shardViewZMax?: Float32Array;
}

/**
 * Register (or refresh) a node's sort inputs. Called after every
 * non-noop commit of an order-dependent node; replaces any previous
 * registration wholesale.
 */
export function registerNode(ctx: SortWorkerCtx, params: RegisterNodeParams): void {
  const { nodeId, generation, centers3, count } = params;
  const safeCount = Math.min(count, Math.floor(centers3.length / 3));
  if (safeCount < count) {
    log.warning(
      Modules.WORKER_POOL,
      `[SortWorker] registerNode(${nodeId}): count ${count} exceeds centers3 capacity; ` +
        `clamping to ${safeCount}`
    );
  }
  ctx.nodes.set(nodeId, { generation, centers3, count: safeCount });
}

/**
 * Compute the back-to-front ordering for a registered node under the
 * given model-view. Returns `null` when the request is stale (the node
 * was re-registered with a newer generation) or the node is unknown
 * (released, or never order-dependent).
 */
export function sortNode(ctx: SortWorkerCtx, params: SortParams): SortResult | null {
  const bodyStart = performance.now();
  const wasm = requireWasm(ctx);
  const node = ctx.nodes.get(params.nodeId);
  if (!node || node.generation !== params.generation) {
    return null;
  }

  const ordering = new Uint32Array(node.count);
  // A shard count above the element count would allocate boxes no shard can
  // ever fill; clamp so the outputs stay meaningful (the kernel already leaves
  // surplus shards at the empty sentinel, this just avoids the waste).
  const requestedShards = Math.max(0, Math.trunc(params.shardCount ?? 0));
  const shardCount = Math.min(requestedShards, node.count);
  const shardBoundsMin = new Float32Array(shardCount * 3);
  const shardBoundsMax = new Float32Array(shardCount * 3);
  const shardViewZMin = new Float32Array(shardCount);
  const shardViewZMax = new Float32Array(shardCount);

  // Timing wraps the backend call generically: `ctx.wasm` is either the
  // compiled WASM module or the TypeScript fallback, so both report.
  const kernelStart = performance.now();
  const placed = wasm.sort_splats_by_depth(
    node.centers3,
    params.modelView,
    ordering,
    node.count,
    shardCount,
    shardBoundsMin,
    shardBoundsMax,
    shardViewZMin,
    shardViewZMax
  );
  const kernelEnd = performance.now();

  // `placed === 0` is the identity-ordering fallback: the ordering is fully
  // written but the shards are storage ranges, not depth intervals, so report
  // no shards at all rather than bounds the merge would misread as depth.
  const usableShards = placed === 0 ? 0 : shardCount;
  const transferables: Transferable[] = [ordering.buffer];
  if (usableShards > 0) {
    transferables.push(
      shardBoundsMin.buffer,
      shardBoundsMax.buffer,
      shardViewZMin.buffer,
      shardViewZMax.buffer
    );
  }

  return transfer(
    {
      generation: node.generation,
      ordering,
      kernelMs: kernelEnd - kernelStart,
      workerMs: kernelEnd - bodyStart,
      shardCount: usableShards,
      ...(usableShards > 0 ? { shardBoundsMin, shardBoundsMax, shardViewZMin, shardViewZMax } : {}),
    },
    transferables
  );
}

/** Drop a node's registration (node disposal / pool release). */
export function releaseNode(ctx: SortWorkerCtx, nodeId: string): void {
  ctx.nodes.delete(nodeId);
}

/** Drop every registration (dataset switch / app teardown). */
export function releaseAllNodes(ctx: SortWorkerCtx): void {
  ctx.nodes.clear();
}
