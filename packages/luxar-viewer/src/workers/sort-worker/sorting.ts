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
}

export interface SortResult {
  /** Generation the ordering was computed for (re-checked on the main thread). */
  generation: number;
  /** Back-to-front permutation [count] — TRANSFERRED to the main thread. */
  ordering: Uint32Array;
  /**
   * Time spent inside the backend calls (`sorter.sort` + the
   * `read_ordering_into` readback), in ms. Since the sorter state became
   * backend-resident (perf lever L3) the only per-sort boundary traffic
   * left is the 64-byte model-view copy-in and the 4 B/splat ordering
   * memcpy-out — the per-sort centers copy-in (12 B/splat), the
   * output-ordering copy-in, and the 3 per-call mallocs are gone.
   * Measured with `performance.now()` on the worker's clock.
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
}

/**
 * Register (or refresh) a node's sort inputs. Called after every
 * non-noop commit of an order-dependent node; replaces any previous
 * registration wholesale. Constructs the backend-resident sorter here —
 * the one boundary copy of the centers; the transferred `centers3`
 * buffer is dropped afterwards (for compiled WASM the centers now live
 * in wasm linear memory).
 *
 * Requires an initialized backend (the coordinator always awaits the
 * worker's `initialize()` before its first register RPC).
 */
export function registerNode(ctx: SortWorkerCtx, params: RegisterNodeParams): void {
  const wasm = requireWasm(ctx);
  const { nodeId, generation, centers3, count } = params;
  const safeCount = Math.min(count, Math.floor(centers3.length / 3));
  if (safeCount < count) {
    log.warning(
      Modules.WORKER_POOL,
      `[SortWorker] registerNode(${nodeId}): count ${count} exceeds centers3 capacity; ` +
        `clamping to ${safeCount}`
    );
  }
  // Construct the replacement BEFORE freeing any old sorter: if
  // construction throws (e.g. wasm OOM), the previous registration stays
  // intact and usable (the stale-generation guard already rejects sorts
  // issued for this failed generation).
  const sorter = wasm.create_depth_sorter(centers3, safeCount);
  const previous = ctx.nodes.get(nodeId);
  ctx.nodes.set(nodeId, { generation, sorter, count: safeCount });
  // CRITICAL: free the replaced sorter (generation bump re-registration).
  // Unlike a GC'd Float32Array, a wasm-resident sorter leaks its linear
  // memory (~20 B/splat — MatrixCity-scale scenes hold >180 MB of
  // centers) unless explicitly freed.
  previous?.sorter.free();
}

/**
 * Compute the back-to-front ordering for a registered node under the
 * given model-view. Returns `null` when the request is stale (the node
 * was re-registered with a newer generation) or the node is unknown
 * (released, or never order-dependent).
 */
export function sortNode(ctx: SortWorkerCtx, params: SortParams): SortResult | null {
  const bodyStart = performance.now();
  // Contract guard only (the sorter handle does the work): keeps the
  // "task called before initialize()" error uniform across tasks.
  requireWasm(ctx);
  const node = ctx.nodes.get(params.nodeId);
  if (!node || node.generation !== params.generation) {
    return null;
  }

  const ordering = new Uint32Array(node.count);
  // Timing wraps the backend calls generically: the sorter is either the
  // wasm-bindgen class or the TS twin, so both report. The readback is
  // part of kernelMs — it is the one remaining per-sort boundary copy
  // (4 B/splat into the freshly allocated `ordering`, which is then
  // TRANSFERRED to the main thread).
  const kernelStart = performance.now();
  node.sorter.sort(params.modelView);
  node.sorter.read_ordering_into(ordering);
  const kernelEnd = performance.now();
  return transfer(
    {
      generation: node.generation,
      ordering,
      kernelMs: kernelEnd - kernelStart,
      workerMs: kernelEnd - bodyStart,
    },
    [ordering.buffer]
  );
}

/** Drop a node's registration (node disposal / pool release). */
export function releaseNode(ctx: SortWorkerCtx, nodeId: string): void {
  const node = ctx.nodes.get(nodeId);
  if (!node) return;
  node.sorter.free();
  ctx.nodes.delete(nodeId);
}

/** Drop every registration (dataset switch / app teardown). */
export function releaseAllNodes(ctx: SortWorkerCtx): void {
  for (const node of ctx.nodes.values()) {
    node.sorter.free();
  }
  ctx.nodes.clear();
}
