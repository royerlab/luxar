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
  // Timing wraps the backend call generically: `ctx.wasm` is either the
  // compiled WASM module or the TypeScript fallback, so both report.
  const kernelStart = performance.now();
  wasm.sort_splats_by_depth(node.centers3, params.modelView, ordering, node.count);
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
  ctx.nodes.delete(nodeId);
}

/** Drop every registration (dataset switch / app teardown). */
export function releaseAllNodes(ctx: SortWorkerCtx): void {
  ctx.nodes.clear();
}
