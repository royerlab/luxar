/**
 * SortWorker-local mutable state (depth-sorting Phase 2, spec §5).
 *
 * Mirrors the data-worker's `WasmCtx` idiom: tasks receive the shared
 * `state` object and read `state.wasm` after `initialize()` set it.
 * Additionally holds the per-node center registry — commit-time
 * registrations transfer each order-dependent node's projected 3D
 * centers here so camera-driven re-sorts (Phase 3) never re-copy them
 * from the main thread.
 *
 * The depth-sort kernel is ndim-agnostic (input is always projected 3D
 * centers), so there is no `pickBackend` / 16-dimension routing here —
 * `initWasm()`'s compiled-or-TS-fallback result is used directly.
 */

import type { WasmModule } from '../../wasm';

/** A node's registered sort inputs, keyed by generation. */
export interface RegisteredNode {
  /**
   * Per-node monotonic non-noop commit counter (NOT `loadedViewVersion` —
   * ladder appends share a view version while the splat count grows). An
   * ordering may only ever be applied to the exact commit it was
   * computed for; see the spec §5 generation contract.
   */
  generation: number;
  /** Projected 3D centers [count * 3], transferred from the main thread. */
  centers3: Float32Array;
  /** Number of splats. */
  count: number;
}

export interface SortWorkerCtx {
  wasm: WasmModule | null;
  /** Registered order-dependent nodes by node id (mesh UUID). */
  nodes: Map<string, RegisteredNode>;
}

export const state: SortWorkerCtx = {
  wasm: null,
  nodes: new Map(),
};

/** Single source of truth for the "task called before initialize()" error. */
export const NOT_INITIALIZED_MSG = '[SortWorker] Not initialized - call initialize() first';

export function requireWasm(ctx: SortWorkerCtx): WasmModule {
  if (!ctx.wasm) throw new Error(NOT_INITIALIZED_MSG);
  return ctx.wasm;
}
