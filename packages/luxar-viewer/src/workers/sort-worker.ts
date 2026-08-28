/**
 * Depth-sort worker — entry point bundled by Vite's `?worker` import
 * (depth-sorting Phase 2, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5).
 *
 * A single PERSISTENT Comlink worker (NOT part of the round-robin data
 * worker pool — spec §2.1 pins sorting to one dedicated worker so node
 * registrations and their transferred center buffers live in exactly one
 * place). Ongoing and camera-driven sorting lives exclusively here,
 * never inside projection (the plain-3D projection fast path doesn't
 * run in a worker at all). The first ordering after an eligible
 * instanced commit may also be computed on the main thread within the
 * shared syncSortMaxElements frame budget; async registration and all
 * subsequent sorting remain here.
 *
 * Task bodies live in `./sort-worker/<file>.ts` and receive the shared
 * `state: SortWorkerCtx` from `./sort-worker/state`. This module's only
 * job is to wire the task functions into `workerAPI` and `expose()` it —
 * the same layout as `data-worker.ts`.
 *
 * Main-thread lifecycle (spawn, single-in-flight-per-node, ordering
 * application, disposal) lives in `rendering/depth-sort-coordinator.ts`.
 */

import { expose } from 'comlink';
import { state } from './sort-worker/state';
import { initialize as initializeImpl, type SortWorkerInitResult } from './sort-worker/initialize';
import {
  registerNode as registerNodeImpl,
  sortNode as sortNodeImpl,
  releaseNode as releaseNodeImpl,
  releaseAllNodes as releaseAllNodesImpl,
  type RegisterNodeParams,
  type SortParams,
  type SortResult,
} from './sort-worker/sorting';

export type { SortWorkerInitResult, RegisterNodeParams, SortParams, SortResult };

/**
 * Comlink RPC surface. Each entry binds `state` to its task helper.
 */
export const workerAPI = {
  // `wasmPath` forwards the embedder's LuxarAppOptions.wasmPath into the
  // worker module scope (the main-thread setWasmJsUrl override does NOT
  // cross into the worker), exactly like the data worker.
  initialize: (wasmPath?: string): Promise<SortWorkerInitResult> => initializeImpl(state, wasmPath),
  registerNode: (p: RegisterNodeParams): void => registerNodeImpl(state, p),
  sort: (p: SortParams): SortResult | null => sortNodeImpl(state, p),
  releaseNode: (nodeId: string): void => releaseNodeImpl(state, nodeId),
  releaseAllNodes: (): void => releaseAllNodesImpl(state),
};

expose(workerAPI);

export type SortWorkerAPI = typeof workerAPI;
