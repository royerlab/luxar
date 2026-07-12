/**
 * Data processing worker — entry point bundled by Vite's `?worker` import.
 *
 * Loads the compiled WASM module (with TypeScript fallback) once at startup,
 * then exposes a Comlink-RPC surface to the main thread: spatial-index queries,
 * nD→3D projection kernels, and array decoders.
 *
 * Worker responsibilities (CPU-heavy, offloaded from main thread):
 * - Spatial index queries (chunk bounding-box tests)
 * - nD→3D projection of Lines / GSplats (WASM batch kernels; per-element
 *   nD visibility/culling is computed INSIDE projection — clip mask,
 *   effective radius, attenuation — not as a standalone task)
 * - Array decoding (LUT, quantization, log-space)
 *
 * NOT handled here (stays on main thread):
 * - Zarr chunk fetching (needs caching store)
 * - Accumulator buffer management
 * - GPU buffer updates
 *
 * Task bodies live in `./data-worker/<concern>/<file>.ts` and receive the
 * shared `state: WasmCtx` from `./data-worker/state`. This module's only
 * job is to wire the task functions into `workerAPI` and `expose()` it.
 */

import { expose } from 'comlink';
import { state } from './data-worker/state';
import { initialize as initializeImpl, type WorkerInitResult } from './data-worker/initialize';
import { querySpatialIndex as querySpatialIndexImpl } from './data-worker/spatial-index/query';
import { projectLinesTo3D as projectLinesTo3DImpl } from './data-worker/projection/lines';
import { projectGSplatsTo3D as projectGSplatsTo3DImpl } from './data-worker/projection/gsplats';
import { decodeQuantized as decodeQuantizedImpl } from './data-worker/decode/quantized';
import { decodeLogScalar as decodeLogScalarImpl } from './data-worker/decode/log-scalar';
import { decodeGeologScalar as decodeGeologScalarImpl } from './data-worker/decode/geolog-scalar';
import { decodePerChannel as decodePerChannelImpl } from './data-worker/decode/perchannel';
import { decodeLUT as decodeLUTImpl } from './data-worker/decode/lut';
import { decodeBroadcasted as decodeBroadcastedImpl } from './data-worker/decode/broadcasted';

// Re-export the projection/effective-radius types that the loader needs.
// Points projection runs on the main thread (WASM-accelerated, see
// data/points/projection.ts), so the worker no longer exposes a Points
// projection task — only Lines and GSplats (worker-offloaded for large data).
import type { EffectiveRadiusConfig, ProjectionViewState } from './data-worker/types';
export type { EffectiveRadiusConfig, ProjectionViewState };
export type { WorkerInitResult };

/**
 * Comlink RPC surface. Each entry binds `state` to its task helper.
 */
export const workerAPI = {
  // `wasmPath` forwards the embedder's LuxarAppOptions.wasmPath into the worker
  // module scope (the main-thread setWasmJsUrl override does NOT cross into the
  // worker), so a relocated WASM binary is loaded here instead of silently
  // falling back to the slower TS implementation.
  initialize: (wasmPath?: string): Promise<WorkerInitResult> => initializeImpl(state, wasmPath),
  querySpatialIndex: (p: Parameters<typeof querySpatialIndexImpl>[1]) =>
    querySpatialIndexImpl(state, p),
  // Decoding functions (main thread fetches, worker decodes)
  decodeQuantized: (p: Parameters<typeof decodeQuantizedImpl>[1]) => decodeQuantizedImpl(state, p),
  decodeLogScalar: (p: Parameters<typeof decodeLogScalarImpl>[1]) => decodeLogScalarImpl(state, p),
  decodeGeologScalar: (p: Parameters<typeof decodeGeologScalarImpl>[1]) =>
    decodeGeologScalarImpl(state, p),
  decodePerChannel: (p: Parameters<typeof decodePerChannelImpl>[1]) =>
    decodePerChannelImpl(state, p),
  decodeLUT: (p: Parameters<typeof decodeLUTImpl>[1]) => decodeLUTImpl(state, p),
  decodeBroadcasted: (p: Parameters<typeof decodeBroadcastedImpl>[1]) =>
    decodeBroadcastedImpl(state, p),
  // Projection functions (nD → 3D, CPU-intensive). Points project on the
  // main thread (WASM + zero-alloc accumulator), so only Lines/GSplats here.
  projectLinesTo3D: (p: Parameters<typeof projectLinesTo3DImpl>[1]) =>
    projectLinesTo3DImpl(state, p),
  projectGSplatsTo3D: (p: Parameters<typeof projectGSplatsTo3DImpl>[1]) =>
    projectGSplatsTo3DImpl(state, p),
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;
