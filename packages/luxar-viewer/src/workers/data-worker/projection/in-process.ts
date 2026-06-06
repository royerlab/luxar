/**
 * Main-thread (in-process) projection dispatcher.
 *
 * The worker dispatchers in this directory (`gsplats.ts`, `lines.ts`,
 * `points.ts`) take a {@link WasmCtx} and run the WASM kernels. They are
 * normally invoked across the Comlink RPC boundary from the worker pool,
 * but the exact same functions can run on the main thread against a
 * locally-initialized `WasmCtx`. This module provides that path.
 *
 * It serves two callers, replacing the deleted main-thread projection
 * copies (`data/gsplats/projection.ts`, `data/lines/projection.ts`):
 *
 *   1. **`useWebWorkers === false` / below-threshold data** — the
 *      data-processors call the in-process dispatcher directly instead
 *      of a separate hand-written main-thread implementation.
 *   2. **Worker-infrastructure failure** — when a worker RPC throws for
 *      a reason other than a dataset-switch abort, the data-processors
 *      degrade to this in-process path rather than crashing.
 *
 * Both callers go through the *same* kernel code the worker runs, so
 * there is exactly one projection implementation per geometry — no
 * second copy to keep in sync.
 *
 * The Comlink `transfer()` markers the dispatchers attach to their
 * return values are inert here: nothing is posted through a MessagePort,
 * so the returned typed arrays stay fully usable on the main thread.
 *
 * @module workers/data-worker/projection/in-process
 */

import { initWasm, isWasmFallback, getFallback } from '../../../wasm';
import { pickBackend, type WasmCtx } from '../state';
import type { WasmModule } from '../../../wasm/types';
import { projectGSplatsTo3D } from './gsplats';
import { projectLinesTo3D } from './lines';

/** Lazily-built main-thread WASM context (mirrors the worker's `state`). */
let ctx: WasmCtx | null = null;
let ctxPromise: Promise<WasmCtx> | null = null;

/**
 * Build (once) the main-thread `WasmCtx`. Mirrors the worker's
 * `initialize()`: load compiled WASM (with TS fallback), and set the
 * uncapped TS reference used for `ndim > MAX_SUPPORTED_DIMS`.
 */
async function getInProcessCtx(): Promise<WasmCtx> {
  if (ctx) return ctx;
  if (!ctxPromise) {
    ctxPromise = (async () => {
      const wasm = await initWasm();
      const built: WasmCtx = {
        wasm,
        tsFallback: isWasmFallback(wasm) ? wasm : getFallback(),
        visibilityMaskBuffer: null,
      };
      ctx = built;
      return built;
    })();
  }
  return ctxPromise;
}

/** Project GSplats nD → 3D on the main thread via the shared dispatcher. */
export async function projectGSplatsInProcess(
  params: Parameters<typeof projectGSplatsTo3D>[1]
): Promise<Awaited<ReturnType<typeof projectGSplatsTo3D>>> {
  const c = await getInProcessCtx();
  return projectGSplatsTo3D(c, params);
}

/** Project Lines nD → 3D on the main thread via the shared dispatcher. */
export async function projectLinesInProcess(
  params: Parameters<typeof projectLinesTo3D>[1]
): Promise<Awaited<ReturnType<typeof projectLinesTo3D>>> {
  const c = await getInProcessCtx();
  return projectLinesTo3D(c, params);
}

/**
 * Resolve the WASM backend for the main-thread Points projection.
 *
 * Points projection is bandwidth-bound and pairs with a zero-allocation
 * accumulator, so — unlike GSplats/Lines — it runs on the main thread
 * rather than a worker (offloading a memory-bound op would pay transfer
 * cost both ways for negligible compute savings). It still uses the WASM
 * kernels: `data/points/projection.ts` calls `extract_3d_positions` /
 * `calculate_effective_radii` on the module returned here, with the
 * uncapped TS reference selected for `ndim > MAX_SUPPORTED_DIMS` (the
 * same {@link pickBackend} routing the worker dispatchers use).
 */
export async function getPointsBackend(ndim: number): Promise<WasmModule> {
  const c = await getInProcessCtx();
  return pickBackend(c, ndim);
}

/**
 * Reset the cached context. Test-only hook so a suite can force a fresh
 * `initWasm()` (e.g. after swapping the module mock). Not used in
 * production.
 */
export function __resetInProcessCtxForTests(): void {
  ctx = null;
  ctxPromise = null;
}
