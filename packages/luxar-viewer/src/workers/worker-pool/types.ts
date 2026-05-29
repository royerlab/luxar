/**
 * Pool-internal types shared between `worker-pool.ts` and its
 * helper modules under `worker-pool/`. Externally re-exported from
 * `worker-pool.ts` so consumers keep their existing import paths.
 */

import type { Remote } from 'comlink';
import type { DataWorkerAPI } from '../data-worker';

/**
 * One live data worker plus its Comlink-wrapped API surface and the
 * pool's load-balancing counter. The counter is mutated by
 * `markQueryStart` / `markQueryEnd` returned from
 * `WorkerPool.getWorkerWithTracking`.
 */
export interface WorkerInstance {
  worker: Worker;
  api: Remote<DataWorkerAPI>;
  activeQueries: number;
  /** True when this worker is on the TypeScript fallback (compiled WASM not loaded). */
  wasmFallback: boolean;
}
