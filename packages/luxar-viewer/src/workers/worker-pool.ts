/**
 * Worker pool manager for data processing workers
 *
 * Phase 2: Single worker implementation
 * Future: Multi-worker load balancing
 */

import { wrap, Remote } from 'comlink';
import type { DataWorkerAPI } from './data-worker';
import { log, Modules } from '../utils/log';

class WorkerPool {
  private worker: Worker | null = null;
  private workerAPI: Remote<DataWorkerAPI> | null = null;
  private initPromise: Promise<void> | null = null;
  private initLock = false;

  /**
   * Initialize worker pool (lazy initialization)
   */
  async initialize(): Promise<void> {
    // Atomic check-and-set to prevent race conditions
    if (this.initPromise) return this.initPromise;

    if (this.initLock) {
      // Another initialization in progress, wait for it
      while (this.initLock) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      return this.initPromise!;
    }

    this.initLock = true;

    this.initPromise = (async () => {
      try {
        log.info(Modules.WORKER_POOL, 'Creating data worker...');

        // Create worker using Vite's worker import syntax
        this.worker = new Worker(new URL('./data-worker.ts', import.meta.url), {
          type: 'module',
        });

        // Wrap with Comlink for RPC-style communication
        this.workerAPI = wrap<DataWorkerAPI>(this.worker);

        // Initialize worker (loads WASM module or TypeScript fallback)
        try {
          await this.workerAPI.initialize();
          log.info(Modules.WORKER_POOL, 'Data worker ready');
        } catch (error) {
          console.error('[WorkerPool] Worker initialization FAILED:', error);
          throw new Error(
            'Failed to initialize data worker. ' +
              'Luxar requires WebAssembly and Web Workers support.'
          );
        }
      } finally {
        this.initLock = false;
      }
    })();

    return this.initPromise;
  }

  /**
   * Get worker API (initializes if needed)
   */
  async getWorker(): Promise<Remote<DataWorkerAPI>> {
    await this.initialize();
    if (!this.workerAPI) {
      throw new Error('[WorkerPool] Worker not available after initialization');
    }
    return this.workerAPI;
  }

  /**
   * Check if worker is initialized
   */
  isInitialized(): boolean {
    return this.workerAPI !== null;
  }

  /**
   * Clean up worker resources
   */
  dispose(): void {
    if (this.worker) {
      log.info(Modules.WORKER_POOL, 'Terminating data worker');
      this.worker.terminate();
      this.worker = null;
      this.workerAPI = null;
      this.initPromise = null;
      this.initLock = false;
    }
  }
}

// Singleton instance
let workerPoolInstance: WorkerPool | null = null;

/**
 * Get the global worker pool instance
 */
export function getWorkerPool(): WorkerPool {
  if (!workerPoolInstance) {
    workerPoolInstance = new WorkerPool();
  }
  return workerPoolInstance;
}

/**
 * Dispose the global worker pool (for testing/cleanup)
 */
export function disposeWorkerPool(): void {
  if (workerPoolInstance) {
    workerPoolInstance.dispose();
    workerPoolInstance = null;
  }
}
