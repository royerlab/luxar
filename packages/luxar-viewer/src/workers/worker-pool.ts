/**
 * Worker pool manager for data processing workers
 *
 * Supports multiple workers with round-robin load balancing for parallel
 * spatial queries across multiple nodes.
 */

import { wrap, Remote } from 'comlink';
import type { DataWorkerAPI } from './data-worker';
import { log, Modules } from '../utils/log';
import { config } from '../config';

interface WorkerInstance {
  worker: Worker;
  api: Remote<DataWorkerAPI>;
  activeQueries: number;
}

class WorkerPool {
  private workers: WorkerInstance[] = [];
  private initPromise: Promise<void> | null = null;
  private initLock = false;

  /**
   * Get the configured worker count, capped by hardware concurrency
   *
   * - workerCount = 0: Auto mode, uses (hardwareConcurrency - 1)
   * - workerCount > 0: Uses that number, capped at (hardwareConcurrency - 1)
   */
  private getConfiguredWorkerCount(): number {
    const configCount = config.dataLoading.performance.workerCount;
    const hardwareConcurrency =
      typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;

    // Leave one core for main thread (rendering, UI)
    const maxWorkers = Math.max(1, hardwareConcurrency - 1);

    // 0 = auto mode: use all available cores minus one
    if (configCount <= 0) {
      return maxWorkers;
    }

    // Otherwise use config value, capped at max
    return Math.min(configCount, maxWorkers);
  }

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
        const workerCount = this.getConfiguredWorkerCount();
        log.info(Modules.WORKER_POOL, `Creating ${workerCount} data worker(s)...`);

        // Create all workers in parallel
        const workerPromises = Array.from({ length: workerCount }, async (_, index) => {
          const worker = new Worker(new URL('./data-worker.ts', import.meta.url), {
            type: 'module',
          });

          const api = wrap<DataWorkerAPI>(worker);

          try {
            await api.initialize();
            log.info(Modules.WORKER_POOL, `Worker ${index + 1}/${workerCount} ready`);
            return { worker, api, activeQueries: 0 };
          } catch (error) {
            console.error(`[WorkerPool] Worker ${index + 1} initialization FAILED:`, error);
            worker.terminate();
            throw error;
          }
        });

        // Wait for all workers to initialize
        const results = await Promise.allSettled(workerPromises);

        // Collect successful workers
        for (const result of results) {
          if (result.status === 'fulfilled') {
            this.workers.push(result.value);
          }
        }

        if (this.workers.length === 0) {
          throw new Error(
            'Failed to initialize any data workers. ' +
              'Luxar requires WebAssembly and Web Workers support.'
          );
        }

        if (this.workers.length < workerCount) {
          log.warning(
            Modules.WORKER_POOL,
            `Only ${this.workers.length}/${workerCount} workers initialized successfully`
          );
        }

        log.info(Modules.WORKER_POOL, `Worker pool ready with ${this.workers.length} worker(s)`);
      } finally {
        this.initLock = false;
      }
    })();

    return this.initPromise;
  }

  /**
   * Get a worker API using least-busy selection
   *
   * Selects the worker with the fewest active queries for better load distribution.
   */
  async getWorker(): Promise<Remote<DataWorkerAPI>> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    // Single worker - fast path
    if (this.workers.length === 1) {
      return this.workers[0].api;
    }

    // Find worker with least active queries (least-busy selection)
    let leastBusyIndex = 0;
    let minQueries = this.workers[0].activeQueries;

    for (let i = 1; i < this.workers.length; i++) {
      if (this.workers[i].activeQueries < minQueries) {
        minQueries = this.workers[i].activeQueries;
        leastBusyIndex = i;
      }
    }

    return this.workers[leastBusyIndex].api;
  }

  /**
   * Get a worker with query tracking for load balancing
   *
   * Returns the worker API and callbacks to mark query start/end.
   * This enables accurate load balancing across workers.
   */
  async getWorkerWithTracking(): Promise<{
    api: Remote<DataWorkerAPI>;
    markQueryStart: () => void;
    markQueryEnd: () => void;
  }> {
    await this.initialize();

    if (this.workers.length === 0) {
      throw new Error('[WorkerPool] No workers available after initialization');
    }

    // Find worker with least active queries
    let leastBusyIndex = 0;
    let minQueries = this.workers[0].activeQueries;

    for (let i = 1; i < this.workers.length; i++) {
      if (this.workers[i].activeQueries < minQueries) {
        minQueries = this.workers[i].activeQueries;
        leastBusyIndex = i;
      }
    }

    const workerInstance = this.workers[leastBusyIndex];

    return {
      api: workerInstance.api,
      markQueryStart: () => {
        workerInstance.activeQueries++;
      },
      markQueryEnd: () => {
        workerInstance.activeQueries = Math.max(0, workerInstance.activeQueries - 1);
      },
    };
  }

  /**
   * Get the number of active workers
   */
  getWorkerCount(): number {
    return this.workers.length;
  }

  /**
   * Get pool statistics for monitoring
   */
  getStats(): { workerCount: number; activeQueries: number[] } {
    return {
      workerCount: this.workers.length,
      activeQueries: this.workers.map((w) => w.activeQueries),
    };
  }

  /**
   * Check if worker pool is initialized
   */
  isInitialized(): boolean {
    return this.workers.length > 0;
  }

  /**
   * Clean up all worker resources
   */
  dispose(): void {
    if (this.workers.length > 0) {
      log.info(Modules.WORKER_POOL, `Terminating ${this.workers.length} data worker(s)`);
      for (const { worker } of this.workers) {
        worker.terminate();
      }
      this.workers = [];
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
