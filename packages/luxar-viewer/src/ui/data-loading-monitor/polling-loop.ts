/**
 * Polling Loop for periodic task execution.
 *
 * Manages a configurable interval timer that executes a callback
 * on each tick. Supports start/stop operations.
 *
 * @example
 * ```typescript
 * const loop = new PollingLoop({
 *   interval: 100,
 *   onTick: () => {
 *     // Process events, update UI, etc.
 *   }
 * });
 *
 * loop.start();
 * // ... later
 * loop.stop();
 * ```
 */

import { log, Modules } from '../../utils/log';

export interface PollingLoopConfig {
  /** Interval between ticks in milliseconds */
  interval: number;
  /** Callback executed on each tick */
  onTick: () => void;
  /** Optional callback when loop starts */
  onStart?: () => void;
  /** Optional callback when loop stops */
  onStop?: () => void;
}

export class PollingLoop {
  private config: PollingLoopConfig;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;

  constructor(config: PollingLoopConfig) {
    this.config = config;
  }

  /**
   * Start the polling loop.
   * If already running, this is a no-op.
   */
  start(): void {
    if (this.running) return;

    this.running = true;

    this.config.onStart?.();
    this.scheduleNextTick();
  }

  /**
   * Stop the polling loop.
   * Cancels any pending tick.
   */
  stop(): void {
    if (!this.running) return;

    this.running = false;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    this.config.onStop?.();
  }

  /**
   * Schedule the next tick.
   */
  private scheduleNextTick(): void {
    if (!this.running) return;

    this.timer = setTimeout(() => {
      this.executeTick();
      this.scheduleNextTick();
    }, this.config.interval);
  }

  /**
   * Execute a single tick.
   */
  private executeTick(): void {
    try {
      this.config.onTick();
    } catch (error) {
      // Don't let tick errors stop the loop
      log.error(Modules.UI, 'PollingLoop error in onTick callback', error);
    }
  }
}
