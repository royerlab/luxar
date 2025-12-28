/**
 * Polling Loop for periodic task execution.
 *
 * Manages a configurable interval timer that executes a callback
 * on each tick. Supports start/stop/pause operations and provides
 * statistics about execution.
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

export interface PollingLoopStats {
  /** Number of ticks executed since start */
  tickCount: number;
  /** Timestamp of last tick */
  lastTickTime: number;
  /** Average time between ticks (ms) */
  avgInterval: number;
  /** Whether the loop is currently running */
  isRunning: boolean;
}

export class PollingLoop {
  private config: PollingLoopConfig;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private tickCount = 0;
  private lastTickTime = 0;
  private intervalSum = 0;

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
    this.tickCount = 0;
    this.intervalSum = 0;
    this.lastTickTime = Date.now();

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
   * Check if the loop is currently running.
   */
  isRunning(): boolean {
    return this.running;
  }

  /**
   * Get the current polling interval.
   */
  getInterval(): number {
    return this.config.interval;
  }

  /**
   * Update the polling interval.
   * Takes effect on the next tick.
   */
  setInterval(interval: number): void {
    this.config.interval = interval;
  }

  /**
   * Get loop statistics.
   */
  getStats(): PollingLoopStats {
    return {
      tickCount: this.tickCount,
      lastTickTime: this.lastTickTime,
      avgInterval: this.tickCount > 0 ? this.intervalSum / this.tickCount : 0,
      isRunning: this.running,
    };
  }

  /**
   * Reset statistics without stopping the loop.
   */
  resetStats(): void {
    this.tickCount = 0;
    this.intervalSum = 0;
  }

  /**
   * Force an immediate tick (useful for testing or urgent updates).
   * Does not affect the regular tick schedule.
   */
  tickNow(): void {
    this.executeTick();
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
    const now = Date.now();
    if (this.lastTickTime > 0) {
      this.intervalSum += now - this.lastTickTime;
    }
    this.lastTickTime = now;
    this.tickCount++;

    try {
      this.config.onTick();
    } catch (error) {
      // Don't let tick errors stop the loop
      console.error('[PollingLoop] Error in onTick callback:', error);
    }
  }
}
