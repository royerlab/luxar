/**
 * Unit tests for PollingLoop
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { PollingLoop } from '../../../ui/data-loading-monitor/polling-loop';

describe('PollingLoop', () => {
  let loop: PollingLoop;
  let tickCount: number;
  let onTickMock: ReturnType<typeof vi.fn<() => void>>;
  let onStartMock: ReturnType<typeof vi.fn<() => void>>;
  let onStopMock: ReturnType<typeof vi.fn<() => void>>;

  beforeEach(() => {
    vi.useFakeTimers();
    tickCount = 0;
    onTickMock = vi.fn(() => {
      tickCount++;
    });
    onStartMock = vi.fn();
    onStopMock = vi.fn();

    loop = new PollingLoop({
      interval: 100,
      onTick: onTickMock,
      onStart: onStartMock,
      onStop: onStopMock,
    });
  });

  afterEach(() => {
    loop.stop();
    vi.useRealTimers();
  });

  describe('start/stop', () => {
    it('should not be running initially', () => {
      expect(loop.isRunning()).toBe(false);
    });

    it('should start and call onStart callback', () => {
      loop.start();

      expect(loop.isRunning()).toBe(true);
      expect(onStartMock).toHaveBeenCalledTimes(1);
    });

    it('should stop and call onStop callback', () => {
      loop.start();
      loop.stop();

      expect(loop.isRunning()).toBe(false);
      expect(onStopMock).toHaveBeenCalledTimes(1);
    });

    it('should be idempotent - multiple starts do nothing', () => {
      loop.start();
      loop.start();
      loop.start();

      expect(onStartMock).toHaveBeenCalledTimes(1);
    });

    it('should be idempotent - multiple stops do nothing', () => {
      loop.start();
      loop.stop();
      loop.stop();
      loop.stop();

      expect(onStopMock).toHaveBeenCalledTimes(1);
    });

    it('should not call onStop if never started', () => {
      loop.stop();
      expect(onStopMock).not.toHaveBeenCalled();
    });
  });

  describe('tick execution', () => {
    it('should execute onTick after interval', () => {
      loop.start();

      expect(onTickMock).not.toHaveBeenCalled();

      vi.advanceTimersByTime(100);
      expect(onTickMock).toHaveBeenCalledTimes(1);
    });

    it('should execute onTick repeatedly', () => {
      loop.start();

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(1);

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(2);

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(3);
    });

    it('should not tick after stop', () => {
      loop.start();

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(1);

      loop.stop();

      vi.advanceTimersByTime(500);
      expect(tickCount).toBe(1); // No more ticks
    });

    it('should execute multiple ticks over time', () => {
      loop.start();

      vi.advanceTimersByTime(550);
      expect(tickCount).toBe(5);
    });
  });

  describe('interval configuration', () => {
    it('should return the current interval', () => {
      expect(loop.getInterval()).toBe(100);
    });

    it('should allow changing interval', () => {
      loop.setInterval(200);
      expect(loop.getInterval()).toBe(200);
    });

    it('should use new interval after current scheduled tick', () => {
      loop.start();

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(1);

      loop.setInterval(50);

      // The current tick was scheduled with 100ms interval
      // The new interval (50ms) takes effect after this tick executes
      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(2);

      // Now new 50ms interval is in effect
      vi.advanceTimersByTime(50);
      expect(tickCount).toBe(3);

      vi.advanceTimersByTime(50);
      expect(tickCount).toBe(4);
    });
  });

  describe('tickNow', () => {
    it('should execute tick immediately', () => {
      expect(tickCount).toBe(0);

      loop.tickNow();

      expect(tickCount).toBe(1);
    });

    it('should not affect regular tick schedule', () => {
      loop.start();

      loop.tickNow(); // Immediate tick
      expect(tickCount).toBe(1);

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(2); // Regular tick still happens
    });

    // ui.md O7 / Phase E15: previously `'should work when loop is not running'`
    // — vague (P9). Rename to surface the actual contract: tickNow()
    // invokes the callback exactly once even when start() was never
    // called, and the loop's running flag stays false.
    it('tickNow() invokes the callback exactly once when the loop is stopped, without starting it', () => {
      loop.tickNow();
      expect(tickCount).toBe(1);
      expect(loop.isRunning()).toBe(false);
    });
  });

  describe('statistics', () => {
    it('should track tick count', () => {
      loop.start();

      expect(loop.getStats().tickCount).toBe(0);

      vi.advanceTimersByTime(100);
      expect(loop.getStats().tickCount).toBe(1);

      vi.advanceTimersByTime(200);
      expect(loop.getStats().tickCount).toBe(3);
    });

    it('should track running state', () => {
      expect(loop.getStats().isRunning).toBe(false);

      loop.start();
      expect(loop.getStats().isRunning).toBe(true);

      loop.stop();
      expect(loop.getStats().isRunning).toBe(false);
    });

    it('should reset stats', () => {
      loop.start();
      vi.advanceTimersByTime(500);

      expect(loop.getStats().tickCount).toBe(5);

      loop.resetStats();

      expect(loop.getStats().tickCount).toBe(0);
      expect(loop.isRunning()).toBe(true); // Still running
    });
  });

  describe('error handling', () => {
    it('should continue running if onTick throws', () => {
      const errorLoop = new PollingLoop({
        interval: 100,
        onTick: () => {
          tickCount++;
          if (tickCount === 2) {
            throw new Error('Test error');
          }
        },
      });

      // Suppress console.error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      errorLoop.start();

      vi.advanceTimersByTime(100);
      expect(tickCount).toBe(1);

      vi.advanceTimersByTime(100); // This tick throws
      expect(tickCount).toBe(2);

      vi.advanceTimersByTime(100); // Should still continue
      expect(tickCount).toBe(3);

      expect(errorLoop.isRunning()).toBe(true);

      errorLoop.stop();
      consoleSpy.mockRestore();
    });
  });

  describe('restart behavior', () => {
    it('should allow restart after stop', () => {
      loop.start();
      vi.advanceTimersByTime(200);
      expect(tickCount).toBe(2);

      loop.stop();

      loop.start();
      vi.advanceTimersByTime(200);
      expect(tickCount).toBe(4);
    });

    it('should reset tick count on restart', () => {
      loop.start();
      vi.advanceTimersByTime(300);

      const statsBefore = loop.getStats().tickCount;
      expect(statsBefore).toBe(3);

      loop.stop();
      loop.start();

      expect(loop.getStats().tickCount).toBe(0);
    });
  });
});
