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
    it('should start and call onStart callback', () => {
      loop.start();

      expect(onStartMock).toHaveBeenCalledTimes(1);
    });

    it('should stop and call onStop callback', () => {
      loop.start();
      loop.stop();

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

  describe('error handling', () => {
    // Audit G7 (viewer-ui-config-themes-core-utils): onStart/onStop
    // throw paths. Pin the CURRENT behavior (the production code does NOT
    // wrap these in try/catch — exceptions propagate to the caller of
    // start()/stop()).
    it('onStart exception propagates', () => {
      const onStartThrowing = vi.fn(() => {
        throw new Error('start failure');
      });
      const errorLoop = new PollingLoop({
        interval: 100,
        onTick: () => {},
        onStart: onStartThrowing,
      });

      expect(() => errorLoop.start()).toThrow('start failure');
      expect(onStartThrowing).toHaveBeenCalledTimes(1);

      // Cleanup: stop the loop so afterEach is clean.
      errorLoop.stop();
    });

    it('onStop exception propagates', () => {
      const onStopThrowing = vi.fn(() => {
        throw new Error('stop failure');
      });
      const errorLoop = new PollingLoop({
        interval: 100,
        onTick: () => {},
        onStop: onStopThrowing,
      });

      errorLoop.start();

      expect(() => errorLoop.stop()).toThrow('stop failure');
      expect(onStopThrowing).toHaveBeenCalledTimes(1);
    });

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
  });
});
