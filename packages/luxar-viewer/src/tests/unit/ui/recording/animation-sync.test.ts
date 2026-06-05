/**
 * Unit tests for animation-sync helpers.
 *
 * The two pure helpers (getTurntableInfo, getNavigableDimensionOptions)
 * are direct calls; SliderSyncCoordinator owns timer + listener state
 * and is tested with vi's fake timers.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fc from 'fast-check';

// Mock scene-dims-manager BEFORE importing the helpers so the proxy
// doesn't return live bound functions.
const mockGetDims = vi.fn();
const mockGetDimensionNames = vi.fn(() => [] as string[]);
const mockGetDimensionRanges = vi.fn();
const mockSetDimensionValue = vi.fn();
vi.mock('../../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: () => mockGetDims(),
    getDimensionNames: () => mockGetDimensionNames(),
    getDimensionRanges: () => mockGetDimensionRanges(),
    setDimensionValue: (...args: unknown[]) => mockSetDimensionValue(...args),
  },
}));

import {
  getTurntableInfo,
  getNavigableDimensionOptions,
  SliderSyncCoordinator,
} from '../../../../ui/recording-panel/animation-sync';
import type { DimensionAnimationManager } from '../../../../scene/animation/dimension-animation-manager';

describe('getTurntableInfo', () => {
  it('formats duration as 360/speed and frames as ceil(duration * fps)', () => {
    expect(getTurntableInfo(60, 30)).toBe('6.0s, 180 frames');
    expect(getTurntableInfo(45, 30)).toBe('8.0s, 240 frames');
    // Non-integer duration → frame count rounds up.
    // [P7] 360/33 = 10.909..s; toFixed(1) = '10.9'; ceil(10.909 × 30) = 328.
    expect(getTurntableInfo(33, 30)).toBe('10.9s, 328 frames');
  });

  it('[property] frame count always equals ceil((360/speed) × fps)', () => {
    // [P12] The exact invariant across the full speed × fps grid.
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 360 }),
        fc.integer({ min: 1, max: 240 }),
        (speed, fps) => {
          const info = getTurntableInfo(speed, fps);
          const match = info.match(/^(\d+\.\d)s, (\d+) frames$/);
          expect(match).not.toBeNull();
          expect(Number(match![2])).toBe(Math.ceil((360 / speed) * fps));
        }
      )
    );
  });
});

describe('getNavigableDimensionOptions', () => {
  beforeEach(() => {
    mockGetDims.mockReset();
    mockGetDimensionNames.mockReset().mockReturnValue([]);
  });

  it('returns the (no dimensions) placeholder when getDims returns null', () => {
    mockGetDims.mockReturnValue(null);
    expect(getNavigableDimensionOptions()).toEqual({ '(no dimensions)': -1 });
  });

  it('lists only non-displayed dimensions', () => {
    mockGetDims.mockReturnValue({ ndim: 4, displayed: [0, 1, 2] });
    mockGetDimensionNames.mockReturnValue(['x', 'y', 'z', 't']);
    expect(getNavigableDimensionOptions()).toEqual({ t: 3 });
  });

  it('renames empty-name dims to "dim N"', () => {
    mockGetDims.mockReturnValue({ ndim: 4, displayed: [0, 1, 2] });
    mockGetDimensionNames.mockReturnValue(['x', 'y', 'z', '']);
    expect(getNavigableDimensionOptions()).toEqual({ 'dim 3': 3 });
  });

  it('returns the (no dimensions) placeholder when every dim is displayed', () => {
    mockGetDims.mockReturnValue({ ndim: 3, displayed: [0, 1, 2] });
    mockGetDimensionNames.mockReturnValue(['x', 'y', 'z']);
    expect(getNavigableDimensionOptions()).toEqual({ '(no dimensions)': -1 });
  });
});

describe('SliderSyncCoordinator', () => {
  function makeAnimationManager() {
    return {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      play: vi.fn(),
    } as unknown as DimensionAnimationManager;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    mockGetDimensionRanges.mockReset().mockReturnValue([
      [0, 100],
      [0, 50],
    ]);
    mockSetDimensionValue.mockReset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('does nothing when dimIndex < 0', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    sc.start(-1, mgr, vi.fn(), () => true);
    expect(mgr.addEventListener).not.toHaveBeenCalled();
    // [P11/M4] Verify the WHOLE method short-circuits, not just the listener
    // registration — a mutation that did work before the `dimIndex < 0`
    // return (e.g. snapping the dimension) would otherwise slip through.
    expect(mockSetDimensionValue).not.toHaveBeenCalled();
    expect(mgr.play).not.toHaveBeenCalled();
  });

  it('snaps to dimension min, registers complete listener, then plays after delay', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    const onComplete = vi.fn();
    sc.start(0, mgr, onComplete, () => true, 100);

    expect(mockSetDimensionValue).toHaveBeenCalledWith(0, 0);
    expect(mgr.addEventListener).toHaveBeenCalledWith('complete', expect.any(Function));
    expect(mgr.play).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);
    expect(mgr.play).toHaveBeenCalledWith(0, { loopMode: 'once', direction: 'forward' });
  });

  it('skips play when isAlive returns false at the end of the delay', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    let alive = true;
    sc.start(0, mgr, vi.fn(), () => alive, 100);
    alive = false;
    vi.advanceTimersByTime(100);
    expect(mgr.play).not.toHaveBeenCalled();
  });

  it('forwards animation-complete to onComplete', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    const onComplete = vi.fn();
    sc.start(0, mgr, onComplete, () => true);
    const handler = vi.mocked(mgr.addEventListener).mock.calls[0][1] as () => void;
    handler();
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('cleanup cancels the pending play and removes the listener', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    sc.start(0, mgr, vi.fn(), () => true, 100);

    sc.cleanup(mgr);
    expect(mgr.removeEventListener).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(200);
    expect(mgr.play).not.toHaveBeenCalled();
  });

  it('cleanup is idempotent', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    expect(() => sc.cleanup(mgr)).not.toThrow();
    expect(() => sc.cleanup(null)).not.toThrow();
    sc.start(0, mgr, vi.fn(), () => true);
    sc.cleanup(mgr);
    expect(() => sc.cleanup(mgr)).not.toThrow();
    expect(mgr.removeEventListener).toHaveBeenCalledTimes(1);
  });

  it('start re-arms cleanly after a previous start', () => {
    const sc = new SliderSyncCoordinator();
    const mgr = makeAnimationManager();
    sc.start(0, mgr, vi.fn(), () => true);
    sc.start(0, mgr, vi.fn(), () => true);
    // Two adds, one remove from the implicit cleanup inside start().
    expect(mgr.addEventListener).toHaveBeenCalledTimes(2);
    expect(mgr.removeEventListener).toHaveBeenCalledTimes(1);
  });
});
