/**
 * R4: tests for the prefetch view-state predictor. Pure function;
 * no DOM, no I/O, no mocking — just numeric extrapolation rules.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  predictNextViewState,
  dispatchPredictivePrefetch,
  type PrefetchableLoader,
} from '../../../../data/scene-loader/predicted-view-state';
import type { ViewState } from '../../../../data/data-loader-types';

function vs(overrides: Partial<ViewState> = {}): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0, 0, 0, 5],
    ...overrides,
  };
}

describe('predictNextViewState', () => {
  it('returns a copy of `current` when `prev` is null', () => {
    const cur = vs({ slicePosition: [0, 0, 0, 7] });
    const out = predictNextViewState(null, cur);
    expect(out.slicePosition).toEqual([0, 0, 0, 7]);
    // Distinct array — predictor must not return the caller's array.
    expect(out.slicePosition).not.toBe(cur.slicePosition);
  });

  it('extrapolates one step along a single animated axis', () => {
    const prev = vs({ slicePosition: [0, 0, 0, 5] });
    const cur = vs({ slicePosition: [0, 0, 0, 6] });
    const out = predictNextViewState(prev, cur);
    expect(out.slicePosition[3]).toBe(7); // 6 + (6-5)
  });

  it('extrapolates backwards motion', () => {
    const prev = vs({ slicePosition: [0, 0, 0, 10] });
    const cur = vs({ slicePosition: [0, 0, 0, 9] });
    const out = predictNextViewState(prev, cur);
    expect(out.slicePosition[3]).toBe(8); // 9 + (9-10)
  });

  it('extrapolates multiple axes independently', () => {
    const prev = vs({
      displayDims: [0, 1],
      slicePosition: [0, 0, 5, 10],
    });
    const cur = vs({
      displayDims: [0, 1],
      slicePosition: [0, 0, 6, 8],
    });
    const out = predictNextViewState(prev, cur);
    expect(out.slicePosition[2]).toBe(7);
    expect(out.slicePosition[3]).toBe(6);
  });

  it('leaves displayDims axes alone even when their slice value changed', () => {
    // Display-axis changes are user-driven (e.g. user pans the
    // camera) and are not appropriate to extrapolate.
    const prev = vs({ displayDims: [0, 1, 2], slicePosition: [0, 0, 0, 5] });
    const cur = vs({ displayDims: [0, 1, 2], slicePosition: [10, 0, 0, 6] });
    const out = predictNextViewState(prev, cur);
    expect(out.slicePosition[0]).toBe(10); // display dim — not extrapolated
    expect(out.slicePosition[3]).toBe(7); // non-display dim — extrapolated
  });

  it('preserves tolerance and dimensions verbatim', () => {
    const prev = vs({ tolerance: [0, 0, 0, 5] });
    const cur = vs({ tolerance: [0, 0, 0, 7] });
    const out = predictNextViewState(prev, cur);
    expect(out.tolerance).toEqual([0, 0, 0, 7]);
  });

  it('returns current when no axis moved', () => {
    const prev = vs({ slicePosition: [0, 0, 0, 7] });
    const cur = vs({ slicePosition: [0, 0, 0, 7] });
    const out = predictNextViewState(prev, cur);
    expect(out.slicePosition).toEqual([0, 0, 0, 7]);
  });

  it('skips extrapolation when a NaN appears (defensive)', () => {
    const prev = vs({ slicePosition: [0, 0, 0, NaN] });
    const cur = vs({ slicePosition: [0, 0, 0, 7] });
    const out = predictNextViewState(prev, cur);
    // Predictor must NOT propagate NaN; falls back to current value.
    expect(out.slicePosition[3]).toBe(7);
  });

  it('produces a fresh array (mutating output must not affect input)', () => {
    const prev = vs({ slicePosition: [0, 0, 0, 5] });
    const cur = vs({ slicePosition: [0, 0, 0, 6] });
    const out = predictNextViewState(prev, cur);
    (out.slicePosition as number[])[3] = 999;
    expect(cur.slicePosition[3]).toBe(6);
  });
});

describe('dispatchPredictivePrefetch', () => {
  it('returns false and does NOT call prefetchChunks on the first tick (prev=null)', () => {
    const spy = vi.fn().mockResolvedValue(undefined);
    const loaders: PrefetchableLoader[] = [{ prefetchChunks: spy }];
    const dispatched = dispatchPredictivePrefetch(null, vs(), loaders);
    expect(dispatched).toBe(false);
    expect(spy).not.toHaveBeenCalled();
  });

  it('returns false when no axis moved since the previous tick', () => {
    const spy = vi.fn().mockResolvedValue(undefined);
    const loaders: PrefetchableLoader[] = [{ prefetchChunks: spy }];
    const dispatched = dispatchPredictivePrefetch(
      vs({ slicePosition: [0, 0, 0, 7] }),
      vs({ slicePosition: [0, 0, 0, 7] }),
      loaders
    );
    expect(dispatched).toBe(false);
    expect(spy).not.toHaveBeenCalled();
  });

  it('calls prefetchChunks on every loader with the predicted (one-step-ahead) state', () => {
    const a = vi.fn().mockResolvedValue(undefined);
    const b = vi.fn().mockResolvedValue(undefined);
    const c = vi.fn().mockResolvedValue(undefined);
    const loaders: PrefetchableLoader[] = [
      { prefetchChunks: a },
      { prefetchChunks: b },
      { prefetchChunks: c },
    ];
    const dispatched = dispatchPredictivePrefetch(
      vs({ slicePosition: [0, 0, 0, 5] }),
      vs({ slicePosition: [0, 0, 0, 6] }),
      loaders
    );
    expect(dispatched).toBe(true);
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
    expect(c).toHaveBeenCalledTimes(1);
    // Predicted slicePosition[3] = 6 + (6-5) = 7
    const aArg = a.mock.calls[0][0];
    expect(aArg.slicePosition[3]).toBe(7);
  });

  it('skips loaders that do not expose prefetchChunks', () => {
    const spy = vi.fn().mockResolvedValue(undefined);
    const loaders: PrefetchableLoader[] = [{}, { prefetchChunks: spy }, {}];
    const dispatched = dispatchPredictivePrefetch(
      vs({ slicePosition: [0, 0, 0, 5] }),
      vs({ slicePosition: [0, 0, 0, 6] }),
      loaders
    );
    expect(dispatched).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('swallows synchronous throws from prefetchChunks', () => {
    const throwy = vi.fn(() => {
      throw new Error('boom');
    });
    const ok = vi.fn().mockResolvedValue(undefined);
    const loaders: PrefetchableLoader[] = [
      { prefetchChunks: throwy as unknown as PrefetchableLoader['prefetchChunks'] },
      { prefetchChunks: ok },
    ];
    expect(() =>
      dispatchPredictivePrefetch(
        vs({ slicePosition: [0, 0, 0, 5] }),
        vs({ slicePosition: [0, 0, 0, 6] }),
        loaders
      )
    ).not.toThrow();
    // The second loader still got dispatched despite the first throwing.
    expect(ok).toHaveBeenCalledTimes(1);
  });

  it('swallows promise rejections from prefetchChunks', async () => {
    const reject = vi.fn().mockRejectedValue(new Error('boom'));
    const loaders: PrefetchableLoader[] = [{ prefetchChunks: reject }];
    dispatchPredictivePrefetch(
      vs({ slicePosition: [0, 0, 0, 5] }),
      vs({ slicePosition: [0, 0, 0, 6] }),
      loaders
    );
    // Let the rejection settle without unhandled-rejection noise.
    await new Promise((r) => setTimeout(r, 0));
    expect(reject).toHaveBeenCalled();
  });

  it('does not extrapolate display-axis changes (matches predictor contract)', () => {
    const spy = vi.fn().mockResolvedValue(undefined);
    dispatchPredictivePrefetch(
      vs({ displayDims: [0, 1, 2], slicePosition: [0, 0, 0, 5] }),
      // Only the display axis [0] changed → not animated.
      vs({ displayDims: [0, 1, 2], slicePosition: [10, 0, 0, 5] }),
      [{ prefetchChunks: spy }]
    );
    expect(spy).not.toHaveBeenCalled();
  });
});
