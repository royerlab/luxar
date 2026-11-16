// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';
import type { Page } from '@playwright/test';
import { stepDimension, summarizePlaybackSteps } from '../e2e/perf-audit-helpers';

function pageWithStepState(state: 'full' | 'capped' | 'timeout'): Page {
  let current = state;
  (window as unknown as { __luxarDebug: unknown }).__luxarDebug = {
    app: {
      setDimensionValue: (_dim: number, _value: number) => {
        current = state;
      },
      awaitDimensionUpdate: async () => undefined,
    },
    getPerf: () => ({
      isSettled: current === 'timeout' ? false : true,
      ladderFull: current === 'full',
    }),
    getState: () => ({ totalElements: 17 }),
  };
  return { evaluate: async (fn: (args: unknown) => unknown, args: unknown) => fn(args) } as Page;
}

describe('performance audit playback steps', () => {
  it('distinguishes full, settled partial, and timed-out steps', async () => {
    const full = await stepDimension(pageWithStepState('full'), 3, 1, 20);
    const capped = await stepDimension(pageWithStepState('capped'), 3, 2, 20);
    const timeout = await stepDimension(pageWithStepState('timeout'), 3, 3, 20);

    expect(full).toMatchObject({ fullStop: 'full', elements: 17 });
    expect(full.fullMs).not.toBeNull();
    expect(capped).toMatchObject({ fullStop: 'capped', fullMs: null, elements: 17 });
    expect(timeout).toMatchObject({ fullStop: 'timeout', fullMs: null, elements: 17 });
  });

  it('keeps an all-capped sample visible when the full-ladder median is null', () => {
    const sample = summarizePlaybackSteps([
      { firstMs: 3, fullMs: null, fullStop: 'capped', elements: 17 },
      { firstMs: 5, fullMs: null, fullStop: 'capped', elements: 17 },
      { firstMs: 7, fullMs: null, fullStop: 'timeout', elements: 17 },
    ]);

    expect(sample).toEqual({ firstMs: 5, fullMs: null, cappedSteps: 2, timedOutSteps: 1 });
  });

  it('times only full steps and reports zero stops for a full sample', () => {
    expect(
      summarizePlaybackSteps([
        { firstMs: 2, fullMs: 6, fullStop: 'full', elements: 17 },
        { firstMs: 4, fullMs: 8, fullStop: 'full', elements: 17 },
      ])
    ).toEqual({ firstMs: 4, fullMs: 8, cappedSteps: 0, timedOutSteps: 0 });
  });
});
