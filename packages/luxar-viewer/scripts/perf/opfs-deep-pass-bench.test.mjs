import { describe, expect, it } from 'vitest';

import { advanceDimensionValue } from '../../src/scene/animation/advance-value';
import { snapDiscreteValue } from '../../src/scene/scene-dims-manager';
import {
  arg,
  booleanOption,
  collectStageDurations,
  deriveMeasuredDurations,
  enrichResult,
  missedPollEvents,
  parseConcurrencyValues,
  parseOptions,
  parsePositiveInteger,
  resolveStartCoordinate,
} from './opfs-deep-pass-bench.mjs';

describe('opfs deep-pass benchmark helpers', () => {
  it('falls back for a missing or valueless argument', () => {
    expect(arg('concurrency', '64', ['node', 'bench'])).toBe('64');
    expect(arg('concurrency', '64', ['node', 'bench', '--concurrency'])).toBe('64');
    expect(arg('concurrency', '64', ['node', 'bench', '--concurrency', '--headless'])).toBe('64');
  });

  it('accepts bare and explicit boolean options', () => {
    expect(booleanOption('clear-first', ['node', 'bench', '--clear-first'])).toBe(true);
    expect(booleanOption('headless', ['node', 'bench', '--headless', 'true'])).toBe(true);
    expect(booleanOption('headless', ['node', 'bench', '--headless', 'false'])).toBe(false);
    expect(booleanOption('headless', ['node', 'bench'])).toBe(false);
    expect(() => booleanOption('headless', ['node', 'bench', '--headless', 'sometimes'])).toThrow(
      /true or false/
    );
  });

  it('parses explicit false for prefetch', () => {
    expect(parseOptions(['node', 'bench', '--prefetch']).prefetch).toBe(true);
    expect(parseOptions(['node', 'bench', '--prefetch', 'true']).prefetch).toBe(true);
    expect(parseOptions(['node', 'bench', '--prefetch', 'false']).prefetch).toBe(false);
  });

  it('requires a positive integer ladder depth', () => {
    expect(parsePositiveInteger('ladder-depth', '6')).toBe(6);
    expect(() => parsePositiveInteger('ladder-depth', '0.5')).toThrow(/positive integer/);
    expect(() => parsePositiveInteger('ladder-depth', '2.5')).toThrow(/positive integer/);
  });

  it('rejects concurrency arms that the viewer would silently ignore', () => {
    expect(parseConcurrencyValues('8,64,512')).toEqual([8, 64, 512]);
    expect(() => parseConcurrencyValues('8,nope,512')).toThrow(/positive integers/);
    expect(() => parseConcurrencyValues('8,0,512')).toThrow(/positive integers/);
    expect(() => parseConcurrencyValues('8,1.5,512')).toThrow(/positive integers/);
  });

  it('excludes stale profiler rows and their descendants', () => {
    const tree = {
      name: 'Total Update',
      lastMs: 50,
      children: [
        { name: 'Load Arrays', lastMs: 5, stale: false },
        {
          name: 'LOD 2',
          lastMs: 20,
          stale: true,
          children: [{ name: 'Load Arrays', lastMs: 20, stale: false }],
        },
      ],
    };

    expect(collectStageDurations(tree, 'Load Arrays')).toEqual([5]);
  });

  it('starts one grid step before the last coordinate reachable in once mode', () => {
    expect(resolveStartCoordinate([10, 20], 2, null)).toBe(16);
    expect(resolveStartCoordinate([10, 20], 2, 16)).toBe(16);
    expect(() => resolveStartCoordinate([10, 20], 2, 18)).toThrow(/one reachable advance/);
    expect(() => resolveStartCoordinate([10, 20], 2, 20)).toThrow(/one reachable advance/);
  });

  it('anchors fractional and offset starts to the discrete grid', () => {
    expect(resolveStartCoordinate([0, 3], 0.3, null)).toBeCloseTo(2.7, 10);
    expect(resolveStartCoordinate([0, 3], 0.3, 2.7)).toBeCloseTo(2.7, 10);
    expect(resolveStartCoordinate([100, 200], 7, null)).toBe(191);
  });

  it.each([
    { range: [0, 50], step: 1 },
    { range: [0, 3], step: 0.3 },
    { range: [100, 200], step: 7 },
  ])('leaves exactly one applied once-mode advance on $range/$step', ({ range, step }) => {
    const [min, max] = range;
    const start = resolveStartCoordinate(range, step, null);
    const first = advanceDimensionValue({
      current: start,
      min,
      max,
      step,
      direction: 'forward',
      loopMode: 'once',
      targetFPS: 2,
      continuousTraverseMs: 1000,
    });
    expect(first.shouldStop).toBe(false);

    const landed = snapDiscreteValue(first.value, step, min, max);
    const second = advanceDimensionValue({
      current: landed,
      min,
      max,
      step,
      direction: 'forward',
      loopMode: 'once',
      targetFPS: 2,
      continuousTraverseMs: 1000,
    });
    expect(second.shouldStop).toBe(true);
  });

  it('rejects a range with no reachable once-mode transition', () => {
    expect(() => resolveStartCoordinate([0, 1], 1, null)).toThrow(/no reachable transition/);
  });

  it('keeps transition and settle stage timings separate', () => {
    const loadArrays = (lastMs) => ({ name: 'Load Arrays', lastMs });
    const result = {
      transitionUpdateCount: 1,
      transitionStartedMs: 10,
      transitionCompletedMs: 17,
      lastActivityMs: 28,
      updateCountDelta: 2,
      updates: [loadArrays(7), loadArrays(11)],
    };

    const enriched = enrichResult(result);
    expect(enriched.loadArraysMs).toEqual({
      transition: [[7]],
      settle: [[11]],
    });
    expect(enriched).toMatchObject({ animationMs: 7, settleMs: 11, missedUpdates: 0 });
  });

  it('excludes cadence pre-roll and quiet-window dead time from measured work', () => {
    expect(
      deriveMeasuredDurations({
        transitionStartedMs: 500,
        transitionCompletedMs: 725,
        lastActivityMs: 925,
      })
    ).toEqual({ animationMs: 225, settleMs: 200 });
  });

  it('reports updates collapsed between polls', () => {
    expect(missedPollEvents(1, 1)).toBe(0);
    expect(missedPollEvents(3, 1)).toBe(2);
    expect(missedPollEvents(0, 0)).toBe(0);
  });
});
