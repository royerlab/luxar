import { describe, expect, it } from 'vitest';

import {
  arg,
  collectStageDurations,
  enrichResult,
  flag,
  parseConcurrencyValues,
  resolveStartCoordinate,
} from './opfs-deep-pass-bench.mjs';

describe('opfs deep-pass benchmark helpers', () => {
  it('falls back for a missing or valueless argument', () => {
    expect(arg('concurrency', '64', ['node', 'bench'])).toBe('64');
    expect(arg('concurrency', '64', ['node', 'bench', '--concurrency'])).toBe('64');
    expect(arg('concurrency', '64', ['node', 'bench', '--concurrency', '--headless'])).toBe('64');
  });

  it('recognizes bare boolean flags', () => {
    expect(flag('prefetch', ['node', 'bench', '--prefetch'])).toBe(true);
    expect(flag('prefetch', ['node', 'bench'])).toBe(false);
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

  it('derives a penultimate coordinate and rejects a clamped last-frame start', () => {
    expect(resolveStartCoordinate([10, 20], 2, null)).toBe(18);
    expect(resolveStartCoordinate([10, 20], 2, 16)).toBe(16);
    expect(() => resolveStartCoordinate([10, 20], 2, 20)).toThrow(/no forward transition/);
  });

  it('keeps transition and settle stage timings separate', () => {
    const loadArrays = (lastMs) => ({ name: 'Load Arrays', lastMs });
    const result = {
      transitionUpdateCount: 1,
      updates: [loadArrays(7), loadArrays(11)],
    };

    expect(enrichResult(result).loadArraysMs).toEqual({
      transition: [[7]],
      settle: [[11]],
    });
  });
});
