import { describe, expect, it } from 'vitest';

import {
  bumpFailedLoadsVersion,
  FailedLoadsMap,
  failedLoadsVersion,
} from '../../../utils/failed-loads-version';

describe('failed-loads version', () => {
  it('changes when a failure is recorded, replaced, or deleted', () => {
    const map = new FailedLoadsMap<string, string>();
    const initial = failedLoadsVersion();
    map.set('/a', 'boom');
    expect(failedLoadsVersion()).toBe(initial + 1);
    map.set('/a', 'new reason');
    expect(failedLoadsVersion()).toBe(initial + 2);
    expect(map.get('/a')).toBe('new reason');
    expect(map.delete('/a')).toBe(true);
    expect(failedLoadsVersion()).toBe(initial + 3);
  });

  it('ignores reads, missing deletes, and empty clears but versions nonempty clears', () => {
    const map = new FailedLoadsMap<string, string>();
    map.set('/a', 'boom');
    const before = failedLoadsVersion();
    expect(map.get('/a')).toBe('boom');
    expect(map.has('/a')).toBe(true);
    expect([...map.keys()]).toEqual(['/a']);
    expect(map.delete('/missing')).toBe(false);
    expect(failedLoadsVersion()).toBe(before);
    map.clear();
    expect(map.size).toBe(0);
    expect(failedLoadsVersion()).toBe(before + 1);
    map.clear();
    expect(failedLoadsVersion()).toBe(before + 1);
  });

  it('bumpFailedLoadsVersion increments the shared version', () => {
    const before = failedLoadsVersion();
    bumpFailedLoadsVersion();
    expect(failedLoadsVersion()).toBe(before + 1);
  });
});
