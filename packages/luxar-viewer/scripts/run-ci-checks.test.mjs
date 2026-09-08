import { describe, expect, it, vi } from 'vitest';

import { CI_CHECKS, runChecks } from './run-ci-checks.mjs';

describe('runChecks', () => {
  it('runs every check and reports all failures', () => {
    const statuses = new Map([
      ['check:format', 1],
      ['test:coverage', 2],
    ]);
    const runner = vi.fn((check) => statuses.get(check) ?? 0);

    expect(runChecks(CI_CHECKS, runner)).toEqual(['check:format', 'test:coverage']);
    expect(runner.mock.calls.map(([check]) => check)).toEqual(CI_CHECKS);
  });

  it('reports success only when every check passes', () => {
    expect(runChecks(CI_CHECKS, () => 0)).toEqual([]);
  });
});
