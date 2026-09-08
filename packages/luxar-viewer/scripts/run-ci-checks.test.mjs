import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';

import { CI_CHECKS, runChecks } from './run-ci-checks.mjs';

describe('runChecks', () => {
  it('runs every check and reports all failures', () => {
    const statuses = new Map([
      ['check:format', 1],
      ['test:coverage', 2],
    ]);
    const runner = vi.fn((check) => ({ status: statuses.get(check) ?? 0, signal: null }));

    expect(runChecks(CI_CHECKS, runner)).toEqual({
      failures: ['check:format', 'test:coverage'],
      killed: null,
    });
    expect(runner.mock.calls.map(([check]) => check)).toEqual(CI_CHECKS);
  });

  it('reports success only when every check passes', () => {
    expect(runChecks(CI_CHECKS, () => ({ status: 0, signal: null }))).toEqual({
      failures: [],
      killed: null,
    });
  });

  it('stops immediately when a check is killed', () => {
    const runner = vi.fn((check) => ({
      status: check === 'lint' ? null : 0,
      signal: check === 'lint' ? 'SIGTERM' : null,
    }));
    expect(runChecks(CI_CHECKS, runner)).toEqual({
      failures: [],
      killed: { check: 'lint', signal: 'SIGTERM' },
    });
    expect(runner.mock.calls.map(([check]) => check)).toEqual(CI_CHECKS.slice(0, 4));
  });

  it('supports local fail-fast without changing the CI default', () => {
    const runner = vi.fn((check) => ({ status: check === 'check:format' ? 1 : 0, signal: null }));
    expect(runChecks(CI_CHECKS, runner, { bail: true })).toEqual({
      failures: ['check:format'],
      killed: null,
    });
    expect(runner.mock.calls.map(([check]) => check)).toEqual(CI_CHECKS.slice(0, 2));
  });
});

describe('CI_CHECKS', () => {
  it('stays in lockstep with check:static plus coverage checks', () => {
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    const staticChecks = pkg.scripts['check:static']
      .split('&&')
      .map((command) => command.trim().replace(/^pnpm run /, ''));
    expect(CI_CHECKS).toEqual([...staticChecks, 'test:coverage', 'check:coverage-slack']);
  });
});
