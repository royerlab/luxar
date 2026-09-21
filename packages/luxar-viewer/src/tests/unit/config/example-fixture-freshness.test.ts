import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  checkExampleFixtureFreshness,
  exposeExampleFixtureFreshnessToWorkers,
  reportExampleFixtureFreshness,
} from '../../../../tools/example-fixture-freshness';

vi.mock('node:child_process', () => ({ execFileSync: vi.fn() }));

const mockedExecFileSync = vi.mocked(execFileSync);

describe('checkExampleFixtureFreshness', () => {
  beforeEach(() => {
    mockedExecFileSync.mockReset();
  });

  it('runs the repository freshness checker through hatch', () => {
    expect(checkExampleFixtureFreshness('/checkout')).toEqual({ status: 'current' });
    expect(mockedExecFileSync).toHaveBeenCalledWith(
      'hatch',
      ['run', 'python', 'scripts/run_examples.py', '--check'],
      { cwd: '/checkout', stdio: 'pipe', timeout: 120_000 }
    );
  });

  it('reports stale fixtures only for the dedicated stale exit code', () => {
    mockedExecFileSync.mockImplementation(() => {
      throw { status: 3, stderr: Buffer.from('fixtures are stale') };
    });

    expect(checkExampleFixtureFreshness('/checkout')).toEqual({ status: 'stale' });
  });

  it('reports other checker failures as unavailable with stderr', () => {
    mockedExecFileSync.mockImplementation(() => {
      throw { status: 1, stderr: Buffer.from('ModuleNotFoundError: numcodecs') };
    });

    expect(checkExampleFixtureFreshness('/checkout')).toEqual({
      status: 'unavailable',
      detail: 'ModuleNotFoundError: numcodecs',
    });
  });

  it('reports an unavailable checker when it cannot be launched', () => {
    mockedExecFileSync.mockImplementation(() => {
      throw new Error('spawn hatch ENOENT');
    });

    expect(checkExampleFixtureFreshness('/checkout')).toEqual({ status: 'unavailable' });
  });
});

describe('reportExampleFixtureFreshness', () => {
  it('warns without throwing when example datasets are stale', () => {
    const reporter = { log: vi.fn(), warn: vi.fn() };

    expect(reportExampleFixtureFreshness({ status: 'stale' }, reporter)).toBe(true);
    expect(reporter.log).not.toHaveBeenCalled();
    expect(reporter.warn).toHaveBeenCalledWith('⚠️  Example datasets are stale.');
    expect(reporter.warn).toHaveBeenCalledWith(
      '   Run "make run-examples" from the repository root to refresh them.'
    );
    expect(reporter.warn).toHaveBeenCalledWith(
      '   Continuing so specs that do not read example datasets can still run.\n'
    );
  });

  it('reports current datasets without a warning', () => {
    const reporter = { log: vi.fn(), warn: vi.fn() };

    expect(reportExampleFixtureFreshness({ status: 'current' }, reporter)).toBe(false);
    expect(reporter.log).toHaveBeenCalledWith(
      '✅ Example datasets match the current fixture producer'
    );
    expect(reporter.warn).not.toHaveBeenCalled();
  });

  it('preserves checker diagnostics when freshness is unavailable', () => {
    const reporter = { log: vi.fn(), warn: vi.fn() };

    expect(
      reportExampleFixtureFreshness(
        { status: 'unavailable', detail: 'ModuleNotFoundError: numcodecs' },
        reporter
      )
    ).toBe(true);
    expect(reporter.log).not.toHaveBeenCalled();
    expect(reporter.warn).toHaveBeenCalledWith(
      '⚠️  Could not run the example fixture freshness checker.'
    );
    expect(reporter.warn).toHaveBeenCalledWith('   ModuleNotFoundError: numcodecs');
    expect(reporter.warn).toHaveBeenCalledWith('   Continuing with presence checks only.\n');
  });
});

describe('Playwright stale-example integration', () => {
  it('forwards every non-current verdict to Playwright workers, distinguishably', () => {
    const environment: Record<string, string | undefined> = {};

    // The regression this pins: under the old boolean wire, `missing` and
    // `unavailable` both cleared the variable and so were indistinguishable
    // from `current`. A checkout with no examples at all looked healthy.
    for (const status of ['stale', 'missing', 'unavailable'] as const) {
      exposeExampleFixtureFreshnessToWorkers({ status }, environment);
      expect(environment.LUXAR_E2E_EXAMPLES_STATUS).toBe(status);
    }

    exposeExampleFixtureFreshnessToWorkers({ status: 'current' }, environment);
    expect(environment.LUXAR_E2E_EXAMPLES_STATUS).toBeUndefined();
  });

  it('names the directory and the remedy when examples were never generated', () => {
    const reporter = { log: vi.fn(), warn: vi.fn() };

    expect(
      reportExampleFixtureFreshness(
        { status: 'missing', detail: '/repo/datasets/examples' },
        reporter
      )
    ).toBe(true);
    expect(reporter.log).not.toHaveBeenCalled();
    expect(reporter.warn).toHaveBeenCalledWith('⚠️  Example datasets have not been generated.');
    expect(reporter.warn).toHaveBeenCalledWith('   Expected them at: /repo/datasets/examples');
    expect(reporter.warn).toHaveBeenCalledWith(
      '   Run "make run-examples" from the repository root to generate them.'
    );
  });

  it('keeps global setup on the warning path without a stale-data throw', () => {
    const globalSetup = readFileSync(new URL('../../e2e/global-setup.ts', import.meta.url), 'utf8');
    const freshnessBlock = globalSetup.slice(
      globalSetup.indexOf('const freshness = checkExampleFixtureFreshness'),
      globalSetup.indexOf('// Check 2: Verify required datasets')
    );

    expect(freshnessBlock).toContain('exposeExampleFixtureFreshnessToWorkers(freshness)');
    expect(freshnessBlock).toContain(
      'const examplesWarned = reportExampleFixtureFreshness(freshness)'
    );
    expect(freshnessBlock).not.toContain('throw');
    expect(globalSetup).toContain("'example datasets stale'");
    expect(globalSetup).toContain("dataset warnings: ${datasetWarnings.join('; ')}");
  });
});
