import { execFileSync } from 'node:child_process';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { checkExampleFixtureFreshness } from '../../../../tools/example-fixture-freshness';

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
      { cwd: '/checkout', stdio: 'pipe' }
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
