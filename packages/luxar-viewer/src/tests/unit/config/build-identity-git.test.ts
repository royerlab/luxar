import { beforeEach, describe, expect, it, vi } from 'vitest';

const { execFileSync } = vi.hoisted(() => ({ execFileSync: vi.fn() }));

vi.mock('child_process', () => ({ execFileSync }));

import { buildIdentity } from '../../../../tools/build-identity';

describe('build identity git probes', () => {
  beforeEach(() => {
    execFileSync.mockReset();
  });

  it('keeps a resolved commit when the dirty-tree probe fails', () => {
    execFileSync.mockReturnValueOnce('abc1234\n').mockImplementationOnce(() => {
      throw new Error('status output exceeded maxBuffer');
    });

    expect(buildIdentity().commit).toBe('abc1234');
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      'git',
      ['status', '--porcelain'],
      expect.any(Object)
    );
  });

  it('still marks a resolved commit when the dirty-tree probe succeeds', () => {
    execFileSync
      .mockReturnValueOnce('abc1234\n')
      .mockReturnValueOnce(' M tools/build-identity.ts\n');

    expect(buildIdentity().commit).toBe('abc1234-dirty');
  });

  it('uses unknown when the commit itself cannot be resolved', () => {
    execFileSync.mockImplementationOnce(() => {
      throw new Error('git is unavailable');
    });

    expect(buildIdentity().commit).toBe('unknown');
    expect(execFileSync).toHaveBeenCalledTimes(1);
  });
});
