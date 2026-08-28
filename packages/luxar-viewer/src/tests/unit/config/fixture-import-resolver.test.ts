import { execFileSync } from 'node:child_process';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  FIXTURE_GENERATOR_TIMEOUT_MS,
  fixtureInputFiles,
} from '../../../../tools/fixture-freshness';

vi.mock('node:child_process', () => ({ execFileSync: vi.fn() }));

const mockedExecFileSync = vi.mocked(execFileSync);
const expectedTimeoutMs = Number(process.env.LUXAR_FIXTURE_GEN_TIMEOUT_MS) || 1_200_000;

describe('fixture import resolver', () => {
  beforeEach(() => {
    mockedExecFileSync.mockReset();
    mockedExecFileSync.mockReturnValue(
      JSON.stringify([
        ['packages/luxar-viewer/tests/fixtures/generate_test_data.py'],
        ['packages/luxar-viewer/tests/fixtures/generate_expectations.py'],
      ])
    );
  });

  it('uses the fixture-generation timeout budget and caches each checkout', () => {
    fixtureInputFiles('/checkout-one', '/checkout-one/fixtures');
    fixtureInputFiles('/checkout-one', '/checkout-one/fixtures');

    expect(FIXTURE_GENERATOR_TIMEOUT_MS).toBe(expectedTimeoutMs);
    expect(mockedExecFileSync).toHaveBeenCalledOnce();
    expect(mockedExecFileSync).toHaveBeenCalledWith(
      'hatch',
      expect.any(Array),
      expect.objectContaining({ timeout: FIXTURE_GENERATOR_TIMEOUT_MS })
    );
  });

  it('explains import-resolution timeouts and the override', () => {
    mockedExecFileSync.mockImplementation(() => {
      throw { code: 'ETIMEDOUT' };
    });

    expect(() => fixtureInputFiles('/checkout-two', '/checkout-two/fixtures')).toThrow(
      `Fixture import resolution exceeded the ${expectedTimeoutMs} ms budget. ` +
        'Raise it with LUXAR_FIXTURE_GEN_TIMEOUT_MS if this machine is slower.'
    );
  });

  it('rejects malformed import-resolution payloads', () => {
    mockedExecFileSync.mockReturnValue('[[]]');

    expect(() => fixtureInputFiles('/checkout-invalid', '/checkout-invalid/fixtures')).toThrow(
      'Fixture import resolver returned an invalid source list'
    );
  });
});
