import { describe, expect, it } from 'vitest';
import { expectQuietHostThroughput } from '../../helpers/quiet-host-throughput';

describe('quiet-host throughput policy', () => {
  const message = 'throughput below floor';

  it('reports without failing when the quiet-host opt-in is unset', () => {
    expect(() => expectQuietHostThroughput(34e6, 50e6, message, {})).not.toThrow();
  });

  it('enforces the floor when the quiet-host opt-in is exactly 1', () => {
    expect(() =>
      expectQuietHostThroughput(34e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: '1' })
    ).toThrowError(message);
  });

  it('accepts a passing measurement on an opted-in quiet host', () => {
    expect(() =>
      expectQuietHostThroughput(50e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: '1' })
    ).not.toThrow();
  });

  it('does not treat other non-empty values as opt-in', () => {
    expect(() =>
      expectQuietHostThroughput(34e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: 'true' })
    ).not.toThrow();
  });
});
