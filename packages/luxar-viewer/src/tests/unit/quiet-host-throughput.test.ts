import { afterEach, describe, expect, it, vi } from 'vitest';
import { expectQuietHostThroughput } from '../helpers/quiet-host-throughput';

describe('expectQuietHostThroughput', () => {
  const message = 'throughput floor missed';

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('reports but does not fail a floor breach by default', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    expect(() => expectQuietHostThroughput(34e6, 50e6, message, {})).not.toThrow();
    expect(warn).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledWith(
      `${message} — set LUXAR_PERF_QUIET_HOST=1 to enforce this floor`
    );
  });

  it('fails below the floor when the quiet-host opt-in is exact', () => {
    expect(() =>
      expectQuietHostThroughput(34e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: '1' })
    ).toThrowError(message);
  });

  it('accepts a measurement exactly at the enforced floor', () => {
    expect(() =>
      expectQuietHostThroughput(50e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: '1' })
    ).not.toThrow();
  });

  it('does not treat other truthy values as opt-in', () => {
    expect(() =>
      expectQuietHostThroughput(34e6, 50e6, message, { LUXAR_PERF_QUIET_HOST: 'true' })
    ).not.toThrow();
  });

  it('does not report a passing measurement without opt-in', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    expectQuietHostThroughput(50e6, 50e6, message, {});

    expect(warn).not.toHaveBeenCalled();
  });
});
