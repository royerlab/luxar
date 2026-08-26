import { expect } from 'vitest';

type PerfEnvironment = {
  LUXAR_PERF_QUIET_HOST?: string;
};

/** Enforce an absolute throughput floor only on an explicitly quiet host. */
export function expectQuietHostThroughput(
  actual: number,
  floor: number,
  message: string,
  env: PerfEnvironment = process.env
): void {
  if (env.LUXAR_PERF_QUIET_HOST === '1') {
    expect(actual, message).toBeGreaterThanOrEqual(floor);
  }
}
