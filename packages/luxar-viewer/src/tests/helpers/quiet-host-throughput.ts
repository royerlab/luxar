/**
 * Quiet-host policy for absolute performance floors.
 *
 * @module tests/helpers/quiet-host-throughput
 */
import { expect } from 'vitest';

/** Environment values used by the quiet-host performance policy. */
export interface PerfEnvironment {
  LUXAR_PERF_QUIET_HOST?: string;
}

/**
 * Enforce an absolute throughput floor only on an explicitly quiet host.
 *
 * @param actual - Measured throughput.
 * @param floor - Minimum acceptable throughput on a quiet host.
 * @param message - Measurement details included in failures and breach reports.
 * @param env - Environment values used to resolve explicit enforcement.
 */
export function expectQuietHostThroughput(
  actual: number,
  floor: number,
  message: string,
  env: PerfEnvironment = process.env
): void {
  if (env.LUXAR_PERF_QUIET_HOST === '1') {
    expect(actual, message).toBeGreaterThanOrEqual(floor);
  } else if (actual < floor) {
    console.warn(`${message} — set LUXAR_PERF_QUIET_HOST=1 to enforce this floor`);
  }
}
