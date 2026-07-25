/**
 * Vitest global setup for the PERF suite (`vitest.perf.config.ts`) — WASM only.
 *
 * The perf benches (`perf-budget.test.ts`, `sort-fullpath-perf.test.ts`)
 * consume no zarr fixtures, and dedicated bench boxes often lack the
 * hatch/Python environment the fixture generator requires — the full
 * `global-setup.ts` would hard-fail there for fixtures nothing reads.
 */

import { ensureWasmBuilt } from './global-setup';

export async function setup(): Promise<void> {
  ensureWasmBuilt();
}
