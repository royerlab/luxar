/**
 * Separate Vitest config for the WASM-vs-TypeScript perf-budget suite.
 *
 * The main `vitest.config.ts` excludes
 * `src/tests/unit/wasm/perf-budget.test.ts` because the
 * timing-sensitive tests occasionally flake on shared CI runners
 * — gating every PR on them produced false-positive failures and
 * pushed contributors toward `--retries`.
 *
 * This config picks only the opt-in WASM performance suites. Run via
 * `pnpm test:perf` locally after a fresh WASM rebuild. Absolute floors are
 * report-only unless `pnpm test:perf:strict` explicitly enforces them on a
 * quiet host.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Deliberately still 'jsdom', matching the environment in which these
    // thresholds were established. Switching environments would change the
    // measurement substrate, so treat that as a separate benchmark-policy change.
    environment: 'jsdom',
    globals: true,
    // WASM-only setup: the perf benches read no zarr fixtures, and bench
    // boxes may lack the hatch/Python env the full setup's fixture
    // generation requires.
    globalSetup: ['./src/tests/global-setup-perf.ts'],
    setupFiles: ['./src/tests/setup.ts'],
    include: [
      'src/tests/unit/wasm/perf-budget.test.ts',
      'src/tests/unit/wasm/sort-fullpath-perf.test.ts',
    ],
  },
});
