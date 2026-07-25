/**
 * Separate Vitest config for the WASM-vs-TypeScript perf-budget suite.
 *
 * The main `vitest.config.ts` excludes
 * `src/tests/unit/wasm/perf-budget.test.ts` because the
 * timing-sensitive tests occasionally flake on shared CI runners
 * — gating every PR on them produced false-positive failures and
 * pushed contributors toward `--retries`.
 *
 * This config picks ONLY the perf-budget suite. Run via
 * `pnpm test:perf` locally or as a scheduled / nightly CI job
 * after a fresh WASM rebuild.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    globalSetup: ['./src/tests/global-setup.ts'],
    setupFiles: ['./src/tests/setup.ts'],
    include: [
      'src/tests/unit/wasm/perf-budget.test.ts',
      'src/tests/unit/wasm/sort-fullpath-perf.test.ts',
    ],
  },
});
