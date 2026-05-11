import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    globalSetup: ['./src/tests/global-setup.ts'],
    setupFiles: ['./src/tests/setup.ts'],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/*.spec.ts', // Exclude E2E tests (Playwright)
      // WASM-vs-TS perf-budget tests are timing-sensitive and
      // occasionally flake on shared CI runners. They run via
      // `pnpm test:perf` (separate suite) — recommended as a
      // scheduled / nightly job rather than gating every PR.
      'src/tests/unit/wasm/perf-budget.test.ts',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/',
        'src/tests/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData/*',
        'dist/',
        // Exclude generated WASM glue. `public/wasm/` is .gitignored
        // and the contents are produced by the Rust wasm-pack build;
        // counting them inflates coverage by their accidental
        // presence locally vs absence in CI, and makes thresholds
        // artifact-sensitive. Coverage reflects only source-
        // controlled, hand-written app code.
        'public/wasm/**',
      ],
      // Coverage thresholds: ratcheted floor that should always be at
      // or below the actual measured coverage. Bumped upward in a
      // dedicated commit when a batch of new tests crosses the next
      // band. Long-term target: 80% across the board.
      thresholds: {
        lines: 71,
        functions: 74,
        branches: 61,
        statements: 71,
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
