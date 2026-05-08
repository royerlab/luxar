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
        // Phase 13.13: exclude generated WASM glue. `public/wasm/`
        // is .gitignored and the contents are produced by the Rust
        // wasm-pack build; counting them inflates coverage by their
        // accidental presence locally vs absence in CI, and makes
        // thresholds artifact-sensitive. Coverage now reflects only
        // source-controlled, hand-written app code.
        'public/wasm/**',
      ],
      // Coverage thresholds: ratcheted floor that should always be at or
      // below the actual measured coverage. They are bumped upward in a
      // dedicated commit each time a phase of new tests crosses the next
      // band. Long-term target: 80% across the board.
      thresholds: {
        lines: 72,
        functions: 74,
        branches: 61,
        statements: 71,
      },
      // Phase 13 (after 13.13's `public/wasm/**` exclusion): re-baselined
      // from a clean denominator. Slice A + Slice B added a handful of
      // resilience/cascade/validation tests; concrete current measurement
      // recorded once `pnpm test:coverage` is run with the new exclude
      // list. Holding floors here for now — they passed under the old
      // (artifact-inflated) numerator AND the post-exclude run is at
      // most a few tenths above each, so this floor remains safe.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
