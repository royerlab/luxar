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
      // Phase 12 review-driven hardening (12.1–12.9) added ~70 unit
      // tests across worker validation, Result<T,E> migration, monitor
      // reset, panel coordinator dataset-browser handle, and worker-
      // pool runWithTimeout. Measurement is now 71.59 % S /
      // 61.07 % B / 74.43 % F / 72.15 % L. Floor advances 1pp on
      // branches and 1pp on lines; statements + functions hold at
      // 71 / 74 (~0.5pp safety margin retained on each). Long-term
      // target stays 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
