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
      // below the actual measured coverage. The values below match the
      // current actuals (~56% lines, ~56% statements, ~57% functions,
      // ~45% branches) with a small safety margin. They are bumped upward
      // in a dedicated commit each time a phase of new tests crosses the
      // next band. Long-term target: 80% across the board.
      thresholds: {
        lines: 57,
        functions: 57,
        branches: 46,
        statements: 56,
      },
      // Note: actuals as of the loadColorRanges share land at ~57.3% lines /
      // 57.9% functions / 56.7% statements / 46.4% branches. The floor here
      // is intentionally 0.3-0.5pp below those numbers so re-runs don't
      // flicker around the threshold.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
