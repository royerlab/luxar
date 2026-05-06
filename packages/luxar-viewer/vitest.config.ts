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
        lines: 58,
        functions: 60,
        branches: 48,
        statements: 58,
      },
      // Note: actuals after the D2/D3/D4 helper extractions and tests
      // land at ~58.8% lines / 60.8% functions / 58.2% statements /
      // 48.5% branches. The floor here is set 0.2-0.8pp below those
      // numbers so re-runs don't flicker around the threshold.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
