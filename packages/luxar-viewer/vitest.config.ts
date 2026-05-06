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
        functions: 61,
        branches: 48,
        statements: 57,
      },
      // Note: actuals after Phases 1, 2, 3.1-3.4, 7.3-7.4 land at
      // ~58.6% lines / 61.3% functions / 57.9% statements / 48.3%
      // branches. Coverage gain comes from the new ManagerRegistry,
      // Result<T,E>, MaterialManager LRU, GPU pool batch-eviction,
      // and worker-timeout test suites (~60 new tests). Floor moves
      // accordingly.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
