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
        lines: 66,
        functions: 69,
        branches: 55,
        statements: 66,
      },
      // Phase 5.2 body landed 42 new lines/gsplats spatial-index loader
      // full-flow tests (initialization, spatial-index queries,
      // extend_to_all, data loading, monitoring, error handling,
      // updateView, resource cleanup, data-type handling, plus
      // gsplats-only prefetchChunks). Measurement is now 66.88 % L /
      // 69.59 % F / 56.04 % B / 66.30 % S. Floor advances 1pp on each
      // metric (~1pp safety margin retained); long-term target stays
      // 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
