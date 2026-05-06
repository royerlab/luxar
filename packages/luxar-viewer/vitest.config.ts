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
        lines: 60.5,
        functions: 63,
        branches: 49.3,
        statements: 60,
      },
      // Note: actuals after Phases 1, 2, 3.1-3.4, 7.3-7.4, plus the
      // Phase-4.1 rendering-controls extractions and the 5.4 RangeSlider
      // / focus-manager / clipping-display / cinematic-mode +
      // ColormapLegend + ResolutionIndicator suites (~120 new tests
      // across this branch) land at ~60.9% lines / 63.3% functions /
      // 60.2% statements / 49.4% branches. Floor moves up by ~1pp
      // per band; long-term target stays 80% across the board.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
