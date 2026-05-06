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
        lines: 61,
        functions: 63.5,
        branches: 49.5,
        statements: 60.3,
      },
      // Note: after the 5.4 ColormapLegend / ResolutionIndicator /
      // PerformanceMonitor / base-types / wasm-loader / scene-loader-
      // helpers / chunk-bounds-loader suites (~62 new tests in this
      // session), measured coverage rose to 61.24 % lines /
      // 63.93 % functions / 49.69 % branches / 60.54 % statements.
      // Floor moves up another ~0.3-0.5pp per band; long-term target
      // stays 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
