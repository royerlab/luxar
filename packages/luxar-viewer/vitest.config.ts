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
        lines: 64,
        functions: 67,
        branches: 54,
        statements: 64,
      },
      // Phase 5.x continued: ~280 new tests landed across multiple
      // commits — WASM-fallback projection / visibility, layer-attrs,
      // dataset-url, debug-console formatters, fov-utils, value-
      // formatting, browser-decision, extend-tolerance, scene-graph-
      // converter, auto-blur dispatch, types/{points,lines,gsplats}
      // type guards. Measurement is now 64.49 % L / 67.87 % F /
      // 54.68 % B / 65.08 % S. Floor advances 1pp on each metric
      // (~1pp safety margin retained); long-term target stays 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
