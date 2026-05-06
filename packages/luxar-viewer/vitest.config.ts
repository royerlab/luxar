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
        statements: 57,
      },
      // Note: actuals after Phase 1 hardening (WASM bounds for
      // projection/decode entry points + worker timeout helper +
      // MaterialManager rebuild + InputContextManager guard) land at
      // ~58.3% lines / 60.8% functions / 57.7% statements / 48.1%
      // branches. Statements dropped slightly because the new
      // data-worker validation paths add code to a test-stub-mocked
      // file; lines/functions held steady. Floor moves accordingly.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
