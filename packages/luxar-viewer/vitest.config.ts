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
        lines: 70,
        functions: 72,
        branches: 58,
        statements: 69,
      },
      // Phase 5.4v + 5.4w landed post-processing-setup (+33) and
      // loading-advisor (+26) tests on top of 5.4s/t/u. Measurement is
      // now 70.48 % L / 73.13 % F / 58.89 % B / 69.89 % S. Floor
      // advances 1–2pp on lines/functions/statements (branches hold —
      // bigger jump deferred to next phase). Long-term target stays
      // 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
