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
        lines: 71,
        functions: 74,
        branches: 60,
        statements: 71,
      },
      // Phase 5.4ab extracted debug cache helpers from app.ts (+16
      // unit tests). Measurement is now 71.77 % L / 74.08 % F /
      // 60.48 % B / 71.24 % S. Floor advances 1pp on functions; lines
      // / branches / statements hold (~0.5pp safety margin retained).
      // Long-term target stays 80 %.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
