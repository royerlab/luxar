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
        lines: 63,
        functions: 65,
        branches: 52,
        statements: 62,
      },
      // Note: after the Phase 4.5 + 4.6 decompositions added ~81 unit
      // tests for the scene-loader / recording-panel sub-modules,
      // measured coverage rose to 63.98 % lines / 66.28 % functions /
      // 53.19 % branches / 63.28 % statements. Floor moves up ~2pp per
      // band with a ~1pp safety margin; long-term target stays 80 %.
      //
      // After Phase 8.x reorganisation + 9.2/9.3/9.5 (≈22 new tests)
      // the measurement is 64.37 % L / 66.36 % F / 53.31 % B / 63.62 % S.
      // Each metric is only ~1.4 pp above its floor — not yet a full
      // band, so the floor stays where it is. Holding for the next
      // batch of unit tests (5.x) to push past 65 / 67 / 54 / 64 before
      // the next ratchet.
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
