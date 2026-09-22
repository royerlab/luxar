import { defineConfig } from 'vitest/config';

import { COVERAGE_THRESHOLDS } from './coverage-thresholds.mjs';
import { resolve } from 'path';
import { viewerVersionDefine } from './tools/build-identity.ts';

export default defineConfig({
  // `VIEWER_VERSION` is the package.json version under vitest too, so
  // `version.test.ts` can prove the define → module → barrel wiring against
  // the file itself. Deliberately NOT the `__LUXAR_BUILD__` stamp: the
  // build-info tests rely on vitest being the UNSTAMPED context.
  define: viewerVersionDefine(),
  test: {
    // `node` is the DEFAULT; a file that needs a DOM opts in with a
    // `// @vitest-environment jsdom` docblock on its first line.
    //
    // Constructing a jsdom document costs ~1.8 s of CPU per test FILE and is
    // paid whether or not the file touches the DOM. Measured on the 42-file
    // wasm+cache subset, same 798 passed / 2 skipped either way:
    //     jsdom  127.15 s  (environment 75.72 s)
    //     node    13.85 s  (environment 12 ms)
    // Across the whole suite only 125 of 561 files need a browser global (a
    // document in most cases), so the other 436 were paying for a DOM they
    // never touched.
    //
    // Inverting the default (rather than listing directories) is deliberate:
    // the need is not directory-aligned — `ui/` is 65% jsdom while
    // `rendering/` and `data/` are 92-94% node — and a missing docblock normally
    // fails loudly with `ReferenceError: document is not defined` rather than
    // silently running in the wrong environment. The exception is a global read
    // through `typeof` behind a non-browser branch, which degrades quietly (see
    // #1642), so the empirical list is derived, not trusted. To regenerate it
    // after a large refactor: `vitest run --environment node` and take the
    // failures.
    environment: 'node',
    globals: true,
    globalSetup: ['./src/tests/global-setup.ts'],
    setupFiles: ['./src/tests/setup.ts'],
    // Per-test / per-hook ceilings. The defaults (5s) are too tight for the
    // heavier jsdom + zarr/scene-loader suites under whole-suite execution,
    // where worker contention stretches individual tests; a too-tight ceiling
    // turned transient slowness into hard `Test timed out` / `Hook timed out`
    // failures. Use the same 60s budget everywhere: off-CI promotion runs share
    // the host with other work and can be more contended than capped CI workers.
    // Worker sizing remains environment-specific below.
    testTimeout: 60_000,
    hookTimeout: 60_000,
    // Worker threads, not forked processes. A thread reuses the host process's
    // heap and module machinery instead of paying a full V8 + Vite-runtime
    // bootstrap per worker, which is most of the fixed cost when the suite is
    // 561 small files.
    pool: 'threads',
    // Cap worker concurrency ON CI ONLY. The `maxWorkers: 4` that used to apply
    // everywhere was a response to `[vitest-pool-runner]: Timeout waiting for
    // worker to respond` in the full coverage run — a symptom of the FORKS
    // pool's per-worker memory footprint on a high-core box. Threads have a far
    // smaller footprint, and with `environment: 'node'` as the default the jsdom
    // documents that dominated that footprint are gone too. Hosted CI runners
    // are 2-core and genuinely need the cap (ci.yml documents an OOM kill from
    // over-parallelising there), so keep it there and let Vitest size the pool
    // to the machine locally.
    maxWorkers: process.env.CI ? 4 : undefined,
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/*.spec.ts', // Exclude E2E tests (Playwright)
      // WASM-vs-TS perf-budget tests are timing-sensitive and
      // occasionally flake on shared CI runners. They run via
      // `pnpm test:perf` (separate suite) — recommended as a
      // scheduled / nightly job rather than gating every PR.
      'src/tests/unit/wasm/perf-budget.test.ts',
      // Full-path sortNode throughput bench (record-only, multi-minute at
      // 5 M splats) — perf suite only, same rationale as perf-budget.
      'src/tests/unit/wasm/sort-fullpath-perf.test.ts',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/**',
        'src/tests/**',
        '**/*.d.ts',
        '**/*.config.*',
        'dist/**',
        // Exclude generated WASM glue. `public/wasm/` is .gitignored
        // and the contents are produced by the Rust wasm-pack build;
        // counting them inflates coverage by their accidental
        // presence locally vs absence in CI, and makes thresholds
        // artifact-sensitive. Coverage reflects only source-
        // controlled, hand-written app code.
        'public/wasm/**',
      ],
      // Vitest removed `coverage.all`, and `include` defaults to "only files
      // some test imported" -- which left 45 source files (including
      // core/app/options.ts at 191 LOC) outside every threshold, so adding a
      // brand-new untested module could RAISE the reported number. Naming the
      // whole tree closes that hole; `exclude` above still applies.
      include: ['src/**/*.ts'],
      // Floors live in ./coverage-thresholds.mjs so that this config and
      // scripts/check-coverage-slack.mjs cannot drift apart. Read that file
      // before editing a number — it documents the coverage semantics that
      // constrain the design (exclusions must be globs; threshold glob keys
      // are additive-only; `perFile` and `autoUpdate` must stay unset; a
      // zero-match glob passes silently).
      thresholds: COVERAGE_THRESHOLDS,
    },
  },
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, './src'),
    },
  },
});
