/**
 * The single source of truth for the viewer's coverage floors.
 *
 * Two consumers need this object and neither may duplicate it — a second copy
 * would drift, which is the exact failure this module exists to prevent:
 *
 *   1. `vitest.config.ts` passes it to `coverage.thresholds`, so vitest fails
 *      the run when a floor is breached.
 *   2. `scripts/check-coverage-slack.mjs` compares each floor against the
 *      measured value and fails when the gap grows past its budget, so a floor
 *      cannot quietly decay back into decoration.
 *
 * `COVERAGE_RECORDED` stores the last accepted measurement for each floor
 * (2026-09, 686 files). Re-derive with `pnpm test:coverage`, then use
 * `pnpm check:coverage-slack -- --print` to emit a paste-ready replacement.
 * Lowering a recorded value resets the erosion baseline, so call that decision
 * out in the PR rather than treating it as routine housekeeping.
 *
 * ## vitest 4 semantics — read before editing
 *
 * - A glob key ADDS a stricter sub-gate. It never removes its files from the
 *   global pool; only `coverage.exclude` exempts anything.
 * - `perFile` is GLOBAL-only. Setting it would apply the global floor to each
 *   of ~690 files individually. Do not set it.
 * - `autoUpdate` rewrites `vitest.config.ts` on disk after every run, CI
 *   included. Do not set it.
 * - A glob matching ZERO files reports pct `"Unknown"`, and `"Unknown" < 86`
 *   is `false` — so renaming a directory turns its gate into one that inspects
 *   nothing and passes. Verified empirically. `check-coverage-slack.mjs`
 *   asserts every glob key below matches at least one file for this reason.
 */

/** Metrics every threshold entry may constrain. */
export const METRICS = ['lines', 'statements', 'functions', 'branches'];

/**
 * Maximum tolerated gap, in percentage points, between a floor and its
 * measured value before `check-coverage-slack.mjs` fails. Before this guard
 * existed the global floors sat 17 points under measured and gated nothing.
 */
export const MAX_SLACK_POINTS = 3;

/** Warn when measured coverage moves this far from its recorded baseline. */
export const MAX_EROSION_POINTS = 1;

export const COVERAGE_THRESHOLDS = {
  // Global — every file in the report, the subtrees below included.
  lines: 88,
  statements: 87,
  functions: 84,
  branches: 80,

  // Crown jewels: high floors so a refactor cannot quietly erode them.
  'src/types/**': { lines: 97, functions: 94, branches: 96 },
  'src/wasm/**': { lines: 96, functions: 98, branches: 95 },
  'src/config/**': { lines: 94, functions: 98, branches: 91 },
  'src/utils/**': { lines: 98, functions: 99, branches: 94 },
  // functions 85 -> 86 after the projected-density tracker / density guard
  // tests (2026-09): the subtree went 87.50 -> 88.06 and check-coverage-slack
  // flagged the old floor as stale — the ratchet working.
  'src/scene/**': { lines: 93, functions: 86, branches: 89 },
  // branches 82 -> 84 after the OPFS write-queue / heap-budget tests
  // (2026-09, #2561): the subtree went 84.88 -> 85.25 and check-coverage-slack
  // flagged the old floor as stale — the ratchet working.
  'src/cache/**': { lines: 93, functions: 93, branches: 84 },
  'src/controls/**': { lines: 93, functions: 88, branches: 86 },

  // The bulk of the codebase.
  // functions 87 -> 90 after the L0-cache-wiring + spatial-extend-dims
  // characterization tests (2026-09) lifted the subtree 89.2 -> 91.2. Raised
  // because check-coverage-slack.mjs flagged the old floor as stale, which is
  // the ratchet working: tests move the measurement, the guard moves the floor.
  'src/data/**': { lines: 92, functions: 90, branches: 86 },
  // functions 88 -> 90 after the worker-pool startup tests (first-worker-ready,
  // warm-up, shared-module) reached the gate/publish/warm-up paths nothing had
  // called: the subtree went 90.60 -> 91.41 and check-coverage-slack.mjs flagged
  // the old floor as stale. The ratchet working, same as the src/data bump above.
  'src/workers/**': { lines: 89, functions: 90, branches: 85 },
  'src/ui/**': { lines: 89, functions: 84, branches: 77 },
  // branches 83 -> 85 after the #2508 capture-readiness tests reached the
  // version-skew branches nothing had exercised (a cap refusing on a partial
  // snapshot, the unreadable-figure paths, the hostile-string guard): the
  // subtree went 85.69 -> 86.22 and check-coverage-slack.mjs flagged the old
  // floor as stale. The ratchet working, same as the src/data and src/workers
  // bumps above.
  'src/core/**': { lines: 88, functions: 74, branches: 85 },
  // input jumped when ui-actions-surface.test.ts began invoking the command
  // table InputHandler builds in registerAllKeyBindings (27 thunks no test
  // had ever called): functions 76.0 -> 90.39.
  'src/input/**': { lines: 91, functions: 89, branches: 84 },

  // NOT a health floor — a CEILING ON THE DEBT. This subtree holds the
  // hand-written TSL/GLSL shader bodies, exercised by
  // `src/tests/e2e/tsl-shader-parity.spec.ts` (a browser cross-backend pixel
  // comparison) and therefore unreachable from vitest. They are deliberately
  // NOT excluded from coverage: that spec runs in CI as the non-required
  // `tsl-parity` job, and excluding code on the strength of a non-required gate
  // is how a metric starts lying. Revisit once the job is required.
  'src/rendering/**': { lines: 74, functions: 75, branches: 70 },
};

/** Last accepted coverage measurements for each floor. */
export const COVERAGE_RECORDED = {
  lines: 89.09,
  statements: 88.18,
  functions: 86.25,
  branches: 82.57,
  'src/types/**': { lines: 98.99, functions: 96.77, branches: 98.05 },
  'src/wasm/**': { lines: 98.58, functions: 100, branches: 96.46 },
  'src/config/**': { lines: 95.37, functions: 100, branches: 92.73 },
  'src/utils/**': { lines: 99.55, functions: 100, branches: 96.32 },
  'src/scene/**': { lines: 95.48, functions: 86.67, branches: 91.09 },
  'src/cache/**': { lines: 95.06, functions: 95.22, branches: 85.28 },
  'src/controls/**': { lines: 95.83, functions: 89.66, branches: 88.36 },
  'src/data/**': { lines: 93.5, functions: 91.87, branches: 87.42 },
  'src/workers/**': { lines: 91.39, functions: 91.41, branches: 86.74 },
  'src/ui/**': { lines: 91.77, functions: 86.72, branches: 79.25 },
  'src/core/**': { lines: 89.61, functions: 74.76, branches: 85.3 },
  'src/input/**': { lines: 93.09, functions: 90.39, branches: 85.26 },
  'src/rendering/**': { lines: 76.3, functions: 77.75, branches: 72.61 },
};
