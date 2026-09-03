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
 * Trailing comments record the measurement each floor was set from
 * (2026-09, 686 files). Re-derive with `pnpm vitest run --coverage`.
 * Any test change can move these measurements, so updating tests and refreshing
 * this file with `pnpm check:coverage-slack` is one atomic change.
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

export const COVERAGE_THRESHOLDS = {
  // Global — every file in the report, the subtrees below included.
  lines: 86, //       measured 88.57
  statements: 85, //  measured 87.69
  functions: 84, //   measured 85.67
  branches: 80, //    measured 82.12

  // Crown jewels: high floors so a refactor cannot quietly erode them.
  'src/types/**': { lines: 97, functions: 94, branches: 96 }, //     98.87 / 96.61 / 97.93
  'src/wasm/**': { lines: 96, functions: 98, branches: 95 }, //      98.61 /   100 / 96.88
  'src/config/**': { lines: 94, functions: 98, branches: 91 }, //    96.30 /   100 / 93.47
  'src/utils/**': { lines: 97, functions: 97, branches: 93 }, //     98.36 / 98.39 / 95.51
  'src/scene/**': { lines: 93, functions: 85, branches: 89 }, //     95.59 / 87.50 / 91.15
  'src/cache/**': { lines: 93, functions: 93, branches: 82 }, //     94.99 / 95.15 / 84.88
  'src/controls/**': { lines: 93, functions: 85, branches: 86 }, //  95.01 / 87.68 / 88.01

  // The bulk of the codebase.
  // functions 87 -> 90 after the L0-cache-wiring + spatial-extend-dims
  // characterization tests (2026-09) lifted the subtree 89.2 -> 91.2. Raised
  // because check-coverage-slack.mjs flagged the old floor as stale, which is
  // the ratchet working: tests move the measurement, the guard moves the floor.
  'src/data/**': { lines: 92, functions: 90, branches: 86 }, //      93.40 / 91.66 / 87.36
  'src/workers/**': { lines: 89, functions: 88, branches: 85 }, //   91.58 / 90.60 / 87.27
  'src/ui/**': { lines: 89, functions: 84, branches: 76 }, //        91.41 / 86.15 / 78.60
  'src/core/**': { lines: 86, functions: 70, branches: 83 }, //      88.41 / 72.03 / 85.69
  // input jumped when ui-actions-surface.test.ts began invoking the command
  // table InputHandler builds in registerAllKeyBindings (27 thunks no test
  // had ever called): functions 76.0 -> 90.39.
  'src/input/**': { lines: 91, functions: 89, branches: 84 }, //     93.09 / 90.39 / 85.26

  // NOT a health floor — a CEILING ON THE DEBT. This subtree holds the
  // hand-written TSL/GLSL shader bodies, exercised by
  // `src/tests/e2e/tsl-shader-parity.spec.ts` (a browser cross-backend pixel
  // comparison) and therefore unreachable from vitest. They are deliberately
  // NOT excluded from coverage: that spec runs in CI as the non-required
  // `tsl-parity` job, and excluding code on the strength of a non-required gate
  // is how a metric starts lying. Revisit once the job is required.
  'src/rendering/**': { lines: 72, functions: 73, branches: 70 }, // 74.25 / 75.41 / 71.07
};
