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
  lines: 86, //       measured 88.08
  statements: 85, //  measured 87.22
  functions: 84, //   measured 85.43
  branches: 80, //    measured 81.71

  // Crown jewels: high floors so a refactor cannot quietly erode them.
  'src/types/**': { lines: 97, functions: 94, branches: 96 }, //     98.9 / 96.6 / 97.9
  'src/wasm/**': { lines: 96, functions: 98, branches: 95 }, //      98.6 /  100 / 96.9
  'src/config/**': { lines: 94, functions: 98, branches: 91 }, //    96.3 /  100 / 93.5
  'src/utils/**': { lines: 94, functions: 92, branches: 93 }, //     96.2 / 94.4 / 95.1
  'src/scene/**': { lines: 93, functions: 85, branches: 89 }, //     95.6 / 87.8 / 91.0
  'src/cache/**': { lines: 93, functions: 93, branches: 82 }, //     95.0 / 95.1 / 84.8
  'src/controls/**': { lines: 93, functions: 85, branches: 86 }, //  95.0 / 87.7 / 88.0

  // The bulk of the codebase.
  // functions 87 -> 90 after the L0-cache-wiring + spatial-extend-dims
  // characterization tests (2026-09) lifted the subtree 89.2 -> 91.2. Raised
  // because check-coverage-slack.mjs flagged the old floor as stale, which is
  // the ratchet working: tests move the measurement, the guard moves the floor.
  'src/data/**': { lines: 92, functions: 90, branches: 84 }, //      93.0 / 91.3 / 86.7
  'src/workers/**': { lines: 89, functions: 88, branches: 85 }, //   91.6 / 90.6 / 87.5
  'src/ui/**': { lines: 89, functions: 84, branches: 76 }, //        91.4 / 86.1 / 78.5
  'src/core/**': { lines: 86, functions: 70, branches: 83 }, //      88.4 / 72.0 / 85.6
  // input jumped when ui-actions-surface.test.ts began invoking the command
  // table InputHandler builds in registerAllKeyBindings (27 thunks no test
  // had ever called): functions 76.0 -> 88.2.
  'src/input/**': { lines: 91, functions: 87, branches: 84 }, //     92.4 / 88.2 / 85.3

  // NOT a health floor — a CEILING ON THE DEBT. This subtree holds the
  // hand-written TSL/GLSL shader bodies, exercised by
  // `src/tests/e2e/tsl-shader-parity.spec.ts` (a browser cross-backend pixel
  // comparison) and therefore unreachable from vitest. They are deliberately
  // NOT excluded from coverage: that spec does not yet run in CI, and
  // excluding code on the strength of a gate that never fires is how a metric
  // starts lying. Revisit once the `tsl-parity` job is green and required.
  'src/rendering/**': { lines: 72, functions: 73, branches: 70 }, // 74.1 / 75.2 / 71.1
};
