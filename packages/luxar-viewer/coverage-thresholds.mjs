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
 * (2026-09). Re-derive with `pnpm test:coverage`, then use
 * `pnpm check:coverage-slack -- --print` to emit a paste-ready replacement.
 * Lowering a recorded value resets the erosion baseline, so call that decision
 * out in the PR rather than treating it as routine housekeeping.
 *
 * ## Vitest coverage semantics — read before editing
 *
 * - Directory exclusions must be explicit globs (for example,
 *   `src/tests/**`); bare directory forms do not exclude executed files in
 *   Vitest 5.
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
  // functions 84 -> 86 after the chunk-boundary prefetch planner/loader tests
  // (#2686): measured 86.61 -> 87.01 and check-coverage-slack flagged the old
  // floor as stale.
  functions: 86,
  // branches 80 -> 82 (2026-09): the GPU-budget tests here and the SSAA
  // framebuffer-clamp tests in #2661 each moved measured branch coverage past
  // the slack budget, and check-coverage-slack flagged the old floor as stale.
  branches: 82,

  // Crown jewels: high floors so a refactor cannot quietly erode them.
  'src/types/**': { lines: 97, functions: 94, branches: 96 },
  'src/wasm/**': { lines: 96, functions: 98, branches: 95 },
  // branches 91 -> 93 after the control-panel / kiosk config tests (2026-09,
  // #2714): `control-panel/fit-grid`, `zarr-bridge/control-panel` and
  // `kiosk.ts` are pure and table-tested, so the subtree went 93.17 -> 94.05
  // and check-coverage-slack flagged the old floor as stale — the ratchet
  // working, same as the bumps below.
  'src/config/**': { lines: 94, functions: 98, branches: 93 },
  'src/utils/**': { lines: 98, functions: 99, branches: 94 },
  // functions 85 -> 86 after the projected-density tracker / density guard
  // tests (2026-09): the subtree went 87.50 -> 88.06 and check-coverage-slack
  // flagged the old floor as stale — the ratchet working.
  'src/scene/**': { lines: 93, functions: 86, branches: 89 },
  // branches 82 -> 84 after the OPFS write-queue / heap-budget tests
  // (2026-09, #2561): the subtree went 84.88 -> 85.25 and check-coverage-slack
  // flagged the old floor as stale — the ratchet working.
  'src/cache/**': { lines: 93, functions: 93, branches: 84 },
  // lines 93 -> 95 and branches 86 -> 88 after the fly-touch orchestration
  // and input/touch.ts tests moved the subtree past the slack budget.
  // functions 88 -> 90 after the shared view-axis roll regression tests
  // lifted the subtree from 90.72 -> 91.21.
  'src/controls/**': { lines: 95, functions: 90, branches: 88 },

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
  // functions 84 -> 86 after the promise-failure and formatter regressions
  // (2026-09, #2600) made check-coverage-slack flag the old floor as stale.
  // lines 89 -> 91 after the control-panel renderer tests (2026-09, #2714):
  // `ui/control-panel/render-panel` is port-injected, so the grid fitting, the
  // centred final row and the active-tile marking are all reachable from
  // jsdom; the subtree went 91.90 -> 92.00 and the old floor went stale.
  'src/ui/**': { lines: 91, functions: 86, branches: 77 },
  // branches 83 -> 85 after the #2508 capture-readiness tests reached the
  // version-skew branches nothing had exercised (a cap refusing on a partial
  // snapshot, the unreadable-figure paths, the hostile-string guard): the
  // subtree went 85.69 -> 86.22 and check-coverage-slack.mjs flagged the old
  // floor as stale. The ratchet working, same as the src/data and src/workers
  // bumps above.
  // Remote-control orchestration and wire-value tests lifted this subtree;
  // keep the floor within the three-point slack budget.
  'src/core/**': { lines: 90, functions: 76, branches: 85 },
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
  // lines 74 -> 76 after the bloom live-texture binding tests moved measured
  // line coverage to 77.18 and check-coverage-slack flagged the old floor.
  'src/rendering/**': { lines: 76, functions: 77, branches: 72 },
};

/** Last accepted coverage measurements for each floor. */
export const COVERAGE_RECORDED = {
  lines: 89.38,
  statements: 88.43,
  functions: 87.01,
  branches: 83.04,
  'src/types/**': { lines: 99, functions: 96.77, branches: 98.05 },
  'src/wasm/**': { lines: 98.58, functions: 100, branches: 96.46 },
  'src/config/**': { lines: 96.36, functions: 100, branches: 94.05 },
  'src/utils/**': { lines: 99.6, functions: 100, branches: 96.19 },
  'src/scene/**': { lines: 95.5, functions: 86.74, branches: 91.15 },
  'src/cache/**': { lines: 95.03, functions: 95.2, branches: 85.41 },
  'src/controls/**': { lines: 96.32, functions: 91.21, branches: 89.44 },
  'src/data/**': { lines: 93.6, functions: 92.03, branches: 87.31 },
  'src/workers/**': { lines: 91.39, functions: 91.41, branches: 86.74 },
  'src/ui/**': { lines: 92.0, functions: 86.95, branches: 79.58 },
  'src/core/**': { lines: 91.11, functions: 78.49, branches: 86.94 },
  'src/input/**': { lines: 93.09, functions: 90.39, branches: 85.26 },
  'src/rendering/**': { lines: 77.18, functions: 78.13, branches: 73.22 },
};
