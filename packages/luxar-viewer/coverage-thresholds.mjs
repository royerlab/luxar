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
 * (last updated 2026-10). Re-derive with `pnpm test:coverage`, then use
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
  // lines 90 -> 92 (2026-10): both the render-gate/HDR test additions (#3171,
  // #3173) and the flat-varying codegen tests here raised the merged dev line
  // measurement past the 3-point slack budget.
  lines: 92,
  // The picking tests in #3004 raised measured coverage to 90.00, leaving
  // the old floor outside the slack budget. The flat-varying codegen tests
  // raised it again to 92.36.
  statements: 91,
  // functions 84 -> 86 after the chunk-boundary prefetch planner/loader tests
  // (#2686): measured 86.61 -> 87.01 and check-coverage-slack flagged the old
  // floor as stale.
  // functions 86 -> 88 after the pick-depth camera-kind test started building
  // the five TSL pick graphs with a real WGSLNodeBuilder (measured 89.74).
  // functions 88 -> 90 (#3140): the gsplat WGSL codegen tests raised measured
  // functions to 91.03, past the 3-point slack budget.
  functions: 90,
  // branches 80 -> 82 (2026-09): the GPU-budget tests here and the SSAA
  // framebuffer-clamp tests in #2661 each moved measured branch coverage past
  // the slack budget. The picking tests in #3004 raised it again to 85.10.
  branches: 84,

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
  // The frame-loop driver and callback tests (#2923) raised measured lines
  // to 96.28 and functions to 89.36, leaving the prior floors stale.
  'src/scene/**': { lines: 95, functions: 90, branches: 89 },
  // Cache tests in #2946 raised these floors. The folded-in loading paths
  // lowered function coverage; shared-root tests keep it above the 95 floor.
  'src/cache/**': { lines: 93, functions: 95, branches: 86 },
  // lines 93 -> 95 and branches 86 -> 88 after the fly-touch orchestration
  // and input/touch.ts tests moved the subtree past the slack budget.
  // functions 88 -> 90 after the shared view-axis roll regression tests
  // lifted the subtree from 90.72 -> 91.21; 90 -> 92 after the unused orbit
  // keyboard-pan path was deleted, lifting it to 93.19.
  'src/controls/**': { lines: 95, functions: 92, branches: 88 },

  // The bulk of the codebase.
  // functions 87 -> 90 after the L0-cache-wiring + spatial-extend-dims
  // characterization tests (2026-09) lifted the subtree 89.2 -> 91.2. Raised
  // because check-coverage-slack.mjs flagged the old floor as stale, which is
  // the ratchet working: tests move the measurement, the guard moves the floor.
  'src/data/**': { lines: 92, functions: 92, branches: 86 },
  // functions 88 -> 90 after the worker-pool startup tests (first-worker-ready,
  // warm-up, shared-module) reached the gate/publish/warm-up paths nothing had
  // called: the subtree went 90.60 -> 91.41 and check-coverage-slack.mjs flagged
  // the old floor as stale. The ratchet working, same as the src/data bump above.
  // The folded-in worker decode tests lifted the three measurements to
  // 93.80/94.05/88.52, so the old floors fell beyond the slack budget.
  // 92/93/87 -> 94/94/89 after the eviction, init-respawn and API-surface
  // cleanup (review 3: C1, C4, C12) measured 95.00/95.58/90.04.
  'src/workers/**': { lines: 94, functions: 94, branches: 89 },
  // functions 84 -> 86 after the promise-failure and formatter regressions
  // (2026-09, #2600) made check-coverage-slack flag the old floor as stale.
  // lines 89 -> 91 after the control-panel renderer tests (2026-09, #2714):
  // `ui/control-panel/render-panel` is port-injected, so the grid fitting, the
  // centred final row and the active-tile marking are all reachable from
  // jsdom; the subtree went 91.90 -> 92.00 and the old floor went stale.
  'src/ui/**': { lines: 91, functions: 86, branches: 79 },
  // branches 83 -> 85 after the #2508 capture-readiness tests reached the
  // version-skew branches nothing had exercised (a cap refusing on a partial
  // snapshot, the unreadable-figure paths, the hostile-string guard): the
  // subtree went 85.69 -> 86.22 and check-coverage-slack.mjs flagged the old
  // floor as stale. The ratchet working, same as the src/data and src/workers
  // bumps above.
  // Remote-control orchestration and wire-value tests lifted this subtree;
  // keep the floor within the three-point slack budget.
  // The environment capture wiring tests in #3075 raised branch coverage to 88.02.
  'src/core/**': { lines: 92, functions: 80, branches: 87 },
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
  // lines 78 -> 84 and functions 79 -> 85 after
  // `tests/unit/rendering/picking/pick-depth-camera-kind.test.ts` began building
  // the five TSL pick graphs with a real WGSLNodeBuilder (no GPU), which runs
  // their factory bodies: measured lines 85.93, functions 86.27. The visual
  // shader bodies were still reached only by the browser parity spec. The HDR
  // mega-shader/tone-curve tests (#3173), the WebGPU DPR render-target tests
  // (#3171), and the flat-varying codegen tests here each raised measured
  // rendering coverage, pushing the old 84/85 floor past the slack budget.
  // branches 78 -> 80 (2026-10): #3186 recorded 80.95 before its final rebase
  // onto #3196/#3128, and the merged tree measures 81.04, past the slack budget.
  // lines 89 -> 91: the keeper-draw regression tests (#3204) raised measured
  // rendering line coverage to 92.01, beyond the old floor's slack budget.
  'src/rendering/**': { lines: 91, functions: 88, branches: 80 },

  // The last two measured top-level source directories without a sub-gate;
  // until 2026-10 their coverage could erode freely inside the global pool.
  'src/audio/**': { lines: 95, functions: 94, branches: 83 },
  'src/profiling/**': { lines: 95, functions: 95, branches: 88 },
};

/** Last accepted coverage measurements for each floor. */
export const COVERAGE_RECORDED = {
  lines: 93.79,
  statements: 92.47,
  functions: 91.03,
  branches: 86.01,
  'src/types/**': { lines: 99.1, functions: 96.67, branches: 97.96 },
  'src/wasm/**': { lines: 98.61, functions: 100, branches: 96.64 },
  'src/config/**': { lines: 96.8, functions: 100, branches: 94.03 },
  'src/utils/**': { lines: 99.49, functions: 99.38, branches: 95.15 },
  'src/scene/**': { lines: 96.69, functions: 92.07, branches: 91.55 },
  'src/cache/**': { lines: 95.92, functions: 95.38, branches: 87.34 },
  'src/controls/**': { lines: 96.81, functions: 92.92, branches: 89.82 },
  'src/data/**': { lines: 94.58, functions: 93.54, branches: 88.36 },
  'src/workers/**': { lines: 95.07, functions: 95.65, branches: 90 },
  'src/ui/**': { lines: 92.64, functions: 88.12, branches: 80.49 },
  'src/core/**': { lines: 92.99, functions: 82.41, branches: 88.16 },
  'src/input/**': { lines: 92.96, functions: 89.41, branches: 86.11 },
  'src/rendering/**': { lines: 92.01, functions: 90.14, branches: 81.04 },
  'src/audio/**': { lines: 96.5, functions: 95.48, branches: 84.11 },
  'src/profiling/**': { lines: 96.63, functions: 96.25, branches: 88.75 },
};
