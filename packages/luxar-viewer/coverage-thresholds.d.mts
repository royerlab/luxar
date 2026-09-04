/**
 * Types for `coverage-thresholds.mjs`.
 *
 * Without this, `vitest.config.ts` imports the thresholds as `any` and TypeScript
 * checks nothing about them: a floor written under a misspelled metric key
 * (`branch` for `branches`) is accepted by vitest, gates nothing, and looks
 * exactly like a floor that passes. That is the same fail-open shape the module's
 * own header warns about for zero-match glob keys, so it should not be reachable
 * through the type system either.
 *
 * Declared here rather than by converting the module to TypeScript because
 * `scripts/check-coverage-slack.mjs` imports it too, at plain Node with no
 * transform step.
 */

/** The four coverage metrics vitest reports, in the order they are printed. */
export declare const METRICS: readonly ['lines', 'statements', 'functions', 'branches'];

/** How far a floor may sit below the measured value before the guard fails. */
export declare const MAX_SLACK_POINTS: number;

/** A floor per metric. Every key is optional; an omitted metric is ungated. */
export interface MetricFloors {
  lines?: number;
  statements?: number;
  functions?: number;
  branches?: number;
}

/**
 * Global floors plus one entry per glob key.
 *
 * A glob key ADDS a stricter sub-gate over its files; it never removes them
 * from the global pool.
 */
export declare const COVERAGE_THRESHOLDS: MetricFloors & Record<string, MetricFloors | number>;
