/**
 * Types for `coverage-thresholds.mjs`.
 *
 * Without this, `vitest.config.ts` imports the thresholds as `any`, so its
 * consumers get no checking for the module's exports or value shapes. Runtime
 * validation of metric names and zero-match glob keys remains the responsibility
 * of `scripts/check-coverage-slack.mjs`.
 *
 * Declared here rather than by converting the module to TypeScript because
 * `scripts/check-coverage-slack.mjs` imports it too, at plain Node with no
 * transform step.
 */

/** The four coverage metrics vitest reports, in the order they are printed. */
export declare const METRICS: readonly ['lines', 'statements', 'functions', 'branches'];

/** How far a floor may sit below the measured value before the guard fails. */
export declare const MAX_SLACK_POINTS: number;

/** How far coverage may move from its recorded baseline before warning. */
export declare const MAX_EROSION_POINTS: number;

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

/** Measurements from the run that established each coverage floor. */
export declare const COVERAGE_RECORDED: MetricFloors & Record<string, MetricFloors | number>;
