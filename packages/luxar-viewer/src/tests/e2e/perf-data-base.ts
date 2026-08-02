/**
 * Shared dataset-server origin for the perf-bench specs.
 *
 * `playwright.perf.config.ts` boots the dataset HTTP server on
 * `Number(process.env.LUXAR_PERF_DATA_PORT ?? 9000)`. Every perf-bench
 * spec must navigate to that SAME origin, otherwise a run started with a
 * non-default `LUXAR_PERF_DATA_PORT` boots the server on one port while
 * the specs fetch datasets from a stale (or absent) :9000 — scenarios
 * then skip, fail, or silently reuse a foreign document root.
 *
 * This module is the single source of truth for that derivation, shared
 * by `line-perf-bench.spec.ts`, `gsplat-perf-bench.spec.ts`, and
 * `performance-tracking-perf-bench.spec.ts`.
 *
 * @module tests/e2e/perf-data-base
 */

/**
 * Resolve the dataset-server origin the perf-bench specs should fetch
 * from.
 *
 * Two environment overrides are honored, matching
 * `playwright.perf.config.ts`:
 *   - `LUXAR_PERF_DATA_BASE` — an explicit origin (e.g. when port 9000
 *     is already held by a foreign document root that Playwright's
 *     `reuseExistingServer` would otherwise silently reuse and 404).
 *     Takes precedence over the port override when set.
 *   - `LUXAR_PERF_DATA_PORT` — the port the perf config boots the
 *     dataset server on. Interpolated verbatim into a localhost origin.
 *
 * Defaults to `http://localhost:9000` when neither is set.
 *
 * @param env - Environment to read overrides from (defaults to
 *   `process.env`); passed explicitly by unit tests.
 * @returns The dataset-server origin. The default has no trailing
 *   slash; an explicit `LUXAR_PERF_DATA_BASE` is returned verbatim.
 */
export function resolvePerfDataBase(env: NodeJS.ProcessEnv = process.env): string {
  return env.LUXAR_PERF_DATA_BASE ?? `http://localhost:${env.LUXAR_PERF_DATA_PORT ?? 9000}`;
}

/** Dataset-server origin resolved once from the ambient environment. */
export const PERF_DATA_BASE = resolvePerfDataBase();
