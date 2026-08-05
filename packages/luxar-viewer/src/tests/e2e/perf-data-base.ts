/**
 * Shared dataset-server port + origin for the perf runs.
 *
 * `playwright.perf.config.ts` boots the dataset HTTP server on
 * `resolvePerfDataPort()`, and every perf-bench spec navigates to the
 * origin from `resolvePerfDataBase()`, which embeds that same port.
 * Deriving both from one place keeps them in lockstep — otherwise a run
 * started with a non-default `LUXAR_PERF_DATA_PORT` boots the server on
 * one port while the specs fetch datasets from a stale (or absent)
 * :9000 — scenarios then skip, fail, or silently reuse a foreign
 * document root.
 *
 * Consumed by `playwright.perf.config.ts`, `line-perf-bench.spec.ts`,
 * `gsplat-perf-bench.spec.ts`, and
 * `performance-tracking-perf-bench.spec.ts`.
 *
 * @module tests/e2e/perf-data-base
 */

/**
 * Resolve the port the perf config boots the dataset server on.
 *
 * `LUXAR_PERF_DATA_PORT` is coerced with `Number(...)` (so `'09000'`
 * normalizes to `9000`); defaults to `9000` when unset.
 *
 * @param env - Environment to read the override from (defaults to
 *   `process.env`); passed explicitly by unit tests.
 * @returns The dataset-server port number.
 */
export function resolvePerfDataPort(env: NodeJS.ProcessEnv = process.env): number {
  return Number(env.LUXAR_PERF_DATA_PORT ?? 9000);
}

/**
 * Resolve the dataset-server origin the perf-bench specs should fetch
 * from.
 *
 * Two environment overrides are honored:
 *   - `LUXAR_PERF_DATA_BASE` — an explicit origin (e.g. when port 9000
 *     is already held by a foreign document root that Playwright's
 *     `reuseExistingServer` would otherwise silently reuse and 404).
 *     Takes precedence over the port override when set.
 *   - `LUXAR_PERF_DATA_PORT` — the port the perf config boots the
 *     dataset server on, normalized via {@link resolvePerfDataPort} so
 *     the origin always names the port the server actually got.
 *
 * Defaults to `http://localhost:9000` when neither is set.
 *
 * @param env - Environment to read overrides from (defaults to
 *   `process.env`); passed explicitly by unit tests.
 * @returns The dataset-server origin. The default has no trailing
 *   slash; an explicit `LUXAR_PERF_DATA_BASE` is returned verbatim.
 */
export function resolvePerfDataBase(env: NodeJS.ProcessEnv = process.env): string {
  return env.LUXAR_PERF_DATA_BASE ?? `http://localhost:${resolvePerfDataPort(env)}`;
}

/** Dataset-server origin resolved once from the ambient environment. */
export const PERF_DATA_BASE = resolvePerfDataBase();
