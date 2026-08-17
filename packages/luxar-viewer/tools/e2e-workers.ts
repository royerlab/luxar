/**
 * Local Playwright parallelism for the E2E suite, sized from the box's spare capacity.
 *
 * `playwright.config.ts` used to hardcode `workers: process.env.CI ? 1 : 4`. That ceiling is a
 * real constraint (see `MAX_LOCAL_WORKERS`), but it is a CEILING, not a target: a developer
 * workstation shared with CI runner slots and other agents can be at a 1-minute load average of
 * 24 on 16 cores, and four Chromium instances on top of that starve the very input handling
 * Playwright's action timeouts measure. Measured on a 16-core box at load 12-24:
 * `dimension-animation.spec.ts` failed 15 of 21 tests with the config default of 4 workers and
 * passed 21 of 21 at `--workers=1`, every failure a bare wall-clock timeout
 * (`page.click: Timeout 10000ms exceeded` with the element already reported visible, enabled and
 * stable) with no product cause. Two issues were filed as viewer regressions off such timeouts.
 *
 * Sizing the count from spare capacity turns that class of false failure into a slower run.
 *
 * Overrides, in precedence order: the Playwright CLI's own `--workers=N` beats anything this
 * module returns (that is how the E2E daemon and the promotion job pin themselves to 1), then
 * `LUXAR_E2E_WORKERS=N`, then this heuristic.
 *
 * @module tools/e2e-workers
 */

import * as os from 'node:os';

/**
 * Most workers a local run may use, whatever the box's capacity.
 *
 * The binding constraint is NOT the GPU — it is the dataset server, a GIL-bound
 * `python3 -m http.server 9000` (`playwright.config.ts` `webServer[1]`) streaming thousands of
 * small zarr chunks to every worker at once. Evidence already in the tree:
 * `all-examples-smoke-test.spec.ts` raised its own timeout to 120 s to "absorb HTTP-server
 * contention when several worker-pool tabs decode mid-size datasets concurrently". Raise this
 * past 4 only together with a measurement, and if the server saturates, replace it rather than
 * adding workers.
 */
export const MAX_LOCAL_WORKERS = 4;

/**
 * Cores budgeted per worker.
 *
 * Each worker costs a whole Chromium (renderer process + GPU process + the viewer's own
 * DedicatedWorker pool) plus a share of the single-threaded dataset server, so one worker per
 * spare core oversubscribes badly; ~2 cores per worker matches the measurements above.
 */
export const CORES_PER_WORKER = 2;

/** Why a particular worker count was chosen — printed in the run's parallelism line. */
export type WorkerReason = 'override' | 'capacity' | 'unknown-load' | 'ci';

/** A worker count together with the reason it was chosen. */
export interface WorkerDecision {
  /** Workers Playwright should run locally (always >= 1). */
  workers: number;
  /** Short human-readable justification, for the stamp printed at config load. */
  reason: WorkerReason;
}

/** Inputs to the pure worker-count decision. */
export interface WorkerInputs {
  /** Logical CPU count, e.g. `os.cpus().length`. */
  cpus?: number;
  /** 1-minute load average, e.g. `os.loadavg()[0]`. */
  load1?: number;
  /** Explicit pin, e.g. a parsed `LUXAR_E2E_WORKERS`. */
  override?: number;
}

/** What the resolver returns: the decision plus the inputs it was taken from. */
export interface E2EWorkerResolution extends WorkerDecision {
  /** Logical CPU count observed on this machine. */
  cpus: number;
  /** 1-minute load average observed on this machine. */
  load1: number;
}

/**
 * Decide a local worker count from explicit inputs — pure, so it is the tested surface.
 *
 * Rules, in order:
 *
 * 1. A finite integer `override` wins outright, clamped to `>= 1` (Playwright rejects 0) and
 *    deliberately NOT capped at {@link MAX_LOCAL_WORKERS}: an explicit pin is a measurement
 *    someone took on their own box. A fractional or non-numeric override is ignored rather
 *    than rounded, because `LUXAR_E2E_WORKERS=two` silently meaning "4" is worse than being
 *    dropped.
 * 2. If `load1` is not a finite number `>= 0`, or `cpus` is not a finite number `>= 1`, return
 *    the {@link MAX_LOCAL_WORKERS} ceiling — the pre-existing behaviour, which is the right
 *    default when capacity is simply unknown.
 * 3. Otherwise budget {@link CORES_PER_WORKER} cores per worker out of the box's spare cores:
 *    `clamp(floor((cpus - load1) / CORES_PER_WORKER), 1, MAX_LOCAL_WORKERS)`.
 *
 * A load of exactly `0` is treated as a genuinely idle box rather than as missing data, which
 * is what Windows reports from `os.loadavg()` and also the right answer there.
 *
 * @param inputs CPU count, 1-minute load average, and any explicit override.
 * @returns The worker count and the reason for it.
 */
export function chooseLocalWorkers({ cpus, load1, override }: WorkerInputs): WorkerDecision {
  if (typeof override === 'number' && Number.isInteger(override)) {
    return { workers: Math.max(1, override), reason: 'override' };
  }

  // The `typeof` guards are inline (rather than hoisted into named booleans) so the fall-through
  // narrows both values to `number`: a caller reached from untyped JS — a Playwright config, a
  // script — can hand this function a string, and `'busy' - 16` is silently NaN.
  if (
    typeof cpus !== 'number' ||
    !Number.isFinite(cpus) ||
    cpus < 1 ||
    typeof load1 !== 'number' ||
    !Number.isFinite(load1) ||
    load1 < 0
  ) {
    return { workers: MAX_LOCAL_WORKERS, reason: 'unknown-load' };
  }

  const budgeted = Math.floor((cpus - load1) / CORES_PER_WORKER);
  return {
    workers: Math.min(MAX_LOCAL_WORKERS, Math.max(1, budgeted)),
    reason: 'capacity',
  };
}

/**
 * Read this machine and the environment, then apply {@link chooseLocalWorkers}.
 *
 * CI short-circuits to a single worker (software rendering is slower and less stable under
 * concurrency), keeping the historical `process.env.CI ? 1 : …` behaviour exactly.
 *
 * @param env Environment to read; injectable so tests need not mutate `process.env`.
 * @returns The decision plus the `cpus` / `load1` it was taken from, for logging.
 */
export function resolveE2EWorkers(env: NodeJS.ProcessEnv = process.env): E2EWorkerResolution {
  const cpus = os.cpus().length;
  // Windows has no load average and returns [0, 0, 0]; that reads as "idle", which is both
  // safe here and the honest answer on a platform that cannot report contention.
  const load1 = os.loadavg()[0];

  if (env.CI) {
    return { workers: 1, reason: 'ci', cpus, load1 };
  }

  const raw = env.LUXAR_E2E_WORKERS?.trim();
  const parsed = raw ? Number(raw) : Number.NaN;
  const override = Number.isFinite(parsed) ? parsed : undefined;

  return { ...chooseLocalWorkers({ cpus, load1, override }), cpus, load1 };
}
