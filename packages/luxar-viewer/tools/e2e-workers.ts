/**
 * Local Playwright parallelism for the E2E suite, scaled by how idle the box is.
 *
 * `playwright.config.ts` used to hardcode `workers: process.env.CI ? 1 : 4`. That ceiling is a
 * real constraint (see `MAX_LOCAL_WORKERS`), but it is a CEILING, not a target: a developer
 * workstation shared with CI runner slots and other jobs can be at a 1-minute load average of
 * 22 on 16 cores, and four Chromium instances on top of that starve the very input handling
 * Playwright's action timeouts measure. Measured on a 16-core box at load 12-24:
 * `dimension-animation.spec.ts` failed 15 of 21 tests with the config default of 4 workers and
 * passed 21 of 21 at `--workers=1`, every failure a bare wall-clock timeout
 * (`page.click: Timeout 10000ms exceeded` with the element already reported visible, enabled and
 * stable) with no product cause.
 *
 * Scaling the count by the box's free fraction turns that class of false failure into a slower
 * run. Only two counts have actually been measured (4 → 15/21 fail, 1 → 21/21 pass, both at load
 * 12-24 on 16 cores); the intermediate 2 and 3 this heuristic can also pick are interpolation,
 * not measurement.
 *
 * Known limits of the signal, deliberately accepted:
 *
 *  - The 1-minute load average LAGS by construction and is read ONCE, at config load. A run
 *    started just after a burst ends crawls on a box that is already idle again; one started
 *    into a lull takes the full ceiling straight into the next burst. The protection is
 *    probabilistic; the slowdown it costs is certain.
 *  - On Linux the load average counts uninterruptible-sleep tasks, so heavy I/O elsewhere (a
 *    fixture generation, a `git lfs pull`) throttles the suite for contention it does not
 *    actually compete for.
 *  - Sampling `os.cpus()[].times` twice would be a better, lag-free signal, but it needs a
 *    ~150 ms wait, and a Playwright config module is re-evaluated in EVERY worker process as
 *    well as in the parent — so that delay would be paid N+1 times on every run. (Nothing stops
 *    a config from awaiting: this package is `"type": "module"` and Playwright loads the config
 *    with `await import()`. The cost is the reason, not the platform.) The lag is the price of
 *    keeping startup free.
 *
 * Overrides, in precedence order: the Playwright CLI's own `--workers=N` beats anything this
 * module returns (and `--debug` forces 1), then `LUXAR_E2E_WORKERS=N`, then this
 * heuristic. Because the CLI wins AFTER the config is evaluated, the run's parallelism line is
 * printed from the E2E global setup, which is handed the resolved `FullConfig` — and reports its
 * `workers` as the run's ceiling, since the concurrency actually reached is decided later still
 * (see {@link formatE2EParallelismStamp}).
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

/** Why a particular worker count was chosen — printed in the run's parallelism line. */
export type WorkerReason = 'override' | 'capacity' | 'unknown-capacity' | 'ci';

/** A worker count together with the reason it was chosen. */
export interface WorkerDecision {
  /** Workers Playwright should run locally (always >= 1). */
  workers: number;
  /** Short human-readable justification, for the run's parallelism line. */
  reason: WorkerReason;
}

/** Inputs to the pure worker-count decision. */
export interface WorkerInputs {
  /** Logical CPU count, e.g. `os.availableParallelism()`. */
  cpus?: number;
  /** 1-minute load average, e.g. `os.loadavg()[0]`; non-finite means "no signal". */
  load1?: number;
  /** Explicit pin, e.g. a parsed `LUXAR_E2E_WORKERS`. */
  override?: number;
}

/** What the resolver returns: the decision plus the inputs it was taken from. */
export interface E2EWorkerResolution extends WorkerDecision {
  /** Logical CPU count observed on this machine. */
  cpus: number;
  /** 1-minute load average observed on this machine; `NaN` where the platform has none. */
  load1: number;
}

/** Whether a CPU count can be used as a divisor and a hard ceiling. */
function isUsableCpuCount(cpus: number | undefined): cpus is number {
  return typeof cpus === 'number' && Number.isFinite(cpus) && cpus >= 1;
}

/**
 * Decide a local worker count from explicit inputs — pure, so it is the tested surface.
 *
 * Rules, in order:
 *
 * 1. A finite integer `override` wins outright, clamped to `[1, cpus]`. The lower clamp is
 *    Playwright's (it rejects 0); the upper one is a TYPO GUARD and nothing more —
 *    `LUXAR_E2E_WORKERS=40` on a 16-core box is a plausible slip for 4, and `1e21` is an integer
 *    that would have Playwright pre-allocate 1e21 worker slots. It is not a claim that one
 *    worker per core is the right amount: the heuristic below is deliberately NOT capped by the
 *    core count (a 1-core idle box still gets {@link MAX_LOCAL_WORKERS}), because the binding
 *    resource is the GIL-bound dataset server rather than the cores, and capping by cores would
 *    lower the historical default on small machines that nothing here has measured. An override
 *    may likewise exceed {@link MAX_LOCAL_WORKERS} — that is the point of having it, since a pin
 *    encodes a measurement someone took on their own box. A fractional or non-numeric override
 *    is ignored rather than rounded, because `LUXAR_E2E_WORKERS=two` silently meaning "4" is
 *    worse than being dropped.
 * 2. If `load1` is not a finite number `>= 0`, or `cpus` is not a finite number `>= 1`, return
 *    the {@link MAX_LOCAL_WORKERS} ceiling with reason `unknown-capacity` — the pre-existing
 *    behaviour, which is the right default when either half of the capacity signal is missing.
 *    A platform without a load average at all reaches this branch through a `NaN` `load1` (see
 *    {@link resolveE2EWorkers}); on Linux a load of exactly `0` is a genuinely idle box and
 *    takes the capacity path below.
 * 3. Otherwise scale the ceiling by the box's FREE FRACTION:
 *    `freeFraction = (cpus - load1) / cpus`,
 *    `workers = clamp(round(MAX_LOCAL_WORKERS * freeFraction), 1, MAX_LOCAL_WORKERS)`.
 *
 *    Rounding to NEAREST, not down: with `floor` the top count was reachable only at a load
 *    average of exactly 0, so an ordinary developer box at load 1.8 got 3 where it had always
 *    had 4. An oversubscribed box makes the product negative; the lower clamp is what turns that
 *    into 1, which is why no `max(0, …)` term appears above.
 *
 *    At `MAX_LOCAL_WORKERS = 4` the bands are fractions of the box, so they hold at any size —
 *    4 while at least 7/8 of it is free, 3 down to 5/8, 2 down to 3/8, 1 below that:
 *
 *    | free fraction  | workers | 16-core example       |
 *    | -------------- | ------- | --------------------- |
 *    | >= 7/8 (0.875) |       4 | load1 <= 2            |
 *    | >= 5/8 (0.625) |       3 | load1 in (2, 6]       |
 *    | >= 3/8 (0.375) |       2 | load1 in (6, 10]      |
 *    | anything less  |       1 | load1 > 10            |
 *
 *    The scaling is deliberately scale-FREE: an idle box of ANY size keeps the historical
 *    ceiling, so this only ever backs OFF under load — a 2-core laptop with nothing running is
 *    not penalised for being small, which nothing here has measured. The one configuration
 *    measured green is the bottom band: 16 cpus at load 12 → 1 worker (at load 22.2 the default
 *    sizing chose 1 and the two specs that had been failing passed in 27 s). The counts in
 *    between are interpolation.
 *
 * @param inputs CPU count, 1-minute load average, and any explicit override.
 * @returns The worker count and the reason for it.
 */
export function chooseLocalWorkers({ cpus, load1, override }: WorkerInputs): WorkerDecision {
  if (typeof override === 'number' && Number.isInteger(override)) {
    // No usable CPU count means no defensible ceiling either, so only the lower clamp applies.
    const hardCeiling = isUsableCpuCount(cpus) ? cpus : Number.POSITIVE_INFINITY;
    return { workers: Math.min(hardCeiling, Math.max(1, override)), reason: 'override' };
  }

  // The `typeof` guard on `load1` is inline (rather than hoisted into a named boolean) so the
  // fall-through narrows it to `number`: a caller reached from untyped JS — a Playwright config,
  // a script — can hand this function a string, and `'busy' - 16` is silently NaN.
  if (
    !isUsableCpuCount(cpus) ||
    typeof load1 !== 'number' ||
    !Number.isFinite(load1) ||
    load1 < 0
  ) {
    return { workers: MAX_LOCAL_WORKERS, reason: 'unknown-capacity' };
  }

  // No `max(0, …)` on the numerator: an oversubscribed box gives a negative product, which the
  // lower clamp below already turns into 1.
  const freeFraction = (cpus - load1) / cpus;
  const scaled = Math.round(MAX_LOCAL_WORKERS * freeFraction);
  return {
    workers: Math.min(MAX_LOCAL_WORKERS, Math.max(1, scaled)),
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
  // Neither of these respects a cgroup CPU quota, so a container on a big host still reads the
  // host's cores (unchanged from before this module existed, and CI pins 1 anyway).
  const cpus = os.availableParallelism?.() ?? os.cpus().length;
  // Windows has no load average and `os.loadavg()` returns a hardcoded [0, 0, 0]. That is NO
  // SIGNAL, not an idle box, so it must not take the capacity path — report it as NaN and let
  // `chooseLocalWorkers` fall through to the `unknown-capacity` ceiling.
  const load1 = os.platform() === 'win32' ? Number.NaN : os.loadavg()[0];

  if (env.CI) {
    return { workers: 1, reason: 'ci', cpus, load1 };
  }

  const raw = env.LUXAR_E2E_WORKERS?.trim();
  const parsed = raw ? Number(raw) : Number.NaN;
  const override = Number.isFinite(parsed) ? parsed : undefined;

  return { ...chooseLocalWorkers({ cpus, load1, override }), cpus, load1 };
}

let memoizedPlan: E2EWorkerResolution | undefined;

/**
 * The run's sizing decision, computed at most once per process.
 *
 * Both `playwright.config.ts` (which needs the count) and the E2E global setup (which reports
 * it) go through this, so the printed line cannot disagree with the count the config asked for —
 * `os.loadavg()` moves between the two calls, and the memo is the only thing preventing that.
 *
 * Per process by design: Playwright re-evaluates the config module in every worker, so "decided
 * once" means once per process, not once per run. Nothing depends on the workers agreeing with
 * the parent — only the parent's count configures the run, and only the global setup prints it.
 *
 * @returns The memoized {@link resolveE2EWorkers} result for `process.env`.
 */
export function e2eWorkerPlan(): E2EWorkerResolution {
  memoizedPlan ??= resolveE2EWorkers();
  return memoizedPlan;
}

/**
 * Build the one-line parallelism stamp for a run.
 *
 * `configuredWorkers` comes from the RESOLVED `FullConfig`, not from `plan`: Playwright applies
 * `--workers=N` (and `--debug`, which forces 1) after the config module has been
 * evaluated, so the two can legitimately differ — and that is exactly the case a reader needs
 * told, since the docs recommend `--workers=1` for diagnosing a flake.
 *
 * It is a CEILING for the run, not the concurrency reached: Playwright's own reporter prints
 * `min(config.workers, maxConcurrentTestGroups)`, and that product is computed after global
 * setup, so a one-file run of a spec pinned to `mode: 'default'` is stamped `3 max` here while
 * Playwright goes on to say "using 1 worker". Hence "max", and hence no cause is claimed for a
 * divergence: `--workers=N` is one route to it, but `playwright.perf.config.ts` sets
 * `workers: 1` in the config, `--ui` takes the count from its own panel, and a connected watch /
 * VS-Code session forces 1 — none of them a command-line flag.
 *
 * @param configuredWorkers Worker ceiling Playwright resolved for the run (`config.workers`).
 * @param plan Sizing decision and the inputs it was taken from.
 * @returns A single line in the house `[emoji] [Module] message` format.
 */
export function formatE2EParallelismStamp(
  configuredWorkers: number,
  plan: E2EWorkerResolution
): string {
  const load = Number.isFinite(plan.load1) ? plan.load1.toFixed(1) : 'n/a';
  const sizing =
    configuredWorkers === plan.workers
      ? `(${plan.reason})`
      : `(${plan.reason} sized ${plan.workers}, run with ${configuredWorkers})`;
  const plural = configuredWorkers === 1 ? '' : 's';
  return (
    `[🧵] [E2E] parallelism: max ${configuredWorkers} worker${plural} — ` +
    `${plan.cpus} cpus, load1 ${load} ${sizing}`
  );
}
