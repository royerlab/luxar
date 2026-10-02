/**
 * The depth-sort coordinator's SortWorker lifecycle: spawn + guarded init,
 * the starved/dead init classification and its bounded retry (issue #1694),
 * the observable status, and the post-retry re-registration sweep.
 *
 * Owns the single persistent Comlink SortWorker the page's coordinators share
 * (spawn at app init via {@link warmUpSortWorker}, terminate when the LAST
 * attached coordinator is disposed — {@link detachCoordinator}). NOT part of
 * the round-robin data-worker pool — node registrations and their transferred
 * center buffers must live in exactly one worker.
 *
 * @module rendering/depth-sort-coordinator/worker-lifecycle
 */

import { wrap } from 'comlink';
// Vite's `?worker` import emits a bundled worker chunk (see
// worker-pool.ts for why `new Worker(new URL(...))` is not used).
import SortWorker from '../../workers/sort-worker?worker';
import type { SortWorkerAPI } from '../../workers/sort-worker';
import { hasCommittedData } from '../../types/committed-data';
import { config } from '../../config';
import { initializeWithGuard } from '../../workers/worker-pool/lifecycle/init-with-guard';
import { WorkerInitTimeoutError } from '../../workers/worker-pool/errors';
import { log, Modules } from '../../utils/log';
import {
  anyNodeWantsSorting,
  invalidateSortedNodeCommitStamps,
  isLiveOrderDependent,
  liveBlendingMode,
  workerHost,
  type CoordinatorState,
} from './state';

/**
 * Optional override for the sort-worker module URL (embedders whose
 * bundler lacks `?worker` support). Mirrors {@link setDataWorkerUrl};
 * set via `LuxarAppOptions.workerPath` alongside the data worker's.
 */
let sortWorkerUrlOverride: string | undefined;

/**
 * Override the URL used to construct the sort worker.
 *
 * An INTERNAL entry point, reachable only from inside the source tree (a
 * vendored/bundled viewer, the standalone bootstrap): it is not re-exported
 * from `src/index.ts`, and `package.json`'s `exports` map publishes only `.`
 * and `./styles.css`, so an npm consumer cannot import it at all.
 *
 * Call before `LuxarApp.init()`. That window used to run to the first
 * order-dependent commit, but the worker is now warmed up during app init
 * ({@link warmUpSortWorker}), so a later call would arrive after the
 * spawn it is meant to redirect. `LuxarAppOptions.workerPath` deliberately
 * does NOT reach here (it names the DATA worker bundle, a different chunk —
 * see `applyModuleOverrides`), so relocating the sort worker means calling
 * this setter yourself, before `init()`.
 */
export function setSortWorkerUrl(url: string): void {
  sortWorkerUrlOverride = url;
}

/**
 * WASM JS-shim URL forwarded into the worker's `initialize()` (the
 * main-thread `setWasmJsUrl` override does not cross the worker
 * boundary). Mirrors {@link setDataWorkerWasmPath}.
 */
let sortWorkerWasmPathOverride: string | undefined;

/**
 * Override the WASM JS-shim URL used inside the sort worker.
 *
 * Same timing contract as {@link setSortWorkerUrl}: before
 * `LuxarApp.init()`.
 */
export function setSortWorkerWasmPath(url: string): void {
  sortWorkerWasmPathOverride = url;
}

/**
 * Total init attempts for a STARVED (deadline-missing) worker, including
 * the first — so 3 means the initial attempt plus 2 retries. Bounded on
 * purpose: each attempt opens one more deadline-long window during which
 * commits attach continuations to a pending promise, and an unbounded retry
 * loop would restore exactly the leak the deadline was added to prevent.
 * Three attempts span at worst three init deadlines plus the 2 s and 4 s
 * backoffs between them (~96 s of wall clock at the default 30 s deadline),
 * which comfortably outlives a load-induced main-thread stall while staying
 * a fixed, small ceiling.
 */
const SORT_WORKER_INIT_MAX_ATTEMPTS = 3;

/**
 * Backoff before a starved retry, multiplied by the number of deadline
 * misses so far (2 s, then 4 s). Measured from the failure, on the
 * monotonic `performance.now()` clock — a wall-clock jump must not skip or
 * freeze a backoff.
 */
const SORT_WORKER_INIT_RETRY_BASE_MS = 2_000;

/**
 * Slack added to the self-wake timer on top of the backoff (see
 * {@link scheduleInitRetryWake}). The retry is gated on
 * `performance.now() >= initRetryNotBeforeMs`, so a wake scheduled for
 * EXACTLY that instant is a coin flip between the timer's own rounding and
 * the clock read — a lost toss costs a whole extra idle window. Small
 * enough to be invisible next to a 2 s backoff.
 */
const SORT_WORKER_INIT_RETRY_WAKE_SLACK_MS = 50;

/**
 * Per-sort RPC deadline (see scheduleSort). Generous — a sort is
 * O(N + buckets) over at most a few million centers, milliseconds on any
 * live worker; the deadline only trips on a crashed/wedged worker thread.
 */
export const SORT_RPC_TIMEOUT_MS = 30_000;

/**
 * Spawn + initialize the persistent sort worker.
 *
 * Init settle guard: a worker whose script dies during ASYNC module
 * evaluation (before `expose()` runs) emits an `error` event but never
 * settles the Comlink `initialize` RPC — and every order-dependent
 * commit attaches a continuation (closing over its centers provider,
 * which for points pins the full `LoadedPointsData`) to the cached
 * `initPromise`. Left pending forever, those closures accumulate one
 * per commit, unbounded. The shared `initializeWithGuard` races the RPC
 * against the `config.depthSort.workerInitTimeoutMs` deadline AND the
 * worker's own `error` / `messageerror` events, so the promise SETTLES on
 * every attempt, draining all queued continuations into the warn-once
 * degrade path. (It is shared with the data pool precisely so the two
 * startup paths cannot drift; the coordinator's hand-rolled copy never
 * grew the `messageerror` arm.)
 *
 * The settle contract is unconditional, but the DEGRADE is not (issue
 * #1694): only the deadline miss is retryable.
 * - A dead script (`error`/`messageerror`), a rejected `initialize`, or a
 *   constructor throw is PERMANENT: nothing about waiting longer would
 *   help, so the cached rejection stands for the session.
 * - Missing the deadline means only that init did not finish in time. At
 *   ~3M points the main thread is saturated long enough during a load for
 *   worker startup to lose that race, and the old unconditional
 *   stays-failed degrade then disabled depth sorting for the whole
 *   session over a condition that would have cleared in seconds. That
 *   case is retried, BOUNDED (see `SORT_WORKER_INIT_MAX_ATTEMPTS`),
 *   so the unbounded closure pile-up the deadline exists to stop cannot
 *   come back: between attempts the cached rejection is still what every
 *   commit sees, and only a finite number of fresh pending promises can
 *   ever exist.
 */
export function ensureWorker(): Promise<void> {
  if (workerHost.initPromise) return workerHost.initPromise;
  // Snapshot for the epoch guard on every write past the awaits below — see
  // `SortWorkerHost.initEpoch` for the hazard this closes.
  const epoch = workerHost.initEpoch;
  workerHost.initPromise = (async () => {
    let w: Worker;
    try {
      w = sortWorkerUrlOverride
        ? new Worker(sortWorkerUrlOverride, { type: 'module' })
        : new SortWorker();
    } catch (error) {
      // A constructor throw (CSP-blocked script, an embedder bundler with
      // no `?worker` support) is permanently fatal — classify it here so it
      // can never be mistaken for a starved deadline and retried.
      noteWorkerInitFailure(error);
      throw error;
    }
    workerHost.worker = w;
    workerHost.api = wrap<SortWorkerAPI>(w);
    try {
      const result = await initializeWithGuard(
        w,
        workerHost.api!,
        'SortWorker',
        config.depthSort.workerInitTimeoutMs,
        // The guard's own handlers are scoped to the init race; this runs on
        // settle, and the coordinator has no permanent ones to restore.
        () => {
          w.onerror = null;
          w.onmessageerror = null;
        },
        sortWorkerWasmPathOverride
      );
      if (epoch !== workerHost.initEpoch) {
        // Stale attempt: whatever started it is gone (a dispose, or a retry
        // that superseded it) while this init was awaited. Terminate the
        // worker so neither the thread nor its transferred centers buffers
        // leak, and write NOTHING module-scoped — the 'ready' stamp, the
        // warn-once re-arm, the retry clear and even the log line all belong
        // to whichever attempt is current NOW. `worker === w` is the same
        // identity check as the catch below: false whenever a live attempt
        // has already re-homed the fields, which is what keeps the live
        // worker/api pair intact. (`terminate()` is idempotent, so racing a
        // dispose that already terminated this worker is harmless.)
        w.terminate();
        if (workerHost.worker === w) {
          workerHost.worker = null;
          workerHost.api = null;
        }
        return;
      }
      workerHost.workerInitState = 'ready';
      // The unavailability episode is over: re-arm the warn-once so a
      // LATER genuine failure is reported rather than swallowed by the flag
      // an earlier starved attempt set. Also drop any armed retry — this
      // worker is live, there is nothing left to retry.
      workerHost.warnedWorkerUnavailable = false;
      workerHost.initTimeoutRetryPending = false;
      log.info(
        Modules.WORKER_POOL,
        `SortWorker ready (${result.wasmFallback ? 'TypeScript fallback' : 'WASM'})`
      );
    } catch (error) {
      // Terminate the wedged/failed worker so it can't hold resources.
      // `initPromise` stays rejected either way — every later commit keeps
      // landing in the warn-once catch, which is what BOUNDS the closure
      // accumulation the deadline exists to prevent. Whether that rejection
      // is the end of the story is noteWorkerInitFailure's call; only
      // maybeRetryStarvedWorkerInit ever clears it.
      w.terminate();
      if (workerHost.worker === w) {
        workerHost.worker = null;
        workerHost.api = null;
      }
      // Epoch guard (see `SortWorkerHost.initEpoch`): a STALE attempt's failure —
      // classically its orphaned init deadline firing long after a dispose —
      // must not classify, log, or arm anything against the attempt that is
      // current now. The worker above is still terminated; only the
      // bookkeeping is skipped.
      if (epoch === workerHost.initEpoch) noteWorkerInitFailure(error);
      throw error;
    }
  })();
  workerHost.initPromise.catch(() => {
    // Classified + logged by noteWorkerInitFailure above; this handler
    // exists only so the CACHED rejection is never an unhandled one (the
    // cache is deliberately kept — see the catch block).
  });
  return workerHost.initPromise;
}

/** Cancel a pending self-wake, if any (idempotent). */
export function clearInitRetryWake(): void {
  if (workerHost.initRetryWakeTimer !== null) {
    clearTimeout(workerHost.initRetryWakeTimer);
    workerHost.initRetryWakeTimer = null;
  }
}

/**
 * Request a frame from every attached coordinator that can take one. Returns
 * whether at least one wake was delivered.
 */
function wakeAttachedCoordinators(): boolean {
  let woken = false;
  for (const c of workerHost.coordinators) {
    if (!c.requestRender) continue;
    c.requestRender();
    woken = true;
  }
  return woken;
}

/**
 * Arm exactly ONE wake-up so an armed retry's backoff expiry is guaranteed
 * to be observed.
 *
 * {@link maybeRetryStarvedWorkerInit} runs only from the per-frame
 * scheduler, which `core/app/init/pipeline.ts` registers as a
 * NON-continuous per-frame callback — so the on-demand render loop
 * `stopAnimation()`s `config.animation.idleTimeoutMs` (default 2000 ms,
 * user-settable down to 500 ms) after the last `requestRender`, taking the
 * scheduler with it. Nothing else asks for a frame once a load finishes, and
 * the first backoff is 2000 ms: on a static "load it and look at it" scene
 * the loop can pause at or before the first eligible retry instant, so the
 * recovery would simply never fire (and the second attempt's 4 s backoff
 * would be unreachable without user interaction). `requestRender` is what
 * re-arms the loop, hence the per-frame callback, hence the retry.
 *
 * Bounded by construction: one pending wake at most, replaced on each
 * arming, cancelled when an attempt starts, when the retry is abandoned, and
 * by the host reset ({@link detachCoordinator}) — so it can never outlive its
 * arming nor wake a disposed app. A wake that cannot be DELIVERED re-arms itself instead of
 * being spent (see below).
 */
function scheduleInitRetryWake(delayMs: number): void {
  clearInitRetryWake();
  workerHost.initRetryWakeTimer = setTimeout(() => {
    workerHost.initRetryWakeTimer = null;
    // Wake EVERY attached coordinator: the retry runs from whichever one's
    // per-frame pass next finds something visible that wants sorting, and the
    // starved worker is every coordinator's. Read each CURRENT `requestRender`
    // (a dispose nulls it), never a captured one — an old app's closure must
    // not be resurrected here.
    if (wakeAttachedCoordinators()) return;
    // There is nobody to wake: `requestRender` is null before `configure` has
    // run, and `resortForCapture` nulls it DELIBERATELY for the duration of
    // an offline capture. Firing into the
    // void would consume the one wake while the retry stays armed — exactly
    // the never-recovers hole this wake exists to close — so re-arm the same
    // delay instead. Bounded: an undeliverable wake is a no-op tick (one
    // timer, still at most one pending), and delivery resumes as soon as the
    // capture's `finally` restores `requestRender`; the host reset clears
    // `initTimeoutRetryPending`, which stops the chain for good.
    //
    // The LOAD case needs no such re-arm, and deliberately gets none: there
    // `requestRender` IS wired, so the wake is delivered and the frame simply
    // declines to spend an attempt while `isLoadInProgress`. That sweep's own
    // commits each call the render wake-up the app installed via
    // `SceneLoaderManager.setRequestRender` (`core/app/init/pipeline.ts`;
    // `SceneLoader` fires it on every commit), so a natural frame — and with
    // it another retry chance — arrives when the sweep ends.
    if (workerHost.initTimeoutRetryPending) scheduleInitRetryWake(delayMs);
  }, delayMs);
}

/**
 * Classify an init failure and arm — or refuse — the bounded retry
 * (issue #1694). The starved/dead distinction is the whole point: a
 * deadline miss says nothing about the worker's health, every other
 * failure says the script will never run.
 */
export function noteWorkerInitFailure(error: unknown): void {
  if (error instanceof WorkerInitTimeoutError) {
    workerHost.initTimeoutCount++;
    if (workerHost.initTimeoutCount < SORT_WORKER_INIT_MAX_ATTEMPTS) {
      workerHost.initTimeoutRetryPending = true;
      const backoffMs = SORT_WORKER_INIT_RETRY_BASE_MS * workerHost.initTimeoutCount;
      workerHost.initRetryNotBeforeMs = performance.now() + backoffMs;
      workerHost.workerInitState = 'starved';
      // Arming a retry is useless if no frame ever comes to run it.
      scheduleInitRetryWake(backoffMs + SORT_WORKER_INIT_RETRY_WAKE_SLACK_MS);
      log.warning(
        Modules.WORKER_POOL,
        `SortWorker init missed its ${config.depthSort.workerInitTimeoutMs}ms deadline ` +
          `(attempt ${workerHost.initTimeoutCount}/${SORT_WORKER_INIT_MAX_ATTEMPTS}) — the main thread was ` +
          'likely starved by a large load; depth sorting is off (identity order drawn) until a ' +
          `retry succeeds, next attempt in ${backoffMs}ms`
      );
      return;
    }
    workerHost.initTimeoutRetryPending = false;
    clearInitRetryWake();
    workerHost.workerInitState = 'failed';
    log.error(
      Modules.WORKER_POOL,
      `SortWorker init missed its deadline ${workerHost.initTimeoutCount} times — giving up; depth ` +
        'sorting stays off (identity order drawn) until the next app re-init'
    );
    return;
  }
  // Not a deadline miss: the worker script is dead / unusable. Nothing to
  // retry — the cached rejection is the final answer, so no wake either.
  workerHost.initTimeoutRetryPending = false;
  clearInitRetryWake();
  workerHost.workerInitState = 'failed';
  log.error(Modules.WORKER_POOL, 'SortWorker failed to initialize', error);
}

/**
 * Observable depth-sort worker state, for the debug surface and tests —
 * the degrade used to be visible only as a console line (issue #1694).
 *
 * - `state`: `'idle'` = never spawned, or an init still in flight;
 *   `'ready'` = the worker initialized and sorts are flowing;
 *   `'starved'` = init missed its deadline and a bounded retry is armed or
 *   in flight (depth sorting is off MEANWHILE, not for the session);
 *   `'failed'` = permanently unavailable (dead script, or the starved
 *   retries were exhausted).
 * - `initTimeouts`: how many init attempts missed the startup deadline
 *   (`config.depthSort.workerInitTimeoutMs`, default 30 s) this session.
 *
 * The verdict describes INIT state only: a worker that dies AFTER a successful
 * init keeps reporting `'ready'` while every sort silently burns the
 * `SORT_RPC_TIMEOUT_MS` deadline instead (a code span, not a `{@link}` — it is
 * module-private, and this function is exported).
 */
export function getDepthSortWorkerStatus(): {
  state: 'idle' | 'ready' | 'starved' | 'failed';
  initTimeouts: number;
} {
  return { state: workerHost.workerInitState, initTimeouts: workerHost.initTimeoutCount };
}

/**
 * After a LATE (retried) init succeeds, get every sorted node's centers to
 * the worker again — the nodes of EVERY attached coordinator, since the
 * worker they all registered with is the one that was replaced.
 *
 * The worker is brand new and holds no registrations, and the coordinator
 * retains no centers by design (the buffers were transferred, or their
 * thunks were never paid because the commit's continuation drained into
 * the failure catch). So the only way back is a re-commit — exactly the
 * situation the blending-mode switch's switch-to-sorted branch solves, and by
 * exactly the same means: invalidate the stamps that would let the
 * memoized-concat fast path skip re-projection, then request ONE reprocess per
 * coordinator for its whole set.
 *
 * Gated on `requestReprocess` being wired: with nothing able to re-commit,
 * invalidating stamps would strand the nodes stamp-less for no gain. (The
 * RETRY itself is deliberately NOT gated on it — a fresh worker still lets
 * future commits register naturally.)
 */
function reregisterAfterLateWorkerInit(): void {
  for (const c of workerHost.coordinators) reregisterCoordinatorNodes(c);
}

/** {@link reregisterAfterLateWorkerInit} for one coordinator's nodes. */
function reregisterCoordinatorNodes(c: CoordinatorState): void {
  if (!c.requestReprocess) return;
  let anyInvalidated = false;
  for (const state of c.nodeStates.values()) {
    // A registered node already has its centers in the worker, so
    // re-committing it would be pure waste. This is a cheap guard, not a
    // race guard: `maybeRetryStarvedWorkerInit` attaches this function to
    // the fresh `initPromise` synchronously, so it always runs BEFORE any
    // commit's own continuation on that promise — no commit can have
    // registered by now. What the ordering does mean is that a commit
    // landing inside the retry window has its stamps invalidated one
    // microtask before its own continuation registers + sorts: it pays one
    // discarded sort and is re-committed by the reprocess below. Self-
    // healing waste, not corruption.
    if (state.registered) continue;
    // A tracked node with no `committedData` stamp is one whose stamps were
    // ALREADY invalidated and whose re-commit is still pending — the
    // switch-to-sorted branch of the blending-mode switch
    // firing inside the retry window, or an earlier run of this very sweep.
    // Both have already requested the reprocess that re-commits (and
    // re-registers) the node, so there is nothing to add here.
    // Notably NOT the LOD-demotion case, tempting as that reading is: every
    // demotion path pairs its `clearCommittedDataStamp` with
    // `releaseDepthSortNode` (`data/scene-loader.ts`'s `releaseLazy*`
    // callbacks), which deletes the node from `nodeStates` entirely — a
    // demoted level is never seen by this loop. The guard therefore stays as
    // the defensive peer of the module's other `!hasCommittedData` skips
    // rather than as the demotion filter.
    if (!hasCommittedData(state.mesh)) continue;
    if (!isLiveOrderDependent(liveBlendingMode(state.mesh))) continue;
    invalidateSortedNodeCommitStamps(state.mesh);
    anyInvalidated = true;
  }
  // `requestReprocess` is `SceneLoader.updateView({})`, and by now a view
  // sweep may well be in flight — the retry is only dispatched from a frame
  // where none was, and the init it awaited took real time. That is not a lost
  // call: the loader's serialization branch parks this state as pending and
  // re-enters `updateView` with it once the in-flight pass unwinds, so the
  // re-commit these stamp-less nodes need always happens. The in-flight pass
  // is usually ABORTED first (its commit skipped); only one that has gone
  // `DRAG_COMMIT_INTERVAL_MS` without a commit, and is itself younger than
  // `DRAG_COMMIT_MAX_HOLD_MS`, is let through to commit before the parked
  // state runs. The accepted cost of the abort is the
  // aborted pass's fetch/decode work, which the winning pass redoes. Gating on
  // `isLoadInProgress` instead would be the worse trade: the stamps are
  // already cleared at this point, so a skipped reprocess leaves the nodes
  // stamp-less and unsorted indefinitely — the bug itself.
  if (anyInvalidated) c.requestReprocess();
}

/**
 * Retry a STARVED init, at most `SORT_WORKER_INIT_MAX_ATTEMPTS`
 * times per session (issue #1694). Driven from the per-frame scheduler
 * rather than a timer: the frame loop is precisely where "the main thread
 * has room again" becomes observable, and it is already gated on the
 * conditions a retry must respect.
 *
 * Every gate below is about not WASTING one of the few attempts:
 * - the failure must be the retryable (deadline) kind and attempts must
 *   remain — `initTimeoutRetryPending` carries both,
 * - the backoff must have elapsed (a retry issued into the same stall
 *   would just miss the deadline again),
 * - no offline capture of the calling coordinator may be in flight: it drains
 *   synchronously against a time bound and cannot await an init that may run
 *   to its deadline,
 * - and SOMETHING of the calling coordinator must actually want sorting right
 *   now — a visible, still committed, live-order-dependent node
 *   (`anyNodeWantsSorting`, shared with {@link isDepthSortAvailable} so the
 *   gate and the monitor's note can never disagree about what "wants sorting"
 *   means). An idle or all-additive scene
 *   would otherwise burn the budget before the scene that needs it loads.
 *
 * Deliberately NOT gated on `requestReprocess`: a recovered worker is
 * worth having even when nothing can force a re-commit, because every
 * FUTURE commit then registers naturally. Only the re-registration sweep
 * needs that callback, and it checks for itself.
 *
 * (The caller explicitly suppresses this helper while `isLoadInProgress`
 * is true, so a retry never fires into an in-flight load sweep — the very
 * condition that starves init in the first place.)
 */
export function maybeRetryStarvedWorkerInit(c: CoordinatorState): void {
  if (!workerHost.initTimeoutRetryPending) return;
  if (workerHost.initTimeoutCount >= SORT_WORKER_INIT_MAX_ATTEMPTS) return;
  if (performance.now() < workerHost.initRetryNotBeforeMs) return;
  if (c.captureSuppressDepth > 0) return;
  if (!anyNodeWantsSorting(c)) return;

  // Consume the arm-flag and the cached rejection TOGETHER: this is the only
  // place the cached rejection is dropped, and consuming the flag in the same
  // synchronous step means a fresh attempt can never be started twice — the
  // failure bookkeeping then re-arms itself from the new attempt's own
  // outcome. Dropping the cached rejection INVALIDATES the previous
  // attempt, so bump the epoch with it (see `SortWorkerHost.initEpoch`) — and the
  // self-wake armed for this backoff has done its job.
  workerHost.initTimeoutRetryPending = false;
  workerHost.initPromise = null;
  workerHost.initEpoch++;
  clearInitRetryWake();
  const epoch = workerHost.initEpoch;
  log.warning(
    Modules.WORKER_POOL,
    `Retrying the starved SortWorker init (attempt ${workerHost.initTimeoutCount + 1}/` +
      `${SORT_WORKER_INIT_MAX_ATTEMPTS})`
  );
  // The re-registration below invalidates both freshness stamps on several
  // nodes at once, with one accepted, transient cost. A cleared
  // `loadedViewVersion` makes the node STALE for the LOD freshness check
  // (`scene/lod-freshness.ts::isFresh`), so a substitutive-LOD group's
  // display falls back to its coarsest ready level (`coarsestFreshOrReadyIndex`
  // in `scene/lod-group-registry.ts`) until the settle-gated `maybeKickReload`
  // climbs back. While `committedData` is absent, the node keeps its exact
  // cross-node `renderOrder`; only its within-mesh permutation stays stale
  // until the re-commit. The LOD fallback is exactly what the switch-to-sorted
  // blending-mode hook has always done for ONE node; an automatic recovery
  // just does it for several, which is why it is written down here rather than
  // left to be discovered.
  void ensureWorker().then(
    () => {
      // Epoch guard: a dispose (or another retry) between the dispatch and
      // this resolve means these nodes belong to a different session — see
      // `SortWorkerHost.initEpoch`.
      if (epoch !== workerHost.initEpoch) return;
      reregisterAfterLateWorkerInit();
    },
    () => {
      // Swallowed: noteWorkerInitFailure already logged and either re-armed
      // the retry or marked the worker permanently failed.
    }
  );
}

/**
 * Spawn + initialize the sort worker AHEAD of any data, at app init.
 *
 * The worker used to be spawned lazily by the first order-dependent commit
 * — which is the worst possible moment, because that commit lands exactly
 * when the main thread and the data-worker pool are saturated decoding the
 * scene. Starting here instead means `initWasm()` runs while the app is
 * still idle and finishes long before a million-element commit exists.
 *
 * Deliberately NOT folded into `DepthSortCoordinator.configure`: that is pure
 * wiring, every unit test calls it, and spawning there would change observable
 * behaviour across the whole suite.
 *
 * Fire-and-forget and idempotent — the commit path's own `ensureWorker()`
 * remains the correctness path (it dedupes on `initPromise`) and still
 * covers embedders that configure late.
 *
 * This spends the FIRST of the bounded init attempts
 * (`SORT_WORKER_INIT_MAX_ATTEMPTS`), before any node has committed. If it
 * misses the deadline, recovery is not immediate: nothing retries here, and
 * `maybeRetryStarvedWorkerInit` only spends an attempt once something visible
 * actually wants sorting — which is the point, since burning the budget on an
 * empty scene would leave none for the load that needs it. (Both names stay
 * code spans: they are module-private, and a doc link from an EXPORTED symbol
 * to one of those trips the TypeDoc warning ratchet.)
 *
 * The cost is unconditional: the sort-worker chunk and its WASM are fetched on
 * every page load, including scenes that never sort (all-additive points, an
 * opaque mesh, `?debug` with no dataset). `?depthSort=0` is the opt-out — it
 * spawns nothing at all.
 */
export function warmUpSortWorker(c: CoordinatorState): void {
  if (!c.depthSortEnabled) return;
  // No `Worker` constructor in this environment (the unit suite's default
  // `node` env, jsdom, SSR): there is nothing to warm up, and trying anyway
  // defined' and latches `workerInitState` at 'failed' page-wide — for the
  // rest of that test file, since the worker host is a singleton. The COMMIT path is
  // deliberately left unguarded: a node that actually asks for sorting must
  // still report honestly.
  if (typeof Worker === 'undefined') return;
  void ensureWorker().catch(() => {
    // Already logged by ensureWorker; a failed warm-up must not become an
    // unhandled rejection, and the commit path will retry.
  });
}

/**
 * False only when BOTH halves hold: depth sorting has GIVEN UP for this
 * session (`'failed'` — a dead worker script, or the starved retry budget
 * exhausted) AND something visible actually wants sorting right now
 * (`anyNodeWantsSorting`). A `'starved'` init still reports available, because
 * a retry is armed and expected to recover.
 *
 * The demand half is not cosmetic. The worker is warmed up unconditionally at
 * app init, so a CSP-blocked chunk latches `'failed'` on a scene with NO
 * order-dependent geometry at all (all-additive points, an opaque mesh,
 * `?debug` with no dataset) — announcing a degrade there would report on a
 * subsystem that session never uses, and drag the performance panel out of its
 * "No timing data yet" empty state to do it. Sharing the predicate with the
 * retry gate is what keeps the two readings of "wants sorting" identical.
 *
 * Exact contract, and it is asymmetric: a SHOWN note proves that visible,
 * committed, order-dependent geometry is being drawn in storage order. Its
 * ABSENCE proves nothing — a `'starved'` init reports available for the whole
 * retry window while identity order is drawn, `?depthSort=0` pins identity
 * order and reports available by definition, and a node whose commit could not
 * reach a HEALTHY worker (detached buffer, throwing centers thunk) is
 * invisible here. {@link getDepthSortWorkerStatus} is the finer-grained read.
 */
export function isDepthSortAvailable(c: CoordinatorState): boolean {
  return !(workerHost.workerInitState === 'failed' && anyNodeWantsSorting(c));
}

/**
 * Attach a coordinator to the shared worker (idempotent). Every path that may
 * need the worker attaches first, so the retry wake and the post-retry
 * re-registration reach it, and so a sibling's dispose cannot terminate the
 * worker under it.
 */
export function attachCoordinator(c: CoordinatorState): void {
  workerHost.coordinators.add(c);
}

/**
 * Detach a disposed coordinator. The LAST detach tears the worker down and
 * resets every host field: an embedder that disposes and re-inits in one page
 * must start with a full retry budget and a truthful status, not a stale
 * 'failed'/'starved' verdict about the worker just terminated (issue #1694).
 * While another coordinator is still attached the worker stays up — its nodes
 * are registered there.
 */
export function detachCoordinator(c: CoordinatorState): void {
  workerHost.coordinators.delete(c);
  if (workerHost.coordinators.size > 0) return;
  workerHost.worker?.terminate();
  workerHost.worker = null;
  workerHost.api = null;
  workerHost.initPromise = null;
  // ORPHAN any init attempt still in flight — kept adjacent to the
  // `initPromise = null` it invalidates, because the two are one invariant.
  // This is the reset that cannot be done by nulling a variable: the attempt's
  // init deadline timer lives in its own closure and nothing here can cancel
  // it, so the epoch is what stops it from classifying a miss against the NEXT
  // app's healthy worker (see `SortWorkerHost.initEpoch` for the failure chain).
  workerHost.initEpoch++;
  workerHost.warnedWorkerUnavailable = false;
  workerHost.workerInitState = 'idle';
  workerHost.initTimeoutRetryPending = false;
  workerHost.initTimeoutCount = 0;
  workerHost.initRetryNotBeforeMs = 0;
  // The armed self-wake must not survive the app that armed it (it would
  // request a frame from a re-inited app for a retry that no longer exists).
  clearInitRetryWake();
}
