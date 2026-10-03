/**
 * Serialization of a SceneLoader's view passes.
 *
 * One lock covers every kind of work that touches the loaders' shared buffers:
 * a view pass (`updateView`'s sweep + atomic commit), the frame the lock is
 * held across between two passes, a progressive-refinement drain, and a
 * failed-loader retry. This class owns that lock and everything that decides
 * who runs next:
 *
 *   - the pending slot (one latest-wins view state, in {@link ViewStateQueue});
 *   - request generations and the waiters parked on them (#2943), which make
 *     `waitForUpdate()` and the dimension-animation pacing gate wait for the
 *     view they asked for to COMMIT, not merely to be accepted;
 *   - the targeted-resync stash (partition parts re-entering the frustum);
 *   - the B5 drag commit guarantee and its hold-expiry timer;
 *   - refinement start, hand-off, pre-emption by a retry (A10) and release.
 *
 * The state is explicit in {@link PassScheduler.phase}:
 *
 *   idle      lock free.
 *   pass      a view pass is running (generation {@link PassScheduler.passGen}).
 *   yielding  the lock is held across a frame with no pass in flight (between
 *             two serialized passes, or a refinement cancellation hand-off).
 *   refining  a refinement drain holds the lock it inherited from a pass tail
 *             or a load.
 *   retrying  a failed-loader retry holds the lock.
 *
 * What runs, and for which view, stays with the host (`SceneLoader`): this
 * class never reads geometry or loaders except through {@link PassSchedulerHost}.
 *
 * @module data/scene-loader/update-view/pass-scheduler
 */

import { log, Modules } from '../../../utils/log';
import { getErrorMessage } from '../../../utils/format-error';
import { scheduleFrame } from '../../../utils/schedule-frame';
import { noteRefinementComplete } from '../../../profiling/load-timeline';
import type { ViewState } from '../../data-loader-types';
import type { ViewStateQueue } from '../view-state/view-state-queue';

/**
 * Longest a superseding view may keep aborting in-flight passes without one
 * committing (B5): past it, the pass in flight — whose streaming policy stops
 * at the first cold rung, so it is the coarsest data for its slice — commits
 * before the newer view runs. Settled frames are unaffected (the last view
 * always runs to completion); only how often a drag shows an intermediate
 * slice changes.
 */
export const DRAG_COMMIT_INTERVAL_MS = 150;

/**
 * Longest a view pass may run and still be owed its commit (B5). A pass older
 * than this is waiting on something slow — typically a stalled chunk download
 * (a cold hosted edge can take tens of seconds) — and holding it would queue
 * every newer view behind that one download, freezing the scrub. Past the cap
 * the superseding view aborts it as it would without the guarantee.
 */
export const DRAG_COMMIT_MAX_HOLD_MS = 1000;

/**
 * Delay before {@link PassScheduler.kickRefinementIfIdle} re-checks a held
 * lock. Frame-scale-ish: responsive after the holder finishes, cheap while it
 * runs (one timer at a time).
 */
const REFINEMENT_KICK_RECHECK_MS = 100;

/** The explicit lock states; see the module doc. */
export type PassPhase = 'idle' | 'pass' | 'yielding' | 'refining' | 'retrying';

/** Per-request directives the scheduler routes (see `UpdateViewOptions`). */
export interface PassRequestOptions {
  resyncPaths?: ReadonlySet<string>;
}

/** What the scheduler needs from its owner. */
export interface PassSchedulerHost {
  /** The pending slot (shared with the host's build ctxs and prefetch). */
  readonly queue: ViewStateQueue;
  isDisposed(): boolean;
  /** An archive fault is latched: no refinement may start. */
  isFaulted(): boolean;
  /**
   * Whether a request names exactly the view the running pass is loading, with
   * no playback directive on either side (#2943) — the view half of a join;
   * the scheduler checks that a joinable pass is actually running.
   */
  matchesRunningPass(viewState: Partial<ViewState>): boolean;
  /** Display axes of the view the running pass projects for (B5 guard). */
  runningDisplayDims(): readonly number[];
  /** The host's view version, for the queue log lines. */
  viewVersion(): number;
  /** Re-enter `updateView` with a drained pending state. */
  runPass(state: Partial<ViewState>, opts?: PassRequestOptions): Promise<void>;
  /** Any sweep-registered loader with additive rungs left to stream. */
  anyHasMoreLODs(): boolean;
  /** Run the refinement orchestrator; it inherits the held lock. */
  runRefinement(): Promise<void>;
}

/** The handle of a running view pass, from {@link PassScheduler.beginPass}. */
export interface RunningPass {
  /** Request generation this pass carries. */
  readonly gen: number;
  /** Aborted when a newer view supersedes the pass (or on dispose). */
  readonly controller: AbortController;
}

/** How a view pass ended, for {@link PassScheduler.endPass}. */
export interface PassOutcome {
  /** The sweep hit an archive fault: its commit was discarded. */
  discarded: boolean;
  /** An archive fault is latched on the loader (from this pass or a lazy level). */
  faulted: boolean;
}

/** Owner of the view-pass lock, the pending slot's lifecycle and refinement hand-offs. */
export class PassScheduler {
  /** The serialization lock. Read through {@link phase} where the kind matters. */
  locked = false;
  /**
   * A refinement run is live. Not the same as `phase === 'refining'`: the final
   * phase's release opens the lock while the run is still unwinding, and a kick
   * landing in that window must still read busy.
   */
  refining = false;
  private retrying = false;

  private requestSeq = 0;
  private readonly requestGens = new WeakMap<object, number>();
  /** Generation of the view pass now running (0 while none is). */
  private passGen = 0;
  /** Newest generation whose view pass has finished. */
  private completedGen = 0;
  private passWaiters: Array<{ gen: number; resolve: () => void }> = [];

  /** Targeted resyncs that arrived while a pass was running (folded in after it). */
  private pendingResyncPaths: Set<string> | null = null;
  /** Resync paths handed to the follow-up pass with the pending state they queued. */
  queuedResyncPaths: Set<string> | null = null;
  /** The pending slot holds the empty state a resync queued (mergeable). */
  private pendingIsResyncOnly = false;

  /** `performance.now()` of the last committed view pass (B5). */
  lastCommitAt = 0;
  private passChainStartedAt = 0;
  private passStartedAt = 0;
  private holdExpiryTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Abort controller of the work holding the lock: the running pass, or the
   * refinement run that published its own. Aborted by a superseding view, a
   * pre-empting retry and dispose.
   */
  controller: AbortController | null = null;

  private retryLockWaiters: Array<(owned: boolean) => void> = [];
  private kickTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly host: PassSchedulerHost) {}

  /** The current lock state; see the module doc. */
  get phase(): PassPhase {
    if (!this.locked) return 'idle';
    if (this.passGen !== 0) return 'pass';
    if (this.retrying) return 'retrying';
    if (this.refining) return 'refining';
    return 'yielding';
  }

  /**
   * A view pass is running or queued — the data for the requested view has not
   * landed. A refinement drain does not count (it runs after the commit); a
   * queued state does, from the moment it is parked until a pass takes it.
   */
  get isLoadPassInProgress(): boolean {
    return (this.locked && !this.refining) || this.host.queue.hasPending();
  }

  /** Every generation minted so far has finished a view pass. */
  get allRequestsCompleted(): boolean {
    return this.requestSeq <= this.completedGen;
  }

  /** A failed-loader retry is waiting for the refinement run to hand it the lock. */
  get retryPreemptPending(): boolean {
    return this.retryLockWaiters.length > 0;
  }

  /** A lock-busy refinement re-check is scheduled. */
  get kickPending(): boolean {
    return this.kickTimer !== null;
  }

  // ---------------------------------------------------------------------------
  // Requests and view passes
  // ---------------------------------------------------------------------------

  /**
   * Gate a view request on the lock. Returns `null` when the lock is free: the
   * caller runs the pass now ({@link beginPass}). Otherwise the request is
   * parked, joined or queued, and the returned promise settles when a pass of
   * its generation (or newer) has committed.
   */
  request(viewState: Partial<ViewState>, opts: PassRequestOptions): Promise<void> | null {
    if (!this.locked) return null;
    if (opts.resyncPaths) {
      this.parkResync(opts.resyncPaths);
      return Promise.resolve();
    }
    // The running pass is already loading exactly this view: wait for ITS
    // commit rather than aborting it and loading the same slice again.
    if (this.canJoinRunningPass(viewState)) return this.waitFor(this.passGen);

    // Supersede: abort the work in flight so its reads bail and its commit is
    // skipped — unless the drag chain is owed a commit (B5), then hold it.
    if (this.runningPassOwesCommit(viewState)) this.armHoldExpiry();
    else this.controller?.abort();

    const supersededPrevious = this.host.queue.hasPending();
    // Every minted generation must finish a pass or be covered by a newer one.
    const gen = ++this.requestSeq;
    // A refinement hold has no view pass in flight: its first queued view
    // starts a new drag chain.
    if (this.refining && (!supersededPrevious || this.pendingIsResyncOnly)) {
      this.passChainStartedAt = performance.now();
    }
    this.requestGens.set(viewState, gen);
    this.host.queue.setPending(viewState);
    // A real change or an untargeted reprocess sweeps everything: a stashed
    // resync is moot.
    this.queuedResyncPaths = null;
    this.pendingIsResyncOnly = false;
    // Supersedes are logged as such, so a drag reads "supersedes previous
    // pending" rather than a run of identical "Update queued" lines.
    const version = this.host.viewVersion();
    log.info(
      Modules.SCENE_LOADER,
      supersededPrevious
        ? `Update queued - supersedes previous pending; in-flight v${version}`
        : `Update queued - in-flight v${version}`
    );
    return this.waitFor(gen);
  }

  /**
   * A targeted resync carries no new view state, so it never supersedes. During
   * a refinement hold it queues an empty state (merged with an earlier resync's,
   * dropped under a real pending view, which sweeps a superset); mid-pass it is
   * parked and folded in once the pass ends.
   */
  private parkResync(paths: ReadonlySet<string>): void {
    if (!this.refining) {
      this.pendingResyncPaths ??= new Set<string>();
      for (const path of paths) this.pendingResyncPaths.add(path);
      return;
    }
    if (this.host.queue.hasPending() && !this.pendingIsResyncOnly) return;
    this.queuedResyncPaths ??= new Set<string>();
    for (const path of paths) this.queuedResyncPaths.add(path);
    if (!this.host.queue.hasPending()) {
      this.host.queue.setPending({});
      this.pendingIsResyncOnly = true;
    }
  }

  private canJoinRunningPass(viewState: Partial<ViewState>): boolean {
    if (this.phase !== 'pass' || this.host.queue.hasPending()) return false;
    if (this.controller?.signal.aborted) return false;
    return this.host.matchesRunningPass(viewState);
  }

  /**
   * Whether a superseding view must let the running VIEW pass commit rather than
   * abort it (B5): the pass chain has gone {@link DRAG_COMMIT_INTERVAL_MS}
   * without a commit and the pass is younger than {@link DRAG_COMMIT_MAX_HOLD_MS}.
   * Only a running view pass can be owed anything (A8): across a frame yield,
   * a refinement drain or a retry the timing fields describe a pass that has
   * ended. A `displayDims` change never holds — geometry projected for the old
   * axes is not a truthful frame of a drag.
   */
  private runningPassOwesCommit(superseding: Partial<ViewState>): boolean {
    if (this.phase !== 'pass') return false;
    const next = superseding.displayDims;
    const current = this.host.runningDisplayDims();
    if (next && (next.length !== current.length || next.some((d, i) => d !== current[i]))) {
      return false;
    }
    const now = performance.now();
    return (
      now - this.passStartedAt < DRAG_COMMIT_MAX_HOLD_MS &&
      now - Math.max(this.lastCommitAt, this.passChainStartedAt) >= DRAG_COMMIT_INTERVAL_MS
    );
  }

  /**
   * Abort the held pass once its {@link DRAG_COMMIT_MAX_HOLD_MS} runs out, so
   * the view queued behind it runs even when no newer view arrives to re-check
   * the cap. One timer per pass; a no-op if the pass ended, or nothing is
   * pending, by the time it fires.
   */
  private armHoldExpiry(): void {
    if (this.holdExpiryTimer !== null) return;
    const controller = this.controller;
    const remaining = DRAG_COMMIT_MAX_HOLD_MS - (performance.now() - this.passStartedAt);
    this.holdExpiryTimer = setTimeout(
      () => {
        this.holdExpiryTimer = null;
        if (this.controller !== controller) return;
        if (this.host.queue.hasPending()) controller?.abort();
      },
      Math.max(0, remaining)
    );
  }

  private clearHoldExpiry(): void {
    if (this.holdExpiryTimer === null) return;
    clearTimeout(this.holdExpiryTimer);
    this.holdExpiryTimer = null;
  }

  /**
   * Take the lock for a view pass of `viewState`: its generation is the one it
   * was queued with, else a fresh one (a direct call, which also starts a new
   * drag chain). Stale resync stashes and hold timers die here.
   */
  beginPass(viewState: Partial<ViewState>): RunningPass {
    this.locked = true;
    if (!this.requestGens.has(viewState)) this.passChainStartedAt = performance.now();
    // Paths stashed for a pending state travel in its opts (`reenter`); any
    // still parked here are stale and must not narrow a later pass.
    this.queuedResyncPaths = null;
    this.pendingIsResyncOnly = false;
    const gen = this.requestGens.get(viewState) ?? ++this.requestSeq;
    this.requestGens.delete(viewState);
    this.passGen = gen;
    // A hold timer armed for an earlier pass must not outlive it (A8).
    this.clearHoldExpiry();
    const controller = new AbortController();
    this.controller = controller;
    this.passStartedAt = performance.now();
    return { gen, controller };
  }

  /** The running pass committed its frame (B5 commit clock). */
  noteCommitted(): void {
    this.lastCommitAt = performance.now();
  }

  /**
   * Close a view pass. A pass that committed settles the waiters it covers; a
   * superseded one parks the direct caller until the pass that supersedes it
   * commits (the returned promise, for the caller to await). Then decide what
   * runs next.
   */
  endPass(pass: RunningPass, outcome: PassOutcome): Promise<void> | undefined {
    this.clearHoldExpiry();
    this.passGen = 0;
    this.completedGen = Math.max(this.completedGen, pass.gen);
    const aborted = pass.controller.signal.aborted;
    let supersededWait: Promise<void> | undefined;
    if (!aborted && !outcome.discarded && !outcome.faulted) {
      this.resolveWaiters(pass.gen);
    } else if (aborted && !this.host.isDisposed() && !outcome.faulted) {
      supersededWait = this.waitFor(pass.gen);
    }
    if (outcome.faulted) this.abandonQueued();
    else this.next();
    return supersededWait;
  }

  /** Archive fault: nothing queued can run; release everything. */
  private abandonQueued(): void {
    this.host.queue.takePending();
    this.pendingResyncPaths = null;
    this.queuedResyncPaths = null;
    this.resolveWaiters();
    this.locked = false;
  }

  /**
   * After a view pass: a pending state runs after one frame; else refinement
   * drains the ladders; else the lock is released. Resyncs parked mid-pass ride
   * an empty pending state unless a real one (a superset sweep) is queued.
   */
  private next(): void {
    if (this.pendingResyncPaths) {
      if (!this.host.queue.hasPending()) {
        this.queuedResyncPaths = this.pendingResyncPaths;
        this.host.queue.setPending({});
        this.pendingIsResyncOnly = true;
      }
      this.pendingResyncPaths = null;
    }
    if (this.host.queue.hasPending()) {
      this.yieldThenRunPending('Queued updateView re-entry failed');
      return;
    }
    // The pass that just finished is the latest-wins winner: settle the parked
    // waiters now, at refinement ENTRY — the pacing gate needs first-commit
    // latency, not full-ladder latency.
    this.resolveCompletedWaiters();
    if (this.host.anyHasMoreLODs()) {
      log.info(
        Modules.SCENE_LOADER,
        'Scheduling progressive LOD refinement (hasMoreLODs=true after update)'
      );
      this.startRefinement('Progressive refinement scheduling failed');
    } else {
      // Nothing to stream: the "refinement complete" milestone is trivial.
      noteRefinementComplete();
      this.locked = false;
    }
  }

  /**
   * Hold the lock across one frame (so at least one frame paints between
   * passes), then release it and run the NEWEST pending state. The state stays
   * in the slot across the yield (A8): a view arriving meanwhile supersedes it
   * through {@link request}, so the newest state runs and the resync-stash
   * rules apply unchanged. `scheduleFrame` degrades to a timer in hidden tabs
   * and runs synchronously without rAF.
   */
  private yieldThenRunPending(failureLabel: string): void {
    scheduleFrame(() => {
      this.locked = false;
      const state = this.host.queue.takePending();
      if (state === null) {
        // Flushed by a teardown during the frame: nothing re-enters.
        this.resolveCompletedWaiters();
        return;
      }
      this.reenter(state).catch((error: unknown) => {
        log.error(Modules.SCENE_LOADER, `${failureLabel}: ${getErrorMessage(error)}`, error);
      });
    });
  }

  /**
   * Re-enter a drained pending state, consuming the resync paths stashed for
   * it — the ONLY way a pending state may be re-entered, or the paths leak into
   * a later, unrelated same-view pass and narrow it.
   */
  private reenter(state: Partial<ViewState>): Promise<void> {
    const paths = this.queuedResyncPaths;
    this.queuedResyncPaths = null;
    // Keep the plain call shape when nothing is stashed.
    return paths ? this.host.runPass(state, { resyncPaths: paths }) : this.host.runPass(state);
  }

  // ---------------------------------------------------------------------------
  // Waiters
  // ---------------------------------------------------------------------------

  private waitFor(gen: number): Promise<void> {
    return new Promise<void>((resolve) => {
      this.passWaiters.push({ gen, resolve });
    });
  }

  /**
   * Settle the waiters covered by generation `upToGen` (all of them by default
   * — dispose / archive-fault flushes); newer ones stay parked.
   */
  resolveWaiters(upToGen = Infinity): void {
    if (this.passWaiters.length === 0) return;
    const covered = this.passWaiters.filter((w) => w.gen <= upToGen);
    if (covered.length === 0) return;
    this.passWaiters = this.passWaiters.filter((w) => w.gen > upToGen);
    for (const w of covered) w.resolve();
  }

  /** Settle the waiters the newest finished view pass covers. */
  resolveCompletedWaiters(): void {
    this.resolveWaiters(this.completedGen);
  }

  // ---------------------------------------------------------------------------
  // Refinement
  // ---------------------------------------------------------------------------

  /**
   * Run the refinement orchestrator under the lock (taken here if not already
   * held). An error escaping the orchestrator glue — each loop releases the
   * lock in its own `finally`, so this is a double fault — releases the lock
   * and drains whatever queued meanwhile, or the viewer would freeze.
   *
   * A disposed or archive-faulted loader runs no drain: the lock is released
   * (and what queued drained) instead. The post-load kick calls in
   * unconditionally, and a lazy LOD level can latch the fault during the load —
   * a lock taken for a run that never starts would be held forever, parking
   * every view and refusing every retry.
   */
  startRefinement(failureLabel: string): void {
    if (this.host.isDisposed() || this.host.isFaulted()) {
      this.releaseAndDrain();
      return;
    }
    this.locked = true;
    this.host.runRefinement().catch((error: unknown) => {
      log.error(Modules.SCENE_LOADER, `${failureLabel}: ${getErrorMessage(error)}`);
      this.releaseAndDrain();
    });
  }

  /**
   * Kick refinement from OUTSIDE a pass (a deferred subtree registered loaders
   * mid-session, a retry committed part of a ladder, an abandoned-rung re-drain).
   * Busy — the lock held, or a run still unwinding — re-checks on a plain timer
   * (one at a time; `scheduleFrame` would recurse synchronously without rAF).
   */
  kickRefinementIfIdle(): void {
    if (this.host.isDisposed() || this.host.isFaulted()) return;
    if (!this.host.anyHasMoreLODs()) return;
    if (this.locked || this.refining) {
      if (this.kickTimer !== null) return;
      this.kickTimer = setTimeout(() => {
        this.kickTimer = null;
        this.kickRefinementIfIdle();
      }, REFINEMENT_KICK_RECHECK_MS);
      return;
    }
    log.info(Modules.SCENE_LOADER, 'Kicking progressive LOD refinement (loader idle)');
    this.startRefinement('Deferred-activation refinement failed');
  }

  /** A refinement run starts, publishing its own abort controller. */
  beginRefinement(controller: AbortController): void {
    this.refining = true;
    this.controller = controller;
  }

  /**
   * A refinement run unwound. A retry that pre-empted it inherits the lock,
   * unless the lock went to a view hand-off (`lockHandedOff`) or a dispose.
   */
  endRefinement(lockHandedOff: boolean): void {
    this.refining = false;
    this.settleRetryWaiters(lockHandedOff);
  }

  /**
   * A refinement run cancelled by a pending view hands the lock over across one
   * frame. The state it took goes back into the slot until the frame fires (A8)
   * — synchronous with the loop's take, so nothing can interleave.
   */
  handOff(pendingState: Partial<ViewState>): void {
    this.host.queue.setPending(pendingState);
    this.yieldThenRunPending('Refinement cancellation re-entry failed');
  }

  /**
   * The refinement run's final release: open the lock and drain a state queued
   * during the last pass (else settle the waiters). A pre-empting retry takes
   * the lock over in {@link endRefinement} instead.
   */
  releaseRefinementLock(): void {
    if (this.retryPreemptPending && !this.host.isDisposed()) return;
    this.releaseAndDrain();
  }

  /**
   * Release the lock and run what queued while it was held — or, with nothing
   * queued, settle the parked waiters (a drained state's own pass settles them
   * at its commit; settling here too would release the pacing gate early).
   */
  releaseAndDrain(): void {
    this.locked = false;
    if (this.host.isDisposed()) return;
    if (!this.drainPending()) this.resolveCompletedWaiters();
  }

  /**
   * Re-enter a state queued while the lock was held (on a microtask, see
   * `ViewStateQueue.drain`), with its stashed resync paths.
   *
   * @returns Whether a state was queued (its pass then settles the waiters).
   */
  drainPending(): boolean {
    return this.host.queue.drain((state) => this.reenter(state));
  }

  // ---------------------------------------------------------------------------
  // Retries
  // ---------------------------------------------------------------------------

  /**
   * Take the lock for a failed-loader retry (A10). A free lock is taken at once.
   * A refinement drain holding it, with no view queued to run next, is
   * pre-empted the way a view change pre-empts it — its reads are aborted and
   * it hands the lock over as it unwinds. A view pass holding it (or one queued
   * behind the drain) keeps it: the retry is reported deferred.
   *
   * The pending check is made HERE, once: a view arriving after the
   * pre-emption queues behind the retry (one loader's retry pass) instead of
   * taking the lock from it. Yielding would turn a retry already accepted into
   * a deferred one the user has to repeat, for a wait the size of one pass.
   *
   * @returns Whether the caller now holds the lock (release with {@link releaseRetry}).
   */
  acquireForRetry(): Promise<boolean> {
    if (!this.locked) {
      this.locked = true;
      this.retrying = true;
      return Promise.resolve(true);
    }
    if (this.phase !== 'refining' || this.host.queue.hasPending()) return Promise.resolve(false);
    const owned = new Promise<boolean>((resolve) => this.retryLockWaiters.push(resolve));
    this.controller?.abort();
    return owned;
  }

  /** Release a lock taken by {@link acquireForRetry}. */
  releaseRetry(): void {
    this.retrying = false;
    this.locked = false;
  }

  private settleRetryWaiters(lockHandedOff: boolean): void {
    const waiters = this.retryLockWaiters;
    if (waiters.length === 0) return;
    this.retryLockWaiters = [];
    const lockHeld = !lockHandedOff && !this.host.isDisposed();
    if (lockHeld) this.retrying = true;
    waiters.forEach((resolve, i) => resolve(lockHeld && i === 0));
  }

  // ---------------------------------------------------------------------------
  // Teardown
  // ---------------------------------------------------------------------------

  /**
   * Release everything for a disposed loader: timers, the lock, the pending
   * slot and resync stashes. Every waiter settles (resolve-only, never reject)
   * so no caller hangs across a dataset switch. The controller is left for the
   * host's resource teardown to abort.
   */
  dispose(): void {
    this.clearHoldExpiry();
    if (this.kickTimer !== null) clearTimeout(this.kickTimer);
    this.kickTimer = null;
    this.settleRetryWaiters(true);
    this.locked = false;
    this.refining = false;
    this.retrying = false;
    this.host.queue.takePending();
    this.pendingResyncPaths = null;
    this.queuedResyncPaths = null;
    this.pendingIsResyncOnly = false;
    this.resolveWaiters();
  }
}
