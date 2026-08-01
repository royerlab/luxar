/**
 * SlicePrefetcher — projected t+1 prefetch during dimension playback.
 *
 * While a dimension is PLAYING, refinement is suppressed (the progressive
 * loaders report `hasMoreLODs === false` under a frame budget), leaving an
 * idle window between a tick's commit and the next tick. This prefetcher
 * uses that window to build the NEXT timepoint's decoded ladder and store
 * it in the shared SliceCache under the t+1 view signature, so the next
 * real tick's `restoreLadder` hits instantly and its whole frame budget
 * goes to DEEPENING the prefix instead of rebuilding it. Being driven by
 * `DimensionAnimationManager.peekNextValue`, it is loop/bounce/backward
 * aware — in particular it warms the loop WRAP (t=max → t=min), which the
 * linear-extrapolation chunk prefetch cannot predict.
 *
 * ## Why shadow loader instances
 *
 * The foreground loaders CANNOT be reused for a concurrent prefetch: they
 * hold mutable per-instance state (`_activeSignal` read by the RangeLoader
 * signal-source and the L0 proxy, a REUSED accumulator whose returned
 * arrays the next load overwrites, `loadedLODs`/`lastViewState` in the
 * progressive wrappers). And `SceneLoader.updateView` is single-flight —
 * routing a prefetch through it would abort/supersede the foreground pass
 * and merge t+1 into the persistent view state (stuck-display hazard).
 *
 * So each registered node gets a lazily-built SHADOW loader from the same
 * factory helpers: own accumulator, own signal, zero shared mutable state
 * with the foreground. The S-cache IS the handoff — the shadow's own
 * `updateView` does restore→deepen→store (it carries a `frameBudgetMs`, which
 * bounds how far one pass deepens ≈ one cold level and makes the progressive
 * loaders store the deepened prefix), and the foreground restores on the real
 * tick. Shadows are deliberately NOT monitor-connected (the factory is
 * side-effect-free by design), so shadow loads never double-count metrics.
 *
 * ## Persisting across ticks (background deepening)
 *
 * A cold LOD level takes longer than one playback frame to decode, so the
 * shadow deepen must NOT be aborted every tick — the foreground therefore
 * does NOT preempt it. Instead {@link prefetch} is a no-op while a batch is
 * still in flight (see {@link inFlight}): the running deepen completes and
 * caches its level, and the next FREE tick re-targets at the then-current
 * predicted view. Across playback loops this fills the SliceCache toward full
 * ladders, so cached quality climbs while the foreground stays responsive.
 *
 * This is safe because the shadow runs on its own loader instances (own
 * accumulator + signal), decodes on the worker pool, and only writes the
 * shared SliceCache under content-keyed entries — it cannot stall or corrupt
 * a foreground tick. `prefetch()` is fire-and-forget (never awaited; every
 * rejection, including expected AbortErrors, is swallowed); its fetches share
 * the global 64-wide fetch gate and are bounded by the budget + abort.
 * `abortInFlight()` / `releaseShadows()` (playback end) / `dispose()` tear it
 * down.
 *
 * @module data/scene-loader/prefetch/slice-prefetcher
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode, ViewState, DataLoader, GeometryKind } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { LoaderRegistry } from '../loaders/loader-registry';
import type { LoaderFactoryDeps } from '../loaders/loader-factory';
import { GEOMETRY_DESCRIPTORS } from '../geometry-descriptors';
import { deriveNodeViewState } from '../view-state/derive-node-view-state';
import { isAbortError } from '../../loaders';

/** Everything the prefetcher may read off its owning SceneLoader. */
export interface SlicePrefetcherCtx {
  /** Live scene graph root (null before a scene is loaded). */
  getSceneGraph(): SceneNode | null;
  /** Per-call dependency snapshot for the loader factory (incl. sliceCache). */
  factoryDeps(): LoaderFactoryDeps;
  /** The foreground loader registry — its keys are the nodes to prefetch. */
  registry: Pick<LoaderRegistry, 'loaders' | 'linesLoaders' | 'gsplatLoaders'>;
  /** Compose a node's effective rendering attrs up the scene-graph ancestry. */
  applyEffectiveAttrs(node: SceneNode): SceneNode['attrs'];
}

type AnyShadowLoader = DataLoader | LinesDataLoader | GSplatsDataLoader;

/** Depth-first exact-path lookup in the scene graph. */
function findNodeByPath(root: SceneNode | null, path: string): SceneNode | null {
  if (!root) return null;
  if (root.path === path) return root;
  for (const child of root.children ?? []) {
    const found = findNodeByPath(child, path);
    if (found) return found;
  }
  return null;
}

/** True when at least one dimension is hidden (S-cache eligibility gate). */
function hasHiddenDims(view: ViewState): boolean {
  return view.displayDims.length < view.slicePosition.length;
}

/**
 * Upper bound (ms) on how long an in-flight prefetch batch may pin the gate
 * before a new tick supersedes it. A normal deepen batch settles in well under
 * a second (one budget-bounded cold level); this only fires when a shadow
 * fetch/build genuinely stalls (a hung connection has no timeout of its own).
 * Since foreground ticks no longer abort the batch every frame, this staleness
 * guard is what keeps a stalled task from disabling prefetch for the rest of
 * the playback session. Generous so it never supersedes healthy deepening.
 */
const MAX_BATCH_STALL_MS = 5000;

export class SlicePrefetcher {
  /** Lazily-built shadow loaders, keyed by node path (async: progressive
   *  factories open zarr subgroups). A failed build is dropped so the next
   *  prefetch retries instead of caching the rejection forever. */
  private shadows = new Map<string, Promise<AnyShadowLoader>>();

  /** Abort controller for the in-flight prefetch batch (null when idle). */
  private controller: AbortController | null = null;

  /**
   * Number of shadow passes still running in the current batch. A new
   * `prefetch()` call is a no-op while this is > 0: a cold LOD level outlives
   * one playback frame, so letting the in-flight batch finish (each pass
   * deepens its node's ladder by ~1 level per pass and stores it) — instead of
   * aborting + restarting every tick — is what lets cached quality climb
   * across loops. The next free tick re-targets at the current predicted view.
   * A stall guard ({@link MAX_BATCH_STALL_MS}) supersedes a batch that never
   * settles, so a hung task can't pin this > 0 for the whole session.
   */
  private inFlight = 0;

  /** `performance.now()` when the in-flight batch started (for the stall guard). */
  private batchStartMs = 0;

  private disposed = false;

  constructor(private readonly ctx: SlicePrefetcherCtx) {}

  /**
   * Fire one background prefetch batch for `viewState` (the PREDICTED next
   * view) — one shadow pass per registered node. Fire-and-forget; never await
   * this from a foreground path.
   *
   * Persistent across ticks: if the previous batch is still running this is a
   * no-op (see {@link inFlight}). Each shadow pass deepens its node's cached
   * ladder by ~1 cold level per pass (the `budgetMs` deadline stops it after
   * the level it was on, which is stored incrementally) and returns; the next
   * free tick re-targets at the then-current predicted view. Over successive
   * playback loops this fills the SliceCache toward full ladders so revisits
   * hit the fast single-concat path — while the foreground stays responsive.
   *
   * @param viewState - Full predicted view state (t+1 slice position).
   * @param budgetMs - Per-pass LOD time budget. Bounds how far a single pass
   *   deepens (≈1 cold level); progressive loaders store each level as it
   *   lands, so an abort (playback end / dataset switch) keeps the depth
   *   already reached.
   */
  prefetch(viewState: ViewState, budgetMs: number): void {
    if (this.disposed) return;
    if (this.inFlight > 0) {
      // A batch is still deepening — let it finish (persist across ticks) so a
      // cold level completes. But if it has STALLED past the stall bound (a
      // hung fetch/build that will never settle its `allSettled`), reclaim the
      // gate so prefetch isn't disabled for the rest of the session.
      if (performance.now() - this.batchStartMs < MAX_BATCH_STALL_MS) return;
      this.abortInFlight();
    }

    const graph = this.ctx.getSceneGraph();
    if (!graph) return;

    const controller = new AbortController();
    this.controller = controller;
    this.batchStartMs = performance.now();

    const { registry } = this.ctx;
    const tasks: Array<Promise<void>> = [];
    for (const path of registry.loaders.keys()) {
      tasks.push(this.prefetchNode(path, 'points', viewState, budgetMs, controller.signal));
    }
    for (const path of registry.linesLoaders.keys()) {
      tasks.push(this.prefetchNode(path, 'lines', viewState, budgetMs, controller.signal));
    }
    for (const path of registry.gsplatLoaders.keys()) {
      tasks.push(this.prefetchNode(path, 'gsplats', viewState, budgetMs, controller.signal));
    }

    this.inFlight = tasks.length;
    // Reopen the gate once the whole batch settles so the next tick re-targets.
    // Guard against a newer controller (an abort + fresh batch raced ahead).
    void Promise.allSettled(tasks).then(() => {
      if (this.controller === controller) this.inFlight = 0;
    });
  }

  /**
   * Abort the in-flight shadow batch. Called on playback end / dataset switch
   * / dispose (NOT on every foreground tick — the batch is meant to persist so
   * background deepening completes; see the note in `SceneLoader.updateView`).
   */
  abortInFlight(): void {
    this.controller?.abort();
    this.controller = null;
    this.inFlight = 0;
  }

  /**
   * Abort + dispose all shadow loaders (frees their accumulators). Called
   * when playback ends; shadows are rebuilt lazily on the next play.
   */
  releaseShadows(): void {
    this.abortInFlight();
    const pending = [...this.shadows.values()];
    this.shadows.clear();
    for (const p of pending) {
      p.then(
        (loader) => loader.dispose(),
        () => undefined
      );
    }
  }

  dispose(): void {
    this.disposed = true;
    this.releaseShadows();
  }

  /**
   * Derive the per-node state and run one shadow load. Returns a promise that
   * settles when the pass finishes (or is skipped/aborted) so the batch can
   * reopen the {@link inFlight} gate; it never rejects (errors are logged).
   */
  private prefetchNode(
    path: string,
    kind: GeometryKind,
    viewState: ViewState,
    budgetMs: number,
    signal: AbortSignal
  ): Promise<void> {
    const graph = this.ctx.getSceneGraph();
    const node = findNodeByPath(graph, path);
    if (!node) return Promise.resolve();

    // Same derivation the handlers apply (extend_to_all + nd_transform);
    // lines skip the partial-extend tolerance override, like its handler.
    const derived = deriveNodeViewState(path, node.attrs, viewState, graph, {
      applyPartialExtendTolerance: kind !== 'lines',
    });
    if (derived.skip) return Promise.resolve();
    if (!hasHiddenDims(derived.viewState)) return Promise.resolve(); // S-cache would skip it anyway

    const shadowViewState: ViewState = {
      ...derived.viewState,
      frameBudgetMs: budgetMs,
      prefetch: true, // deepen mode: store each level, pin until foreground restores
    };

    return this.getShadow(path, kind, node)
      .then((shadow) => {
        if (signal.aborted || this.disposed) return;
        // Structurally identical view-state shapes across the three
        // geometry loader interfaces (same cast the handlers perform).
        return (shadow as DataLoader).updateView(shadowViewState, undefined, signal);
      })
      .then(() => undefined)
      .catch((err) => {
        if (!isAbortError(err)) {
          log.info(Modules.SCENE_LOADER, `t+1 prefetch skipped for ${path}: ${String(err)}`);
        }
      });
  }

  /** Get or lazily build the shadow loader for a node. */
  private getShadow(path: string, kind: GeometryKind, node: SceneNode): Promise<AnyShadowLoader> {
    const existing = this.shadows.get(path);
    if (existing) return existing;

    const deps = this.ctx.factoryDeps();
    const loc = zarr.root(deps.zarrStore);
    const effectiveAttrs = this.ctx.applyEffectiveAttrs(node);
    const nAdditive = (node.attrs.n_additive_sublods as number | undefined) ?? 0;

    const descriptor = GEOMETRY_DESCRIPTORS[kind];
    const build = async (): Promise<AnyShadowLoader> =>
      nAdditive > 1
        ? descriptor.createProgressiveLoader(node, nAdditive, effectiveAttrs, deps)
        : descriptor.createLoader(node, loc, deps);

    const promise = build();
    this.shadows.set(path, promise);
    // Drop failed builds so a transient error doesn't poison the map.
    promise.catch(() => {
      if (this.shadows.get(path) === promise) this.shadows.delete(path);
    });
    return promise;
  }
}
