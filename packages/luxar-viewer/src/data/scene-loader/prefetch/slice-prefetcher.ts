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
 * `updateView` does restore→deepen→store-prefix (it always carries a
 * `frameBudgetMs`, which is ALSO what makes the progressive loaders store
 * prefix ladders), and the foreground restores on the real tick. Shadows
 * are deliberately NOT monitor-connected (the factory is side-effect-free
 * by design), so shadow loads never double-count metrics.
 *
 * ## Never disturbing the foreground
 *
 * `prefetch()` is fire-and-forget (never awaited; every rejection —
 * including expected AbortErrors — is swallowed). `SceneLoader.updateView`
 * calls `abortInFlight()` at its very top, so a shadow pass never overlaps
 * a foreground pass start and can never delay a tick. Network fetches
 * share the global 64-wide fetch gate and are bounded by the budget +
 * abort. `releaseShadows()` (playback end) frees the shadow accumulators.
 *
 * @module data/scene-loader/prefetch/slice-prefetcher
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode, ViewState, DataLoader, GeometryKind } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { LoaderRegistry } from '../loaders/loader-registry';
import {
  createPointsLoader,
  createLinesLoader,
  createGSplatsLoader,
  createProgressivePointsLoader,
  createProgressiveLinesLoader,
  createProgressiveGSplatsLoader,
  type LoaderFactoryDeps,
} from '../loaders/loader-factory';
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

export class SlicePrefetcher {
  /** Lazily-built shadow loaders, keyed by node path (async: progressive
   *  factories open zarr subgroups). A failed build is dropped so the next
   *  prefetch retries instead of caching the rejection forever. */
  private shadows = new Map<string, Promise<AnyShadowLoader>>();

  /** Abort controller for the in-flight prefetch pass (null when idle). */
  private controller: AbortController | null = null;

  private disposed = false;

  constructor(private readonly ctx: SlicePrefetcherCtx) {}

  /**
   * Fire one background prefetch pass for `viewState` (the PREDICTED next
   * view). Single-flight: an in-flight pass is aborted first. Fire-and-
   * forget — never await this from a foreground path.
   *
   * @param viewState - Full predicted view state (t+1 slice position).
   * @param budgetMs - Per-pass LOD time budget. ALWAYS set on the shadow
   *   pass: it bounds the work AND makes the progressive loaders store
   *   prefix ladders (a budget-free shadow pass aborted mid-ladder would
   *   store nothing and the whole handoff would silently fail).
   */
  prefetch(viewState: ViewState, budgetMs: number): void {
    if (this.disposed) return;
    this.abortInFlight();
    const controller = new AbortController();
    this.controller = controller;

    const graph = this.ctx.getSceneGraph();
    if (!graph) return;

    const { registry } = this.ctx;
    for (const path of registry.loaders.keys()) {
      this.prefetchNode(path, 'points', viewState, budgetMs, controller.signal);
    }
    for (const path of registry.linesLoaders.keys()) {
      this.prefetchNode(path, 'lines', viewState, budgetMs, controller.signal);
    }
    for (const path of registry.gsplatLoaders.keys()) {
      this.prefetchNode(path, 'gsplats', viewState, budgetMs, controller.signal);
    }
  }

  /** Abort the in-flight shadow pass (called at every foreground pass start). */
  abortInFlight(): void {
    this.controller?.abort();
    this.controller = null;
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

  /** Derive the per-node state and run one shadow load (fire-and-forget). */
  private prefetchNode(
    path: string,
    kind: GeometryKind,
    viewState: ViewState,
    budgetMs: number,
    signal: AbortSignal
  ): void {
    const graph = this.ctx.getSceneGraph();
    const node = findNodeByPath(graph, path);
    if (!node) return;

    // Same derivation the handlers apply (extend_to_all + nd_transform);
    // lines skip the partial-extend tolerance override, like its handler.
    const derived = deriveNodeViewState(path, node.attrs, viewState, graph, {
      applyPartialExtendTolerance: kind !== 'lines',
    });
    if (derived.skip) return;
    if (!hasHiddenDims(derived.viewState)) return; // S-cache would skip it anyway

    const shadowViewState: ViewState = { ...derived.viewState, frameBudgetMs: budgetMs };

    void this.getShadow(path, kind, node)
      .then((shadow) => {
        if (signal.aborted || this.disposed) return;
        // Structurally identical view-state shapes across the three
        // geometry loader interfaces (same cast the handlers perform).
        return (shadow as DataLoader).updateView(shadowViewState, undefined, signal);
      })
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

    const build = async (): Promise<AnyShadowLoader> => {
      switch (kind) {
        case 'points':
          return nAdditive > 1
            ? createProgressivePointsLoader(node, nAdditive, effectiveAttrs, deps)
            : createPointsLoader(node, loc, deps);
        case 'lines':
          return nAdditive > 1
            ? createProgressiveLinesLoader(node, nAdditive, effectiveAttrs, deps)
            : createLinesLoader(node, loc, deps);
        case 'gsplats':
          return nAdditive > 1
            ? createProgressiveGSplatsLoader(node, nAdditive, effectiveAttrs, deps)
            : createGSplatsLoader(node, loc, deps);
      }
    };

    const promise = build();
    this.shadows.set(path, promise);
    // Drop failed builds so a transient error doesn't poison the map.
    promise.catch(() => {
      if (this.shadows.get(path) === promise) this.shadows.delete(path);
    });
    return promise;
  }
}
