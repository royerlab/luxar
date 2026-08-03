/**
 * Initial-load path for a single Points leaf node.
 *
 * Builds the spatial-index loader, registers it in the loader registry,
 * attaches an empty placeholder mesh so commit / retry / update can find
 * a target even when the initial fetch fails, then routes the data fetch
 * through the same `deriveNodeViewState` + `processPointsData` /
 * `commitPointsGeometry` path the main update loop and retry path use.
 *
 * Sibling of `data/points/handler.ts`, which handles the *update* path
 * (`loadAndStage` + `label`). This file is for the *initial* load.
 */

import type * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { LoaderError, classifyLoaderError } from './load-leaf-error-dispatch';
import {
  createPointsLoader as createPointsLoaderHelper,
  createProgressivePointsLoader as createProgressivePointsLoaderHelper,
} from '../loaders/loader-factory';
import type { SceneNode, DataLoader, ViewState } from '../../data-loader-types';
import type { PointsDataLoader, PointsMetadata } from '../../../types/points';
import type { NodeBuildCtx } from './build-ctx';

/**
 * Construct the points spatial-index loader and wire it to the monitor.
 */
function createPointsLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): DataLoader {
  const loader = createPointsLoaderHelper(node, loc, ctx.factoryDeps);
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/**
 * Construct the progressive (multi-LOD) Points loader. Composes the
 * parent's effective rendering attrs up the scene-graph ancestry so
 * each sub-LOD synthetic node inherits opacity/intensity/etc. Mirrors
 * the gsplats equivalent.
 */
async function createProgressivePointsLoader(
  node: SceneNode,
  nAdditive: number,
  ctx: NodeBuildCtx
): Promise<PointsDataLoader> {
  const loader = await createProgressivePointsLoaderHelper(
    node,
    nAdditive,
    ctx.applyEffectiveAttrs(node),
    ctx.factoryDeps
  );
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/** Result of the cheap half of Points node loading. */
export interface PointsCheapLoad {
  /** The empty placeholder mesh, already attached to the parent. */
  placeholder: THREE.Mesh;
  /** The constructed loader (NOT yet registered — the caller registers it). */
  loader: DataLoader;
}

/**
 * Cheap half of Points node loading: build the spatial-index loader and
 * attach an empty placeholder mesh — no array fetch, no commit, and
 * crucially **no registry registration**. Splitting this from the
 * expensive half lets `load-lod-group-node.ts` attach a deferred Points
 * level's placeholder up front while deferring the costly geometry load
 * to `loadPointsNodeExpensive` (per-level lazy loading), exactly as the
 * gsplats loader does. Registering an unloaded lazy level would pull it
 * into the scene-wide `updateView` sweep and defeat the deferral, so the
 * caller owns registration: the combined `loadPointsNode` (eager path)
 * registers immediately; lod_group lazy levels are NEVER registered — the
 * `LODGroupRegistry` drives their (re)loads instead.
 */
export async function loadPointsNodeCheap(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<PointsCheapLoad> {
  log.custom('📍', Modules.SCENE_LOADER, `Loading points: ${node.path}`);
  log.info(Modules.SCENE_LOADER, `  Has spatial index: ${node.hasSpatialIndex}`);
  log.info(Modules.SCENE_LOADER, `  Total points: ${node.attrs.n_points || 'unknown'}`);

  // Progressive multi-additive-LOD Points: walks `additive_<i>/` subgroups
  // and wraps them in a `PointsProgressiveLoader`. Single-LOD nodes
  // (no `n_additive_sublods` attr) take the standard path below.
  const nAdditive = (node.attrs as { n_additive_sublods?: number }).n_additive_sublods ?? 0;
  if (nAdditive > 1) {
    log.info(
      Modules.SCENE_LOADER,
      `  Additive sub-LODs: ${nAdditive} (progressive loading enabled)`
    );
  }

  const loader =
    nAdditive > 1
      ? ((await createProgressivePointsLoader(node, nAdditive, ctx)) as DataLoader)
      : createPointsLoader(node, loc, ctx);

  // Construct + attach an empty placeholder before fetching data, so an
  // initial-load failure leaves a recoverable scene state.
  // commit-points-geometry finds the placeholder by name and populates
  // it once data arrives (initial fetch or future retry/update); the
  // 0-points → N-points transition naturally takes the "different size"
  // branch in commitPointsGeometry. retryFailedLoader() reads the
  // placeholder's `userData.attrs` to derive the retry view state.
  const attrs = ctx.applyEffectiveAttrs(node) as unknown as PointsMetadata;
  // Pass the RAW uncomposed attrs alongside the composed ones: on a
  // colormapped node the authored gain is the scalar window, and telling a
  // leaf-authored window apart from an inherited ancestor gain needs both
  // (see `resolveColormapWindow`).
  const placeholder = ctx.nodeFactory.createEmptyPointsNode(
    node.path,
    attrs,
    loader,
    node.attrs as unknown as Partial<PointsMetadata>
  );
  parentThree.add(placeholder);

  return { placeholder, loader };
}

/**
 * Expensive half of Points node loading: derive the view state, fetch
 * the points via the loader, and commit geometry into the placeholder
 * (found by name). Safe to call after initial scene load returns (lazy
 * lod_group levels) — it checks `isDatasetLive()` before committing so a
 * deferred load still in flight when the user switches datasets never
 * writes into a disposed/replaced scene.
 */
export async function loadPointsNodeExpensive(
  node: SceneNode,
  ctx: NodeBuildCtx,
  loader: DataLoader
): Promise<void> {
  try {
    // Route initial load through deriveNodeViewState (same helper as
    // the main update path and retry) so initial / update / retry can
    // never silently load different query regions.
    const derived = ctx.deriveNodeViewState(node.path, node.attrs, {
      applyPartialExtendTolerance: true,
    });
    // Capture the version at DERIVE time (see loadGSplatsNodeExpensive) so a
    // deferred reload is stamped for the slice it actually loaded.
    const loadedViewVersion = ctx.getViewVersion();
    // For a fully-extended node `derived.viewState` carries the extend-to-all
    // tolerance sentinel (and pinned slice) on every non-displayed dim, so this
    // first load pulls the whole slice-independent node instead of only the
    // coincidental current-slice subset.
    const pointsViewState: ViewState = derived.viewState;

    const data = await (loader as PointsDataLoader).loadPoints(pointsViewState);

    if (data.pointCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible points for ${node.path} - object created for future updates`
      );
    }

    // Liveness gate: a deferred (lazy lod_group) load may resolve after the
    // dataset was switched/disposed; committing then would write into a stale
    // root group. Drop the commit silently — the abort-discard policy, matching
    // loadGSplatsNodeExpensive.
    if (!ctx.isDatasetLive()) return;

    // Commit data into the placeholder via the same path future
    // updateView() / retry calls use.
    ctx.commitPointsGeometry(ctx.processPointsData(node.path, data), undefined, loadedViewVersion);

    // A settled successful (re)load clears any prior failure record so a
    // recovered lazy level stops counting against the outcome report and the
    // auto-retry budget. No-op on a first successful load. Symmetric with the
    // catch's recordFailure. Unconditional (unlike the lines/gsplats twins,
    // which clear only when their `staged` commit landed): staging points
    // always yields a commit, so there is no landed signal to gate on —
    // matching the sweep handler's markPathHealthy.
    ctx.registry.clearFailure(node.path);
    log.success(Modules.SCENE_LOADER, `Loaded ${data.pointCount} points for ${node.path}`);
  } catch (error) {
    // Expected dispose-crossing: a read that passed its abort check can
    // still throw a non-AbortError against a torn-down store (dataset
    // switch aborts the signal FIRST, then disposes the store). The
    // dataset is dead — a failure record + error-level LoaderError would
    // be pure noise (and a spurious user toast). Symmetric with the
    // success path's liveness gate above.
    if (!ctx.isDatasetLive()) return;
    // Record the failure so `retryFailedLoader(path)` can target this node.
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}

/**
 * Load a single Points node on initial scene construction. Returns the
 * placeholder mesh once data has been committed (or empty when no
 * points are currently visible — the slice-update path will populate
 * it later). Throws `LoaderError` on failure so `loadLeafNode` can
 * dispatch by kind and keep the rest of the scene alive.
 *
 * Composition of the cheap (placeholder + loader) and expensive (fetch +
 * commit) halves; the non-LOD dispatch path uses this combined form so its
 * behaviour is unchanged. Registers the loader once the
 * initial load settles, so the update sweep can never overlap the initial
 * load on the same loader instance.
 */
export async function loadPointsNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
  const { placeholder, loader } = await loadPointsNodeCheap(node, parentThree, loc, ctx);
  try {
    await loadPointsNodeExpensive(node, ctx, loader);
  } finally {
    // Register only once the initial load has SETTLED (success or failure).
    // Registering before the await let a concurrent updateView sweep call
    // loader.updateView while the initial load was mid-flight on the same
    // instance — interleaving the shared accumulator buffers and clobbering
    // the per-update _activeSignal slot (routine during deferred-group
    // activation, where zoom-triggered loads overlap slice scrubs). Nothing
    // during the load resolves the loader through the registry maps (commit
    // helpers use rootGroup.getObjectByName), and load-scene's post-load
    // consumers run after every loadXNode has been awaited, so the deferral
    // is invisible to them. Registering on FAILURE too is deliberate:
    // retryFailedLoader resolves eager loaders through these maps, so a
    // failed initial load must stay retryable.
    ctx.registry.registerPointsLoader(node.path, loader);
  }
  return placeholder;
}
