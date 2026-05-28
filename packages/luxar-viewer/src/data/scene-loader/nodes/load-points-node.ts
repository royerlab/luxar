/**
 * Initial-load path for a single Points leaf node.
 *
 * Builds the spatial-index loader, registers it in the loader registry,
 * attaches an empty placeholder mesh so commit / retry / update can find
 * a target even when the initial fetch fails, then routes the data fetch
 * through the same `deriveNodeViewState` + `updatePointsGeometry` path
 * the main update loop and retry path use.
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

/**
 * Load a single Points node on initial scene construction. Returns the
 * placeholder mesh once data has been committed (or empty when no
 * points are currently visible — the slice-update path will populate
 * it later). Throws `LoaderError` on failure so `loadLeafNode` can
 * dispatch by kind and keep the rest of the scene alive.
 */
export async function loadPointsNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
  log.custom('📍', Modules.SCENE_LOADER, `Loading points: ${node.path}`);
  log.info(Modules.SCENE_LOADER, `  Has spatial index: ${node.hasSpatialIndex}`);
  log.info(Modules.SCENE_LOADER, `  Total points: ${node.attrs.n_points || 'unknown'}`);

  // Progressive multi-additive-LOD Points: walks `additive_<i>/` subgroups
  // and wraps them in a `PointsProgressiveLoader`. Single-LOD nodes
  // (no `n_additive_sublods` attr) take the standard path below.
  const nAdditive =
    (node.attrs as { n_additive_sublods?: number }).n_additive_sublods ?? 0;
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

  // Store loader for updates (route through the registry's
  // register* methods rather than mutating its internal map).
  ctx.registry.registerPointsLoader(node.path, loader);

  // Construct + attach an empty placeholder before fetching data, so an
  // initial-load failure leaves a recoverable scene state.
  // commit-points-geometry finds the placeholder by name and populates
  // it once data arrives (initial fetch or future retry/update); the
  // 0-points → N-points transition naturally takes the "different size"
  // branch in commitPointsGeometry. retryFailedLoader() reads the
  // placeholder's `userData.attrs` to derive the retry view state.
  const attrs = ctx.applyEffectiveAttrs(node) as unknown as PointsMetadata;
  const placeholder = ctx.nodeFactory.createEmptyPointsNode(node.path, attrs, loader);
  parentThree.add(placeholder);

  try {
    log.info(Modules.SCENE_LOADER, 'Initial ViewState for loading:');
    log.info(Modules.SCENE_LOADER, `  displayDims: [${ctx.viewState.displayDims.join(', ')}]`);
    log.info(Modules.SCENE_LOADER, `  slicePosition: [${ctx.viewState.slicePosition.join(', ')}]`);
    log.info(Modules.SCENE_LOADER, `  tolerance: [${ctx.viewState.tolerance.join(', ')}]`);

    // Route initial load through deriveNodeViewState (same helper as
    // the main update path and retry) so initial / update / retry can
    // never silently load different query regions. Initial load doesn't
    // apply the full-extend skip — we still want to construct the THREE
    // node so future slice changes can populate it; the skip return only
    // happens on update/retry where there's an existing node to leave
    // alone.
    const derived = ctx.deriveNodeViewState(node.path, node.attrs, {
      applyPartialExtendTolerance: true,
    });
    let pointsViewState: ViewState;
    if (derived.skip) {
      // Full-extend on initial load: behave as if extend_to_all
      // weren't set (load with the base view state) so the empty
      // node still gets constructed.
      pointsViewState = ctx.viewState;
    } else {
      pointsViewState = derived.viewState;
    }

    const data = await loader.loadPoints(pointsViewState);

    // Log if no initial points are visible (this is normal for nD slicing)
    if (data.pointCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible points for ${node.path} - object created for future updates`
      );
    }

    // Commit data into the placeholder via the same path future
    // updateView() / retry calls use. Unifies initial-load and update
    // through one geometry-commit code path.
    ctx.updatePointsGeometry(node.path, data);

    log.success(Modules.SCENE_LOADER, `Loaded ${data.pointCount} points for ${node.path}`);

    return placeholder;
  } catch (error) {
    // Record the failure so `retryFailedLoader(path)` can target this
    // node. The placeholder stays attached to the scene (added before
    // this try/catch), so retry can populate it.
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}
