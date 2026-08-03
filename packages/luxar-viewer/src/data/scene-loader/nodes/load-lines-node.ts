/**
 * Initial-load path for a single Lines leaf node.
 *
 * Mirrors `load-points-node.ts`: builds the spatial-index loader,
 * registers it, attaches an empty placeholder, fetches the line-segment
 * data using the per-node-derived view state, then routes through the
 * shared `processLinesData` + `commitLinesGeometry` pipeline.
 *
 * Lines differ from Points/GSplats in one place: the initial data fetch
 * uses `applyPartialExtendTolerance: false` because line bounds already
 * encode their non-displayed spatial extent (the partial-extend
 * tolerance override would double-apply during clipping).
 *
 * Sibling of `data/lines/handler.ts` (update path).
 */

import type * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { LoaderError, classifyLoaderError } from './load-leaf-error-dispatch';
import {
  createLinesLoader as createLinesLoaderHelper,
  createProgressiveLinesLoader as createProgressiveLinesLoaderHelper,
} from '../loaders/loader-factory';
import type { SceneNode } from '../../data-loader-types';
import type { LinesMetadata, LinesDataLoader, LinesViewState } from '../../../types/lines';
import type { NodeBuildCtx } from './build-ctx';

/** Construct the lines spatial-index loader and wire it to the monitor. */
function createLinesLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): LinesDataLoader {
  const loader = createLinesLoaderHelper(node, loc, ctx.factoryDeps);
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/** Progressive multi-LOD Lines loader; mirrors the points equivalent. */
async function createProgressiveLinesLoader(
  node: SceneNode,
  nAdditive: number,
  ctx: NodeBuildCtx
): Promise<LinesDataLoader> {
  const loader = await createProgressiveLinesLoaderHelper(
    node,
    nAdditive,
    ctx.applyEffectiveAttrs(node),
    ctx.factoryDeps
  );
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/** Result of the cheap half of Lines node loading. */
export interface LinesCheapLoad {
  /** The empty placeholder mesh, already attached to the parent. */
  placeholder: THREE.Mesh;
  /** The constructed loader (NOT yet registered — the caller registers it). */
  loader: LinesDataLoader;
}

/**
 * Cheap half of Lines node loading: build the spatial-index loader and attach an
 * empty placeholder mesh — no array fetch, no commit, **no registry
 * registration**. Splitting this from the expensive half lets
 * `load-lod-group-node.ts` defer a Lines lod-group child (the finest level of a
 * lines-substitutive ladder), mirroring the points/gsplats loaders. The caller
 * owns registration: the combined `loadLinesNode` (eager path) registers
 * immediately; lod_group lazy levels are NEVER registered (so they don't join
 * the scene-wide `updateView` sweep and defeat the deferral) — the
 * `LODGroupRegistry` drives their (re)loads instead.
 */
export async function loadLinesNodeCheap(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<LinesCheapLoad> {
  log.custom('📐', Modules.SCENE_LOADER, `Loading lines: ${node.path}`);

  const attrs = node.attrs as unknown as LinesMetadata;
  log.info(Modules.SCENE_LOADER, `  Segments: ${attrs.n_segments || 'unknown'}`);
  log.info(Modules.SCENE_LOADER, `  Vertices: ${attrs.n_vertices || 'unknown'}`);

  const nAdditive = (node.attrs as { n_additive_sublods?: number }).n_additive_sublods ?? 0;
  if (nAdditive > 1) {
    log.info(
      Modules.SCENE_LOADER,
      `  Additive sub-LODs: ${nAdditive} (progressive loading enabled)`
    );
  }

  const loader =
    nAdditive > 1
      ? await createProgressiveLinesLoader(node, nAdditive, ctx)
      : createLinesLoader(node, loc, ctx);

  // Construct + attach empty placeholder before fetching.
  // processLinesData / commitLinesGeometry look up the mesh by name
  // and populate it on success; on failure the placeholder remains
  // for retry to target. Same path is used by every future update.
  const placeholder = ctx.nodeFactory.createEmptyLinesNode(
    node.path,
    ctx.applyEffectiveAttrs(node),
    attrs,
    loader
  );
  parentThree.add(placeholder);

  return { placeholder, loader };
}

/**
 * Expensive half of Lines node loading: derive the view state, fetch the line
 * segments, project, and commit geometry into the placeholder (found by name).
 * Safe to call after initial scene load returns (lazy lod_group levels) — it
 * checks `isDatasetLive()` before committing so a deferred load still in flight
 * when the user switches datasets never writes into a disposed/replaced scene.
 */
export async function loadLinesNodeExpensive(
  node: SceneNode,
  ctx: NodeBuildCtx,
  loader: LinesDataLoader
): Promise<void> {
  const attrs = node.attrs as unknown as LinesMetadata;
  try {
    // Lines path does not apply the partial-extend tolerance override during the
    // data fetch (only during clipping), so applyPartialExtendTolerance=false.
    const derivedLines = ctx.deriveNodeViewState(node.path, attrs, {
      applyPartialExtendTolerance: false,
    });
    // Capture the version at DERIVE time (see loadGSplatsNodeExpensive) so a
    // deferred reload is stamped for the slice it actually loaded.
    const loadedViewVersion = ctx.getViewVersion();
    // Always load with the derived view state — including the fully-extended
    // (`derivedLines.skip === 'extend_to_all'`) case. deriveNodeViewState derives
    // from the loader's live view state, so the derived state already IS the
    // current slice with the extend handling applied. A prior version fell back
    // to the raw live slice on skip, which sliced the fully-extended node's
    // segments away (issue #1157).
    const linesViewState: LinesViewState = derivedLines.viewState;

    const data = await loader.loadLines(linesViewState);

    if (data.segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible segments for ${node.path} - object created for future updates`
      );
    }

    const staged = await ctx.processLinesData(node.path, data, linesViewState);
    // Liveness gate: a deferred (lazy lod_group) load may resolve after the
    // dataset was switched/disposed; committing then would write into a stale
    // root group. Drop silently (no commit, no success log) — the abort-discard
    // policy, matching loadPointsNodeExpensive / loadGSplatsNodeExpensive's
    // early return.
    if (!ctx.isDatasetLive()) return;
    if (staged) {
      ctx.commitLinesGeometry(staged, undefined, loadedViewVersion);
      // A settled successful (re)load clears any prior failure record so a
      // recovered lazy level stops counting against the outcome report and
      // the auto-retry budget. No-op on a first successful load. Symmetric
      // with the catch's recordFailure. Inside the `staged` guard: a null
      // staged means the placeholder is gone and nothing was committed, so a
      // load that landed nowhere must not clear the failure — the same rule
      // as the retry path's verifyAndClear.
      ctx.registry.clearFailure(node.path);
    }
    log.success(Modules.SCENE_LOADER, `Loaded ${data.segmentCount} segments for ${node.path}`);
  } catch (error) {
    // Expected dispose-crossing — see the load-points-node.ts twin.
    if (!ctx.isDatasetLive()) return;
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}

/**
 * Load a single Lines node on initial scene construction. Composition of the
 * cheap (placeholder + loader) and expensive (fetch + commit) halves; the non-LOD
 * dispatch path uses this combined form so its behaviour is unchanged. Registers
 * the loader immediately — this node loads eagerly. See `load-points-node.ts`.
 */
export async function loadLinesNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
  const { placeholder, loader } = await loadLinesNodeCheap(node, parentThree, loc, ctx);
  try {
    await loadLinesNodeExpensive(node, ctx, loader);
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
    ctx.registry.registerLinesLoader(node.path, loader);
  }
  return placeholder;
}
