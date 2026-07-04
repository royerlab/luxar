/**
 * Initial-load path for a single GSplats leaf node.
 *
 * Mirrors `load-points-node.ts` and `load-lines-node.ts`. The wrinkle
 * specific to GSplats is the LOD branch: multi-additive nodes
 * (`n_additive_sublods > 1`) get a progressive loader that composes
 * effective rendering attrs up the scene-graph ancestry so LOD
 * synthetic nodes inherit opacity/intensity from ancestors. Single-LOD
 * nodes get the standard loader.
 *
 * Sibling of `data/gsplats/handler.ts` (update path).
 */

import type * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { LoaderError, classifyLoaderError } from './load-leaf-error-dispatch';
import {
  createGSplatsLoader as createGSplatsLoaderHelper,
  createProgressiveGSplatsLoader as createProgressiveGSplatsLoaderHelper,
} from '../loaders/loader-factory';
import { timeLodStage, timeLodStageSync } from '../lod-load-stats';
import type { SceneNode } from '../../data-loader-types';
import type { GSplatsMetadata, GSplatsDataLoader, GSplatsViewState } from '../../../types/gsplats';
import type { NodeBuildCtx } from './build-ctx';

/** Construct the single-LOD gsplats loader and wire it to the monitor. */
function createGSplatsLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): GSplatsDataLoader {
  const loader = createGSplatsLoaderHelper(node, loc, ctx.factoryDeps);
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/**
 * Construct the progressive (multi-LOD) gsplats loader. The parent's
 * effective rendering attrs are composed up the scene-graph ancestry
 * here (not in the helper) so the LOD synthetic nodes see ancestor
 * opacity/intensity/etc.
 */
async function createProgressiveGSplatsLoader(
  node: SceneNode,
  nAdditive: number,
  ctx: NodeBuildCtx
): Promise<GSplatsDataLoader> {
  const loader = await createProgressiveGSplatsLoaderHelper(
    node,
    nAdditive,
    ctx.applyEffectiveAttrs(node),
    ctx.factoryDeps
  );
  ctx.connectLoaderToMonitor(node.path, loader);
  return loader;
}

/** Result of the cheap half of gsplats node loading. */
export interface GSplatsCheapLoad {
  /** The empty placeholder mesh, already attached to the parent. */
  placeholder: THREE.Mesh;
  /** The constructed loader, already registered for updates. */
  loader: GSplatsDataLoader;
}

/**
 * Cheap half of GSplats node loading: construct the loader and attach
 * an empty placeholder mesh — no array fetch, no GPU commit, and
 * crucially **no registry registration**. Splitting this from the
 * expensive half lets the lod_group loader attach every level's
 * placeholder up front while deferring the costly geometry load to
 * ``loadGSplatsNodeExpensive`` (per-level lazy loading). Branches on
 * ``n_additive_sublods`` for progressive vs single-LOD loaders, exactly
 * as the combined path did.
 *
 * Registration is the caller's responsibility. Registering an unloaded
 * lazy level would pull it into the scene-wide ``updateView`` sweep
 * (``runLoaderUpdates`` over every registered gsplat loader), which would
 * load+commit every level and defeat the lazy deferral. The combined
 * ``loadGSplatsNode`` (eager path) registers immediately; the lod_group's
 * lazy levels are NEVER registered — they stay out of the sweep for their
 * whole lifetime and the ``LODGroupRegistry`` drives their (re)loads
 * (settle-gated ``ensureLoaded``; see ``load-lod-group-node.ts``).
 */
export async function loadGSplatsNodeCheap(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<GSplatsCheapLoad> {
  const attrs = node.attrs as unknown as GSplatsMetadata;
  const nAdditive = attrs.n_additive_sublods ?? 0;
  log.custom('🔮', Modules.SCENE_LOADER, `Loading gsplats: ${node.path}`);
  log.info(Modules.SCENE_LOADER, `  Splats: ${attrs.n_splats?.toLocaleString() || 'unknown'}`);
  log.info(Modules.SCENE_LOADER, `  Dimensions: ${attrs.ndim || 'unknown'}D`);
  if (nAdditive > 1) {
    log.info(
      Modules.SCENE_LOADER,
      `  Additive sub-LODs: ${nAdditive} (progressive loading enabled)`
    );
  }

  // Create gsplats loader — progressive for multi-additive, standard otherwise
  const loader =
    nAdditive > 1
      ? await createProgressiveGSplatsLoader(node, nAdditive, ctx)
      : createGSplatsLoader(node, loc, ctx);

  // Empty placeholder + same-flow commit. See loadPoints/loadLines
  // for the rationale.
  const placeholder = ctx.nodeFactory.createEmptyGSplatsNode(
    node.path,
    ctx.applyEffectiveAttrs(node),
    attrs,
    loader
  );
  parentThree.add(placeholder);

  return { placeholder, loader };
}

/**
 * Expensive half of GSplats node loading: fetch the splat arrays via
 * the loader, process them, and commit geometry into the placeholder
 * (found by name in the root group). Safe to call after initial scene
 * load returns (lazy lod_group levels) — it checks ``isDatasetLive()``
 * before committing so a load still in flight when the user switches
 * datasets never writes into a disposed/replaced scene.
 */
export async function loadGSplatsNodeExpensive(
  node: SceneNode,
  ctx: NodeBuildCtx,
  loader: GSplatsDataLoader
): Promise<void> {
  try {
    // GSplats path mirrors Points: applyPartialExtendTolerance=true so
    // tolerance overrides + nd_transform inversion both happen up front.
    const derivedGSplats = ctx.deriveNodeViewState(node.path, node.attrs, {
      applyPartialExtendTolerance: true,
    });
    // Capture the version at DERIVE time so a deferred reload that finishes
    // after a further scrub is stamped for the slice it actually loaded (the
    // registry then re-reloads for the newer version) — not mis-stamped fresh.
    const loadedViewVersion = ctx.getViewVersion();
    // Use the LIVE view-state (not the build-time snapshot) for the
    // extend_to_all skip-fallback so a deferred reload queries the current slice.
    const live = ctx.getLiveViewState();
    const gsplatsViewState: GSplatsViewState = derivedGSplats.skip
      ? {
          displayDims: live.displayDims,
          slicePosition: live.slicePosition,
          tolerance: live.tolerance,
          dimensions: live.dimensions,
        }
      : derivedGSplats.viewState;

    // Debug-only per-stage timing (no-op unless ?debug). Buckets the
    // three meaningful costs — fetch+decode, CPU process (project+pack),
    // GPU commit — so navigation hitches can be attributed to a stage
    // before deciding what (if anything) to move off the main thread.
    const data = await timeLodStage('lazy:loadGSplats', () => loader.loadGSplats(gsplatsViewState));

    if (data.splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible gsplats for ${node.path} - object created for future updates`
      );
    }

    // Process + commit through the same helpers used by every update
    // and retry. Helpers find the placeholder by name and read its
    // userData for truncate/attrs.
    const staged = await timeLodStage('lazy:process', () =>
      ctx.processGSplatsData(node.path, data, gsplatsViewState)
    );
    if (staged) {
      // Liveness gate: a deferred (lazy lod_group) load may resolve
      // after the dataset was switched/disposed; committing then would
      // write into a stale root group. Drop the commit silently — the
      // worker results are discarded, matching the abort-discard policy.
      if (!ctx.isDatasetLive()) return;
      timeLodStageSync('lazy:commit', () =>
        ctx.commitGSplatsGeometry(staged, undefined, loadedViewVersion)
      );
    }

    log.success(
      Modules.SCENE_LOADER,
      `Loaded ${data.splatCount.toLocaleString()} gsplats for ${node.path}`
    );
  } catch (error) {
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}

/**
 * Load a single GSplats node on initial scene construction. Branches
 * on `n_additive_sublods` to pick the progressive vs single-LOD
 * loader. See `load-points-node.ts` for the shared placeholder +
 * commit rationale.
 *
 * Composition of the cheap (placeholder + loader) and expensive
 * (fetch + commit) halves; the non-LOD dispatch path uses this combined
 * form so its behavior is unchanged.
 */
export async function loadGSplatsNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
  const { placeholder, loader } = await loadGSplatsNodeCheap(node, parentThree, loc, ctx);
  // Register immediately — this node loads eagerly, so it should
  // participate in subsequent updateView sweeps right away.
  ctx.registry.registerGSplatsLoader(node.path, loader);
  await loadGSplatsNodeExpensive(node, ctx, loader);
  return placeholder;
}
