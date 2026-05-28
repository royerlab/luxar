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

/**
 * Load a single GSplats node on initial scene construction. Branches
 * on `n_additive_sublods` to pick the progressive vs single-LOD
 * loader. See `load-points-node.ts` for the shared placeholder +
 * commit rationale.
 */
export async function loadGSplatsNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
  const attrs = node.attrs as unknown as GSplatsMetadata;
  const nAdditive = attrs.n_additive_sublods ?? 0;
  log.custom('🔮', Modules.SCENE_LOADER, `Loading gsplats: ${node.path}`);
  log.info(
    Modules.SCENE_LOADER,
    `  Splats: ${attrs.n_splats?.toLocaleString() || 'unknown'}`
  );
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

  // Store loader for updates (route through registry).
  ctx.registry.registerGSplatsLoader(node.path, loader);

  // Empty placeholder + same-flow commit. See loadPoints/loadLines
  // for the rationale.
  const placeholder = ctx.nodeFactory.createEmptyGSplatsNode(
    node.path,
    ctx.applyEffectiveAttrs(node),
    attrs,
    loader
  );
  parentThree.add(placeholder);

  try {
    // GSplats path mirrors Points: applyPartialExtendTolerance=true so
    // tolerance overrides + nd_transform inversion both happen up front.
    const derivedGSplats = ctx.deriveNodeViewState(node.path, node.attrs, {
      applyPartialExtendTolerance: true,
    });
    const gsplatsViewState: GSplatsViewState = derivedGSplats.skip
      ? {
          displayDims: ctx.viewState.displayDims,
          slicePosition: ctx.viewState.slicePosition,
          tolerance: ctx.viewState.tolerance,
          dimensions: ctx.viewState.dimensions,
        }
      : derivedGSplats.viewState;

    const data = await loader.loadGSplats(gsplatsViewState);

    if (data.splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible gsplats for ${node.path} - object created for future updates`
      );
    }

    // Process + commit through the same helpers used by every update
    // and retry. Helpers find the placeholder by name and read its
    // userData for truncate/attrs.
    const staged = await ctx.processGSplatsData(node.path, data, gsplatsViewState);
    if (staged) ctx.commitGSplatsGeometry(staged);

    log.success(
      Modules.SCENE_LOADER,
      `Loaded ${data.splatCount.toLocaleString()} gsplats for ${node.path}`
    );

    return placeholder;
  } catch (error) {
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}
