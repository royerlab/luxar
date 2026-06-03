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

/**
 * Load a single Lines node on initial scene construction. See
 * `load-points-node.ts` for the shared placeholder + commit rationale.
 */
export async function loadLinesNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Mesh | null> {
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

  // Store loader for updates (route through registry).
  ctx.registry.registerLinesLoader(node.path, loader);

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

  try {
    // Lines path does not apply the partial-extend tolerance override
    // during the data fetch (only during clipping below), so
    // applyPartialExtendTolerance=false. This call still validates
    // extend_to_all dim names and applies the inverse nd_transform.
    const derivedLines = ctx.deriveNodeViewState(node.path, attrs, {
      applyPartialExtendTolerance: false,
    });
    const linesViewState: LinesViewState = derivedLines.skip
      ? ctx.viewState
      : derivedLines.viewState;

    const data = await loader.loadLines(linesViewState);

    if (data.segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `No initially visible segments for ${node.path} - object created for future updates`
      );
    }

    // Project + commit through the same helpers used by every update
    // and retry. processLinesData reads the placeholder's userData
    // (extend_to_all etc.) and finds the mesh by name; commit step
    // writes into the existing geometry.
    const staged = await ctx.processLinesData(node.path, data, linesViewState);
    if (staged) ctx.commitLinesGeometry(staged);

    log.success(Modules.SCENE_LOADER, `Loaded ${data.segmentCount} segments for ${node.path}`);

    return placeholder;
  } catch (error) {
    // See loadPoints catch — same record-failure-then-throw shape.
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}
