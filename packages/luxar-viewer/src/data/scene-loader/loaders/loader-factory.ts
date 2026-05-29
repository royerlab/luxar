/**
 * Loader-factory concern extracted from `scene-loader.ts`.
 *
 * Four constructor helpers, one per geometry kind:
 *   - `createPointsLoader` → `PointsSpatialIndexLoader`
 *   - `createLinesLoader` → `LinesSpatialIndexLoader`
 *   - `createGSplatsLoader` → `GSplatsSpatialIndexLoader`
 *   - `createProgressiveGSplatsLoader` → `GSplatsProgressiveLoader`
 *     wrapping N `GSplatsSpatialIndexLoader`s (one per LOD)
 *
 * All four follow the same path-resolution pattern: a leaf at `/` uses
 * the supplied `loc`; otherwise we resolve from the store root so
 * absolute paths in the scene graph map to zarr group locations
 * correctly.
 *
 * Behavior matches the inline original — same constructor argument
 * order, same logging, same null-coalescing of optional cache /
 * profiler / prefetcher dependencies.
 *
 * @module data/scene-loader/loaders/loader-factory
 */

import * as zarr from '../../zarr';
import { PointsSpatialIndexLoader } from '../../points/points-spatial-index-loader';
import { PointsProgressiveLoader } from '../../points/points-progressive-loader';
import type { PointsDataLoader } from '../../../types/points';
import { LinesProgressiveLoader } from '../../lines/lines-progressive-loader';
export type { PointsSpatialIndexLoader } from '../../points/points-spatial-index-loader';
import { LinesSpatialIndexLoader } from '../../lines/lines-spatial-index-loader';
import { GSplatsSpatialIndexLoader } from '../../gsplats/gsplats-spatial-index-loader';
import { GSplatsProgressiveLoader } from '../../gsplats/gsplats-progressive-loader';
import type { SceneNode } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import { ArrayRefRegistry } from '../../array-decoder/decoder';
import { log, Modules } from '../../../utils/log';
import { UpdateProfiler } from '../../../profiling/update-profiler';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';

/** Common dependencies needed by every loader factory call. */
export interface LoaderFactoryDeps {
  zarrStore: zarr.Readable;
  arrayRefRegistry: ArrayRefRegistry;
  profiler: UpdateProfiler | null;
  l0Cache: DecompressedChunkCache | null;
  cachingStore: MultiLevelCachingStore | null;
}

/** Resolve the node's zarr location: root path uses `loc` directly. */
function resolveNodeLoc(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  zarrStore: zarr.Readable
): zarr.Location<zarr.Readable> {
  return node.path === '/' ? loc : zarr.root(zarrStore).resolve(node.path.slice(1));
}

/**
 * Create the spatial index loader for a points node. The points
 * loader handles 3D datasets without spatial indices itself, so this
 * is the universal points entry point.
 *
 * Note: the SceneLoader still owns monitor-connection — this factory
 * only constructs the loader, deliberately not the side-effect of
 * registering it with the data monitor.
 */
export function createPointsLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  deps: LoaderFactoryDeps
): PointsSpatialIndexLoader {
  const nodeLoc = resolveNodeLoc(node, loc, deps.zarrStore);
  log.query(Modules.SCENE_LOADER, `Using PointsSpatialIndexLoader for ${node.path}`);
  return new PointsSpatialIndexLoader(
    nodeLoc,
    node,
    deps.arrayRefRegistry,
    deps.zarrStore,
    deps.profiler ?? undefined,
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined
  );
}

/** Create the spatial index loader for a lines node. */
export function createLinesLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  deps: LoaderFactoryDeps
): LinesDataLoader {
  const nodeLoc = resolveNodeLoc(node, loc, deps.zarrStore);
  log.query(Modules.SCENE_LOADER, `Using LinesSpatialIndexLoader for ${node.path}`);
  return new LinesSpatialIndexLoader(
    nodeLoc,
    node,
    deps.arrayRefRegistry,
    deps.zarrStore,
    deps.profiler ?? undefined,
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined
  );
}

/** Create the spatial index loader for a single-LOD gsplats node. */
export function createGSplatsLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  deps: LoaderFactoryDeps
): GSplatsDataLoader {
  const nodeLoc = resolveNodeLoc(node, loc, deps.zarrStore);
  log.query(Modules.SCENE_LOADER, `Using GSplatsSpatialIndexLoader for ${node.path}`);
  return new GSplatsSpatialIndexLoader(
    nodeLoc,
    node,
    deps.arrayRefRegistry,
    deps.zarrStore,
    deps.profiler ?? undefined,
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined
  );
}

/**
 * Create a progressive gsplats loader for a multi-additive-LOD scene
 * node.
 *
 * Walks ``additive_<i>/`` zarr subgroups directly under the node path,
 * reads each one's attrs to build a synthetic SceneNode, and creates a
 * ``GSplatsSpatialIndexLoader`` per additive sub-LOD. Wraps them all in
 * a ``GSplatsProgressiveLoader``.
 *
 * The caller supplies a pre-composed ``parentEffectiveAttrs`` snapshot —
 * this matches the inline original which calls
 * ``applyEffectiveAttrs(node)`` once before the loop so each sub-LOD's
 * synthetic node sees rendering attrs already composed through the
 * scene-graph ancestry (not just the parent's raw zarr attrs).
 */
export async function createProgressiveGSplatsLoader(
  node: SceneNode,
  nAdditive: number,
  parentEffectiveAttrs: SceneNode['attrs'],
  deps: LoaderFactoryDeps
): Promise<GSplatsDataLoader> {
  const parentLoc = zarr.root(deps.zarrStore).resolve(node.path === '/' ? '' : node.path.slice(1));

  log.query(
    Modules.SCENE_LOADER,
    `Creating progressive GSplats loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const lodLoaders: GSplatsSpatialIndexLoader[] = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);

    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;

    const lodNode: SceneNode = {
      path: `${node.path}/additive_${i}`,
      type: 'gsplats',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        gamma: parentEffectiveAttrs.gamma,
        intensity: parentEffectiveAttrs.intensity,
        offset: parentEffectiveAttrs.offset,
        blending_mode: parentEffectiveAttrs.blending_mode,
        extend_to_all: node.attrs.extend_to_all,
      },
      hasSpatialIndex: false,
      children: [],
    };

    lodLoaders.push(
      new GSplatsSpatialIndexLoader(
        lodLoc,
        lodNode,
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.profiler ?? undefined,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new GSplatsProgressiveLoader(lodLoaders, nAdditive, node.path);
}

/**
 * Create a progressive points loader for a multi-additive-LOD scene
 * node. Mirrors :func:`createProgressiveGSplatsLoader` for Points.
 *
 * Walks ``additive_<i>/`` zarr subgroups directly under the node path
 * and builds one `PointsSpatialIndexLoader` per subgroup. Wraps all
 * of them in a `PointsProgressiveLoader`.
 */
export async function createProgressivePointsLoader(
  node: SceneNode,
  nAdditive: number,
  parentEffectiveAttrs: SceneNode['attrs'],
  deps: LoaderFactoryDeps
): Promise<PointsDataLoader> {
  const parentLoc = zarr.root(deps.zarrStore).resolve(
    node.path === '/' ? '' : node.path.slice(1)
  );

  log.query(
    Modules.SCENE_LOADER,
    `Creating progressive Points loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const lodLoaders: PointsSpatialIndexLoader[] = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);
    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;

    const lodNode: SceneNode = {
      path: `${node.path}/additive_${i}`,
      type: 'points',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        gamma: parentEffectiveAttrs.gamma,
        intensity: parentEffectiveAttrs.intensity,
        offset: parentEffectiveAttrs.offset,
        blending_mode: parentEffectiveAttrs.blending_mode,
        extend_to_all: node.attrs.extend_to_all,
      },
      hasSpatialIndex: false,
      children: [],
    };

    lodLoaders.push(
      new PointsSpatialIndexLoader(
        lodLoc,
        lodNode,
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.profiler ?? undefined,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new PointsProgressiveLoader(lodLoaders, nAdditive, node.path);
}

/**
 * Create a progressive lines loader for a multi-additive-LOD scene
 * node. Mirrors :func:`createProgressivePointsLoader` for Lines.
 */
export async function createProgressiveLinesLoader(
  node: SceneNode,
  nAdditive: number,
  parentEffectiveAttrs: SceneNode['attrs'],
  deps: LoaderFactoryDeps
): Promise<LinesDataLoader> {
  const parentLoc = zarr.root(deps.zarrStore).resolve(
    node.path === '/' ? '' : node.path.slice(1)
  );

  log.query(
    Modules.SCENE_LOADER,
    `Creating progressive Lines loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const lodLoaders: LinesSpatialIndexLoader[] = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);
    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;

    const lodNode: SceneNode = {
      path: `${node.path}/additive_${i}`,
      type: 'lines',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        gamma: parentEffectiveAttrs.gamma,
        intensity: parentEffectiveAttrs.intensity,
        offset: parentEffectiveAttrs.offset,
        blending_mode: parentEffectiveAttrs.blending_mode,
        extend_to_all: node.attrs.extend_to_all,
      },
      hasSpatialIndex: false,
      children: [],
    };

    lodLoaders.push(
      new LinesSpatialIndexLoader(
        lodLoc,
        lodNode,
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.profiler ?? undefined,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new LinesProgressiveLoader(lodLoaders, nAdditive, node.path);
}
