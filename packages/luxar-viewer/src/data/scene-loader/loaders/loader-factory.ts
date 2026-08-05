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
 * prefetcher dependencies.
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
import { MeshWholeNodeLoader } from '../../mesh/mesh-whole-node-loader';
import { GSplatsProgressiveLoader } from '../../gsplats/gsplats-progressive-loader';
import type { SceneNode } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { MeshDataLoader, MeshMetadata } from '../../../types/mesh';
import { ArrayRefRegistry } from '../../array-decoder/decoder';
import { log, Modules } from '../../../utils/log';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { SliceCache } from '../../../cache/slice-cache';
import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';

/** Common dependencies needed by every loader factory call. */
export interface LoaderFactoryDeps {
  zarrStore: zarr.Readable;
  arrayRefRegistry: ArrayRefRegistry;
  l0Cache: DecompressedChunkCache | null;
  /** Shared SliceCache; passed to progressive loaders for per-slice reuse. */
  sliceCache: SliceCache | null;
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
 * Read a sub-LOD's `lod_stats.energy_fraction_cum` stamp — the cumulative
 * self-energy fraction e(k) the ladder prefix up to and including this
 * sub-LOD carries. Stamped at build time (or by `luxar gsplat
 * annotate-quality`); `null` on unstamped (legacy) datasets. The progressive
 * loaders fold the per-sub-LOD table into `committedEnergyFraction`, which
 * the LOD display gate uses for energy-threshold upgrade releases.
 */
function energyFractionFromAttrs(lodAttrs: Record<string, unknown>): number | null {
  const stats = lodAttrs.lod_stats as Record<string, unknown> | undefined;
  const e = stats?.energy_fraction_cum;
  return typeof e === 'number' ? e : null;
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
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined,
    // Plain-leaf nodes cache their decoded slice in the S-cache as a
    // 1-element ladder. Progressive sub-LOD loaders (created below) must
    // NOT receive it — their wrapper owns the whole-ladder cache entry.
    deps.sliceCache ?? undefined
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
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined,
    // Plain-leaf nodes cache their decoded slice in the S-cache as a
    // 1-element ladder. Progressive sub-LOD loaders (created below) must
    // NOT receive it — their wrapper owns the whole-ladder cache entry.
    deps.sliceCache ?? undefined
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
    deps.l0Cache ?? undefined,
    deps.cachingStore?.getPrefetcher() ?? undefined,
    // Plain-leaf nodes cache their decoded slice in the S-cache as a
    // 1-element ladder. Progressive sub-LOD loaders (created below) must
    // NOT receive it — their wrapper owns the whole-ladder cache entry.
    deps.sliceCache ?? undefined
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
  const energyTable: Array<number | null> = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);

    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;
    energyTable.push(energyFractionFromAttrs(lodAttrs));

    const lodNode: SceneNode = {
      path: `${node.path === '/' ? '' : node.path}/additive_${i}`,
      type: 'gsplats',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        absorption: parentEffectiveAttrs.absorption,
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
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new GSplatsProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    energyTable,
    deps.sliceCache ?? undefined
  );
}

/**
 * Create the loader for a mesh node.
 *
 * The whole-node loader — no spatial-index variant to choose between, because a
 * mesh has no per-slice working set to skip (see `data/mesh/README.md`). So unlike
 * the three sibling factories this one has a single branch.
 */
export function createMeshLoader(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>,
  deps: LoaderFactoryDeps
): MeshDataLoader {
  const nodeLoc = resolveNodeLoc(node, loc, deps.zarrStore);
  log.query(Modules.SCENE_LOADER, `Using MeshWholeNodeLoader for ${node.path}`);
  return new MeshWholeNodeLoader(node.path, node.attrs as unknown as MeshMetadata, nodeLoc, {
    zarrStore: deps.zarrStore,
    arrayRefRegistry: deps.arrayRefRegistry,
  });
}

/**
 * Refuse to build a progressive (multi-LOD) mesh loader.
 *
 * Mesh has no LOD path at all — no decimation, no additive ladder
 * (`docs/specs/MESH_NODE_SPEC.md` §9) — and the Python side already refuses to
 * write a mesh under a `kind=lod` group. This exists because
 * `GeometryDescriptor` requires the factory for every drawable kind, and the
 * honest implementation of "this kind cannot do that" is a clear throw rather than
 * a silent fallback to the single-LOD loader.
 *
 * Reaching this means a store declared `n_additive_sublods > 1` on a mesh node,
 * which no Luxar writer produces; failing loudly is what turns that into one lost
 * node with an explanation instead of a mesh that quietly renders its coarsest
 * level forever.
 */
export function createProgressiveMeshLoader(
  node: SceneNode,
  _nAdditive: number,
  _parentEffectiveAttrs: SceneNode['attrs'],
  _deps: LoaderFactoryDeps
): Promise<MeshDataLoader> {
  return Promise.reject(
    new Error(
      `Mesh node ${node.path} declares additive sub-LODs, but mesh has no LOD path ` +
        '(MESH_NODE_SPEC.md §9): there is no decimation and no additive ladder. ' +
        'Write the mesh as a plain leaf.'
    )
  );
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
  const parentLoc = zarr.root(deps.zarrStore).resolve(node.path === '/' ? '' : node.path.slice(1));

  log.query(
    Modules.SCENE_LOADER,
    `Creating progressive Points loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const lodLoaders: PointsSpatialIndexLoader[] = [];
  const energyTable: Array<number | null> = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);
    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;
    energyTable.push(energyFractionFromAttrs(lodAttrs));

    const lodNode: SceneNode = {
      path: `${node.path === '/' ? '' : node.path}/additive_${i}`,
      type: 'points',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        absorption: parentEffectiveAttrs.absorption,
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
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new PointsProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    energyTable,
    deps.sliceCache ?? undefined
  );
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
  const parentLoc = zarr.root(deps.zarrStore).resolve(node.path === '/' ? '' : node.path.slice(1));

  log.query(
    Modules.SCENE_LOADER,
    `Creating progressive Lines loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const lodLoaders: LinesSpatialIndexLoader[] = [];
  const energyTable: Array<number | null> = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);
    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;
    energyTable.push(energyFractionFromAttrs(lodAttrs));

    const lodNode: SceneNode = {
      path: `${node.path === '/' ? '' : node.path}/additive_${i}`,
      type: 'lines',
      attrs: {
        ...lodAttrs,
        opacity: parentEffectiveAttrs.opacity,
        absorption: parentEffectiveAttrs.absorption,
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
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
    );
  }

  return new LinesProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    energyTable,
    deps.sliceCache ?? undefined
  );
}
