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
        // A sub-LOD's label CSR lives on `additive_<i>` but the pick path only
        // ever resolves labels against the PARENT node's path, so no reader can
        // key by a sub-LOD's on-disk index. Clearing the flags here keeps the
        // spatial-index loader from publishing per-level `ranges` — and the
        // projection from composing a per-level slot → on-disk map — that the
        // ladder concat then has to discard. (The parent's own missing
        // `has_labels` is #1422.) These two keys therefore CONTRADICT the store:
        // whoever implements per-level labels must decide whether a level has a
        // CSR from the real `lodGroup.attrs` / the `additive_<i>` group itself,
        // never from this synthesized node.
        has_labels: false,
        has_image_labels: false,
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
 * Mesh has no ADDITIVE (prefix) ladder: a prefix of an index buffer is a surface with
 * holes in it, not a coarser one, so that flavour is excluded on principle
 * (`docs/specs/MESH_NODE_SPEC.md` §9). SUBSTITUTIVE levels are a different shape and DO
 * work — sibling children of a `kind=lod` group, written by
 * `add_mesh(substitutive_lod=…)` — and they never route through here. This exists because
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
      `Mesh node ${node.path} declares additive sub-LODs, but mesh has no additive ` +
        '(prefix) ladder (MESH_NODE_SPEC.md §9): a prefix of an index buffer is a ' +
        'surface with holes, not a coarser one. Write the mesh as a plain leaf, or use ' +
        'substitutive levels — add_mesh(substitutive_lod=...) — which are separate ' +
        'kind=lod children, not sub-LODs inside this node.'
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

  // PASS 1 — open every `additive_<i>` group (one open per level, in order)
  // and collect what the level offsets are decided from. The synthesized nodes
  // are built in pass 2, AFTER `levelOffsets` is known: their label flags
  // depend on it, and stamping them first would make every level build a
  // per-level picking map that the concat is going to discard.
  const levels: Array<{ loc: zarr.Location<zarr.Readable>; attrs: Record<string, unknown> }> = [];
  const energyTable: Array<number | null> = [];
  // Per-level ON-DISK row counts (`n_points` on each `additive_<i>` group, NOT
  // the loaded count) — the offsets into the parent CSR's index space.
  const onDiskCounts: Array<number | undefined> = [];

  for (let i = 0; i < nAdditive; i++) {
    const lodLoc = parentLoc.resolve(`additive_${i}`);
    const lodGroup = await zarr.open(lodLoc, { kind: 'group' });
    const lodAttrs = lodGroup.attrs as Record<string, unknown>;
    levels.push({ loc: lodLoc, attrs: lodAttrs });
    energyTable.push(energyFractionFromAttrs(lodAttrs));
    onDiskCounts.push(typeof lodAttrs.n_points === 'number' ? lodAttrs.n_points : undefined);
  }

  // Does the PARENT node carry the ladder's UNION label CSR? That CSR is keyed
  // by the concatenation `additive_0 || additive_1 || …`, each level in its own
  // stored order (#1422), so it is the parent — never a sub-LOD — that decides
  // whether a ladder has readable labels at all.
  const parentDeclaresLabels =
    node.attrs.has_labels === true || node.attrs.has_image_labels === true;
  // CSR-style bounds over the on-disk counts, length `nAdditive + 1`:
  // `levelOffsets[i]` is where level `i` starts inside the parent's union CSR
  // index space (so [0] === 0) and `levelOffsets[i + 1]` is where it ends, so
  // the composer can also BOUND each level's ids instead of only shifting
  // them. The last entry is the union's total row count.
  // Only meaningful when the parent declares labels; on anything unusable stay
  // null (hover then reports the raw slot — no better than before #1439, but
  // never an id composed into someone else's CSR row).
  let levelOffsets: number[] | null = null;
  if (parentDeclaresLabels) {
    const usable = onDiskCounts.every((n) => n !== undefined && Number.isSafeInteger(n) && n >= 0);
    if (!usable) {
      log.warning(
        Modules.SCENE_LOADER,
        `Progressive Points ${node.path} declares labels but a sub-LOD is missing a valid ` +
          '`n_points`; per-level picking maps are disabled (hover falls back to the ' +
          'visible-buffer slot).'
      );
    } else {
      const total = (onDiskCounts as number[]).reduce((s, n) => s + n, 0);
      // Free cross-check: the writer's parent `n_points` IS the sum of the
      // levels' row counts (`n_points_total`), so a mismatch means the CSR and
      // the levels come from different builds — composing would land in
      // someone else's row. Only checked when the parent actually carries the
      // attr; its absence is not evidence of anything.
      if (typeof node.attrs.n_points === 'number' && node.attrs.n_points !== total) {
        log.warning(
          Modules.SCENE_LOADER,
          `Progressive Points ${node.path}: sub-LOD row counts sum to ${total} but the parent ` +
            `declares n_points=${node.attrs.n_points}; the label CSR and the levels disagree, ` +
            'so per-level picking maps are disabled (hover falls back to the visible-buffer slot).'
        );
      } else if (total > 0xffffffff) {
        // The composed map is a `Uint32Array` of union indices, so a union
        // wider than 2^32 rows would wrap silently into another CSR row (and
        // past 2^53 the prefix sums stop being exact at all). Individually
        // safe-integer counts can still sum past both bounds, so the SUM is
        // what has to be checked.
        log.warning(
          Modules.SCENE_LOADER,
          `Progressive Points ${node.path}: sub-LOD row counts sum to ${total}, past the ` +
            '2^32 index range the picking map is stored in; per-level picking maps are ' +
            'disabled (hover falls back to the visible-buffer slot).'
        );
      } else {
        levelOffsets = [];
        let acc = 0;
        for (const n of onDiskCounts as number[]) {
          levelOffsets.push(acc);
          acc += n;
        }
        // Closing bound: `levelOffsets[nAdditive]` === the union row count, so
        // every level — the last one included — has an end to be checked against.
        levelOffsets.push(acc);
      }
    }
  }
  // Per-level maps are only ever USED when the offsets exist to place them.
  const levelsBuildMaps = parentDeclaresLabels && levelOffsets !== null;

  // PASS 2 — synthesize each sub-LOD node and its loader.
  const lodLoaders: PointsSpatialIndexLoader[] = levels.map(
    ({ loc: lodLoc, attrs: lodAttrs }, i) => {
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
          // A sub-LOD carries no CSR of ITS OWN that any reader can key by — the
          // pick path resolves labels against the PARENT node's path — so a
          // sub-LOD's stored flags are never trusted and are always overridden
          // here. They are overridden with the parent's declaration, AND only
          // when the composition can actually run: with `levelOffsets` each level
          // builds its own level-space slot → on-disk map and
          // `concatenatePointsData` offsets it into the parent's index space by
          // the preceding levels' on-disk counts (#1439). Without them (no parent
          // CSR, or a failed cross-check) both stay false, so no per-level map is
          // built only to be discarded, and picking stays allocation-free exactly
          // as before.
          has_labels: levelsBuildMaps && node.attrs.has_labels === true,
          has_image_labels: levelsBuildMaps && node.attrs.has_image_labels === true,
        },
        hasSpatialIndex: false,
        children: [],
      };

      return new PointsSpatialIndexLoader(
        lodLoc,
        lodNode,
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      );
    }
  );

  return new PointsProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    energyTable,
    deps.sliceCache ?? undefined,
    levelOffsets
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
        // A sub-LOD's label CSR lives on `additive_<i>` but the pick path only
        // ever resolves labels against the PARENT node's path, so no reader can
        // key by a sub-LOD's on-disk index. Clearing the flags here keeps the
        // spatial-index loader from publishing per-level `vertexRangeBounds` — and
        // the projection from composing a per-level slot → on-disk map — that
        // the ladder concat then has to discard. (The parent's own missing
        // `has_labels` is #1422.) These two keys therefore CONTRADICT the store:
        // whoever implements per-level labels must decide whether a level has a
        // CSR from the real `lodGroup.attrs` / the `additive_<i>` group itself,
        // never from this synthesized node.
        has_labels: false,
        has_image_labels: false,
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
