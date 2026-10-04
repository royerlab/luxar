/**
 * Loader-factory concern extracted from `scene-loader.ts`.
 *
 * Eight constructor helpers, two per geometry kind:
 *   - single-LOD: `createPointsLoader` / `createLinesLoader` /
 *     `createGSplatsLoader` → the matching `*SpatialIndexLoader`, and
 *     `createMeshLoader` → `MeshWholeNodeLoader`;
 *   - progressive (additive ladder): `createProgressivePointsLoader` /
 *     `createProgressiveLinesLoader` / `createProgressiveGSplatsLoader` wrap one
 *     per-level loader per `additive_<i>` subgroup in the matching
 *     `*ProgressiveLoader`, and `createProgressiveMeshLoader` builds the mesh
 *     reveal ladder (`MeshProgressiveLoader`).
 *
 * A leaf at `/` uses the supplied `loc`; any other path is resolved from the
 * store root, so absolute scene-graph paths map to zarr group locations.
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
import { MeshProgressiveLoader } from '../../mesh/mesh-progressive-loader';
import { GSplatsProgressiveLoader } from '../../gsplats/gsplats-progressive-loader';
import type { SceneNode } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { KTX2TextureDecoder, MeshDataLoader, MeshMetadata } from '../../../types/mesh';
import { ArrayRefRegistry } from '../../array-decoder/decoder';
import { MAX_MESH_VERTICES } from '../../../config/constants';
import { LoaderError } from '../nodes/load-leaf-error-dispatch';
import { log, LogEmoji, Modules } from '../../../utils/log';
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
  decodeKTX2?: KTX2TextureDecoder | null;
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

/** One `additive_<i>` subgroup of a laddered node: its location and stored attrs. */
interface AdditiveLevel {
  readonly loc: zarr.Location<zarr.Readable>;
  readonly attrs: Record<string, unknown>;
}

/**
 * Open a laddered node's `additive_<i>` subgroups in ladder order (one
 * metadata open each, sequentially — the order the ladder is read in).
 */
async function openAdditiveLevels(
  node: SceneNode,
  nAdditive: number,
  deps: LoaderFactoryDeps
): Promise<AdditiveLevel[]> {
  const parentLoc = zarr.root(deps.zarrStore).resolve(node.path === '/' ? '' : node.path.slice(1));
  const levels: AdditiveLevel[] = [];
  for (let i = 0; i < nAdditive; i++) {
    const loc = parentLoc.resolve(`additive_${i}`);
    const group = await zarr.open(loc, { kind: 'group' });
    levels.push({ loc, attrs: group.attrs as Record<string, unknown> });
  }
  return levels;
}

/** Which per-element metadata channels a sub-LOD may publish (see the callers). */
interface LevelLabelFlags {
  readonly has_labels: boolean;
  readonly has_image_labels: boolean;
  readonly has_keys: boolean;
}

/** No per-level label channel: a sub-LOD carries no CSR any reader can key by. */
const NO_LEVEL_LABELS: LevelLabelFlags = {
  has_labels: false,
  has_image_labels: false,
  has_keys: false,
};

/**
 * A sub-LOD's attrs: its own stored attrs, with the parent's rendering attrs as
 * composed through the scene-graph ancestry (`parentEffectiveAttrs`), the
 * parent's `extend_to_all`, and the label flags the caller decided.
 */
function composeLevelAttrs(
  level: AdditiveLevel,
  node: SceneNode,
  parentEffectiveAttrs: SceneNode['attrs'],
  labels: LevelLabelFlags
): Record<string, unknown> {
  return {
    ...level.attrs,
    opacity: parentEffectiveAttrs.opacity,
    absorption: parentEffectiveAttrs.absorption,
    gamma: parentEffectiveAttrs.gamma,
    intensity: parentEffectiveAttrs.intensity,
    offset: parentEffectiveAttrs.offset,
    blending_mode: parentEffectiveAttrs.blending_mode,
    extend_to_all: node.attrs.extend_to_all,
    ...labels,
  };
}

/** The synthetic scene node a progressive loader's sub-LOD loader is built over. */
function buildAdditiveLevelNode(
  node: SceneNode,
  index: number,
  type: SceneNode['type'],
  attrs: Record<string, unknown>
): SceneNode {
  return {
    path: `${node.path === '/' ? '' : node.path}/additive_${index}`,
    type,
    attrs: attrs as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Using PointsSpatialIndexLoader for ${node.path}`
  );
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Using LinesSpatialIndexLoader for ${node.path}`
  );
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Using GSplatsSpatialIndexLoader for ${node.path}`
  );
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Creating progressive GSplats loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const levels = await openAdditiveLevels(node, nAdditive, deps);
  // No sub-LOD carries a label CSR of its own, and the pick path only ever
  // resolves labels against the PARENT node's path, so no reader can key by a
  // sub-LOD's on-disk index. Clearing the flags keeps the spatial-index loader
  // from publishing per-level `ranges` — and the projection from composing a
  // per-level slot → on-disk map — that the ladder concat then has to discard.
  // (A gsplat ladder carries no labels at any level: the authoring path has no
  // `labels` channel. The Points and Lines ladders write one parent-level union
  // CSR instead — #1422.)
  const lodLoaders = levels.map(
    (level, i) =>
      new GSplatsSpatialIndexLoader(
        level.loc,
        buildAdditiveLevelNode(
          node,
          i,
          'gsplats',
          composeLevelAttrs(level, node, parentEffectiveAttrs, NO_LEVEL_LABELS)
        ),
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
  );

  return new GSplatsProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    levels.map((level) => energyFractionFromAttrs(level.attrs)),
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
  log.verbose(LogEmoji.QUERY, Modules.SCENE_LOADER, `Using MeshWholeNodeLoader for ${node.path}`);
  return new MeshWholeNodeLoader(node.path, node.attrs as unknown as MeshMetadata, nodeLoc, {
    zarrStore: deps.zarrStore,
    arrayRefRegistry: deps.arrayRefRegistry,
    decodeKTX2: deps.decodeKTX2 ?? undefined,
  });
}

/**
 * Create a progressive mesh loader for a REVEAL-ladder node.
 *
 * Walks `additive_<i>/` subgroups under the node path and builds one
 * {@link MeshWholeNodeLoader} per level, wrapped in a `MeshProgressiveLoader`.
 * Structurally the sibling of the three factories above; two differences, both
 * following from what a mesh ladder is:
 *
 *  - **No energy table.** The other three read each level's
 *    `lod_stats.energy_fraction_cum` so the display gate can release an upgrade
 *    early and `energyCompensation` can brighten an incomplete emissive prefix.
 *    A reveal prefix is a partial object at FULL brightness, so that multiplier
 *    would blow out the first shell and fade it as the surface completes — the
 *    inverse of growing in. Not reading the stamps is the second of the three
 *    latches (`MeshProgressiveLoader.committedEnergyFraction` and the Python
 *    writer's refusal are the others; `MESH_NODE_SPEC.md` §9.1).
 *  - **No `SliceCache`.** A mesh level is whole-node resident, so its own decode
 *    is the cache and lasts the node's lifetime; a per-slice ladder entry would
 *    store the same bytes under every key.
 *
 * This function REPLACED a deliberate rejection. Mesh has no additive
 * level-of-detail ladder and still never will — a prefix of an arbitrary index
 * buffer is a holed surface, not a coarser one — but a REVEAL is a different
 * claim: the writer admits only orderings whose every prefix is one connected
 * patch (`MESH_ADDITIVE_METHODS = {"radial"}`), so what streams in is a growing
 * surface rather than lace. See `MESH_NODE_SPEC.md` §9 for the distinction.
 */
export async function createProgressiveMeshLoader(
  node: SceneNode,
  nAdditive: number,
  parentEffectiveAttrs: SceneNode['attrs'],
  deps: LoaderFactoryDeps
): Promise<MeshDataLoader> {
  // The vertex cap has to be charged against the LADDER, not each level.
  //
  // `MAX_MESH_VERTICES` (2^27) exists because above it the pick vote key aliases
  // across nodes (spec §6.5), and what the pick path reads is the CONCATENATED
  // prefix — so the total is the quantity that must respect it. Every level runs
  // its own `preflightMesh`, but a level is only a fraction of the surface (and a
  // face-partition duplicates boundary vertices, so the levels sum to MORE than
  // the source mesh): four levels of 40M each pass individually while the ladder
  // they compose is 160M, past the cap, with picking silently aliasing. Charged
  // here, before a single subgroup is opened, so the refusal costs no fetch —
  // the same discipline as the per-level preflight it complements.
  //
  // The parent's `n_vertices` IS that sum (`write_mesh_multi_lod` stamps it from
  // what the levels actually wrote). A store that omits it is not second-guessed:
  // the per-level preflights still run, and the cap then binds per level as it did
  // before, which is the pre-existing behaviour rather than a new hole.
  const declaredVertices = node.attrs.n_vertices;
  if (typeof declaredVertices === 'number' && declaredVertices > MAX_MESH_VERTICES) {
    throw new LoaderError(
      'Validation',
      node.path,
      new Error(
        `Mesh reveal ladder declares ${declaredVertices.toLocaleString()} vertices across ` +
          `its ${nAdditive} levels, above the ${MAX_MESH_VERTICES.toLocaleString()} cap ` +
          '(the pick vote-key stride bound, spec §6.5). The levels are concatenated into ' +
          'one buffer, so the cap applies to their total — write fewer levels, or a ' +
          'coarser surface.'
      )
    );
  }

  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Creating progressive Mesh loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const levels = await openAdditiveLevels(node, nAdditive, deps);
  assertLevelsUntextured(node, levels);
  // Per-level DECLARED counts, for the parent-total cross-check below.
  const levelVertices = levels.map((level) =>
    typeof level.attrs.n_vertices === 'number' ? level.attrs.n_vertices : undefined
  );
  const levelFaces = levels.map((level) =>
    typeof level.attrs.n_faces === 'number' ? level.attrs.n_faces : undefined
  );
  // A laddered mesh has NO labels — `write_mesh_multi_lod` refuses a level that
  // carries them, because the three sibling ladders put one union CSR on the
  // parent and a face-partition duplicates boundary vertices, so one source
  // vertex maps to two slots and that index space is ill-defined. Cleared here
  // as well so a hand-written store cannot make a level publish per-level label
  // ranges the concat would then have to discard.
  const lodLoaders = levels.map(
    (level, i) =>
      new MeshWholeNodeLoader(
        `${node.path === '/' ? '' : node.path}/additive_${i}`,
        composeLevelAttrs(
          level,
          node,
          parentEffectiveAttrs,
          NO_LEVEL_LABELS
        ) as unknown as MeshMetadata,
        level.loc,
        {
          zarrStore: deps.zarrStore,
          arrayRefRegistry: deps.arrayRefRegistry,
          // No `decodeKTX2`: a level is refused above if it declares a texture.
        }
      )
  );

  assertLadderTotalsBackedByLevels(node, declaredVertices, node.attrs.n_faces, {
    levelVertices,
    levelFaces,
  });

  return new MeshProgressiveLoader(lodLoaders, nAdditive, node.path);
}

/**
 * Refuse a mesh reveal ladder whose levels declare a texture or UVs.
 *
 * The writer never authors one (`add_mesh` refuses `texture=` with
 * `additive_lod=`: the image would be stored once per shell), and the level
 * concat carries neither — so a hand-written textured ladder would render
 * untextured, and level 0's release would close the bitmap its material
 * samples. Refused before any level is fetched, like the totals check below.
 */
function assertLevelsUntextured(node: SceneNode, levels: AdditiveLevel[]): void {
  for (const [i, level] of levels.entries()) {
    if (level.attrs.has_texture !== true && level.attrs.has_uvs !== true) continue;
    throw new LoaderError(
      'Validation',
      node.path,
      new Error(
        `Mesh reveal ladder level additive_${i} declares a texture or UVs. A reveal ` +
          'ladder carries no texture (the writer refuses texture= with additive_lod=); ' +
          'write the textured surface as a plain mesh leaf.'
      )
    );
  }
}

/**
 * Refuse a mesh ladder whose PARENT declares more geometry than its levels hold.
 *
 * The parent's `n_vertices` / `n_faces` are not merely descriptive here: the commit
 * passes them as the geometry's buffer CAPACITY (`commit-mesh-geometry.ts`, #1521),
 * so they are what `position` / `color` / `normal` / `aScalar` and the index buffer
 * are all sized from — on the FIRST commit, before any level past the first has
 * been fetched.
 *
 * Nothing else checks them. A ladder's parent group carries no `vertices`/`faces`
 * array of its own, so it never reaches `preflightMesh`: every per-level preflight
 * validates its OWN counts against its OWN arrays and against the byte budget, and
 * the vertex cap above bounds the parent's vertex total, but `n_faces` has no cap at
 * all. A store pairing a tiny `additive_0` with an enormous parent `n_faces` would
 * therefore have the first commit allocate an index buffer sized to the declaration
 * — past any budget, and large enough to fail the allocation outright, which is a
 * `RangeError` out of the commit rather than the node-scoped `LoaderError` the
 * two-stage gate exists to produce.
 *
 * So the totals are cross-checked against what the levels declare, which is exactly
 * what `write_mesh_multi_lod` stamps them from (the sums over its levels). A
 * consistent store passes untouched, and the capacity can then never exceed what the
 * fully-revealed ladder would allocate anyway.
 *
 * Two asymmetries are deliberate:
 *
 * - Only the parent declaring MORE is refused. Declaring less is handled by
 *   `resolveCapacity`'s `Math.max` against the live count — it costs the
 *   allocate-once property, not correctness, and refusing it would reject a store
 *   that renders fine.
 * - A parent that declares NEITHER total is not second-guessed (the capacity then
 *   falls back to the committed counts). But once it declares one, every level must
 *   declare both, because a level missing them is a level whose share of the total
 *   cannot be verified — and it would fail its own preflight anyway, just later and
 *   only once it is fetched, which is after the allocation.
 */
function assertLadderTotalsBackedByLevels(
  node: SceneNode,
  declaredVertices: unknown,
  declaredFaces: unknown,
  levels: { levelVertices: Array<number | undefined>; levelFaces: Array<number | undefined> }
): void {
  const declared = { n_vertices: declaredVertices, n_faces: declaredFaces };
  if (typeof declared.n_vertices !== 'number' && typeof declared.n_faces !== 'number') return;

  const usable = (n: number | undefined): n is number =>
    n !== undefined && Number.isSafeInteger(n) && n >= 0;
  const perLevel = { n_vertices: levels.levelVertices, n_faces: levels.levelFaces };
  for (const key of ['n_vertices', 'n_faces'] as const) {
    if (perLevel[key].every(usable)) continue;
    throw new LoaderError(
      'Validation',
      node.path,
      new Error(
        `Mesh reveal ladder declares ${key}=${String(declared[key])} on the parent, but a ` +
          `sub-LOD does not declare its own ${key}. The parent's totals size every GPU ` +
          'buffer the node ever binds, and only the levels can vouch for them — so a ' +
          'ladder whose levels do not is refused rather than trusted.'
      )
    );
  }

  for (const key of ['n_vertices', 'n_faces'] as const) {
    const total = declared[key];
    if (typeof total !== 'number') continue;
    const sum = (perLevel[key] as number[]).reduce((s, n) => s + n, 0);
    if (total > sum) {
      throw new LoaderError(
        'Validation',
        node.path,
        new Error(
          `Mesh reveal ladder declares ${key}=${total.toLocaleString()} on the parent but its ` +
            `${perLevel[key].length} levels hold ${sum.toLocaleString()}. The parent total is ` +
            'what every buffer is sized from, on the first commit, so a declaration its own ' +
            'levels do not back would allocate for geometry that will never arrive.'
        )
      );
    }
  }
}

/**
 * Decide a Points or Lines ladder's per-level label flags and its picking-map
 * level offsets over the PARENT's union string/image CSR (#1422, #1439).
 *
 * `countKey` names each level's on-disk element count in that CSR's index
 * space: `n_points` for Points, `n_vertices` for Lines (whose CSR is
 * per-VERTEX). `levelOffsets` is CSR-style (`levels.length + 1` entries) or
 * `null` when the composition cannot run; the label flags are on only when it
 * can, so no level builds a map the concat would discard.
 */
function ladderPickingLayout(
  node: SceneNode,
  levels: AdditiveLevel[],
  countKey: 'n_points' | 'n_vertices',
  label: 'Points' | 'Lines'
): { levelOffsets: number[] | null; labels: LevelLabelFlags } {
  // Per-level ON-DISK row counts (`countKey` on each `additive_<i>` group, NOT
  // the loaded count) — the offsets into the parent CSR's index space.
  const onDiskCounts = levels.map((level) =>
    typeof level.attrs[countKey] === 'number' ? (level.attrs[countKey] as number) : undefined
  );

  // Does the PARENT node carry a UNION string/image CSR? Each CSR is keyed
  // by the concatenation `additive_0 || additive_1 || …`, each level in its own
  // stored order (#1422), so it is the parent — never a sub-LOD — that decides
  // whether a ladder has readable per-element metadata at all.
  const parentDeclaresStringChannel =
    node.attrs.has_labels === true ||
    node.attrs.has_image_labels === true ||
    node.attrs.has_keys === true;
  // CSR-style bounds over the on-disk counts, length `nAdditive + 1`:
  // `levelOffsets[i]` is where level `i` starts inside the parent's union CSR
  // index space (so [0] === 0) and `levelOffsets[i + 1]` is where it ends, so
  // the composer can also BOUND each level's ids instead of only shifting
  // them. The last entry is the union's total row count.
  // Only meaningful when the parent declares a string/image channel; otherwise stay
  // null (hover then reports the raw slot — no better than before #1439, but
  // never an id composed into someone else's CSR row).
  let levelOffsets: number[] | null = null;
  if (parentDeclaresStringChannel) {
    const usable = onDiskCounts.every((n) => n !== undefined && Number.isSafeInteger(n) && n >= 0);
    if (!usable) {
      log.warning(
        Modules.SCENE_LOADER,
        `Progressive ${label} ${node.path} declares a string/image channel but a sub-LOD is missing a valid ` +
          `\`${countKey}\`; per-level picking maps are disabled (hover falls back to the ` +
          'visible-buffer slot).'
      );
    } else {
      const total = (onDiskCounts as number[]).reduce((s, n) => s + n, 0);
      // Free cross-check: the writer's parent count IS the sum of the
      // levels' row counts, so a mismatch means the CSR and
      // the levels come from different builds — composing would land in
      // someone else's row. Only checked when the parent actually carries the
      // attr; its absence is not evidence of anything.
      if (typeof node.attrs[countKey] === 'number' && node.attrs[countKey] !== total) {
        log.warning(
          Modules.SCENE_LOADER,
          `Progressive ${label} ${node.path}: sub-LOD row counts sum to ${total} but the parent ` +
            `declares ${countKey}=${String(node.attrs[countKey])}; the parent CSR and the levels disagree, ` +
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
          `Progressive ${label} ${node.path}: sub-LOD row counts sum to ${total}, past the ` +
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
  const levelsBuildMaps = parentDeclaresStringChannel && levelOffsets !== null;

  // A ladder's string/image CSR lives on the PARENT node, spanning the levels in
  // `additive_<i>` order (#1422), and that is also the only path the pick path
  // ever resolves per-element metadata against — a sub-LOD carries no CSR of ITS
  // OWN that any reader can key by, so a sub-LOD's stored flags are never trusted
  // and are always overridden here. They are overridden with the parent's
  // declaration, AND only when the composition can actually run: with
  // `levelOffsets` each level publishes its own level-space map (Points: slot →
  // on-disk; Lines: its loaded on-disk vertex ranges) and the ladder concat
  // offsets it into the parent's index space by the preceding levels' on-disk
  // counts (#1439). Without them (no parent CSR, or a
  // failed cross-check) all stay false, so no per-level map is built only to be
  // discarded, and picking stays allocation-free exactly as before.
  const labels: LevelLabelFlags = {
    has_labels: levelsBuildMaps && node.attrs.has_labels === true,
    has_image_labels: levelsBuildMaps && node.attrs.has_image_labels === true,
    has_keys: levelsBuildMaps && node.attrs.has_keys === true,
  };
  return { levelOffsets, labels };
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Creating progressive Points loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  // PASS 1 — open every `additive_<i>` group (one open per level, in order)
  // and collect what the level offsets are decided from. The synthesized nodes
  // are built in pass 2, AFTER `levelOffsets` is known: their label flags
  // depend on it, and stamping them first would make every level build a
  // per-level picking map that the concat is going to discard.
  const levels = await openAdditiveLevels(node, nAdditive, deps);
  const energyTable = levels.map((level) => energyFractionFromAttrs(level.attrs));
  const { levelOffsets, labels } = ladderPickingLayout(node, levels, 'n_points', 'Points');

  // PASS 2 — synthesize each sub-LOD node and its loader.
  const lodLoaders = levels.map(
    (level, i) =>
      new PointsSpatialIndexLoader(
        level.loc,
        buildAdditiveLevelNode(
          node,
          i,
          'points',
          composeLevelAttrs(level, node, parentEffectiveAttrs, labels)
        ),
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
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
  log.verbose(
    LogEmoji.QUERY,
    Modules.SCENE_LOADER,
    `Creating progressive Lines loader for ${node.path} (${nAdditive} additive sub-LODs)`
  );

  const levels = await openAdditiveLevels(node, nAdditive, deps);
  // The ladder's string/image CSR lives on the PARENT, per-VERTEX and spanning
  // the levels in `additive_<i>` order (#1422), so the offsets are over each
  // level's on-disk VERTEX count. With them each level publishes its loaded
  // on-disk vertex ranges and `concatenateLinesData` shifts them into the
  // parent's index space — the lines twin of the Points composition (#1439).
  const { levelOffsets, labels } = ladderPickingLayout(node, levels, 'n_vertices', 'Lines');
  const lodLoaders = levels.map(
    (level, i) =>
      new LinesSpatialIndexLoader(
        level.loc,
        buildAdditiveLevelNode(
          node,
          i,
          'lines',
          composeLevelAttrs(level, node, parentEffectiveAttrs, labels)
        ),
        deps.arrayRefRegistry,
        deps.zarrStore,
        deps.l0Cache ?? undefined,
        deps.cachingStore?.getPrefetcher() ?? undefined
      )
  );

  return new LinesProgressiveLoader(
    lodLoaders,
    nAdditive,
    node.path,
    levels.map((level) => energyFractionFromAttrs(level.attrs)),
    deps.sliceCache ?? undefined,
    levelOffsets
  );
}
