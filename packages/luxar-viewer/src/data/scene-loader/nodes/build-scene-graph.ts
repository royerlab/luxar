/**
 * Build the hierarchical SceneNode tree from a zarr root's contents.
 *
 * Enumerates the store, sorts by path depth (parents before children),
 * and for each group opens it to read its attrs, skips overlay groups
 * (those live in a parallel screen-space tree), skips internal
 * subgroups of gsplats nodes (e.g. ``additive_<i>/`` LOD subgroups —
 * those belong to the gsplats loader, not the scene graph), and — for
 * nodes that declare `colormap === 'custom'` — eagerly loads the
 * sibling `colormap_lut` zarr array and attaches the raw bytes to the
 * node's attrs so the NodeFactory can build a
 * `getColormapTexture('custom', lut)` later without a second async hop.
 *
 * Extracted from SceneLoader.buildSceneGraph so the orchestrator can
 * stay focused on lifecycle and the graph construction can be tested
 * directly against zarr fixtures.
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { ZarrSceneAttrs, ZarrNodeAttrs } from '../../../types/zarr';
import type { SceneNode } from '../../data-loader-types';
import { isGeometryType } from '../../../types/geometry-capabilities';
import { enumerateStore } from './enumerate-store';
import { normalizeExtendDims } from '../view-state/extend-tolerance';

/**
 * Normalize + log a node's `extend_to_all`. The raw attr is NOT trusted to be a
 * `string[]`: an older producer could stamp the unresolved `'all'` sentinel, and
 * `'all'.join` is undefined — the `log.data` line below is evaluated eagerly, so
 * that used to throw and abort the whole scene-graph build. The normalized array
 * is written BACK onto the node's attrs (same idiom as `customLutBytes`), which
 * is what makes this the coercion point for the whole graph: every downstream
 * consumer (loader-factory's synthesized LOD children, the spatial-index
 * loaders' `attrs.extend_to_all || []`, `announceExtendToAllOnce`,
 * `SpatialQueryBuilder`) reads the node attrs and would otherwise re-inherit the
 * raw value and throw at query time instead. Written back only when the attr was
 * PRESENT — a node without one keeps no key at all (nothing downstream
 * distinguishes absent from `[]`, but there is no reason to invent one).
 * `deriveNodeViewState` normalizes again as belt-and-braces.
 */
function normalizeNodeExtendDims(node: SceneNode): void {
  const rawExtendDims: unknown = (node.attrs as ZarrNodeAttrs | undefined)?.extend_to_all;
  if (rawExtendDims === undefined || rawExtendDims === null) return;

  const extendDims = normalizeExtendDims(rawExtendDims);
  const malformed = !Array.isArray(rawExtendDims) || extendDims.length < rawExtendDims.length;
  if (malformed) {
    log.warning(
      Modules.SCENE_LOADER,
      `Node ${node.path} has a malformed extend_to_all attr ` +
        `(${JSON.stringify(rawExtendDims)}) — expected a list of dimension names. ` +
        (extendDims.length === 0
          ? 'Treating the node as not extended.'
          : `Extending across [${extendDims.join(', ')}] only.`)
    );
  }
  (node.attrs as ZarrNodeAttrs).extend_to_all = extendDims;
  if (extendDims.length > 0) {
    log.data(Modules.SCENE_LOADER, `Node ${node.path} has extend_to_all: ${extendDims.join(', ')}`);
  }
}

/**
 * Load a node-authored custom colormap LUT without making scene loading depend
 * on the optional sibling array being valid.
 *
 * The Python writer stores LUTs as uint8 arrays of shape [256, 3] or [256, 4],
 * but tolerate other typed arrays so a malformed producer degrades to the same
 * texture path instead of aborting the whole scene. A missing or unreadable LUT
 * is likewise non-fatal: `getColormapTexture` owns the warning-backed viridis
 * fallback for `colormap='custom'` without bytes.
 */
async function loadCustomColormapLut(
  node: SceneNode,
  loc: zarr.Location<zarr.Readable>
): Promise<void> {
  if ((node.attrs as ZarrNodeAttrs).colormap !== 'custom') return;

  try {
    const lutArr = await zarr.open(loc.resolve('colormap_lut'), { kind: 'array' });
    const lutResult = await zarr.readArray(lutArr);
    const data = lutResult.data;
    let bytes: Uint8Array;
    if (data instanceof Uint8Array) {
      bytes = data;
    } else if (data instanceof Int8Array || data instanceof Uint8ClampedArray) {
      bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    } else {
      // Float / int16 etc. — unexpected for a LUT but recover by copying the bytes view.
      bytes = new Uint8Array((data as ArrayBufferView).buffer);
    }
    (node.attrs as ZarrNodeAttrs).customLutBytes = bytes;
    log.info(
      Modules.SCENE_LOADER,
      `${node.path}: loaded custom colormap LUT (${bytes.length} bytes)`
    );
  } catch (e: unknown) {
    // A producer may declare `colormap='custom'` without the sibling array.
    // Keep loading the scene; the texture helper falls back to viridis and this
    // warning preserves the diagnosis without turning appearance into I/O failure.
    log.warning(
      Modules.SCENE_LOADER,
      `${node.path}: colormap='custom' but failed to load colormap_lut zarr array — falling back to viridis. ${e instanceof Error ? e.message : ''}`
    );
  }
}

/**
 * Group the listing's ARRAY entries by their parent group path (relative
 * array names). Returns `null` when the listing carries no array at all, so a
 * caller can tell "no arrays visible" from "this node has none".
 */
export function collectArraysByParent(
  listing: ReadonlyArray<{ path: string; kind: string }>
): Map<string, Set<string>> | null {
  const byParent = new Map<string, Set<string>>();
  for (const entry of listing) {
    if (entry.kind !== 'array') continue;
    const slash = entry.path.lastIndexOf('/');
    const parent = slash <= 0 ? '/' : entry.path.slice(0, slash);
    const name = entry.path.slice(slash + 1);
    if (!name) continue;
    let set = byParent.get(parent);
    if (!set) {
      set = new Set<string>();
      byParent.set(parent, set);
    }
    set.add(name);
  }
  return byParent.size > 0 ? byParent : null;
}

/**
 * Build the scene graph structure rooted at `rootLoc`. The optional
 * `store` argument is the same store the location was opened from —
 * `enumerateStore` is called with it to get the contents listing.
 */
export async function buildSceneGraph(
  rootLoc: zarr.Location<zarr.Readable>,
  rootAttrs: ZarrSceneAttrs,
  store: zarr.Readable | null
): Promise<SceneNode> {
  // Enumerate all groups in the store
  const listing = await enumerateStore(store);

  // A standalone .gsplats.zarr is a *detached node subtree* — the file root
  // IS the node (a gsplats/points/lines leaf, or a kind=lod / kind=partition
  // group), not a scene container. Adopt the real root type/kind so
  // loadSceneNodes dispatches the root through the right leaf/group loader.
  // A scene root (type 'scene' or absent) keeps the container behaviour.
  const rootType = (rootAttrs as ZarrNodeAttrs)?.type;
  const rootKind = (rootAttrs as ZarrNodeAttrs)?.kind;
  const isBareLeafRoot = isGeometryType(rootType);
  const isBareGroupRoot = rootType === 'group' && (rootKind === 'lod' || rootKind === 'partition');
  const isBareNodeRoot = isBareLeafRoot || isBareGroupRoot;

  // Build hierarchical structure
  const root: SceneNode = {
    path: '/',
    type: isBareNodeRoot ? (rootType as string) : 'scene',
    attrs: rootAttrs,
    hasSpatialIndex: false,
    children: [],
  };

  // The loop below only visits the listing, which excludes '/', so the root is
  // coerced here. It matters for a BARE NODE root (a detached `.gsplats.zarr`
  // subtree, where the file root IS the leaf/lod/partition node): its attrs go
  // straight to a leaf loader, and `deriveNodeViewState`'s own normalization
  // does not reach `SpatialQueryBuilder` / `announceExtendToAllOnce`, which read
  // the node attrs directly. A scene root never carries the attr, so this is a
  // no-op there.
  normalizeNodeExtendDims(root);
  await loadCustomColormapLut(root, rootLoc);

  // Build node map
  const nodeMap = new Map<string, SceneNode>();
  nodeMap.set('/', root);

  // Child arrays per group path, from the same listing. Leaf loaders consult
  // `node.arrays` before opening an OPTIONAL array (colors / radii / widths /
  // sharpnesses): the historical try/open-and-404 probe cost three round
  // trips per node on a 100-node scene before any geometry moved. Only a
  // listing that actually contains arrays is trusted — an array-less listing
  // (fallback enumeration, an odd store) leaves `arrays` undefined so the
  // loaders keep probing rather than silently dropping authored channels.
  const arraysByParent = collectArraysByParent(listing);
  if (arraysByParent) root.arrays = arraysByParent.get('/') ?? new Set();

  // Sort by path depth to ensure parents are created before children
  const sortedPaths = listing
    .filter((e) => e.kind === 'group' && e.path !== '/')
    .sort((a, b) => a.path.split('/').length - b.path.split('/').length);

  // Prefix-path roots whose subtrees are *internal* to a leaf loader and
  // must not appear as scene-graph children. Currently only gsplats LOD
  // subgroups (`additive_<i>/`) — the gsplats loader walks them itself.
  const internalSubtreePrefixes: string[] = [];

  // A bare LEAF root owns its entire subtree: its `additive_<i>/` ladder
  // subgroups are handled by the leaf loader, not the scene graph. Marking
  // the whole store internal leaves the root as a childless leaf node that
  // loadSceneNodes dispatches via loadGSplatsNode/etc. (A bare lod/partition
  // root keeps its `child_<i>/`/`part_<i>/` children — only its grandchildren
  // are marked internal by the per-node logic below.)
  if (isBareLeafRoot) {
    internalSubtreePrefixes.push('/');
  }

  for (const entry of sortedPaths) {
    // Skip overlays group — screen-space overlays are not part of the 3D scene graph
    if (entry.path === '/overlays' || entry.path.startsWith('/overlays/')) {
      continue;
    }

    // Skip anything sitting inside a leaf-loader's internal subtree
    // (e.g. `additive_<i>/` LOD subgroups under a gsplats node).
    if (internalSubtreePrefixes.some((prefix) => entry.path.startsWith(prefix))) {
      continue;
    }

    const loc = rootLoc.resolve(entry.path.slice(1)); // Remove leading /
    const group = await zarr.open(loc, { kind: 'group' });
    const attrs = group.attrs as ZarrNodeAttrs;

    // Skip metadata-sidecar groups. A standalone `.gsplats.zarr` carries a
    // provenance group (`fitting/`, written by `save_gsplats` via a raw
    // `create_group`) alongside the real nodes. It is NOT renderable geometry.
    // Every genuine scene node is stamped with an explicit `type` (leaf →
    // points/lines/gsplats; container → "group" via `add_group`) and lod /
    // partition groups additionally carry `kind`; a raw `create_group` sidecar
    // has neither. Without this skip, a sidecar sibling of a standalone
    // `kind=lod` / `kind=partition` root is adopted as a phantom LOD/partition
    // child — it has no `coverage_fraction` (the selector defaults it to 0) and no
    // `child_index` (so it sorts last), corrupting the ascending `coverage_fraction`
    // ladder and firing a spurious "thresholds not strictly ascending" warning
    // that misblames the producer. Mark the subtree internal so any nested
    // metadata (e.g. `fitting/config`) is skipped too. (The `/overlays` skip
    // above is path-based; this attr-based rule covers any future sidecar.)
    const kind = (attrs as Record<string, unknown> | null)?.kind;
    if (attrs?.type == null && kind == null) {
      internalSubtreePrefixes.push(`${entry.path}/`);
      continue;
    }

    // We no longer check for spatial index here - PointsSpatialIndexLoader handles it
    const node: SceneNode = {
      path: entry.path,
      type: attrs?.type || 'group',
      attrs: attrs || {},
      hasSpatialIndex: false, // Will be determined by the loader
      children: [],
    };
    if (arraysByParent) node.arrays = arraysByParent.get(entry.path) ?? new Set();

    await loadCustomColormapLut(node, loc);

    normalizeNodeExtendDims(node);

    // Find parent and add as child
    const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
    const parent = nodeMap.get(parentPath);
    if (parent) {
      parent.children = parent.children || [];
      parent.children.push(node);
    }

    nodeMap.set(entry.path, node);

    // Geometry nodes are leaves from a scene-graph perspective. Their
    // `additive_<i>/` multi-LOD subgroups carry the same `type` themselves and
    // would otherwise show up as spurious child nodes in the monitor UI; mark
    // the subtree internal so subsequent iterations skip it. Symmetric across
    // the whole geometry vocabulary.
    if (isGeometryType(node.type)) {
      internalSubtreePrefixes.push(`${entry.path}/`);
    }
  }

  // Restore napari-style insertion order. Children are pushed in store
  // enumeration order (consolidated metadata → alphabetical), but the Python
  // `Node` stamps each child's add order as `child_index`. Sort every sibling
  // list by it so the layers panel, picking, and any order-sensitive consumer
  // follow scene-authoring order. Internal `child_<i>`/`part_<i>` gsplat
  // subgroups are likewise stamped (with their numeric index `i`) by the
  // gsplat-tree writers, so they sort correctly too — fixing the alphabetical
  // `part_10`-before-`part_2` misordering for groups with ≥10 children. The
  // sort is stable, so only genuinely legacy/unstamped children (no
  // `child_index`) fall back to enumeration order via the `?? Infinity`
  // sentinel.
  const sortChildrenByInsertionOrder = (node: SceneNode): void => {
    if (!node.children || node.children.length === 0) return;
    node.children.sort((a, b) => {
      const ai = (a.attrs as ZarrNodeAttrs)?.child_index ?? Infinity;
      const bi = (b.attrs as ZarrNodeAttrs)?.child_index ?? Infinity;
      return ai - bi;
    });
    for (const child of node.children) sortChildrenByInsertionOrder(child);
  };
  sortChildrenByInsertionOrder(root);

  return root;
}
