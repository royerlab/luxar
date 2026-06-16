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
import { enumerateStore } from './enumerate-store';

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
  const isBareLeafRoot = rootType === 'gsplats' || rootType === 'points' || rootType === 'lines';
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

  // Build node map
  const nodeMap = new Map<string, SceneNode>();
  nodeMap.set('/', root);

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

    // We no longer check for spatial index here - PointsSpatialIndexLoader handles it
    const node: SceneNode = {
      path: entry.path,
      type: attrs?.type || 'group',
      attrs: attrs || {},
      hasSpatialIndex: false, // Will be determined by the loader
      children: [],
    };

    // when a node's metadata declares colormap='custom', load its
    // colormap_lut zarr array (if present) and attach the bytes to the
    // node's attrs so NodeFactory can pass them into
    // getColormapTexture('custom', lut). Without this step the viewer
    // falls back to the viridis built-in (handled by getColormapTexture).
    if (attrs && (attrs as ZarrNodeAttrs).colormap === 'custom') {
      try {
        const lutArr = await zarr.open(loc.resolve('colormap_lut'), { kind: 'array' });
        const lutResult = await zarr.get(lutArr);
        const data = lutResult.data;
        // Promote whatever typed-array we got into a tightly-typed Uint8Array.
        // The Python writer stores LUTs as uint8 of shape [256,3] or [256,4].
        let bytes: Uint8Array;
        if (data instanceof Uint8Array) {
          bytes = data;
        } else if (data instanceof Int8Array || data instanceof Uint8ClampedArray) {
          bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        } else {
          // Float / int16 etc. — unexpected for a LUT but recover by copying bytes view.
          bytes = new Uint8Array((data as ArrayBufferView).buffer);
        }
        (node.attrs as ZarrNodeAttrs).customLutBytes = bytes;
        log.info(
          Modules.SCENE_LOADER,
          `${entry.path}: loaded custom colormap LUT (${bytes.length} bytes)`
        );
      } catch (e: unknown) {
        // colormap_lut may not exist if a node declared colormap='custom'
        // by mistake. getColormapTexture will fall back to viridis with a
        // warning. We don't fail the scene load.
        log.warning(
          Modules.SCENE_LOADER,
          `${entry.path}: colormap='custom' but failed to load colormap_lut zarr array — falling back to viridis. ${e instanceof Error ? e.message : ''}`
        );
      }
    }

    // Log if extend_to_all is present
    if (attrs?.extend_to_all) {
      log.data(
        Modules.SCENE_LOADER,
        `Node ${entry.path} has extend_to_all: ${attrs.extend_to_all.join(', ')}`
      );
    }

    // Find parent and add as child
    const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
    const parent = nodeMap.get(parentPath);
    if (parent) {
      parent.children = parent.children || [];
      parent.children.push(node);
    }

    nodeMap.set(entry.path, node);

    // Points / Lines / GSplats nodes are leaves from a scene-graph
    // perspective. Their `additive_<i>/` multi-LOD subgroups carry
    // the same `type` themselves and would otherwise show up as
    // spurious child nodes in the monitor UI; mark the subtree
    // internal so subsequent iterations skip it. Symmetric across
    // all three leaf types.
    if (node.type === 'gsplats' || node.type === 'points' || node.type === 'lines') {
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
