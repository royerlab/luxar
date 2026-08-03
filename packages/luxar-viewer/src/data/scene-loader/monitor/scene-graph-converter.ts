/**
 * Pure conversion from a {@link SceneNode} (the loader's representation)
 * to a {@link SceneGraphNode} (the data-monitor UI's representation).
 *
 * Extracted from scene-loader.ts so the recursion + display-name
 * derivation + type-specific stat extraction can be unit-tested
 * without instantiating SceneLoader.
 *
 * @module data/scene-loader/monitor/scene-graph-converter
 */

import type { SceneNode } from '../../data-loader-types';
import type { SceneGraphNode } from '../../../types/data-monitor-types';

/** Valid scene-graph-node display types. */
type GraphNodeType = SceneGraphNode['type'];

/** Whitelist of valid display types — anything else falls back to 'scene'. */
const VALID_TYPES: ReadonlySet<GraphNodeType> = new Set<GraphNodeType>([
  'scene',
  'group',
  'points',
  'lines',
  'gsplats',
  'mesh',
]);

/**
 * Pretty-print the node's display name from its path:
 *   - root path "/" of a true scene → "Scene"
 *   - root path "/" of a bare-node file (a standalone .gsplats.zarr opened
 *     directly) → a type label ("GSplats" / "Points" / "Lines" / "LOD" /
 *     "Partition") instead of the misleading "Scene"
 *   - "/foo/bar"     → "bar"
 *   - "" or weird    → fall back to the raw path string
 */
function deriveDisplayName(path: string, rootType?: string): string {
  if (path === '/') {
    if (!rootType || rootType === 'scene') return 'Scene';
    const labels: Record<string, string> = {
      gsplats: 'GSplats',
      points: 'Points',
      lines: 'Lines',
      lod: 'LOD',
      partition: 'Partition',
      group: 'Group',
    };
    return labels[rootType] ?? rootType;
  }
  return path.split('/').filter(Boolean).pop() || path;
}

/**
 * Coerce the node's `type` field to a valid `SceneGraphNode['type']`.
 *
 * Whitelist explicitly so downstream switch statements can rely on
 * the union being honest. A bare `as` cast would let unknown types
 * (e.g. a future schema's `'volume'`) leak in and lie to TypeScript.
 */
function deriveDisplayType(rawType: string | undefined): GraphNodeType {
  if (!rawType || rawType === 'scene') return 'scene';
  return VALID_TYPES.has(rawType as GraphNodeType) ? (rawType as GraphNodeType) : 'scene';
}

/**
 * Leaf geometry display types a specialized group can resolve to.
 *
 * DO NOT widen to `GEOMETRY_TYPES`. This gates the `display_type` attr of a
 * `kind=lod` / `kind=partition` group, so it is the set of geometry types that
 * actually support those containers — the mirror of the Python-side allowlist in
 * `core/node/specialized_groups.py`. A geometry type with no LOD/partition
 * support must not be admitted here just because it is a valid leaf type.
 */
const LEAF_DISPLAY_TYPES: ReadonlySet<string> = new Set(['points', 'lines', 'gsplats']);

/**
 * Read the specialized-group discriminant (`kind=lod` / `kind=partition`)
 * from a `type === 'group'` node's attrs. Returns `undefined` for plain
 * groups and leaves. Mirrors the resolution in `ui/layers/layer-state.ts`.
 */
function deriveKind(node: SceneNode): SceneGraphNode['kind'] {
  if (node.type !== 'group') return undefined;
  const rawKind = (node.attrs as Record<string, unknown>).kind;
  return rawKind === 'lod' || rawKind === 'partition' ? rawKind : undefined;
}

/**
 * Convert a `SceneNode` to a `SceneGraphNode`. Recursive: children are
 * converted via the same function so the whole tree gets the same
 * shape.
 *
 * Per-type stats:
 *   - points  → `pointCount` from `attrs.n_points`
 *   - lines   → `segmentCount` + `vertexCount` from `attrs.n_*`
 *   - gsplats → `splatCount` from `attrs.n_splats`
 *   - other   → no extra stats
 *
 * Specialized groups (`kind=lod` / `kind=partition`) additionally carry
 * `kind`, the resolved geometry `displayType`, and a `lodGroupChildCount`
 * / `partCount` so the monitor tree can render a kind badge + icon (mirrors
 * `ui/layers/layer-state.ts`). The node's `type` stays `'group'` so the
 * stats aggregator keeps treating it as a container — substitutive-LOD
 * de-duplication lives in `calculateSceneGraphStats`, keyed off `kind`.
 */
export function convertToSceneGraphNode(node: SceneNode): SceneGraphNode {
  // For a bare-node root ("/"), prefer the kind (lod/partition) over the raw
  // "group" type so the monitor shows "LOD"/"Partition" rather than "Scene".
  const rootKind =
    node.type === 'group'
      ? ((node.attrs as Record<string, unknown> | undefined)?.kind as string | undefined)
      : node.type;
  const name = deriveDisplayName(node.path, rootKind);
  const type = deriveDisplayType(node.type);

  const graphNode: SceneGraphNode = {
    path: node.path,
    name,
    type,
    children: [],
    hasSpatialIndex: node.hasSpatialIndex,
  };

  if (node.type === 'points') {
    graphNode.pointCount = node.attrs.n_points;
  } else if (node.type === 'lines') {
    graphNode.segmentCount = node.attrs.n_segments as number | undefined;
    graphNode.vertexCount = node.attrs.n_vertices as number | undefined;
  } else if (node.type === 'gsplats') {
    graphNode.splatCount = node.attrs.n_splats as number | undefined;
  }

  const kind = deriveKind(node);
  if (kind) {
    graphNode.kind = kind;
    const childCount = node.children?.length ?? 0;
    if (kind === 'lod') graphNode.lodGroupChildCount = childCount;
    else graphNode.partCount = childCount;
    // Resolve the user-facing geometry type (points/lines/gsplats) the
    // group represents, written by the Python compiler. Used for the tree
    // icon; absent / non-leaf values are simply left undefined.
    const displayType = (node.attrs as Record<string, unknown>).display_type;
    if (typeof displayType === 'string' && LEAF_DISPLAY_TYPES.has(displayType)) {
      graphNode.displayType = displayType as SceneGraphNode['displayType'];
    }
  }

  // Additive-LOD marker: the parent leaf carries `n_additive_sublods`; its
  // `additive_<i>` subgroups are pruned from the scene graph (they belong
  // to the progressive loader). Record the count so the tree renders a
  // "LOD x/N" slot — live progress comes from the LODProgressProvider.
  const nAdditive = (node.attrs as Record<string, unknown>).n_additive_sublods;
  if (typeof nAdditive === 'number' && nAdditive > 1) {
    graphNode.additiveSublods = nAdditive;
  }

  if (node.children) {
    graphNode.children = node.children.map(convertToSceneGraphNode);
  }

  return graphNode;
}
