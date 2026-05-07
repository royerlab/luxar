/**
 * Pure conversion from a {@link SceneNode} (the loader's representation)
 * to a {@link SceneGraphNode} (the data-monitor UI's representation).
 *
 * Extracted from scene-loader.ts so the recursion + display-name
 * derivation + type-specific stat extraction can be unit-tested
 * without instantiating SceneLoader.
 *
 * @module data/scene-loader/scene-graph-converter
 */

import type { SceneNode } from '../data-loader-types';
import type { SceneGraphNode } from '../../types/data-monitor-types';

/** Valid scene-graph-node display types. */
type GraphNodeType = SceneGraphNode['type'];

/**
 * Pretty-print the node's display name from its path:
 *   - root path "/"  → "Scene"
 *   - "/foo/bar"     → "bar"
 *   - "" or weird    → fall back to the raw path string
 */
function deriveDisplayName(path: string): string {
  if (path === '/') return 'Scene';
  return path.split('/').filter(Boolean).pop() || path;
}

/** Coerce the node's `type` field to a valid `SceneGraphNode['type']`. */
function deriveDisplayType(rawType: string | undefined): GraphNodeType {
  if (!rawType || rawType === 'scene') return 'scene';
  // `as` cast: the union is finite and rawType has already been narrowed
  // to a string by the loader; unknown types fall back to 'scene' above
  // is the conservative default.
  return rawType as GraphNodeType;
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
 */
export function convertToSceneGraphNode(node: SceneNode): SceneGraphNode {
  const name = deriveDisplayName(node.path);
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

  if (node.children) {
    graphNode.children = node.children.map(convertToSceneGraphNode);
  }

  return graphNode;
}
