import type {
  GeometryCounters,
  SceneGraphNode,
  SceneGraphState,
} from '../../types/data-monitor-types';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../types/format-contract';

/**
 * A fresh all-zero per-type counter record, one slot per geometry type.
 *
 * Written as a plain loop, not `Object.fromEntries(GEOMETRY_TYPES.map(...))`:
 * `calculateSceneGraphStats` calls this twice per scene-graph node, and the
 * `fromEntries` form allocates an intermediate array of `[key, 0]` pairs on every
 * call. Measured on a 5000-node tree that shape cost ~6-10 ms per
 * `setSceneGraph` against ~0.4-1.3 ms for the loop — a 5-15x difference for no
 * behavioural gain.
 */
function zeroCounters(): GeometryCounters {
  const counters = {} as GeometryCounters;
  for (const type of GEOMETRY_TYPES) counters[type] = 0;
  return counters;
}

/** The empty scene-graph state (no scene loaded / scene torn down). */
function emptySceneGraphState(): SceneGraphState {
  return {
    root: null,
    totalNodes: 0,
    nodesByType: zeroCounters(),
    totalByType: zeroCounters(),
    visibleByType: zeroCounters(),
    droppedElements: 0,
  };
}

/**
 * This node's own element count for `type`, or 0 when it is not that type.
 *
 * The per-type count fields are named after each type's ELEMENT (points have
 * points, lines have segments, gsplats have splats), so a table cannot key them
 * by type name. The `never` tail makes adding a geometry type a compile error
 * here — a plain `return 0` would leave the new type's elements out of every
 * dataset total with nothing to explain why.
 *
 * The tail still returns 0 rather than the unhandled value: breaking at compile
 * time is the point, but at runtime a count must stay a number or it poisons
 * every total it is summed into.
 *
 * The integer check and not `?? 0` / `|| 0`: these counts are read straight
 * off zarr attrs with a bare cast (`scene-graph-converter.ts`), so only a
 * non-negative integer may enter the totals. This rejects `NaN`, infinity,
 * negative/fractional counts, and truthy non-numbers.
 */
function elementCountOf(node: SceneGraphNode, type: GeometryTypeName): number {
  if (node.type !== type) return 0;
  let count: number | undefined;
  switch (type) {
    case 'points':
      count = node.pointCount;
      break;
    case 'lines':
      count = node.segmentCount;
      break;
    case 'gsplats':
      count = node.splatCount;
      break;
    case 'mesh':
      // Faces, matching the drawn-primitive convention (`lines` counts
      // segments, not vertices). Missing attrs degrade to 0 below.
      count = node.faceCount;
      break;
    default:
      void (type satisfies never);
      return 0;
  }
  return typeof count === 'number' && Number.isInteger(count) && count >= 0 ? count : 0;
}

/** Owns scene-graph state, expansion state, and the memoized path index. */
export class SceneGraphModel {
  /** Per-path visible counts pushed by the SceneLoader's visible-counts walk. */
  private visibleCountsByPath: ReadonlyMap<string, number> = new Map();
  private sceneGraphState: SceneGraphState = emptySceneGraphState();
  /** Track expanded nodes in the scene graph tree by path. */
  private mutableExpandedNodes = new Set<string>(['/']);
  // Rebuilt only when the root reference changes, making per-tick lookups O(1).
  private sceneGraphNodeIndex = new Map<string, SceneGraphNode>();
  private sceneGraphNodeIndexRoot: SceneGraphNode | null = null;

  constructor(private readonly markStructureDirty: () => void) {}

  get expandedNodes(): ReadonlySet<string> {
    return this.mutableExpandedNodes;
  }

  /** Reset tree, visibility, and expansion state for reload or teardown. */
  resetSceneGraphState(): void {
    this.sceneGraphState = emptySceneGraphState();
    this.mutableExpandedNodes = new Set<string>(['/']);
    this.visibleCountsByPath = new Map();
    this.sceneGraphNodeIndex.clear();
    this.sceneGraphNodeIndexRoot = null;
    this.markStructureDirty();
  }

  setSceneGraph(root: SceneGraphNode): void {
    this.sceneGraphState = {
      root,
      ...this.calculateSceneGraphStats(root),
    };
    this.markStructureDirty();
  }

  getSceneGraph(): SceneGraphState {
    return this.sceneGraphState;
  }

  toggleNodeExpansion(path: string): void {
    if (this.mutableExpandedNodes.has(path)) {
      this.mutableExpandedNodes.delete(path);
    } else {
      this.mutableExpandedNodes.add(path);
    }
    this.markStructureDirty();
  }

  isNodeExpanded(path: string): boolean {
    return this.mutableExpandedNodes.has(path);
  }

  updateVisibleCount(type: GeometryTypeName, count: number): void {
    this.sceneGraphState.visibleByType[type] = count;
  }

  updateDroppedElementCount(count: number): void {
    this.sceneGraphState.droppedElements = count;
  }

  updateVisibleCountsByPath(counts: ReadonlyMap<string, number>): void {
    this.visibleCountsByPath = counts;
  }

  /**
   * Merge the latest path counts into geometry nodes in place. Paths absent
   * from the latest walk are reset to `undefined`, preventing stale tooltips.
   *
   * All four types, mesh included: the walk in `monitor/visible-counts.ts`
   * already stamps a mesh node's committed `visibleTriangleCount` into the
   * per-path map, so skipping mesh here dropped a number that had already been
   * measured — and left the tree's mesh badge unable to say how much of the
   * surface the current slab actually indexes.
   *
   * A `never`-tailed switch rather than an if/else chain, matching
   * `elementCountOf` above: the stamp fields are named after each type's own
   * element noun so they cannot be keyed by type name, but the DISPATCH can be
   * compile-checked. An unhandled type would silently leave its badge with no
   * visible count — verbatim how mesh went missing here in the first place.
   */
  syncVisibleCountsIntoTree(): void {
    const index = this.ensureSceneGraphNodeIndex();
    if (!index) return;
    for (const node of index.values()) {
      const visible = this.visibleCountsByPath.get(node.path);
      switch (node.type) {
        case 'points':
          node.visiblePointCount = visible;
          break;
        case 'lines':
          node.visibleSegmentCount = visible;
          break;
        case 'gsplats':
          node.visibleSplatCount = visible;
          break;
        case 'mesh':
          node.visibleFaceCount = visible;
          break;
        case 'scene':
        case 'group':
          // Containers carry no elements of their own.
          break;
        case 'sound':
          // Heard, not drawn: a sound node has no visible element count.
          break;
        default:
          void (node.type satisfies never);
      }
    }
  }

  getSceneGraphNodeByPath(path: string): SceneGraphNode | null {
    return this.ensureSceneGraphNodeIndex()?.get(path) ?? null;
  }

  clearExpandedNodes(): void {
    this.mutableExpandedNodes.clear();
  }

  /** Calculate structural counts and full-detail geometry totals recursively. */
  private calculateSceneGraphStats(node: SceneGraphNode): Omit<SceneGraphState, 'root'> {
    let totalNodes = 1;
    const nodesByType = zeroCounters();
    const totalByType = zeroCounters();
    for (const type of GEOMETRY_TYPES) {
      if (node.type === type) nodesByType[type] = 1;
      totalByType[type] = elementCountOf(node, type);
    }

    const childStats = node.children.map((child) => this.calculateSceneGraphStats(child));
    for (const stats of childStats) {
      totalNodes += stats.totalNodes;
      for (const type of GEOMETRY_TYPES) nodesByType[type] += stats.nodesByType[type];
    }

    // Substitutive LOD children represent the same data at different
    // resolutions. The Python writer guarantees coarsest-to-finest child
    // order, so only the finest (last) level contributes to totals. Partition
    // parts are disjoint, while additive children never reach the scene graph,
    // so only kind === 'lod' needs this de-duplication.
    const contributing =
      node.kind === 'lod' && childStats.length > 0
        ? [childStats[childStats.length - 1]]
        : childStats;
    for (const stats of contributing) {
      for (const type of GEOMETRY_TYPES) totalByType[type] += stats.totalByType[type];
    }

    return {
      totalNodes,
      nodesByType,
      totalByType,
      visibleByType: { ...totalByType },
      droppedElements: 0,
    };
  }

  private ensureSceneGraphNodeIndex(): Map<string, SceneGraphNode> | null {
    const root = this.sceneGraphState.root;
    if (!root) return null;
    if (this.sceneGraphNodeIndexRoot !== root) {
      this.sceneGraphNodeIndex.clear();
      const stack: SceneGraphNode[] = [root];
      while (stack.length > 0) {
        const node = stack.pop()!;
        this.sceneGraphNodeIndex.set(node.path, node);
        for (const child of node.children) stack.push(child);
      }
      this.sceneGraphNodeIndexRoot = root;
    }
    return this.sceneGraphNodeIndex;
  }
}
