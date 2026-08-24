import type {
  GeometryCounters,
  SceneGraphNode,
  SceneGraphState,
} from '../../types/data-monitor-types';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../types/format-contract';

/** A fresh all-zero per-type counter record, one slot per geometry type. */
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
  };
}

/** This node's own element count for `type`, or 0 when it is not that type. */
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
  private visibleCountsByPath: ReadonlyMap<string, number> = new Map();
  private sceneGraphState: SceneGraphState = emptySceneGraphState();
  private mutableExpandedNodes = new Set<string>(['/']);
  private sceneGraphNodeIndex = new Map<string, SceneGraphNode>();
  private sceneGraphNodeIndexRoot: SceneGraphNode | null = null;

  constructor(private readonly markStructureDirty: () => void) {}

  get expandedNodes(): ReadonlySet<string> {
    return this.mutableExpandedNodes;
  }

  resetSceneGraphState(): void {
    this.sceneGraphState = emptySceneGraphState();
    this.mutableExpandedNodes = new Set<string>(['/']);
    this.visibleCountsByPath = new Map();
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

  updateVisibleCountsByPath(counts: ReadonlyMap<string, number>): void {
    this.visibleCountsByPath = counts;
  }

  syncVisibleCountsIntoTree(): void {
    const index = this.ensureSceneGraphNodeIndex();
    if (!index) return;
    for (const node of index.values()) {
      const visible = this.visibleCountsByPath.get(node.path);
      if (node.type === 'points') node.visiblePointCount = visible;
      else if (node.type === 'lines') node.visibleSegmentCount = visible;
      else if (node.type === 'gsplats') node.visibleSplatCount = visible;
    }
  }

  getSceneGraphNodeByPath(path: string): SceneGraphNode | null {
    return this.ensureSceneGraphNodeIndex()?.get(path) ?? null;
  }

  clearExpandedNodes(): void {
    this.mutableExpandedNodes.clear();
  }

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

    const contributing =
      node.kind === 'lod' && childStats.length > 0
        ? [childStats[childStats.length - 1]]
        : childStats;
    for (const stats of contributing) {
      for (const type of GEOMETRY_TYPES) totalByType[type] += stats.totalByType[type];
    }

    return { totalNodes, nodesByType, totalByType, visibleByType: { ...totalByType } };
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
