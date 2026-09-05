import type {
  LODProgressState,
  NodeDensityState,
  NodeDrawOrder,
  SceneGraphNode,
  SceneGraphState,
} from '../../../types/data-monitor-types';
import {
  activeLevelRole,
  chipInnerHtml,
  countAdditiveNodes,
  densityChipContent,
  drawOrderChipContent,
  levelRoleTitleSuffix,
  lodChipContent,
  nodeStatsContent,
  summariseLodStates,
} from '../templates/scene-graph';

export interface SceneGraphBadgeSource {
  getSceneGraph(): SceneGraphState;
  getSceneGraphNodeByPath(path: string): SceneGraphNode | null;
}

/** Patch live scene-graph badges and row state without rebuilding the tree DOM. */
export function updateSceneGraphBadges(
  container: HTMLElement,
  model: SceneGraphBadgeSource,
  lodStates: ReadonlyMap<string, LODProgressState>,
  drawOrderStates: ReadonlyMap<string, NodeDrawOrder>,
  densityStates: ReadonlyMap<string, NodeDensityState> = new Map()
): void {
  const root = model.getSceneGraph().root;
  if (!root) return;

  const badges = container.querySelectorAll('.luxar-scene-graph__badge[data-node-path]');
  badges.forEach((badge) => {
    const path = (badge as HTMLElement).dataset.nodePath;
    if (!path) return;
    const node = model.getSceneGraphNodeByPath(path);
    if (!node) return;
    const stats = nodeStatsContent(node);
    if (stats) {
      badge.textContent = stats.text;
      (badge as HTMLElement).title = stats.title;
    }
  });

  // LOD progress changes every frame without changing tree structure.
  const lodChips = container.querySelectorAll('.luxar-scene-graph__lod[data-lod-path]');
  lodChips.forEach((chip) => {
    const path = (chip as HTMLElement).dataset.lodPath;
    if (!path) return;
    const node = model.getSceneGraphNodeByPath(path);
    if (!node) return;
    const content = lodChipContent(node, lodStates.get(path));
    chip.textContent = content?.text ?? '';
    (chip as HTMLElement).title = content?.title ?? '';
  });

  // Draw order is camera-dependent and may disappear for hidden nodes.
  const drawOrderChips = container.querySelectorAll(
    '.luxar-scene-graph__draworder[data-draworder-path]'
  );
  drawOrderChips.forEach((chip) => {
    const path = (chip as HTMLElement).dataset.draworderPath;
    if (!path) return;
    const content = drawOrderChipContent(drawOrderStates.get(path));
    // Glyph + text: the glyph is trusted `MONITOR_ICONS` markup, the text is escaped.
    chip.innerHTML = chipInnerHtml(content);
    (chip as HTMLElement).title = content?.title ?? '';
  });

  // Density thinning comes and goes with the camera (and the guard toggle).
  const densityChips = container.querySelectorAll('.luxar-scene-graph__density[data-density-path]');
  densityChips.forEach((chip) => {
    const path = (chip as HTMLElement).dataset.densityPath;
    if (!path) return;
    const content = densityChipContent(densityStates.get(path));
    chip.innerHTML = chipInnerHtml(content);
    (chip as HTMLElement).title = content?.title ?? '';
  });

  // The active substitutive level can switch between structural rebuilds.
  const levelRows = container.querySelectorAll('.luxar-scene-graph__node-row[data-level-of]');
  levelRows.forEach((row) => {
    const element = row as HTMLElement;
    const parentPath = element.dataset.levelOf;
    const indexRaw = element.dataset.levelIndex;
    if (!parentPath || indexRaw === undefined) return;
    const role = activeLevelRole(lodStates.get(parentPath), Number(indexRaw));
    element.classList.toggle('luxar-scene-graph__node-row--active-level', role === 'active');
    element.classList.toggle('luxar-scene-graph__node-row--inactive-level', role === 'inactive');
    const baseTitle = element.dataset.baseTitle;
    if (baseTitle !== undefined) {
      element.title = `${baseTitle}${levelRoleTitleSuffix(role)}`;
    }
  });

  // Refresh the header LOD/partition summary from the same live snapshot.
  const summary = container.querySelector('[data-field="lod-summary"]') as HTMLElement | null;
  if (summary) {
    summary.textContent = summariseLodStates(lodStates, countAdditiveNodes(root));
  }
}
