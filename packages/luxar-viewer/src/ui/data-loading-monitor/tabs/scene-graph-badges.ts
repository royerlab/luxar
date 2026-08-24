import type { LODProgressState, NodeDrawOrder } from '../../../types/data-monitor-types';
import { SceneGraphModel } from '../scene-graph-model';
import {
  activeLevelRole,
  countAdditiveNodes,
  drawOrderChipContent,
  levelRoleTitleSuffix,
  lodChipContent,
  nodeStatsContent,
  summariseLodStates,
} from '../templates/scene-graph';

/** Patch live scene-graph badges and row state without rebuilding the tree DOM. */
export function updateSceneGraphBadges(
  container: HTMLElement,
  model: SceneGraphModel,
  lodStates: ReadonlyMap<string, LODProgressState>,
  drawOrderStates: ReadonlyMap<string, NodeDrawOrder>
): void {
  const root = model.getSceneGraph().root;
  if (!root) return;

  model.syncVisibleCountsIntoTree();

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

  const drawOrderChips = container.querySelectorAll(
    '.luxar-scene-graph__draworder[data-draworder-path]'
  );
  drawOrderChips.forEach((chip) => {
    const path = (chip as HTMLElement).dataset.draworderPath;
    if (!path) return;
    const content = drawOrderChipContent(drawOrderStates.get(path));
    chip.textContent = content?.text ?? '';
    (chip as HTMLElement).title = content?.title ?? '';
  });

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

  const summary = container.querySelector('[data-field="lod-summary"]') as HTMLElement | null;
  if (summary) {
    summary.textContent = summariseLodStates(lodStates, countAdditiveNodes(root));
  }
}
