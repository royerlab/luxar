// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';

import { SceneGraphModel } from '../../../../../ui/data-loading-monitor/scene-graph-model';
import { updateSceneGraphBadges } from '../../../../../ui/data-loading-monitor/tabs/scene-graph-badges';

describe('updateSceneGraphBadges', () => {
  it('patches visible counts and clears vanished live chip state without rebuilding', () => {
    const model = new SceneGraphModel(vi.fn());
    model.setSceneGraph({
      path: '/',
      name: 'Scene',
      type: 'scene',
      children: [
        {
          path: '/points',
          name: 'points',
          type: 'points',
          pointCount: 100,
          children: [],
        },
      ],
    });
    model.updateVisibleCountsByPath(new Map([['/points', 25]]));

    const container = document.createElement('div');
    container.innerHTML = `
      <span class="luxar-scene-graph__badge" data-node-path="/points"></span>
      <span class="luxar-scene-graph__lod" data-lod-path="/points">stale</span>
      <span class="luxar-scene-graph__draworder" data-draworder-path="/points">stale</span>
      <span data-field="lod-summary"></span>
    `;

    model.syncVisibleCountsIntoTree();
    updateSceneGraphBadges(container, model, new Map(), new Map());

    const badge = container.querySelector('[data-node-path="/points"]') as HTMLElement;
    expect(badge.textContent).toBe('100');
    expect(badge.title).toContain('25 visible after slicing');
    expect(container.querySelector('[data-lod-path="/points"]')!.textContent).toBe('');
    expect(container.querySelector('[data-draworder-path="/points"]')!.textContent).toBe('');
    expect(container.querySelector('[data-field="lod-summary"]')!.textContent).toBe('');

    model.updateVisibleCountsByPath(new Map());
    model.syncVisibleCountsIntoTree();
    updateSceneGraphBadges(container, model, new Map(), new Map());
    expect(badge.title).not.toContain('visible');
  });
});
