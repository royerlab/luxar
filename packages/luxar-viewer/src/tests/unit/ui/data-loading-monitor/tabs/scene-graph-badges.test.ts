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
      <span class="luxar-scene-graph__density" data-density-path="/points">stale</span>
      <span data-field="lod-summary"></span>
    `;

    model.syncVisibleCountsIntoTree();
    updateSceneGraphBadges(container, model, new Map(), new Map());

    const badge = container.querySelector('[data-node-path="/points"]') as HTMLElement;
    expect(badge.textContent).toBe('100');
    expect(badge.title).toContain('25 visible after slicing');
    expect(container.querySelector('[data-lod-path="/points"]')!.textContent).toBe('');
    expect(container.querySelector('[data-draworder-path="/points"]')!.textContent).toBe('');
    const density = container.querySelector('[data-density-path="/points"]') as HTMLElement;
    expect(density.textContent).toBe('');
    expect(container.querySelector('[data-field="lod-summary"]')!.textContent).toBe('');

    // Thinning appears and disappears with the camera without a rebuild.
    updateSceneGraphBadges(
      container,
      model,
      new Map(),
      new Map(),
      new Map([
        ['/points', { keep: 0.125, elementsPerPixel: 27.4, blendable: true, onScreen: true }],
      ])
    );
    expect(density.textContent).toBe('1/8'); // the glyph is an SVG, no text
    expect(density.querySelector('svg.luxar-micon')).not.toBeNull();
    expect(density.title).toContain('Drawn 1/8 of the resident elements');
    expect(density.title).toContain('27 resident elements per pixel');
    updateSceneGraphBadges(
      container,
      model,
      new Map(),
      new Map(),
      new Map([['/points', { keep: 1, elementsPerPixel: 2, blendable: true, onScreen: true }]])
    );
    expect(density.innerHTML).toBe('');
    expect(density.title).toBe('');

    // The draw-order chip patches the same way: glyph + order text, bucket in the tooltip.
    updateSceneGraphBadges(
      container,
      model,
      new Map(),
      new Map([['/points', { bucket: 'opaque', depthWrite: true, renderOrder: 2 }]])
    );
    const drawOrder = container.querySelector('[data-draworder-path="/points"]') as HTMLElement;
    expect(drawOrder.textContent).toBe('#2');
    expect(drawOrder.querySelector('svg.luxar-micon')).not.toBeNull();
    expect(drawOrder.title).toContain("'opaque' render bucket");

    model.updateVisibleCountsByPath(new Map());
    model.syncVisibleCountsIntoTree();
    updateSceneGraphBadges(container, model, new Map(), new Map());
    expect(badge.title).not.toContain('visible');
  });
});
