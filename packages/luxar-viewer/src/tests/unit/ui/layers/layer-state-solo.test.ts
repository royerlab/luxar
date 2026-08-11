/**
 * Unit tests for LayerStateManager.solo() (toggle-restore) and
 * setVisibleMany() (single-notification batch visibility) — the state
 * primitives behind the layers panel's context-menu visibility verbs.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import type { SceneNode } from '../../../../data/data-loader-types';
import { LayerStateManager } from '../../../../ui/layers/layer-state';

function graph(): SceneNode {
  const leaf = (name: string): SceneNode =>
    ({
      name,
      path: `/${name}`,
      type: 'points',
      attrs: { layer: true },
      children: [],
    }) as unknown as SceneNode;
  return {
    name: 'scene',
    path: '/',
    type: 'scene',
    attrs: {},
    children: [leaf('a'), leaf('b'), leaf('c')],
  } as unknown as SceneNode;
}

describe('LayerStateManager solo + setVisibleMany', () => {
  let state: LayerStateManager;
  // Plain counter instead of a vi.fn(): this repo's vitest typing rejects
  // calling a bare Mock as LayerChangeListener.
  let notifyCount = 0;

  beforeEach(() => {
    state = new LayerStateManager();
    state.initFromSceneGraph(graph());
    notifyCount = 0;
    state.onChange(() => {
      notifyCount++;
    });
  });

  const vis = () => state.getLayers().map((l) => `${l.name}:${l.visible ? 1 : 0}`);

  it('setVisibleMany applies a batch with exactly ONE notification', () => {
    state.setVisibleMany([
      { path: '/a', visible: false },
      { path: '/b', visible: false },
      { path: '/c', visible: false },
    ]);
    expect(vis()).toEqual(['a:0', 'b:0', 'c:0']);
    expect(notifyCount).toBe(1);
  });

  it('setVisibleMany with no effective change does not notify', () => {
    state.setVisibleMany([{ path: '/a', visible: true }]);
    expect(notifyCount).toBe(0);
  });

  it('solo hides all others; soloing the same path restores the capture', () => {
    state.setVisible('/b', false); // pre-existing hand-tuned state
    notifyCount = 0;

    state.solo('/a');
    expect(vis()).toEqual(['a:1', 'b:0', 'c:0']);
    expect(state.soloedPath).toBe('/a');
    expect(notifyCount).toBe(1);

    state.solo('/a'); // toggle off — restores the TRUE pre-solo set
    expect(vis()).toEqual(['a:1', 'b:0', 'c:1']);
    expect(state.soloedPath).toBeNull();
    expect(notifyCount).toBe(2);
  });

  it('re-targeting solo keeps the ORIGINAL capture for the eventual un-solo', () => {
    state.setVisible('/c', false);
    state.solo('/a');
    state.solo('/b'); // re-target without un-soloing
    expect(vis()).toEqual(['a:0', 'b:1', 'c:0']);
    expect(state.soloedPath).toBe('/b');

    state.solo('/b'); // un-solo restores the pre-/a capture, not the /a state
    expect(vis()).toEqual(['a:1', 'b:1', 'c:0']);
    expect(state.soloedPath).toBeNull();
  });

  it('a manual visibility change clears the solo capture', () => {
    state.solo('/a');
    state.setVisible('/b', true);
    expect(state.soloedPath).toBeNull();
    // Soloing /a again is a FRESH capture (of the current, not stale, state).
    state.solo('/a');
    state.solo('/a');
    expect(vis()).toEqual(['a:1', 'b:1', 'c:0']);
  });

  it('a scene reload clears the solo capture', () => {
    state.solo('/a');
    state.initFromSceneGraph(graph());
    expect(state.soloedPath).toBeNull();
  });
});
