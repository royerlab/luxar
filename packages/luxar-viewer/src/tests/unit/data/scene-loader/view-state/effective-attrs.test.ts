/**
 * Unit tests for the scene-loader effective-attrs helper.
 *
 * Validates the fall-through-when-missing-graph behavior and the
 * compose-from-ancestry behavior; the underlying composition rules
 * are tested elsewhere (see attrs-composer.test.ts).
 */

import { describe, it, expect } from 'vitest';
import { applyEffectiveAttrs } from '../../../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../../../data/data-loader-types';

function makeNode(path: string, attrs: SceneNode['attrs'] = {}): SceneNode {
  return { path, type: 'points', attrs, hasSpatialIndex: false };
}

describe('applyEffectiveAttrs', () => {
  it('returns the raw attrs when scene graph is null', () => {
    const node = makeNode('/cloud', { opacity: 0.5, n_points: 10 });
    expect(applyEffectiveAttrs(null, node)).toBe(node.attrs);
  });

  it('returns the raw attrs when scene graph is undefined', () => {
    const node = makeNode('/cloud', { opacity: 0.5 });
    expect(applyEffectiveAttrs(undefined, node)).toBe(node.attrs);
  });

  it('overrides rendering attrs with composed values from the graph', () => {
    // Single-node graph: composition is identity except defaults fill in.
    const root: SceneNode = {
      path: '/',
      type: 'group',
      attrs: { opacity: 0.5, gamma: 2 },
      children: [
        {
          path: '/cloud',
          type: 'points',
          attrs: { opacity: 1, intensity: 3 },
        },
      ],
    } as unknown as SceneNode;

    const child = makeNode('/cloud', { opacity: 1, intensity: 3, n_points: 99 });
    const result = applyEffectiveAttrs(root, child);

    // Effective opacity composes (multiplies): 0.5 × 1 = 0.5.
    expect(result.opacity).toBe(0.5);
    // Effective gamma is the parent's value (child unset).
    expect(result.gamma).toBe(2);
    // Non-rendering attrs pass through unchanged.
    expect(result.n_points).toBe(99);
  });

  it('returns a *new* object — does not mutate the original attrs', () => {
    const root: SceneNode = {
      path: '/',
      type: 'group',
      attrs: { opacity: 0.5 },
      children: [{ path: '/cloud', type: 'points', attrs: { opacity: 1 } }],
    } as unknown as SceneNode;
    const node = makeNode('/cloud', { opacity: 1, n_points: 7 });
    const result = applyEffectiveAttrs(root, node);
    expect(result).not.toBe(node.attrs);
    expect(node.attrs.opacity).toBe(1); // original untouched
  });
});
