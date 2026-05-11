import { describe, it, expect } from 'vitest';
import {
  composeAttrs,
  collectAncestorAttrs,
  getEffectiveAttrs,
} from '../../../data/utils/attrs-composer';
import type { SceneNode } from '../../../data/data-loader-types';

describe('composeAttrs', () => {
  it('returns identity for empty chain', () => {
    const e = composeAttrs([]);
    expect(e.opacity).toBe(1);
    expect(e.gamma).toBe(1);
    expect(e.intensity).toBe(1);
    expect(e.offset).toBe(0);
    expect(e.blending_mode).toBe('additive');
  });

  it('multiplies opacity/gamma/intensity through the chain', () => {
    const e = composeAttrs([
      { opacity: 0.5, intensity: 2.0, gamma: 1.2 },
      { opacity: 0.5, intensity: 3.0, gamma: 1.5 },
    ]);
    expect(e.opacity).toBeCloseTo(0.25);
    expect(e.intensity).toBeCloseTo(6.0);
    expect(e.gamma).toBeCloseTo(1.8);
  });

  it('sums offsets through the chain', () => {
    const e = composeAttrs([{ offset: 0.1 }, { offset: -0.05 }, { offset: 0.2 }]);
    expect(e.offset).toBeCloseTo(0.25);
  });

  it('uses nearest-set blending_mode (later overrides earlier)', () => {
    expect(composeAttrs([{ blending_mode: 'normal' }]).blending_mode).toBe('normal');
    expect(
      composeAttrs([{ blending_mode: 'normal' }, { blending_mode: 'max' }]).blending_mode
    ).toBe('max');
    // Unset child keeps parent's choice
    expect(composeAttrs([{ blending_mode: 'normal' }, { opacity: 0.5 }]).blending_mode).toBe(
      'normal'
    );
  });

  it('clamps opacity to [0, 1]', () => {
    expect(composeAttrs([{ opacity: 2.0 }]).opacity).toBe(1);
    expect(composeAttrs([{ opacity: -0.5 }]).opacity).toBe(0);
  });

  it('clamps gamma to [0.1, 10]', () => {
    expect(composeAttrs([{ gamma: 100 }]).gamma).toBe(10);
    expect(composeAttrs([{ gamma: 0.01 }]).gamma).toBe(0.1);
  });

  it('treats unset fields as identity (not 0)', () => {
    const e = composeAttrs([{ opacity: 0.5 }, {}]);
    expect(e.opacity).toBeCloseTo(0.5);
    expect(e.intensity).toBe(1);
    expect(e.offset).toBe(0);
  });
});

describe('collectAncestorAttrs / getEffectiveAttrs', () => {
  const root: SceneNode = {
    path: '',
    type: 'scene',
    hasSpatialIndex: false,
    attrs: {},
    children: [
      {
        path: 'grp',
        type: 'group',
        hasSpatialIndex: false,
        attrs: { opacity: 0.5, intensity: 2.0 },
        children: [
          {
            path: 'grp/pts',
            type: 'points',
            hasSpatialIndex: true,
            attrs: { opacity: 0.5, offset: 0.1 },
          },
        ],
      },
    ],
  };

  it('walks from root to the target and collects each node attrs', () => {
    const chain = collectAncestorAttrs(root, 'grp/pts');
    expect(chain.length).toBe(2);
    expect(chain[0].opacity).toBe(0.5);
    expect(chain[1].offset).toBe(0.1);
  });

  it('composes an effective opacity of 0.25 for a 0.5×0.5 chain', () => {
    const e = getEffectiveAttrs(root, 'grp/pts');
    expect(e.opacity).toBeCloseTo(0.25);
    expect(e.intensity).toBeCloseTo(2.0);
    expect(e.offset).toBeCloseTo(0.1);
  });

  it('returns identity for a path that does not exist', () => {
    const e = getEffectiveAttrs(root, 'missing/node');
    expect(e.opacity).toBe(1);
  });

  // Regression: the real SceneLoader uses leading-slash paths ('/', '/grp',
  // '/grp/pts'). A naive path-segment join would produce '//grp' and miss
  // every child. Verify the leading-slash format walks correctly.
  describe('with scene-loader leading-slash paths', () => {
    const slashRoot: SceneNode = {
      path: '/',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: '/RedCloud',
          type: 'points',
          hasSpatialIndex: true,
          attrs: { layer: true, opacity: 0.8, gamma: 2.0 },
        },
        {
          path: '/CompositeLayer',
          type: 'group',
          hasSpatialIndex: false,
          attrs: { layer: true, opacity: 0.5 },
          children: [
            {
              path: '/CompositeLayer/GreenPart',
              type: 'points',
              hasSpatialIndex: true,
              attrs: { opacity: 0.5 },
            },
          ],
        },
      ],
    };

    it('walks to a direct child of the slash-root', () => {
      const chain = collectAncestorAttrs(slashRoot, '/RedCloud');
      expect(chain.length).toBe(1);
      expect(chain[0].opacity).toBe(0.8);
      expect(chain[0].gamma).toBe(2.0);
    });

    it('walks to a nested child (group → data leaf)', () => {
      const chain = collectAncestorAttrs(slashRoot, '/CompositeLayer/GreenPart');
      expect(chain.length).toBe(2);
      expect(chain[0].opacity).toBe(0.5);
      expect(chain[1].opacity).toBe(0.5);
    });

    it('composes effective opacity of 0.25 for slash-formatted paths', () => {
      const e = getEffectiveAttrs(slashRoot, '/CompositeLayer/GreenPart');
      expect(e.opacity).toBeCloseTo(0.25);
    });

    it('returns identity for an unknown slash path', () => {
      const e = getEffectiveAttrs(slashRoot, '/does/not/exist');
      expect(e.opacity).toBe(1);
    });
  });
});
