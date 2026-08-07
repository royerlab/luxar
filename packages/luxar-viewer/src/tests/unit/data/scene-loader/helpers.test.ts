/**
 * Unit tests for the small scene-loader pure helpers extracted from
 * the loader facade: URL normalization and effective-attrs composition.
 *
 * (OnceInit has its own dedicated test at
 * `data/loaders/once-init.test.ts` — not duplicated here.)
 */

import { describe, it, expect } from 'vitest';
import { normalizeURL } from '../../../../data/scene-loader/lifecycle/url-normalization';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../../data/data-loader-types';

describe('normalizeURL', () => {
  it('keeps absolute http URLs as-is and adds a trailing slash', () => {
    expect(normalizeURL('http://example.com/data.zarr', 'http://localhost')).toBe(
      'http://example.com/data.zarr/'
    );
    expect(normalizeURL('http://example.com/data.zarr/', 'http://localhost')).toBe(
      'http://example.com/data.zarr/'
    );
  });

  it('keeps absolute https URLs as-is and adds a trailing slash', () => {
    expect(normalizeURL('https://example.com/dataset', 'http://localhost')).toBe(
      'https://example.com/dataset/'
    );
  });

  it('prepends the window origin to a relative path', () => {
    expect(normalizeURL('data/scene.zarr', 'http://localhost:5173')).toBe(
      'http://localhost:5173/data/scene.zarr/'
    );
  });

  it('preserves the leading slash on a path that already has one', () => {
    expect(normalizeURL('/data/scene.zarr', 'http://localhost:5173')).toBe(
      'http://localhost:5173/data/scene.zarr/'
    );
  });

  it('does not duplicate the trailing slash on a relative path that already ends in /', () => {
    expect(normalizeURL('/data/scene.zarr/', 'http://localhost:5173')).toBe(
      'http://localhost:5173/data/scene.zarr/'
    );
  });
});

describe('applyEffectiveAttrs', () => {
  function makeNode(): SceneNode {
    return {
      path: 'group/leaf',
      type: 'points',
      attrs: {
        type: 'points',
        opacity: 0.5,
        gamma: 1.0,
        intensity: 2.0,
        offset: 0.1,
        blending_mode: 'additive',
        // sentinel to prove the spread preserves untouched fields
        n_points: 42,
      } as SceneNode['attrs'],
      children: [],
      hasSpatialIndex: false,
    };
  }

  it('returns the raw attrs unchanged when sceneGraph is null', () => {
    const node = makeNode();
    expect(applyEffectiveAttrs(null, node)).toBe(node.attrs);
  });

  it('returns the raw attrs unchanged when sceneGraph is undefined', () => {
    const node = makeNode();
    expect(applyEffectiveAttrs(undefined, node)).toBe(node.attrs);
  });

  it('replaces rendering attrs with effective values from the graph', () => {
    // Build a graph that contributes an opacity multiplier so the final
    // composed value differs from the leaf's own.
    const leaf = makeNode();
    const root: SceneNode = {
      path: '',
      type: 'group',
      attrs: { type: 'group', opacity: 0.5 } as SceneNode['attrs'],
      children: [
        {
          path: 'group',
          type: 'group',
          attrs: { type: 'group', opacity: 0.5 } as SceneNode['attrs'],
          children: [leaf],
          hasSpatialIndex: false,
        },
      ],
      hasSpatialIndex: false,
    };
    const result = applyEffectiveAttrs(root, leaf);
    // Untouched fields preserved
    expect(result.n_points).toBe(42);
    // Composed opacity: 0.5 (root) * 0.5 (intermediate) * 0.5 (leaf) = 0.125
    expect(result.opacity).toBeCloseTo(0.125, 5);
  });

  it("carries a wrapper's join down to a leaf that does not set one", () => {
    // The partition case: `add_lines(..., join="none", partition={...})`
    // writes `join` on the WRAPPER only (COMPOSITING_ATTRS), so if it does
    // not compose, every part silently renders with the default miter.
    const leaf: SceneNode = {
      path: 'tracks/part_0',
      type: 'lines',
      attrs: { type: 'lines' } as SceneNode['attrs'],
      children: [],
      hasSpatialIndex: false,
    };
    const root: SceneNode = {
      path: '',
      type: 'group',
      attrs: { type: 'group' } as SceneNode['attrs'],
      children: [
        {
          path: 'tracks',
          type: 'group',
          attrs: { type: 'group', kind: 'partition', join: 'none' } as SceneNode['attrs'],
          children: [leaf],
          hasSpatialIndex: false,
        },
      ],
      hasSpatialIndex: false,
    };
    expect(applyEffectiveAttrs(root, leaf).join).toBe('none');
  });
});
