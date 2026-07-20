import { describe, it, expect, beforeEach } from 'vitest';
import {
  LayerStateManager,
  computeUniforms,
  computeDisplayRange,
  isLayerEnabled,
} from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';

// ─── computeUniforms / computeDisplayRange ──────────────

describe('computeUniforms', () => {
  it('returns identity for [0, 1]', () => {
    const { intensity, offset } = computeUniforms(0, 1);
    expect(intensity).toBeCloseTo(1.0, 5);
    expect(offset).toBeCloseTo(0.0, 5);
  });

  it('maps [0.2, 0.8] correctly', () => {
    const { intensity, offset } = computeUniforms(0.2, 0.8);
    // intensity = 1/(0.8 - 0.2) = 1/0.6 ≈ 1.667
    expect(intensity).toBeCloseTo(1.0 / 0.6, 5);
    // offset = -0.2 / 0.6 ≈ -0.333
    expect(offset).toBeCloseTo(-0.2 / 0.6, 5);
  });

  it('handles degenerate range (min ≈ max)', () => {
    const { intensity, offset } = computeUniforms(0.5, 0.5);
    // Should clamp to large value
    expect(intensity).toBe(1000);
    expect(offset).toBe(-0.5 * 1000);
  });

  it('handles HDR range [0, 5]', () => {
    const { intensity, offset } = computeUniforms(0, 5);
    expect(intensity).toBeCloseTo(0.2, 5);
    expect(offset).toBeCloseTo(0.0, 5);
  });
});

describe('computeDisplayRange', () => {
  it('recovers [0, 1] from identity uniforms', () => {
    const { min, max } = computeDisplayRange(1.0, 0.0);
    expect(min).toBeCloseTo(0, 5);
    expect(max).toBeCloseTo(1, 5);
  });

  it('handles zero intensity gracefully', () => {
    const { min, max } = computeDisplayRange(0, 0);
    expect(min).toBe(0);
    expect(max).toBe(1);
  });
});

describe('computeUniforms ↔ computeDisplayRange round-trip', () => {
  const cases = [
    [0, 1],
    [0.1, 0.9],
    [0, 5],
    [-0.5, 0.5],
    [100, 200],
  ] as const;

  for (const [origMin, origMax] of cases) {
    it(`round-trips [${origMin}, ${origMax}]`, () => {
      const { intensity, offset } = computeUniforms(origMin, origMax);
      const { min, max } = computeDisplayRange(intensity, offset);
      expect(min).toBeCloseTo(origMin, 5);
      expect(max).toBeCloseTo(origMax, 5);
    });
  }
});

// ─── LayerStateManager ──────────────────────────────────

function makeSceneGraph(layers: Partial<SceneNode['attrs']>[]): SceneNode {
  return {
    path: '',
    type: 'scene',
    attrs: {},
    hasSpatialIndex: false,
    children: layers.map((attrs, i) => ({
      path: `layer_${i}`,
      type: 'points' as const,
      attrs: { layer: true, ...attrs },
      hasSpatialIndex: true,
    })),
  };
}

describe('isLayerEnabled', () => {
  it('treats strict boolean true as enabled', () => {
    expect(isLayerEnabled(true)).toBe(true);
    expect(isLayerEnabled(false)).toBe(false);
  });

  it('coerces truthy finite numbers (hand-edited zarr) to enabled', () => {
    expect(isLayerEnabled(1)).toBe(true);
    expect(isLayerEnabled(0)).toBe(false);
    expect(isLayerEnabled(NaN)).toBe(false);
  });

  it('treats undefined / strings / objects as not-a-layer', () => {
    expect(isLayerEnabled(undefined)).toBe(false);
    expect(isLayerEnabled('true')).toBe(false);
    expect(isLayerEnabled({})).toBe(false);
  });
});

describe('LayerStateManager', () => {
  let mgr: LayerStateManager;

  beforeEach(() => {
    mgr = new LayerStateManager();
  });

  it('coerces a numeric `layer` attr instead of silently dropping the node', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        // `layer: 1` (e.g. hand-edited zarr) must still surface as a layer.
        { path: 'a', type: 'points', attrs: { layer: 1 } as never, hasSpatialIndex: true },
        { path: 'b', type: 'points', attrs: { layer: 0 } as never, hasSpatialIndex: true },
      ],
    };
    mgr.initFromSceneGraph(graph);
    expect(mgr.count).toBe(1);
    expect(mgr.getLayers().map((l) => l.name)).toEqual(['a']);
  });

  it('collects nodes with layer=true (including groups as composite layers)', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        { path: 'a', type: 'points', attrs: { layer: true }, hasSpatialIndex: true },
        { path: 'b', type: 'points', attrs: {}, hasSpatialIndex: true }, // no layer flag
        { path: 'c', type: 'lines', attrs: { layer: true }, hasSpatialIndex: false },
        { path: 'd', type: 'group', attrs: { layer: true }, hasSpatialIndex: false }, // group — composite layer
      ],
    };
    mgr.initFromSceneGraph(graph);
    expect(mgr.count).toBe(3); // a (points) + c (lines) + d (group)
    const names = mgr.getLayers().map((l) => l.name);
    expect(names).toEqual(['a', 'c', 'd']);
    expect(mgr.getLayer('d')!.type).toBe('group');
  });

  it('honors the `visible` attr for initial visibility', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        { path: 'a', type: 'points', attrs: { layer: true }, hasSpatialIndex: true },
        {
          path: 'b',
          type: 'points',
          attrs: { layer: true, visible: false },
          hasSpatialIndex: true,
        },
      ],
    };
    mgr.initFromSceneGraph(graph);
    expect(mgr.getLayer('a')!.visible).toBe(true);
    expect(mgr.getLayer('b')!.visible).toBe(false);
  });

  it('initializes display range from color_data_range', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ color_data_range: [0.1, 0.9] as [number, number] }]));
    const layer = mgr.getLayers()[0];
    expect(layer.dataMin).toBeCloseTo(0.1, 5);
    expect(layer.dataMax).toBeCloseTo(0.9, 5);
    expect(layer.displayMin).toBeCloseTo(0.1, 5);
    expect(layer.displayMax).toBeCloseTo(0.9, 5);
  });

  it('defaults data range to [0, 1] when absent', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    const layer = mgr.getLayers()[0];
    expect(layer.dataMin).toBe(0);
    expect(layer.dataMax).toBe(1);
  });

  describe('blendingMode init composes along the ancestry', () => {
    it('a layer leaf without its own mode inherits the nearest ancestor mode', () => {
      // The material renders with the COMPOSED mode, so the panel must
      // initialize from it — reading only the node's own attr showed
      // 'additive' for a leaf inside a blending_mode:'max' group.
      const graph: SceneNode = {
        path: '',
        type: 'scene',
        attrs: {},
        hasSpatialIndex: false,
        children: [
          {
            path: 'grp',
            type: 'group',
            attrs: { blending_mode: 'max' },
            hasSpatialIndex: false,
            children: [
              {
                path: 'grp/pts',
                type: 'points',
                attrs: { layer: true },
                hasSpatialIndex: true,
              },
            ],
          },
        ],
      };
      mgr.initFromSceneGraph(graph);
      expect(mgr.getLayer('grp/pts')!.blendingMode).toBe('max');
    });

    it("a malformed blending_mode attr initializes as 'normal' (composed + normalized)", () => {
      mgr.initFromSceneGraph(makeSceneGraph([{ blending_mode: 'bogus' }]));
      expect(mgr.getLayers()[0].blendingMode).toBe('normal');
    });

    it("defaults to 'additive' when no mode is set anywhere in the chain", () => {
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      expect(mgr.getLayers()[0].blendingMode).toBe('additive');
    });
  });

  describe('absorption init is RAW (like opacity), not composed', () => {
    it('a leaf inside a κ-scaled group initializes from its OWN attr only', () => {
      // Multiplicative attrs must init raw: composeEffective substitutes
      // each layer's live values per ancestry node, so a composed init
      // (group 2.0 × leaf 0.5 = 1.0) would multiply the ancestor κ in
      // TWICE at apply time. Contrast blendingMode above, which IS
      // composed (nearest-setter-wins has no double-count).
      const graph: SceneNode = {
        path: '',
        type: 'scene',
        attrs: {},
        hasSpatialIndex: false,
        children: [
          {
            path: 'grp',
            type: 'group',
            attrs: { absorption: 2.0 },
            hasSpatialIndex: false,
            children: [
              {
                path: 'grp/pts',
                type: 'gsplats',
                attrs: { layer: true, absorption: 0.5 },
                hasSpatialIndex: true,
              },
            ],
          },
        ],
      };
      mgr.initFromSceneGraph(graph);
      expect(mgr.getLayer('grp/pts')!.absorption).toBe(0.5); // raw, NOT 1.0
    });

    it('defaults to the identity 1.0 when unset', () => {
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      expect(mgr.getLayers()[0].absorption).toBe(1.0);
    });

    it('setAbsorption mutates and notifies', () => {
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      const path = mgr.getLayers()[0].path;
      mgr.setAbsorption(path, 4.5);
      expect(mgr.getLayer(path)!.absorption).toBe(4.5);
    });
  });

  it('derives a composite group layer range from its finest descendant leaf', () => {
    // Regression: a kind=partition/lod group carries no range of its own; the
    // [0, 1] fallback makes a colormapped gsplat render near-black. Derive from
    // the finest (largest-n_splats) descendant so the layer window is correct.
    const graph: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: {},
      children: [
        {
          path: '/g',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          children: [
            {
              path: '/g/coarse',
              type: 'gsplats',
              attrs: { amplitude_data_range: [0, 0.25] as [number, number], n_splats: 100 },
              children: [],
            },
            {
              path: '/g/fine',
              type: 'gsplats',
              attrs: { amplitude_data_range: [0, 0.03] as [number, number], n_splats: 5000 },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;
    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('/g')!;
    // Finest leaf (n_splats=5000) wins → [0, 0.03], NOT the [0, 1] fallback.
    expect(layer.dataMax).toBeCloseTo(0.03, 5);
  });

  it('expands slider bounds to encompass authored intensity/offset display range', () => {
    // Regression: when a node has authored intensity != 1, the recovered
    // display range can extend beyond color_data_range. If the slider's
    // <input type="range"> bounds stay narrow, the browser clamps the thumb
    // values and the first interaction snaps the layer state from the
    // authored brightness to the slider-implied one — a sudden visible jump.
    mgr.initFromSceneGraph(
      makeSceneGraph([
        {
          color_data_range: [0.157, 0.973] as [number, number],
          intensity: 0.2,
          offset: 0.0,
        },
      ])
    );
    const layer = mgr.getLayers()[0];
    // displayMax = (1 - 0) / 0.2 = 5 — beyond color_data_range's 0.973
    expect(layer.displayMin).toBeCloseTo(0, 5);
    expect(layer.displayMax).toBeCloseTo(5, 5);
    // Slider bounds must include the whole display range so the
    // <input type="range"> doesn't silently clamp the thumb on first render.
    expect(layer.dataMin).toBeLessThanOrEqual(layer.displayMin);
    expect(layer.dataMax).toBeGreaterThanOrEqual(layer.displayMax);
  });

  // ─── Selection ─────────────────────────────────────

  it('select single clears others', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}, {}, {}]));
    mgr.select('layer_0', 'single');
    mgr.select('layer_1', 'single');
    expect(mgr.getSelected().length).toBe(1);
    expect(mgr.getSelected()[0].path).toBe('layer_1');
  });

  it('select add toggles', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}, {}]));
    mgr.select('layer_0', 'single');
    mgr.select('layer_1', 'add');
    expect(mgr.getSelected().length).toBe(2);
    // Toggle off
    mgr.select('layer_1', 'add');
    expect(mgr.getSelected().length).toBe(1);
  });

  it('select range selects contiguous block', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}, {}, {}, {}]));
    mgr.select('layer_0', 'single'); // anchor
    mgr.select('layer_2', 'range'); // range 0..2
    expect(mgr.getSelected().length).toBe(3);
    expect(mgr.getSelected().map((l) => l.path)).toEqual(['layer_0', 'layer_1', 'layer_2']);
  });

  // ─── Mutations ─────────────────────────────────────

  it('setVisible changes layer visibility', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    expect(mgr.getLayers()[0].visible).toBe(true);
    mgr.setVisible('layer_0', false);
    expect(mgr.getLayers()[0].visible).toBe(false);
  });

  it('setDisplayRange updates min/max', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    mgr.setDisplayRange('layer_0', 0.2, 0.8);
    const layer = mgr.getLayers()[0];
    expect(layer.displayMin).toBeCloseTo(0.2, 5);
    expect(layer.displayMax).toBeCloseTo(0.8, 5);
  });

  it('setBlendingMode updates the layer and fires the change listener', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    expect(mgr.getLayer('layer_0')!.blendingMode).toBe('additive'); // composed default
    let called = 0;
    mgr.onChange(() => called++);
    mgr.setBlendingMode('layer_0', 'max');
    expect(mgr.getLayer('layer_0')!.blendingMode).toBe('max');
    expect(called).toBe(1);
  });

  it('applyToSelected mutates all selected layers', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}, {}]));
    mgr.select('layer_0', 'single');
    mgr.select('layer_1', 'add');
    mgr.applyToSelected((l) => {
      l.gamma = 2.0;
    });
    expect(mgr.getLayers()[0].gamma).toBe(2.0);
    expect(mgr.getLayers()[1].gamma).toBe(2.0);
  });

  // ─── Events ────────────────────────────────────────

  it('onChange fires on state mutations', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    let called = 0;
    mgr.onChange(() => called++);
    mgr.setVisible('layer_0', false);
    expect(called).toBe(1);
    mgr.setGamma('layer_0', 2.0);
    expect(called).toBe(2);
  });

  it('unsubscribe stops notifications', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    let called = 0;
    const unsub = mgr.onChange(() => called++);
    mgr.setVisible('layer_0', false);
    expect(called).toBe(1);
    unsub();
    mgr.setVisible('layer_0', true);
    expect(called).toBe(1);
  });

  it('dispose clears everything', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}, {}]));
    mgr.dispose();
    expect(mgr.count).toBe(0);
    expect(mgr.getLayers()).toEqual([]);
  });

  // ─── Specialized groups (kind=partition, kind=lod) ─────────

  it('discovers nested lod_groups under a kind=partition layer', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/partition_root',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          hasSpatialIndex: false,
          children: [
            {
              path: '/partition_root/part_0',
              type: 'group',
              attrs: { kind: 'lod', display_type: 'gsplats' },
              hasSpatialIndex: false,
              children: [
                {
                  path: '/partition_root/part_0/level_0',
                  type: 'gsplats',
                  attrs: {},
                  hasSpatialIndex: true,
                },
                {
                  path: '/partition_root/part_0/level_1',
                  type: 'gsplats',
                  attrs: {},
                  hasSpatialIndex: true,
                },
                {
                  path: '/partition_root/part_0/level_2',
                  type: 'gsplats',
                  attrs: {},
                  hasSpatialIndex: true,
                },
              ],
            },
            {
              path: '/partition_root/part_1',
              type: 'group',
              attrs: { kind: 'lod', display_type: 'gsplats' },
              hasSpatialIndex: false,
              // Ragged ladder: 2 children vs 3 in part_0.
              children: [
                {
                  path: '/partition_root/part_1/level_0',
                  type: 'gsplats',
                  attrs: {},
                  hasSpatialIndex: true,
                },
                {
                  path: '/partition_root/part_1/level_1',
                  type: 'gsplats',
                  attrs: {},
                  hasSpatialIndex: true,
                },
              ],
            },
          ],
        },
      ],
    };
    mgr.initFromSceneGraph(graph);
    const wrapper = mgr.getLayer('/partition_root')!;
    expect(wrapper.kind).toBe('partition');
    expect(wrapper.partCount).toBe(2);
    expect(wrapper.nestedLodGroupPaths).toEqual([
      '/partition_root/part_0',
      '/partition_root/part_1',
    ]);
    // Combined-badge sizing: largest ladder wins.
    expect(wrapper.nestedLodMaxChildCount).toBe(3);
  });

  it('leaves nestedLodGroupPaths undefined for kind=partition with no nested lod', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/partition_root',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          hasSpatialIndex: false,
          children: [
            {
              path: '/partition_root/part_0',
              type: 'gsplats',
              attrs: {},
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };
    mgr.initFromSceneGraph(graph);
    const wrapper = mgr.getLayer('/partition_root')!;
    expect(wrapper.nestedLodGroupPaths).toBeUndefined();
    expect(wrapper.nestedLodMaxChildCount).toBeUndefined();
  });

  it('does not descend into an lod_group looking for nested LODs', () => {
    // A kind=lod's own children are LOD levels, not further wrappers.
    // The walker must stop at the first lod_group per branch.
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/partition_root',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          hasSpatialIndex: false,
          children: [
            {
              path: '/partition_root/inner_lod',
              type: 'group',
              attrs: { kind: 'lod', display_type: 'gsplats' },
              hasSpatialIndex: false,
              children: [
                // Suppose someone constructed a synthetic graph with an
                // lod_group nested inside another lod_group's children:
                // we must NOT collect it.
                {
                  path: '/partition_root/inner_lod/inner_lod',
                  type: 'group',
                  attrs: { kind: 'lod' },
                  hasSpatialIndex: false,
                  children: [],
                },
              ],
            },
          ],
        },
      ],
    };
    mgr.initFromSceneGraph(graph);
    const wrapper = mgr.getLayer('/partition_root')!;
    expect(wrapper.nestedLodGroupPaths).toEqual(['/partition_root/inner_lod']);
  });
});
