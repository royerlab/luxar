import { describe, it, expect, beforeEach } from 'vitest';
import {
  LayerStateManager,
  computeUniforms,
  computeDisplayRange,
  isLayerEnabled,
  resolveLayerBlendingMode,
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

  it('handles degenerate range (min == max) as identity pass-through', () => {
    // A constant data range has nothing to window, so gain/offset must be the
    // identity (color unchanged). The old "high contrast" mapping (gain 1000,
    // offset -1000·min) turned `color·1000 − 1000` into 0 for every non-white
    // color — rendering classical-splat imports (constant amplitude_data_range
    // [1, 1] from per-element-opacity-in-alpha) entirely black.
    expect(computeUniforms(0.5, 0.5)).toEqual({ intensity: 1.0, offset: 0.0 });
    expect(computeUniforms(1.0, 1.0)).toEqual({ intensity: 1.0, offset: 0.0 });
    expect(computeUniforms(0.0, 0.0)).toEqual({ intensity: 1.0, offset: 0.0 });
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

  it('reports an ancestor palette on a nested leaf layer', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'palette',
          type: 'group',
          attrs: { colormap: 'plasma' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'palette/gs',
              type: 'gsplats',
              attrs: { layer: true, amplitude_data_range: [2, 8] },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('palette/gs')!;
    expect(layer.colormap).toBe('plasma');
    expect(layer.scalarWindow).toBe(true);
    expect(layer.scalarDataRange).toEqual([2, 8]);
    expect(layer.displayMin).toBe(2);
    expect(layer.displayMax).toBe(8);
    expect(layer.dataMin).toBe(2);
    expect(layer.dataMax).toBe(8);
  });

  it('prefers a wrapper descendant palette over an inherited ancestor palette', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'outer',
          type: 'group',
          attrs: { colormap: 'plasma' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'outer/partition',
              type: 'group',
              attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
              hasSpatialIndex: false,
              children: [
                {
                  path: 'outer/partition/part_0',
                  type: 'gsplats',
                  attrs: { colormap: 'viridis', amplitude_data_range: [2, 8] },
                  hasSpatialIndex: true,
                },
              ],
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);
    expect(mgr.getLayer('outer/partition')!.colormap).toBe('viridis');
  });

  it('derives a shared label vocabulary for a partition wrapper layer', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'tiles',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'tiles/part_0',
              type: 'gsplats',
              attrs: {
                label_vocabulary: {
                  '9007199254740993': 'cell',
                  '9007199254740995': 'artifact',
                },
              },
              hasSpatialIndex: true,
            },
            {
              path: 'tiles/part_1',
              type: 'gsplats',
              attrs: {
                label_vocabulary: {
                  '9007199254740995': 'artifact',
                  '9007199254740993': 'cell',
                },
              },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);

    expect(mgr.getLayer('tiles')!.labelVocabulary).toEqual([
      { id: '9007199254740993', name: 'cell' },
      { id: '9007199254740995', name: 'artifact' },
    ]);
  });

  it('does not derive a wrapper label vocabulary when descendants disagree', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'tiles',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'tiles/part_0',
              type: 'gsplats',
              attrs: { label_vocabulary: { '7': 'cell' } },
              hasSpatialIndex: true,
            },
            {
              path: 'tiles/part_1',
              type: 'gsplats',
              attrs: { label_vocabulary: { '8': 'cell' } },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);

    expect(mgr.getLayer('tiles')!.labelVocabulary).toBeUndefined();
  });

  it('does not report an inherited palette on a scalarless points wrapper layer', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'outer',
          type: 'group',
          attrs: { colormap: 'plasma' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'outer/partition',
              type: 'group',
              attrs: { layer: true, kind: 'partition', display_type: 'points' },
              hasSpatialIndex: false,
              children: [
                {
                  path: 'outer/partition/part_0',
                  type: 'points',
                  attrs: { color_data_range: [0, 255] },
                  hasSpatialIndex: true,
                },
              ],
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('outer/partition')!;
    expect(layer.colormap).toBeUndefined();
    expect(layer.scalarWindow).toBe(false);
    expect(layer.supportsColormap).toBe(false);
    expect([layer.dataMin, layer.dataMax]).toEqual([0, 255]);
  });

  it('does not report an inherited palette on a scalarless points layer', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'palette',
          type: 'group',
          attrs: { colormap: 'plasma' },
          hasSpatialIndex: false,
          children: [
            {
              path: 'palette/points',
              type: 'points',
              attrs: { layer: true, color_data_range: [0, 255] },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };

    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('palette/points')!;
    expect(layer.colormap).toBeUndefined();
    expect(layer.scalarWindow).toBe(false);
    expect(layer.supportsColormap).toBe(false);
    expect([layer.dataMin, layer.dataMax]).toEqual([0, 255]);
  });

  it('keeps a scalarless points layer own authored palette', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ colormap: 'magma' }]));
    const layer = mgr.getLayer('layer_0')!;
    expect(layer.colormap).toBe('magma');
    expect(layer.scalarWindow).toBe(true);
    expect(layer.supportsColormap).toBe(true);
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

  it('starts a direct-colour layer at the IDENTITY window, not color_data_range', () => {
    // The window maps the rendered value to [0, 1]. For a direct-colour layer
    // that value is authored RGB, whose range already IS [0, 1] — windowing it
    // on color_data_range is an unrequested contrast stretch.
    mgr.initFromSceneGraph(makeSceneGraph([{ color_data_range: [0.1, 0.9] as [number, number] }]));
    const layer = mgr.getLayers()[0];
    expect(layer.displayMin).toBeCloseTo(0, 5);
    expect(layer.displayMax).toBeCloseTo(1, 5);
    // …which is the identity gain/offset — the authored colour reaches the
    // shader untouched.
    const { intensity, offset } = computeUniforms(layer.displayMin, layer.displayMax);
    expect(intensity).toBeCloseTo(1, 5);
    expect(offset).toBeCloseTo(0, 5);
    // The colour range still bounds the slider so stretching stays one drag away.
    expect(layer.dataMin).toBeLessThanOrEqual(0.1);
    expect(layer.dataMax).toBeGreaterThanOrEqual(0.9);
  });

  it('does not blow up the gain on a near-uniform colour (grey → blue regression)', () => {
    // demo_gsplats_recipes_tribolium paints a column a flat grey
    // (0.72, 0.74, 0.78) → color_data_range [0.72, 0.78]. Windowing on it gave
    // gain 16.7 / offset −12, mapping that grey to (0.00, 0.33, 1.00): the
    // column rendered SATURATED BLUE and every other column clipped to white.
    mgr.initFromSceneGraph(
      makeSceneGraph([{ color_data_range: [0.72, 0.78] as [number, number] }])
    );
    const layer = mgr.getLayers()[0];
    const { intensity, offset } = computeUniforms(layer.displayMin, layer.displayMax);
    expect(intensity).toBeCloseTo(1, 5);
    expect(offset).toBeCloseTo(0, 5);
  });

  it('keeps HDR colours (range beyond 1) at the identity window', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ color_data_range: [0, 3.5] as [number, number] }]));
    const layer = mgr.getLayers()[0];
    // Windowing on [0, 3.5] would DIM authored HDR by 3.5× before tone mapping.
    expect(layer.displayMax).toBeCloseTo(1, 5);
    expect(layer.dataMax).toBeGreaterThanOrEqual(3.5);
  });

  it('uses an HDR texture range to bound the direct-colour slider', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ texture_data_range: [0, 6.5] as [number, number] }]));
    const layer = mgr.getLayers()[0];
    expect(layer.displayMin).toBe(0);
    expect(layer.displayMax).toBe(1);
    expect(layer.dataMax).toBeGreaterThanOrEqual(6.5);
  });

  it('still windows a COLORMAPPED layer on its scalar range', () => {
    // The #522 case: a linear [0, 1] window on right-skewed gsplat amplitudes
    // buries ~99% of splats in the bottom few % and renders near-black.
    mgr.initFromSceneGraph(
      makeSceneGraph([
        {
          colormap: 'gray',
          amplitude_data_range: [0.0001, 0.02] as [number, number],
          color_data_range: [0.72, 0.78] as [number, number],
        },
      ])
    );
    const layer = mgr.getLayers()[0];
    expect(layer.displayMin).toBeCloseTo(0.0001, 6);
    expect(layer.displayMax).toBeCloseTo(0.02, 6);
  });

  it('re-defaults the window when the colormap is toggled', () => {
    mgr.initFromSceneGraph(
      makeSceneGraph([
        {
          has_scalars: true,
          scalar_data_range: [0.0001, 0.02] as [number, number],
          color_data_range: [0.2, 0.6] as [number, number],
        },
      ])
    );
    const path = mgr.getLayers()[0].path;
    // Direct colour to start…
    expect(mgr.getLayer(path)!.displayMax).toBeCloseTo(1, 5);
    // …colormap ON windows the SCALAR (a [0, 1] window on it renders near-black)…
    mgr.setColormapWindow(path, true);
    expect(mgr.getLayer(path)!.displayMin).toBeCloseTo(0.0001, 6);
    expect(mgr.getLayer(path)!.displayMax).toBeCloseTo(0.02, 6);
    expect(mgr.getLayer(path)!.dataMin).toBeLessThanOrEqual(0.0001);
    // …and OFF restores the identity.
    mgr.setColormapWindow(path, false);
    expect(mgr.getLayer(path)!.displayMin).toBeCloseTo(0, 5);
    expect(mgr.getLayer(path)!.displayMax).toBeCloseTo(1, 5);
  });

  it('a colormap toggle moves the slider BOUNDS to the mode, not just widens them', () => {
    // Merely widening leaves the useful window as an unusable sliver: an
    // amplitude window of [1e-4, 0.02] inside [0, 1] bounds is 2% of the track.
    // Bounds must land where a natively-authored layer of that mode inits.
    mgr.initFromSceneGraph(
      makeSceneGraph([
        {
          has_scalars: true,
          scalar_data_range: [0.0001, 0.02] as [number, number],
          color_data_range: [0.2, 0.6] as [number, number],
        },
      ])
    );
    const path = mgr.getLayers()[0].path;
    mgr.setColormapWindow(path, true);
    const on = mgr.getLayer(path)!;
    expect(on.dataMin).toBeCloseTo(0.0001, 6);
    expect(on.dataMax).toBeCloseTo(0.02, 6);
    expect((on.displayMax - on.displayMin) / (on.dataMax - on.dataMin)).toBeGreaterThan(0.5);
  });

  it('HDR colour bounds survive a colormap ON→OFF round trip', () => {
    mgr.initFromSceneGraph(
      makeSceneGraph([
        {
          has_scalars: true,
          scalar_data_range: [0, 0.02] as [number, number],
          color_data_range: [0, 3.5] as [number, number],
        },
      ])
    );
    const path = mgr.getLayers()[0].path;
    expect(mgr.getLayer(path)!.dataMax).toBeCloseTo(3.5, 6);
    mgr.setColormapWindow(path, true);
    mgr.setColormapWindow(path, false);
    // Back to the direct-colour bounds — an HDR colour is still reachable.
    expect(mgr.getLayer(path)!.dataMax).toBeCloseTo(3.5, 6);
  });

  it("recognises colormap 'custom' and a colormap on a DESCENDANT, but not an empty string", () => {
    mgr.initFromSceneGraph(
      makeSceneGraph([{ colormap: 'custom', amplitude_data_range: [0, 0.03] as [number, number] }])
    );
    expect(mgr.getLayers()[0].displayMax).toBeCloseTo(0.03, 6);

    mgr.initFromSceneGraph(
      makeSceneGraph([{ colormap: '', amplitude_data_range: [0, 0.03] as [number, number] }])
    );
    expect(mgr.getLayers()[0].displayMax).toBe(1);
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

    it('leaves the mode NON-explicit when no ancestry level sets one (#1272)', () => {
      // The per-type default is only a placeholder — it must NOT be pushed onto
      // descendants via the live-attrs path, so the flag stays false.
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      const layer = mgr.getLayers()[0];
      expect(layer.blendingMode).toBe('additive');
      expect(layer.blendingModeExplicit).toBe(false);
    });

    it('an ancestor-INHERITED mode displays but is NOT owned (#1275)', () => {
      // The leaf's dropdown shows the composed 'max', but ownership reads the
      // node's OWN attr: re-emitting an inherited mode as the leaf's own setter
      // would freeze a snapshot that shadows the ancestor layer's next live
      // pick (the nearer setter wins) — see the LayerInfo.blendingModeExplicit doc.
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
      const leaf = mgr.getLayer('grp/pts')!;
      expect(leaf.blendingMode).toBe('max');
      expect(leaf.blendingModeExplicit).toBe(false);
    });

    it('marks the mode EXPLICIT after a user pick via setBlendingMode (#1272)', () => {
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      expect(mgr.getLayer('layer_0')!.blendingModeExplicit).toBe(false);
      mgr.setBlendingMode('layer_0', 'max');
      expect(mgr.getLayer('layer_0')!.blendingModeExplicit).toBe(true);
    });
  });

  describe('blendingModeExplicit tracks OWNERSHIP (not the displayed mode)', () => {
    /** scene → group (attrs) → mesh leaf. */
    function makeGroupOverMesh(groupAttrs: Record<string, unknown>): SceneNode {
      return {
        path: '',
        type: 'scene',
        attrs: {},
        hasSpatialIndex: false,
        children: [
          {
            path: 'grp',
            type: 'group',
            attrs: { layer: true, ...groupAttrs },
            hasSpatialIndex: false,
            children: [
              {
                path: 'grp/mesh',
                type: 'mesh',
                attrs: { layer: true },
                hasSpatialIndex: true,
              },
            ],
          },
        ],
      };
    }

    it('a geometry leaf owns a mode only when IT authored one', () => {
      // An unauthored leaf shows its per-type default but does not own it —
      // owning would emit the default as a setter and block an ancestor group
      // layer's Blend pick from ever reaching the leaf (nearest-setter-wins).
      mgr.initFromSceneGraph(makeSceneGraph([{}]));
      expect(mgr.getLayers()[0].blendingModeExplicit).toBe(false);
      mgr.initFromSceneGraph(makeSceneGraph([{ blending_mode: 'max' }]));
      expect(mgr.getLayers()[0].blendingModeExplicit).toBe(true);
    });

    it('a plain group that authored no blending_mode does NOT own one', () => {
      mgr.initFromSceneGraph(makeGroupOverMesh({}));
      const grp = mgr.getLayer('grp')!;
      expect(grp.type).toBe('group');
      expect(grp.blendingModeExplicit).toBe(false);
      // The contained (unauthored) mesh leaf doesn't own one either — its
      // `opaque` comes from the per-type fallback at apply time, not from the
      // panel injecting a setter.
      expect(mgr.getLayer('grp/mesh')!.blendingModeExplicit).toBe(false);
      expect(mgr.getLayer('grp/mesh')!.blendingMode).toBe('opaque');
    });

    it('a group that authored blending_mode on disk DOES own one', () => {
      mgr.initFromSceneGraph(makeGroupOverMesh({ blending_mode: 'max' }));
      expect(mgr.getLayer('grp')!.blendingModeExplicit).toBe(true);
    });

    it('setBlendingMode makes a plain group own its mode', () => {
      mgr.initFromSceneGraph(makeGroupOverMesh({}));
      expect(mgr.getLayer('grp')!.blendingModeExplicit).toBe(false);
      mgr.setBlendingMode('grp', 'max');
      expect(mgr.getLayer('grp')!.blendingModeExplicit).toBe(true);
      expect(mgr.getLayer('grp')!.blendingMode).toBe('max');
    });

    it('setBlendingMode stores the MESH-RESOLVED mode (volumetric → opaque)', () => {
      // Same point-of-storage resolution as the panel's dropdown handler: the
      // Blend dropdown displays the stored value raw, so a programmatic
      // 'volumetric' on a mesh must not make the panel claim a mode the mesh
      // shader does not implement.
      mgr.initFromSceneGraph(makeGroupOverMesh({}));
      mgr.setBlendingMode('grp/mesh', 'volumetric');
      expect(mgr.getLayer('grp/mesh')!.blendingMode).toBe('opaque');
      expect(mgr.getLayer('grp/mesh')!.blendingModeExplicit).toBe(true);
    });
  });

  describe('resolveLayerBlendingMode locks the mode the control gates depend on', () => {
    // The layer-controls absorption / alpha-cutoff visibility gates resolve the
    // mode through this before comparing. A mesh maps `volumetric` → `opaque`
    // (no absorption path; the cutout is active), while every other type keeps it.
    it('resolves volumetric → opaque for a mesh (never shows a dead absorption slider)', () => {
      expect(resolveLayerBlendingMode('mesh', 'volumetric')).toBe('opaque');
    });

    it('leaves volumetric alone for gsplats / points / lines', () => {
      expect(resolveLayerBlendingMode('gsplats', 'volumetric')).toBe('volumetric');
      expect(resolveLayerBlendingMode('points', 'volumetric')).toBe('volumetric');
      expect(resolveLayerBlendingMode('lines', 'volumetric')).toBe('volumetric');
    });

    it('passes non-volumetric modes through unchanged for a mesh', () => {
      expect(resolveLayerBlendingMode('mesh', 'opaque')).toBe('opaque');
      expect(resolveLayerBlendingMode('mesh', 'additive')).toBe('additive');
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

  /** A kind=partition group layer over two gsplats leaves, coarse + fine. */
  function makePartitionGraph(leafAttrs: Record<string, unknown>): SceneNode {
    return {
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
              attrs: {
                amplitude_data_range: [0, 0.25] as [number, number],
                n_splats: 100,
                ...leafAttrs,
              },
              children: [],
            },
            {
              path: '/g/fine',
              type: 'gsplats',
              attrs: {
                amplitude_data_range: [0, 0.03] as [number, number],
                n_splats: 5000,
                ...leafAttrs,
              },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;
  }

  it('derives a COLORMAPPED composite group layer range from its finest descendant leaf', () => {
    // Regression: a kind=partition/lod group carries no range of its own; the
    // [0, 1] fallback makes a colormapped gsplat render near-black. Derive from
    // the finest (largest-n_splats) descendant so the layer window is correct.
    // The colormap is what makes the windowed value a SCALAR — a colourful
    // partition instead keeps the identity (twin test below).
    mgr.initFromSceneGraph(makePartitionGraph({ colormap: 'gray' }));
    const layer = mgr.getLayer('/g')!;
    // Finest leaf (n_splats=5000) wins → [0, 0.03], NOT the [0, 1] fallback.
    expect(layer.dataMax).toBeCloseTo(0.03, 5);
    expect(layer.displayMax).toBeCloseTo(0.03, 5);
  });

  it('a composite kind=partition group without an authored mode does NOT own one', () => {
    // Ownership reads the NODE's own attrs, never the layer's display type: a
    // kind=partition/lod wrapper's LayerInfo.type is its display_type (a
    // geometry name, 'gsplats' here), but the wrapper authored no mode, so it
    // must not become a `blending_mode` setter merely because its display type
    // looks like a leaf.
    mgr.initFromSceneGraph(makePartitionGraph({}));
    const layer = mgr.getLayer('/g')!;
    expect(layer.type).toBe('gsplats'); // display_type
    expect(layer.blendingModeExplicit).toBe(false); // authored no mode
  });

  it('surfaces a descendant-authored palette on the wrapper layer', () => {
    // `colormap` is deliberately NOT a compositing attr: the Python writer
    // copies it onto every part, so a kind=partition wrapper from
    // `add_gsplats_from_file(..., colormap=...)` has none of its own while
    // every leaf renders through the LUT. The wrapper layer must report the
    // palette — otherwise the dropdown shows "(direct colors)", the legend
    // omits the layer, and the direct-colours option can never fire a change
    // event to switch the LUT off.
    mgr.initFromSceneGraph(makePartitionGraph({ colormap: 'gray' }));
    const layer = mgr.getLayer('/g')!;
    expect(layer.colormap).toBe('gray');
    expect(layer.scalarWindow).toBe(true);
  });

  it('leaves MIXED descendant palettes unrepresented on the wrapper', () => {
    // Two parts with different palettes: the single dropdown cannot show
    // both, so the wrapper keeps colormap undefined (the window still counts
    // as scalar via scalarWindow).
    const graph = makePartitionGraph({});
    const parts = graph.children![0].children!;
    parts[0].attrs.colormap = 'gray';
    parts[1].attrs.colormap = 'viridis';
    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('/g')!;
    expect(layer.colormap).toBeUndefined();
    expect(layer.scalarWindow).toBe(true);
  });

  it('does not derive a palette from a descendant that is a LAYER of its own', () => {
    // A nested layer owns its colormap control, so its palette can change at
    // any time. Snapshotting it onto the ancestor at init would go stale on
    // the first inner edit: the ancestor would keep claiming a scalar window
    // over a leaf that is back on direct colour, and its range control would
    // route to the identity for that leaf (silently inert).
    const graph = makePartitionGraph({});
    const inner = graph.children![0].children![0];
    inner.attrs.layer = true;
    inner.attrs.colormap = 'viridis';
    mgr.initFromSceneGraph(graph);
    const wrapper = mgr.getLayer('/g')!;
    expect(wrapper.colormap).toBeUndefined();
    expect(wrapper.scalarWindow).toBe(false);
    expect(wrapper.displayMax).toBeCloseTo(1, 5);
    // The nested layer still reports its own palette on its own row.
    expect(mgr.getLayer('/g/coarse')!.colormap).toBe('viridis');
    expect(mgr.getLayer('/g/coarse')!.scalarWindow).toBe(true);
  });

  it('keeps a DIRECT-COLOUR composite group layer at the identity window', () => {
    // The gallery demo's `tiles` / `adaptive` columns: parts carry per-splat
    // RGB and no colormap, so their amplitude range must NOT become the window
    // (that windowed authored colour by a scalar range and blew out the scene).
    mgr.initFromSceneGraph(makePartitionGraph({ color_data_range: [0.35, 0.9] }));
    const layer = mgr.getLayer('/g')!;
    expect(layer.displayMin).toBeCloseTo(0, 5);
    expect(layer.displayMax).toBeCloseTo(1, 5);
  });

  it('unions descendant colour ranges so a later HDR part stays reachable', () => {
    // A direct-colour partition's slider bounds must cover EVERY part's
    // colour spread — taking the first part's range alone would leave a later
    // HDR part's colours (here up to 3.5) beyond the slider's reach.
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
              path: '/g/p0',
              type: 'gsplats',
              attrs: { color_data_range: [0.1, 0.8] as [number, number] },
              children: [],
            },
            {
              path: '/g/p1',
              type: 'gsplats',
              attrs: { color_data_range: [0, 3.5] as [number, number] },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;
    mgr.initFromSceneGraph(graph);
    const layer = mgr.getLayer('/g')!;
    expect(layer.dataMin).toBeLessThanOrEqual(0);
    expect(layer.dataMax).toBeGreaterThanOrEqual(3.5);
    // The window itself stays the direct-colour identity.
    expect(layer.displayMax).toBeCloseTo(1, 5);
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

// `layer_order` — the authored cross-layer draw order
// (docs/guides/specs/LAYER_ORDER_SPEC.md). The panel's whole job here is
// keeping "unset" distinguishable from "0": both resolve to band 0 and preserve
// containment ordering within that band, but only the explicit value remains a
// stated panel value and participates in authored-band diagnostics.
describe('LayerStateManager — layer order', () => {
  const graph = (): SceneNode =>
    ({
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'authored',
          type: 'gsplats',
          attrs: { layer: true, layer_order: 20 } as never,
          hasSpatialIndex: true,
        },
        {
          path: 'bare',
          type: 'gsplats',
          attrs: { layer: true } as never,
          hasSpatialIndex: true,
        },
      ],
    }) as SceneNode;

  let mgr: LayerStateManager;

  beforeEach(() => {
    mgr = new LayerStateManager();
    mgr.initFromSceneGraph(graph());
  });

  it('reads an authored level off the node, and leaves a bare layer unset', () => {
    expect(mgr.getLayer('authored')?.layerOrder).toBe(20);
    expect(mgr.getLayer('authored')?.layerOrderExplicit).toBe(true);
    expect(mgr.getLayer('bare')?.layerOrder).toBeUndefined();
    expect(mgr.getLayer('bare')?.layerOrderExplicit).toBe(false);
  });

  it('ignores a layer order authored on the scene root', () => {
    const rooted = graph();
    rooted.attrs = { layer_order: 5 } as never;

    const rootedManager = new LayerStateManager();
    rootedManager.initFromSceneGraph(rooted);

    expect(rootedManager.getLayer('authored')?.layerOrder).toBe(20);
    expect(rootedManager.getLayer('authored')?.inheritedLayerOrder).toBeUndefined();
    expect(rootedManager.getLayer('bare')?.layerOrder).toBeUndefined();
    expect(rootedManager.getLayer('bare')?.inheritedLayerOrder).toBeUndefined();
  });

  it('keeps a layer order authored on a standalone data root', () => {
    const standaloneManager = new LayerStateManager();
    standaloneManager.initFromSceneGraph({
      path: '/',
      type: 'gsplats',
      attrs: { layer: true, layer_order: 8 } as never,
      hasSpatialIndex: true,
    });

    expect(standaloneManager.getLayer('/')?.layerOrder).toBe(8);
    expect(standaloneManager.getLayer('/')?.layerOrderExplicit).toBe(true);
  });

  it('setLayerOrder marks the layer explicit', () => {
    mgr.setLayerOrder('bare', -5);
    expect(mgr.getLayer('bare')?.layerOrder).toBe(-5);
    expect(mgr.getLayer('bare')?.layerOrderExplicit).toBe(true);
  });

  // Clearing is NOT setting 0. If `layerOrderExplicit` were left true here,
  // `liveLayerAttrs` would keep emitting the stale value as this layer's own
  // composition setter, so the panel and diagnostics would keep reporting an
  // authored order that was just deleted.
  it('setLayerOrder(undefined) clears BOTH the value and the explicit flag', () => {
    mgr.setLayerOrder('authored', undefined);
    expect(mgr.getLayer('authored')?.layerOrder).toBeUndefined();
    expect(mgr.getLayer('authored')?.layerOrderExplicit).toBe(false);
  });

  it('an authored 0 stays explicit (a real band, not an absence)', () => {
    mgr.setLayerOrder('bare', 0);
    expect(mgr.getLayer('bare')?.layerOrder).toBe(0);
    expect(mgr.getLayer('bare')?.layerOrderExplicit).toBe(true);
  });

  it('rejects a fractional level instead of silently changing its order', () => {
    mgr.setLayerOrder('bare', 3.7);
    expect(mgr.getLayer('bare')?.layerOrder).toBeUndefined();
    expect(mgr.getLayer('bare')?.layerOrderExplicit).toBe(false);
  });

  it('rejects an order outside the JavaScript safe-integer range', () => {
    mgr.setLayerOrder('bare', Number.MAX_SAFE_INTEGER + 1);
    expect(mgr.getLayer('bare')?.layerOrder).toBeUndefined();
    expect(mgr.getLayer('bare')?.layerOrderExplicit).toBe(false);
  });

  // A hand-edited store can carry junk. The renderer treats a non-finite level
  // as absent, and the panel must agree or the field would show a value the
  // render is not using.
  // Both fields must derive from the same sanitized read. A loose `!= null` on
  // the raw attr would report explicit=true alongside layerOrder=undefined —
  // the panel claiming this layer authored an order the renderer discards.
  it('treats a non-finite authored order as unset in BOTH fields', () => {
    const mgrJunk = new LayerStateManager();
    mgrJunk.initFromSceneGraph({
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: 'junk',
          type: 'gsplats',
          attrs: { layer: true, layer_order: 'front' } as never,
          hasSpatialIndex: true,
        },
      ],
    } as SceneNode);
    expect(mgrJunk.getLayer('junk')?.layerOrder).toBeUndefined();
    expect(mgrJunk.getLayer('junk')?.layerOrderExplicit).toBe(false);
  });

  // The pair can never be half-set: explicit implies a value, and a value
  // implies explicit. Anything else is a state no consumer knows how to read.
  it('never reports explicit without a value, or a value without explicit', () => {
    for (const raw of [3, 0, -5, 'front', null, Number.NaN, Number.POSITIVE_INFINITY, true]) {
      const m = new LayerStateManager();
      m.initFromSceneGraph({
        path: '',
        type: 'scene',
        attrs: {},
        hasSpatialIndex: false,
        children: [
          {
            path: 'n',
            type: 'gsplats',
            attrs: { layer: true, layer_order: raw } as never,
            hasSpatialIndex: true,
          },
        ],
      } as SceneNode);
      const layer = m.getLayer('n')!;
      expect(layer.layerOrderExplicit).toBe(layer.layerOrder !== undefined);
    }
  });
});
