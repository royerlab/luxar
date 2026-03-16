import { describe, it, expect, beforeEach } from 'vitest';
import {
  LayerStateManager,
  computeUniforms,
  computeDisplayRange,
} from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';

// ─── computeUniforms / computeDisplayRange ──────────────

describe('computeUniforms', () => {
  it('returns identity for [0, 1]', () => {
    const { intensity, offset } = computeUniforms(0, 1);
    expect(intensity).toBeCloseTo(1.0);
    expect(offset).toBeCloseTo(0.0);
  });

  it('maps [0.2, 0.8] correctly', () => {
    const { intensity, offset } = computeUniforms(0.2, 0.8);
    // intensity = 1/(0.8 - 0.2) = 1/0.6 ≈ 1.667
    expect(intensity).toBeCloseTo(1.0 / 0.6);
    // offset = -0.2 / 0.6 ≈ -0.333
    expect(offset).toBeCloseTo(-0.2 / 0.6);
  });

  it('handles degenerate range (min ≈ max)', () => {
    const { intensity, offset } = computeUniforms(0.5, 0.5);
    // Should clamp to large value
    expect(intensity).toBe(1000);
    expect(offset).toBe(-0.5 * 1000);
  });

  it('handles HDR range [0, 5]', () => {
    const { intensity, offset } = computeUniforms(0, 5);
    expect(intensity).toBeCloseTo(0.2);
    expect(offset).toBeCloseTo(0.0);
  });
});

describe('computeDisplayRange', () => {
  it('recovers [0, 1] from identity uniforms', () => {
    const { min, max } = computeDisplayRange(1.0, 0.0);
    expect(min).toBeCloseTo(0);
    expect(max).toBeCloseTo(1);
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

describe('LayerStateManager', () => {
  let mgr: LayerStateManager;

  beforeEach(() => {
    mgr = new LayerStateManager();
  });

  it('collects only nodes with layer=true', () => {
    const graph: SceneNode = {
      path: '',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        { path: 'a', type: 'points', attrs: { layer: true }, hasSpatialIndex: true },
        { path: 'b', type: 'points', attrs: {}, hasSpatialIndex: true }, // no layer flag
        { path: 'c', type: 'lines', attrs: { layer: true }, hasSpatialIndex: false },
        { path: 'd', type: 'group', attrs: { layer: true }, hasSpatialIndex: false }, // group — NOT a data node
      ],
    };
    mgr.initFromSceneGraph(graph);
    expect(mgr.count).toBe(2); // a (points) + c (lines)
    const names = mgr.getLayers().map((l) => l.name);
    expect(names).toEqual(['a', 'c']);
  });

  it('initializes display range from color_data_range', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ color_data_range: [0.1, 0.9] as [number, number] }]));
    const layer = mgr.getLayers()[0];
    expect(layer.dataMin).toBeCloseTo(0.1);
    expect(layer.dataMax).toBeCloseTo(0.9);
    expect(layer.displayMin).toBeCloseTo(0.1);
    expect(layer.displayMax).toBeCloseTo(0.9);
  });

  it('defaults data range to [0, 1] when absent', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    const layer = mgr.getLayers()[0];
    expect(layer.dataMin).toBe(0);
    expect(layer.dataMax).toBe(1);
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
    expect(layer.displayMin).toBeCloseTo(0.2);
    expect(layer.displayMax).toBeCloseTo(0.8);
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
});
