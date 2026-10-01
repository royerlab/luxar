/**
 * The node factory and the Layers panel must agree on a colormapped leaf's
 * scalar window.
 *
 * Two code paths set it. The factory (`resolveColormapWindow`, at node creation)
 * sees only authored attrs; the panel (`LayerApplyEngine.applyComposed`, at
 * `initFromScene` and on every edit) reads the window off the LAYER that owns
 * the control. On any leaf the panel reaches after creation the panel wins, so a
 * disagreement is invisible — until a leaf is built after the panel pushed
 * (`kind=partition` parts activated later, lazy levels), which then renders on
 * the factory's window, or until the frames between creation and the panel's
 * first push.
 *
 * They disagreed exactly when the GAIN sat on the layer node itself above an
 * identity leaf — `luxar gsplat convert --intensity/--offset` and
 * `add_gsplats_from_file(..., layer=True, intensity=…)` both author it there.
 * The panel reads a non-identity gain on a layer as the absolute window
 * (`initialDisplayRange`/`computeDisplayRange`), while the factory folded it onto
 * the leaf's data range as if it were an ancestor gain: on the hosted h2afva
 * timelapse [0.001, 0.025] against [6.95e-5, 0.00174], ~14x narrower.
 *
 * Each case builds the leaf through the real factory, then lets the real engine
 * push the owning layer, and asserts the push changed nothing.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import { __resetMaterialManagerForTests } from '../../../../rendering/material-manager';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import { SceneNodeIndex } from '../../../../data/scene-loader/view-state/scene-node-index';
import { collectAncestorNodes } from '../../../../data/attrs-composer';
import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { GSplatsMetadata, GSplatsDataLoader } from '../../../../types/gsplats';

const RANGE: [number, number] = [5.409804826328468e-10, 0.06948927677778476];

function group(path: string, attrs: Record<string, unknown>, children: SceneNode[]): SceneNode {
  return {
    path,
    type: 'group',
    attrs: { type: 'group', ...attrs } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

function splats(path: string, attrs: Record<string, unknown> = {}): SceneNode {
  return {
    path,
    type: 'gsplats',
    attrs: {
      type: 'gsplats',
      colormap: 'plasma',
      amplitude_data_range: RANGE,
      n_splats: 1000,
      ...attrs,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function scene(child: SceneNode): SceneNode {
  return {
    path: '/',
    type: 'scene',
    attrs: {} as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [child],
  };
}

type Window = { scalarMin: number; scalarScale: number };

/**
 * Build `leafPath` through the factory, snapshot its window, push `layerPath`
 * through the engine, snapshot again.
 */
function factoryThenPanel(graph: SceneNode, leafPath: string, layerPath: string) {
  const chain = collectAncestorNodes(graph, leafPath);
  const leaf = chain[chain.length - 1];
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  const mesh = new NodeFactory().createGSplatsNode(
    leaf.path,
    applyEffectiveAttrs(new SceneNodeIndex(graph), leaf),
    leaf.attrs as unknown as GSplatsMetadata,
    {
      centers: new Float32Array([0, 0, 0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      amplitudes: new Float32Array([0.01]),
      colors: new Float32Array([1, 1, 1]),
      splatCount: 1,
    },
    { dispose: vi.fn() } as unknown as GSplatsDataLoader
  );
  root.add(mesh);
  const uniforms = (mesh.material as THREE.ShaderMaterial).uniforms;
  const read = (): Window => ({
    scalarMin: uniforms.uScalarMin.value as number,
    scalarScale: uniforms.uScalarScale.value as number,
  });
  const factory = read();

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const engine = new LayerApplyEngine({
    getRootGroup: () => root,
    getSceneGraph: () => graph,
    state,
    requestRender: vi.fn(),
    requestReprocess: vi.fn(),
  });
  const layer = state.getLayer(layerPath);
  expect(layer, `${layerPath} is a layer`).toBeDefined();
  engine.applyDisplayRange(layer!);
  return { factory, panel: read(), layer: layer! };
}

function expectSameWindow(a: Window, b: Window): void {
  expect(a.scalarMin).toBeCloseTo(b.scalarMin, 12);
  expect(a.scalarScale / b.scalarScale).toBeCloseTo(1, 9);
}

describe('factory window == panel window for a colormapped leaf', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  it('gain ON the layer node (kind=partition wrapper, the h2afva shape)', () => {
    const graph = scene(
      group(
        '/nuclei',
        {
          kind: 'partition',
          layer: true,
          display_type: 'gsplats',
          intensity: 41.666666666666664,
          offset: -0.041666666666666664,
        },
        [splats('/nuclei/part_0', { intensity: 1, offset: 0 })]
      )
    );
    const { factory, panel, layer } = factoryThenPanel(graph, '/nuclei/part_0', '/nuclei');
    // The panel's window is the authored gain read as a window.
    expect(layer.displayMin).toBeCloseTo(0.001, 12);
    expect(layer.displayMax).toBeCloseTo(0.025, 12);
    expectSameWindow(factory, panel);
  });

  it('gain ON the layer node (kind=lod wrapper)', () => {
    const graph = scene(
      group('/cells', { kind: 'lod', layer: true, display_type: 'gsplats', intensity: 0.5 }, [
        splats('/cells/level_0'),
      ])
    );
    const { factory, panel } = factoryThenPanel(graph, '/cells/level_0', '/cells');
    expectSameWindow(factory, panel);
  });

  it('gain ABOVE the layer (non-layer ancestor): folded onto the data range', () => {
    const graph = scene(
      group('/scaled', { intensity: 2 }, [
        group('/scaled/nuclei', { kind: 'partition', layer: true, display_type: 'gsplats' }, [
          splats('/scaled/nuclei/part_0'),
        ]),
      ])
    );
    const { factory, panel } = factoryThenPanel(graph, '/scaled/nuclei/part_0', '/scaled/nuclei');
    expectSameWindow(factory, panel);
  });

  it('no gain anywhere: the data range', () => {
    const graph = scene(
      group('/nuclei', { kind: 'partition', layer: true, display_type: 'gsplats' }, [
        splats('/nuclei/part_0'),
      ])
    );
    const { factory, panel } = factoryThenPanel(graph, '/nuclei/part_0', '/nuclei');
    expectSameWindow(factory, { scalarMin: RANGE[0], scalarScale: 1 / (RANGE[1] - RANGE[0]) });
    expectSameWindow(factory, panel);
  });

  it('the leaf IS the layer and carries the gain', () => {
    const graph = scene(splats('/cloud', { layer: true, intensity: 0.09 }));
    const { factory, panel } = factoryThenPanel(graph, '/cloud', '/cloud');
    expectSameWindow(factory, panel);
  });

  it('nested: gain on an OUTER layer, the leaf is its own (identity) layer — the nearest layer owns the window', () => {
    const graph = scene(
      group('/wrapper', { layer: true, intensity: 4 }, [splats('/wrapper/cloud', { layer: true })])
    );
    const { factory, panel } = factoryThenPanel(graph, '/wrapper/cloud', '/wrapper/cloud');
    expectSameWindow(factory, panel);
  });
});
