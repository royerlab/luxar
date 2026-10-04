// @vitest-environment jsdom
/**
 * A leaf that MATERIALISES after the Layers panel initialised must carry the
 * owning layer's CURRENT state before it is first drawn.
 *
 * The panel pushes its state into the scene at exactly two moments: once per
 * layer at `initFromScene`, and on every edit — and both skip a leaf that has
 * no scene object yet. Gated partition loading (B4) builds an out-of-slice /
 * out-of-frustum part only when the LOD registry first activates it, long
 * after the panel initialised, and nothing re-applied the layer state then: the
 * part rendered its AUTHORED appearance until the next slider edit re-pushed
 * everything. On the hosted h2afva timelapse a time step brought in parts with
 * the authored absorption/opacity (edits made before the step were lost on
 * them) and a colormap window ~14x narrower than the panel's.
 *
 * The scene below is that store's shape: a `layer=true, kind=partition` wrapper
 * authoring the display window (intensity/offset), absorption, opacity and
 * gamma, over colormapped gsplat parts one timepoint each. The view sits at
 * t = 1, so part_1 loads eagerly and part_0 is deferred behind the registry's
 * activation thunk — the real loader path, real node factory, real panel.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

const createGSplatsLoaderMock = vi.fn();
vi.mock(import('../../../../data/scene-loader/loaders/loader-factory'), async (importOriginal) => ({
  ...(await importOriginal()),
  createGSplatsLoader: (...args: unknown[]) => createGSplatsLoaderMock(...args),
  createProgressiveGSplatsLoader: (...args: unknown[]) => createGSplatsLoaderMock(...args),
}));

import { LayersPanel } from '../../../../ui/layers/layers-panel';
import { NodeFactory } from '../../../../rendering/node-factory';
import { getColormapTexture } from '../../../../rendering/colormap-textures';
import { __resetMaterialManagerForTests } from '../../../../rendering/material-manager';
import { computeScalarRangeUniforms } from '../../../../rendering/materials/_shared/scalar-range';
import { loadSceneNodes } from '../../../../data/scene-loader/nodes/load-scene-nodes';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import { SceneNodeIndex } from '../../../../data/scene-loader/view-state/scene-node-index';
import { findObjectByName } from '../../../../utils/scene-graph-index';
import { log } from '../../../../utils/log';
import { makeTestNodeBuildCtx } from '../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { AnimationController } from '../../../../scene/animation/animation-controller';
import type { PartitionGroupChild } from '../../../../scene/lod-group-registry';
import type { PointsMetadata } from '../../../../types/points';
import type { LinesMetadata, LinesDataLoader } from '../../../../types/lines';
import type { MeshMetadata, MeshDataLoader } from '../../../../types/mesh';
import type { DataLoader } from '../../../../data/data-loader-types';
import { DepthSortCoordinator } from '../../../../rendering/depth-sort-coordinator';

const AMPLITUDE_RANGE: [number, number] = [5.409804826328468e-10, 0.06948927677778476];

function part(index: number, parent = '/nuclei'): SceneNode {
  return {
    path: `${parent}/part_${index}`,
    type: 'gsplats',
    attrs: {
      type: 'gsplats',
      child_index: index,
      position_bounds: { min: [0, 0, 0, index], max: [1, 1, 1, index] },
      colormap: 'plasma',
      amplitude_data_range: AMPLITUDE_RANGE,
      n_splats: 1000 + index,
      intensity: 1.0,
      offset: 0.0,
      absorption: 1.0,
      opacity: 1.0,
      gamma: 1.0,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function sceneGraph(nested = false): SceneNode {
  const wrapperPath = nested ? '/outer/n' : '/nuclei';
  const wrapper: SceneNode = {
    path: wrapperPath,
    type: 'group',
    attrs: {
      type: 'group',
      kind: 'partition',
      layer: true,
      display_type: 'gsplats',
      intensity: 41.666666666666664,
      offset: -0.041666666666666664,
      absorption: 1.09,
      opacity: 0.55,
      gamma: 2.2,
      blending_mode: 'volumetric',
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [part(0, wrapperPath), part(1, wrapperPath)],
  };
  const outer: SceneNode = nested
    ? {
        path: '/outer',
        type: 'group',
        attrs: { type: 'group', layer: true, display_type: 'gsplats' } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [wrapper],
      }
    : wrapper;
  return {
    path: '/',
    type: 'scene',
    attrs: {} as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [outer],
  };
}

function makeAnimationController(): AnimationController {
  return {
    startAnimation: vi.fn(),
    addPerFrameCallback: vi.fn(),
    removePerFrameCallback: vi.fn(() => true),
    hasPerFrameCallback: vi.fn(() => false),
  } as unknown as AnimationController;
}

function stubLoc() {
  return { resolve: () => ({ resolve: () => ({}) }) } as never;
}

interface Harness {
  graph: SceneNode;
  root: THREE.Group;
  panel: LayersPanel;
  deferred: PartitionGroupChild;
}

/**
 * Load the partition through the production loader path, then initialise the
 * panel on the result — the order the app uses (`loadDataset` hydrates the
 * panel after `loadSceneData` resolves).
 */
async function loadAndInitPanel(graph: SceneNode = sceneGraph()): Promise<Harness> {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  const container = document.createElement('div');
  document.body.appendChild(container);
  const panel = new LayersPanel(container, makeAnimationController());
  const registerPartition = vi.fn();
  const ctx: NodeBuildCtx = makeTestNodeBuildCtx({
    nodeFactory: new NodeFactory(),
    applyEffectiveAttrs: (node: SceneNode) => applyEffectiveAttrs(new SceneNodeIndex(graph), node),
    lodGroupRegistry: { registerPartition } as unknown as NodeBuildCtx['lodGroupRegistry'],
    viewState: {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 1],
      tolerance: [0, 0, 0, 0],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1 },
      ],
    } as NodeBuildCtx['viewState'],
    // What `SceneLoader.makeNodeBuildCtx` hands every leaf loader once
    // `loadDataset` has connected the panel.
    onLeafMaterialized: (path: string, object: THREE.Object3D) =>
      panel.applyLayerStateToNewLeaf(graph, path, object),
  } as Partial<NodeBuildCtx>);

  await loadSceneNodes(graph.children![0], root, stubLoc(), ctx);
  panel.initFromScene(root, graph);

  const children = registerPartition.mock.calls[0][0].children as PartitionGroupChild[];
  const deferred = children[0];
  return { graph, root, panel, deferred };
}

type Uniforms = Record<string, { value: number }>;

function uniformsOf(root: THREE.Group, path: string): Uniforms {
  const mesh = findObjectByName(root, path) as THREE.Mesh | undefined;
  expect(mesh, `${path} has a scene object`).toBeDefined();
  return (mesh!.material as THREE.ShaderMaterial).uniforms as unknown as Uniforms;
}

const APPEARANCE = [
  'uAbsorption',
  'uOpacity',
  'uInvGamma',
  'uIntensity',
  'uOffset',
  'uScalarMin',
  'uScalarScale',
] as const;

function appearance(u: Uniforms): Record<string, number> {
  return Object.fromEntries(APPEARANCE.map((k) => [k, u[k]?.value]));
}

describe('LayersPanel — a partition part activated after the panel initialised', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
    createGSplatsLoaderMock.mockReset();
    createGSplatsLoaderMock.mockImplementation(() => ({
      dispose: vi.fn(),
      loadGSplats: vi.fn(async () => ({ splatCount: 0 })),
    }));
  });

  it('the deferred part is not built until the registry activates it', async () => {
    const h = await loadAndInitPanel();
    expect(h.deferred.path).toBe('/nuclei/part_0');
    expect(h.deferred.activate).toBeTypeOf('function');
    expect(findObjectByName(h.root, '/nuclei/part_0')).toBeUndefined();
    expect(findObjectByName(h.root, '/nuclei/part_1')).toBeDefined();
  });

  it('carries panel edits made BEFORE its activation', async () => {
    const h = await loadAndInitPanel();
    h.panel.setLayer('/nuclei', { absorption: 3, opacity: 0.3 });

    await h.deferred.activate!();

    const fresh = uniformsOf(h.root, '/nuclei/part_0');
    const eager = uniformsOf(h.root, '/nuclei/part_1');
    expect(fresh.uAbsorption.value).toBeCloseTo(3, 6);
    expect(fresh.uOpacity.value).toBeCloseTo(eager.uOpacity.value, 6);
    // The window is the one the panel pushes for the layer.
    const layer = h.panel.layerState.getLayer('/nuclei')!;
    const expected = computeScalarRangeUniforms(layer.displayMin, layer.displayMax);
    expect(fresh.uScalarMin.value).toBeCloseTo(expected.scalarMin, 9);
    expect(fresh.uScalarScale.value / expected.scalarScale).toBeCloseTo(1, 6);
    // …and in every respect it matches the part that was there all along.
    expect(appearance(fresh)).toEqual(appearance(eager));
  });

  it('with NO edits, its window equals the panel initial window', async () => {
    const h = await loadAndInitPanel();

    await h.deferred.activate!();

    const fresh = uniformsOf(h.root, '/nuclei/part_0');
    const eager = uniformsOf(h.root, '/nuclei/part_1');
    const layer = h.panel.layerState.getLayer('/nuclei')!;
    const expected = computeScalarRangeUniforms(layer.displayMin, layer.displayMax);
    expect(fresh.uScalarMin.value).toBeCloseTo(expected.scalarMin, 9);
    expect(fresh.uScalarScale.value / expected.scalarScale).toBeCloseTo(1, 6);
    expect(appearance(fresh)).toEqual(appearance(eager));
  });

  it('replays nested colormap edits in push order for a deferred part', async () => {
    const h = await loadAndInitPanel(sceneGraph(true));
    h.panel.setLayer('/outer/n', { colormap: 'viridis' });
    h.panel.setLayer('/outer', { colormap: 'plasma' });

    await h.deferred.activate!();

    const eager = uniformsOf(h.root, '/outer/n/part_1').uColormapTex.value;
    const fresh = uniformsOf(h.root, '/outer/n/part_0').uColormapTex.value;
    expect(eager).toBe(getColormapTexture('plasma'));
    expect(fresh).toBe(eager);
  });

  it('uses the latest push when a nested layer is edited again', async () => {
    const h = await loadAndInitPanel(sceneGraph(true));
    h.panel.setLayer('/outer/n', { colormap: 'viridis' });
    h.panel.setLayer('/outer', { colormap: 'plasma' });
    h.panel.setLayer('/outer/n', { colormap: 'viridis' });

    await h.deferred.activate!();

    const eager = uniformsOf(h.root, '/outer/n/part_1').uColormapTex.value;
    const fresh = uniformsOf(h.root, '/outer/n/part_0').uColormapTex.value;
    expect(eager).toBe(getColormapTexture('viridis'));
    expect(fresh).toBe(eager);
  });

  // geometry-subset: gsplats is the leaf type of every other case in this file
  it.each(['points', 'lines', 'mesh'] as const)(
    'replays a panel-selected colormap to a late %s leaf before its first data commit',
    (type) => {
      const makeLeaf = (index: number): SceneNode => ({
        path: `/nuclei/part_${index}`,
        type,
        attrs: {
          type,
          has_scalars: true,
          scalar_data_range: [0, 10],
          n_points: 1,
          n_segments: 1,
          n_vertices: 3,
          n_faces: 1,
          ndim: 4,
          max_width: 1,
        } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [],
      });
      const leaves = [makeLeaf(0), makeLeaf(1)];
      const wrapper: SceneNode = {
        path: '/nuclei',
        type: 'group',
        attrs: {
          type: 'group',
          kind: 'partition',
          layer: true,
          display_type: type,
        } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: leaves,
      };
      const graph: SceneNode = {
        path: '/',
        type: 'scene',
        attrs: {} as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [wrapper],
      };
      const root = new THREE.Group();
      root.name = 'LuxarScene';
      const factory = new NodeFactory();
      const loader = { dispose: vi.fn() };
      const create = (leaf: SceneNode): THREE.Mesh => {
        const attrs = applyEffectiveAttrs(new SceneNodeIndex(graph), leaf);
        if (type === 'points') {
          return factory.createEmptyPointsNode(
            leaf.path,
            attrs as unknown as PointsMetadata,
            loader as unknown as DataLoader
          );
        }
        if (type === 'lines') {
          return factory.createEmptyLinesNode(
            leaf.path,
            attrs,
            attrs as unknown as LinesMetadata,
            loader as unknown as LinesDataLoader
          );
        }
        return factory.createEmptyMeshNode(
          leaf.path,
          attrs as unknown as MeshMetadata,
          loader as unknown as MeshDataLoader
        );
      };
      const eager = create(leaves[1]);
      const colormapActive = (mesh: THREE.Mesh): boolean =>
        'USE_COLORMAP' in ((mesh.material as THREE.ShaderMaterial).defines ?? {});
      // The eager leaf's geometry has received its scalar data. The late
      // leaf will still have only its placeholder when the panel replays.
      eager.geometry.userData.hasScalars = true;
      root.add(eager);
      const container = document.createElement('div');
      const panel = new LayersPanel(container, makeAnimationController());
      panel.initFromScene(root, graph);
      panel.setLayer('/nuclei', { colormap: 'viridis' });
      expect(colormapActive(eager)).toBe(true);

      const late = create(leaves[0]);
      root.add(late);
      const warning = vi.spyOn(log, 'warning');
      panel.applyLayerStateToNewLeaf(graph, leaves[0].path, late);

      expect(colormapActive(late)).toBe(true);
      expect(warning).not.toHaveBeenCalledWith(
        expect.anything(),
        expect.stringContaining('Scalar colormap suppressed')
      );
      const fresh = uniformsOf(root, leaves[0].path);
      const existing = uniformsOf(root, leaves[1].path);
      expect(fresh.uScalarMin.value).toBeCloseTo(existing.uScalarMin.value, 9);
      expect(fresh.uScalarScale.value).toBeCloseTo(existing.uScalarScale.value, 6);
      warning.mockRestore();
      panel.dispose();
    }
  );

  it('a part activated after a switch INTO a sorted mode asks for no extra pass', async () => {
    // Its first commit registers it with the sorter under the LIVE mode; a
    // switch hook on the empty placeholder would only queue a full re-sweep
    // for every part a playback step activates. (The routed switch hook also
    // ignores a mesh no coordinator has seen, so the guard holds twice over.)
    const requestReprocess = vi.fn();
    const depthSort = new DepthSortCoordinator();
    depthSort.configure({ getCamera: () => null, requestRender: vi.fn(), requestReprocess });
    try {
      const graph = sceneGraph();
      graph.children![0].attrs.blending_mode = 'additive';
      const h = await loadAndInitPanel(graph);
      h.panel.setLayer('/nuclei', { blendingMode: 'normal' });
      requestReprocess.mockClear();

      await h.deferred.activate!();

      const fresh = findObjectByName(h.root, '/nuclei/part_0') as THREE.Mesh;
      expect((fresh.material as THREE.Material).userData.blendingMode).toBe('normal');
      expect(requestReprocess).not.toHaveBeenCalled();
    } finally {
      depthSort.dispose();
    }
  });

  it('a nested layer hidden before its part was activated stays hidden', async () => {
    const graph = sceneGraph();
    const wrapper = graph.children![0];
    const leaf = { ...part(0), path: '/nuclei/part_0/splats' };
    wrapper.children![0] = {
      path: '/nuclei/part_0',
      type: 'group',
      attrs: {
        type: 'group',
        layer: true,
        child_index: 0,
        position_bounds: { min: [0, 0, 0, 0], max: [1, 1, 1, 0] },
      } as SceneNode['attrs'],
      hasSpatialIndex: false,
      children: [leaf],
    };
    const h = await loadAndInitPanel(graph);
    h.panel.setLayer('/nuclei/part_0', { visible: false });

    await h.deferred.activate!();

    const group = findObjectByName(h.root, '/nuclei/part_0')!;
    expect(group.visible).toBe(false);
    expect(group.userData.layerVisible).toBe(false);
  });

  it('a leaf of a DIFFERENT scene graph is left alone (stale panel / dataset switch)', async () => {
    const h = await loadAndInitPanel();
    const other = sceneGraph();
    const mesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.ShaderMaterial({ uniforms: { uAbsorption: { value: 1 } } })
    );
    mesh.name = '/nuclei/part_0';
    h.panel.setLayer('/nuclei', { absorption: 3 });

    h.panel.applyLayerStateToNewLeaf(other, '/nuclei/part_0', mesh);

    expect((mesh.material as THREE.ShaderMaterial).uniforms.uAbsorption.value).toBe(1);
  });
});
