/**
 * What the depth-sort coordinator is told when a layer's blending mode changes.
 *
 * Both arguments to `noteDepthSortBlendingModeSwitch` must be the RESOLVED mode.
 * `prevBlendingMode` always was — it is read off `userData` — but the new one
 * used to be the REQUESTED mode straight from the layer compose. That asymmetry
 * is invisible for the three emissive types, whose request IS their resolved
 * mode, and wrong for mesh, which maps the unsupported `volumetric` onto
 * `opaque`.
 *
 * **The reachable path is narrow, and getting it wrong makes the test vacuous.**
 * Two layers above this one already resolve: `setBlendingMode` resolves at the
 * point of storage, and `initFromSceneGraph` resolves a layer's composed mode
 * for its own type. So neither the Blend dropdown nor an authored attr on the
 * mesh node itself can put `volumetric` into the composition — a test driven
 * either way passes against the bug and proves nothing.
 *
 * What is left is a NON-LAYER ancestor. `composeEffective` reads
 * `node.attrs.blending_mode` raw for any node the panel does not track as a
 * layer, so a plain group authored `blending_mode='volumetric'` composes down
 * onto a mesh leaf that owns no mode of its own, unresolved. That is precisely
 * the inheritance case §6.3 keeps a viewer-side fallback for, and it is the only
 * case here that is load-bearing — see
 * `resolves a mode INHERITED from a non-layer ancestor`. The three
 * authored-on-the-node cases are kept as documentation of the layers that
 * already resolve, and are labelled as such rather than left looking like
 * coverage they do not provide.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const noteDepthSortBlendingModeSwitch = vi.fn();
vi.mock('../../../../rendering/depth-sort-coordinator', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../../../rendering/depth-sort-coordinator')>()),
  noteDepthSortBlendingModeSwitch: (...args: unknown[]) => noteDepthSortBlendingModeSwitch(...args),
}));

const { LayerApplyEngine } = await import('../../../../ui/layers/layer-apply');
const { LayerStateManager } = await import('../../../../ui/layers/layer-state');

import type { SceneNode } from '../../../../data/data-loader-types';

/** Mesh's real resolution rule: `volumetric` has no meaning for a surface. */
function resolveMeshMode(mode: string): string {
  return mode === 'volumetric' ? 'opaque' : mode;
}

function meshMaterialStub(initial: string): Record<string, unknown> {
  const mat: Record<string, unknown> = {
    userData: { blendingMode: initial, _luxarNodeOwned: true },
    uniforms: { uOpacity: { value: 1.0 } },
    defines: {},
    updateIntensity: vi.fn(),
    updateOffset: vi.fn(),
    updateGamma: vi.fn(),
    updateOpacity: vi.fn(),
    // The whole point: stamps the RESOLVED mode, exactly as the real mesh
    // materials do (`materials/mesh/material-glsl.ts` / `-tsl.ts`).
    applyBlendingMode: vi.fn((mode: string) => {
      (mat.userData as Record<string, unknown>).blendingMode = resolveMeshMode(mode);
    }),
  };
  mat.clone = vi.fn(() => mat);
  return mat;
}

function harness(initialMode: string, authoredMode?: string) {
  const mat = meshMaterialStub(initialMode);
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
  mesh.name = '/surf';
  mesh.userData.nodeType = 'mesh';
  // Skip the clone-on-first-use path: it registers the clone with
  // MaterialManager, which needs the full THREE.Material event API. A node-owned
  // material is not cloned in production either, so this is the realistic shape
  // and it keeps the stub about blending resolution rather than about THREE.
  mesh.userData._layerMaterialCloned = true;
  const rootGroup = new THREE.Group();
  rootGroup.add(mesh);

  // `layer: true` is what makes a node listable in the panel at all
  // (`isLayerEnabled`), and the root is `type: 'scene'` so the walk skips it.
  const graph: SceneNode = {
    name: 'root',
    path: '',
    type: 'scene',
    attrs: {},
    children: [
      {
        name: 'surf',
        path: '/surf',
        type: 'mesh',
        // An AUTHORED / inherited mode, which is the only way an unsupported one
        // reaches the composition (see the module doc).
        attrs: authoredMode ? { layer: true, blending_mode: authoredMode } : { layer: true },
        children: [],
      } as unknown as SceneNode,
    ],
  } as unknown as SceneNode;

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const engine = new LayerApplyEngine({
    getRootGroup: () => rootGroup,
    getSceneGraph: () => graph,
    state,
    requestRender: () => {},
  });
  return { engine, state, mat, mesh };
}

/**
 * The same scene, but with the mode authored on a NON-LAYER group ancestor and
 * the mesh leaf owning none — the one arrangement that puts an unresolved mode
 * into the composition (see the module doc).
 */
function inheritedHarness(initialMode: string, ancestorMode: string) {
  const mat = meshMaterialStub(initialMode);
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
  mesh.name = '/g/surf';
  mesh.userData.nodeType = 'mesh';
  mesh.userData._layerMaterialCloned = true;
  const rootGroup = new THREE.Group();
  rootGroup.add(mesh);

  const graph = {
    name: 'root',
    path: '',
    type: 'scene',
    attrs: {},
    children: [
      {
        name: 'g',
        path: '/g',
        type: 'group',
        // NOT a layer, so the panel never resolved this value for a leaf type.
        attrs: { blending_mode: ancestorMode },
        children: [
          { name: 'surf', path: '/g/surf', type: 'mesh', attrs: { layer: true }, children: [] },
        ],
      },
    ],
  } as unknown as SceneNode;

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const engine = new LayerApplyEngine({
    getRootGroup: () => rootGroup,
    getSceneGraph: () => graph,
    state,
    requestRender: () => {},
  });
  return { engine, state, mat, mesh, path: '/g/surf' };
}

/** Re-apply the composed attrs, as any panel edit does. */
function apply(h: ReturnType<typeof harness>): void {
  h.engine.applyBlendingMode(h.state.getLayer('/surf')!);
}

describe('the depth-sort mode-switch hook is told the RESOLVED mode', () => {
  beforeEach(() => {
    noteDepthSortBlendingModeSwitch.mockClear();
  });

  it('resolves a mode INHERITED from a non-layer ancestor', () => {
    // THE load-bearing case. A plain group authored `volumetric` composes down
    // onto a mesh leaf that owns no mode, so the composition hands the hook an
    // unresolved value while the material has already mapped it to `opaque`.
    //
    // Before the fix that read as sorted → sorted, so the coordinator saw no
    // transition: it left the node registered with the SortWorker and kept its
    // retained `triangleSource` for a material that would never sort again.
    const h = inheritedHarness('normal', 'volumetric');
    h.engine.applyBlendingMode(h.state.getLayer('/g/surf')!);

    expect(noteDepthSortBlendingModeSwitch).toHaveBeenCalledTimes(1);
    const [, newMode, prevMode] = noteDepthSortBlendingModeSwitch.mock.calls[0];
    expect(prevMode).toBe('normal');
    expect(newMode).toBe('opaque');
    // Belt: the material really did resolve it, so the assertion above is about
    // the hook rather than about a stub that never mapped anything.
    expect((h.mat.userData as Record<string, unknown>).blendingMode).toBe('opaque');
  });

  it('reports a genuine switch into a sorted mode', () => {
    // The positive control. Without it the case above would pass against a hook
    // that reported `opaque` unconditionally, or was never called at all.
    const h = inheritedHarness('opaque', 'normal');
    h.engine.applyBlendingMode(h.state.getLayer('/g/surf')!);

    const [, newMode, prevMode] = noteDepthSortBlendingModeSwitch.mock.calls[0];
    expect(prevMode).toBe('opaque');
    expect(newMode).toBe('normal');
  });

  it('the panel and the node attr already resolve, one layer up', () => {
    // Documents WHY the cases above go through an ancestor. A mode authored on
    // the mesh node itself is resolved by `initFromSceneGraph`, so it never
    // reaches the hook unresolved — which is also why a test driven this way
    // would pass against the bug and prove nothing.
    const h = harness('normal', 'volumetric');
    apply(h);
    const [, newMode] = noteDepthSortBlendingModeSwitch.mock.calls[0];
    expect(newMode).toBe('opaque');
    expect(h.state.getLayer('/surf')!.blendingMode).toBe('opaque');
  });
});
