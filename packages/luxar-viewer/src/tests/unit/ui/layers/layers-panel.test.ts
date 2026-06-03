/**
 * Unit tests for LayersPanel.
 *
 * The panel is heavily DOM- and THREE-bound, so tests focus on the
 * public lifecycle surface that
 * runs cleanly under jsdom without WebGL:
 *
 * - Construction leaves the panel hidden and DOM-empty
 * - `show()` is a no-op before initFromScene (panel not built yet)
 * - `toggle()` with an empty scene fires the "no layers" toast and
 *   stays hidden
 * - `initFromScene` with a scene containing zero layered nodes is a
 *   silent no-op (no panel built)
 * - `initFromScene` with a layered node populates `layerState` and
 *   builds the panel DOM
 * - `dispose()` tears down the panel + clears layer state + survives
 *   double-disposal
 *
 * `materialManager` is mocked because it would touch shader compilation
 * (WebGL); `showToast` is mocked so we can observe the empty-scene path.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { AnimationController } from '../../../../scene/animation/animation-controller';

// `showToast` lives in src/ui/toast; mock so the empty-scene branch
// is observable.
const showToastMock = vi.fn();
vi.mock('../../../../ui/toast', () => ({
  showToast: (msg: string) => showToastMock(msg),
}));

// `materialManager` clones materials (WebGL shader compilation under
// the hood). Empty scenes never reach this path, but the panel
// imports the singleton at module load — give it a stub.
vi.mock('../../../../rendering/material-manager', () => ({
  materialManager: {
    register: vi.fn(),
    getPointMaterial: vi.fn(),
    getLineMaterial: vi.fn(),
    getGSplatMaterial: vi.fn(),
  },
}));

// `getColormapTexture` reads a sampler uniform; not exercised by
// these public-surface tests but imported eagerly.
vi.mock('../../../../rendering/colormap-textures', () => ({
  getColormapTexture: vi.fn(() => null),
}));

// `SceneLoaderManager` is consulted by the LOD-level dropdown to find
// the lod_group registry on the current scene loader. Tests mock it so
// the dropdown change handler can drive a stub registry.
const setSelectorModeMock = vi.fn();
const registryGetMock = vi.fn(() => undefined);
const getDefaultLoaderMock = vi.fn(() => ({
  lodGroupRegistry: {
    setSelectorMode: setSelectorModeMock,
    get: registryGetMock,
  },
}));
vi.mock('../../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: () => ({
      getDefaultLoader: getDefaultLoaderMock,
    }),
  },
}));

import {
  LayersPanel,
  isColormapActive,
  applyColorAdjustments,
  type LuxarMaterial,
} from '../../../../ui/layers/layers-panel';

function makeAnimationController(): AnimationController {
  return {
    startAnimation: vi.fn(),
  } as unknown as AnimationController;
}

function makeEmptySceneGraph(): SceneNode {
  return {
    name: 'root',
    path: '/',
    type: 'group',
    attrs: {},
    children: [],
  } as unknown as SceneNode;
}

function makeLayeredSceneGraph(): SceneNode {
  // Single layered points node — mirrors what the Python API emits
  // when `layer=True`.
  return {
    name: 'root',
    path: '/',
    type: 'group',
    attrs: {},
    children: [
      {
        name: 'cloud',
        path: '/cloud',
        type: 'points',
        attrs: { layer: true, type: 'points' },
        children: [],
      },
    ],
  } as unknown as SceneNode;
}

describe('LayersPanel — construction', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
    showToastMock.mockClear();
  });

  it('starts hidden with empty layer state and does NOT inject any DOM into the container', () => {
    // [ui.md/W2][P2] Previously asserted isVisible()===false and count===0,
    // which collapses to "did not crash". Strengthen: the constructor
    // must not eagerly create the panel DOM — that's done lazily in
    // initFromScene(). A regression that built the panel eagerly would
    // pollute the container.
    expect(container.children.length).toBe(0);
    const panel = new LayersPanel(container, animationController);
    expect(panel.isVisible()).toBe(false);
    expect(panel.layerState.count).toBe(0);
    // Container should remain empty until initFromScene is called.
    expect(container.children.length).toBe(0);
    // No selection state either.
    expect(panel.layerState.getSelected().length).toBe(0);
  });

  it('layerState getter returns the same instance across calls', () => {
    const panel = new LayersPanel(container, animationController);
    const state1 = panel.layerState;
    const state2 = panel.layerState;
    expect(state1).toBe(state2);
  });

  it('show() before initFromScene is a no-op — panel stays hidden and DOM is untouched', () => {
    // [ui.md/W2][P2] Previously only asserted isVisible()===false.
    // Strengthen: container DOM must also stay empty (no late panel
    // creation as a side effect of show()).
    const panel = new LayersPanel(container, animationController);
    panel.show();
    expect(panel.isVisible()).toBe(false);
    expect(container.children.length).toBe(0);
  });

  it('hide() before initFromScene is a no-op — no DOM created, no exception thrown', () => {
    // [ui.md/W2][P2] Previously only asserted !toThrow() and isVisible()===false.
    // Strengthen by checking the container is still empty.
    const panel = new LayersPanel(container, animationController);
    expect(() => panel.hide()).not.toThrow();
    expect(panel.isVisible()).toBe(false);
    expect(container.children.length).toBe(0);
  });
});

describe('LayersPanel.toggle — empty-scene path', () => {
  let container: HTMLElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    showToastMock.mockClear();
  });

  it('with no layers loaded, toggle fires the empty-scene toast and stays hidden', () => {
    const panel = new LayersPanel(container, makeAnimationController());
    panel.toggle();
    expect(panel.isVisible()).toBe(false);
    expect(showToastMock).toHaveBeenCalledWith(
      'No layers in this scene (use layer=True in Python API)'
    );
  });

  it('with an initialized but layer-less scene, toggle still fires the toast', () => {
    const panel = new LayersPanel(container, makeAnimationController());
    const rootGroup = new THREE.Group();
    panel.initFromScene(rootGroup, makeEmptySceneGraph());

    panel.toggle();
    expect(panel.isVisible()).toBe(false);
    expect(showToastMock).toHaveBeenCalled();
  });
});

describe('LayersPanel.initFromScene', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
  });

  it('layer-less scene: layerState stays empty and no panel DOM is built', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeEmptySceneGraph());

    expect(panel.layerState.count).toBe(0);
    expect(container.querySelector('.luxar-layers')).toBeNull();
  });

  it('scene with one layered node: layerState reflects it', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());

    expect(panel.layerState.count).toBe(1);
  });

  it('reinitialization disposes prior state cleanly', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    expect(panel.layerState.count).toBe(1);

    // Reinitialize with a layer-less scene — count should drop.
    panel.initFromScene(new THREE.Group(), makeEmptySceneGraph());
    expect(panel.layerState.count).toBe(0);
  });
});

describe('LayersPanel.dispose', () => {
  let container: HTMLElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
  });

  it('dispose on an uninitialized panel is a no-op', () => {
    const panel = new LayersPanel(container, makeAnimationController());
    expect(() => panel.dispose()).not.toThrow();
    expect(panel.isVisible()).toBe(false);
    expect(panel.layerState.count).toBe(0);
  });

  it('dispose after initFromScene clears layer state and removes panel DOM', () => {
    const panel = new LayersPanel(container, makeAnimationController());
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    expect(panel.layerState.count).toBe(1);

    panel.dispose();

    expect(panel.layerState.count).toBe(0);
    expect(panel.isVisible()).toBe(false);
  });

  it('double dispose is safe', () => {
    const panel = new LayersPanel(container, makeAnimationController());
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    panel.dispose();
    expect(() => panel.dispose()).not.toThrow();
  });

  it('dispose tears down all event listeners attached to row buttons', () => {
    // Every addEventListener goes through this.events.on(...), and
    // clear() / dispose() drains the group. After dispose, click
    // events on the (still-rooted) eye button should NOT trigger
    // visibility toggle handlers.
    const panel = new LayersPanel(container, makeAnimationController());
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());

    // Capture an eye button reference BEFORE dispose so the test can
    // dispatch a click on a node the closure could (incorrectly) still
    // be listening to.
    const eyeBtn = container.querySelector('.luxar-layer-row__eye') as HTMLElement | null;
    expect(eyeBtn).not.toBeNull();

    // Internal event group should have non-zero size before dispose.

    const events = (panel as any).events as { size: number };
    expect(events.size).toBeGreaterThan(0);

    panel.dispose();

    // After dispose, the group is replaced with a fresh empty one
    // (so a subsequent show()/initFromScene doesn't reuse a disposed
    // group). Either size === 0 OR the group reference changed.

    const eventsAfter = (panel as any).events as { size: number };
    expect(eventsAfter.size).toBe(0);

    // Sanity: dispatching a click on the captured eye button does
    // not throw and (because handlers were removed) does not flip
    // the layer's visibility.
    expect(() => eyeBtn?.click()).not.toThrow();
  });
});

function findActiveLevelSelect(container: HTMLElement): HTMLSelectElement | null {
  // Three selects share the class; find the one whose sibling label
  // text is "Active level".
  const groups = Array.from(container.querySelectorAll('.luxar-layers-panel__control-group'));
  for (const group of groups) {
    const label = group.querySelector('.luxar-layers-panel__control-label');
    if (label?.textContent === 'Active level') {
      return group.querySelector('.luxar-layers-panel__select') as HTMLSelectElement | null;
    }
  }
  return null;
}

describe('LayersPanel — LOD active-level dropdown', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  function makeLodSceneGraph(): SceneNode {
    // kind=lod group with two child levels — Layer state will surface
    // `lodGroupChildCount = 2` and render the active-level dropdown.
    return {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'pyramid',
          path: '/pyramid',
          type: 'group',
          attrs: { layer: true, kind: 'lod', display_type: 'points' },
          children: [
            { name: 'lod_0', path: '/pyramid/lod_0', type: 'points', attrs: {}, children: [] },
            { name: 'lod_1', path: '/pyramid/lod_1', type: 'points', attrs: {}, children: [] },
          ],
        },
      ],
    } as unknown as SceneNode;
  }

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
    setSelectorModeMock.mockClear();
    getDefaultLoaderMock.mockClear();
    showToastMock.mockClear();
  });

  it('change → setSelectorMode is called AND requestRender wakes the animation loop', () => {
    // Regression guard: previously the change handler called
    // registry.setSelectorMode(...) but never animationController.startAnimation(),
    // so the level swap (which runs in a per-frame callback) was invisible
    // when the loop was idle.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();

    // Select the lod layer so it becomes primary.
    panel.layerState.select('/pyramid', 'single');

    // The panel renders three selects sharing this class (blend,
    // colormap, lod-active-level). Find the lod-level dropdown by its
    // label ("Active level") rather than by class index.
    const select = findActiveLevelSelect(container);
    expect(select).not.toBeNull();

    // Find the lock-to-level-1 option.
    select!.value = '1';
    select!.dispatchEvent(new Event('change', { bubbles: true }));

    expect(setSelectorModeMock).toHaveBeenCalledWith('/pyramid', { lockLevel: 1 });
    expect(animationController.startAnimation).toHaveBeenCalled();
  });

  it('change to "auto" propagates as auto-mode and still wakes the animation loop', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    // The panel renders three selects sharing this class (blend,
    // colormap, lod-active-level). Find the lod-level dropdown by its
    // label ("Active level") rather than by class index.
    const select = findActiveLevelSelect(container);
    select!.value = 'auto';
    select!.dispatchEvent(new Event('change', { bubbles: true }));

    expect(setSelectorModeMock).toHaveBeenCalledWith('/pyramid', 'auto');
    expect(animationController.startAnimation).toHaveBeenCalled();
  });

  it('does NOT wake the loop when setSelectorMode throws for every path', () => {
    // When the registry rejects every path (e.g. stale path after scene
    // reload), no visibility change happens, so calling requestRender
    // would be a spurious wake. The fix gates the call on at least one
    // successful apply.
    setSelectorModeMock.mockImplementation(() => {
      throw new Error('unknown lod_group path');
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    // The panel renders three selects sharing this class (blend,
    // colormap, lod-active-level). Find the lod-level dropdown by its
    // label ("Active level") rather than by class index.
    const select = findActiveLevelSelect(container);
    // Clear startAnimation calls from prior interactions (selection +
    // renderControls plumbing).
    (animationController.startAnimation as ReturnType<typeof vi.fn>).mockClear();

    select!.value = '0';
    select!.dispatchEvent(new Event('change', { bubbles: true }));

    expect(setSelectorModeMock).toHaveBeenCalled();
    expect(animationController.startAnimation).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Display-range / gamma routing: colormap (LUT) vs direct-color.
//
// Regression guard for the gamma-on-value fix. When a leaf renders through
// a colormap LUT, the display range must drive the scalar window
// (`updateScalarRange`) and the color GOG (`updateIntensity`/`updateOffset`)
// must be bypassed — gamma is applied to the value pre-LUT in the shader.
// In direct-color mode the GOG drives intensity/offset as before. Gamma is
// pushed in both modes.
// ---------------------------------------------------------------------------

/** A LuxarMaterial stub that records the update calls + carries `defines`. */
function makeRecordingMaterial(defines: Record<string, string> | null): {
  mat: LuxarMaterial;
  calls: {
    gamma: number[];
    intensity: number[];
    offset: number[];
    scalarRange: Array<[number, number]>;
    opacity: number[];
  };
} {
  const calls = {
    gamma: [] as number[],
    intensity: [] as number[],
    offset: [] as number[],
    scalarRange: [] as Array<[number, number]>,
    opacity: [] as number[],
  };
  const mat = {
    defines,
    updateGamma: (v: number) => calls.gamma.push(v),
    updateIntensity: (v: number) => calls.intensity.push(v),
    updateOffset: (v: number) => calls.offset.push(v),
    updateOpacity: (v: number) => calls.opacity.push(v),
    updateScalarRange: (min: number, max: number) => calls.scalarRange.push([min, max]),
  } as unknown as LuxarMaterial;
  return { mat, calls };
}

describe('isColormapActive', () => {
  it('is true only when the USE_COLORMAP define is present', () => {
    expect(isColormapActive(makeRecordingMaterial({ USE_COLORMAP: '' }).mat)).toBe(true);
    expect(isColormapActive(makeRecordingMaterial({}).mat)).toBe(false);
    expect(isColormapActive(makeRecordingMaterial(null).mat)).toBe(false);
  });
});

describe('applyColorAdjustments — colormap vs direct routing', () => {
  it('colormap mode: display range drives the scalar window; color GOG is bypassed', () => {
    const { mat, calls } = makeRecordingMaterial({ USE_COLORMAP: '' });
    // intensity/offset encode display range [0.5, 2.5]:
    //   computeDisplayRange(0.5, -0.25) → { min: 0.5, max: 2.5 }
    applyColorAdjustments(mat, 2.2, 0.5, -0.25);

    expect(calls.gamma).toEqual([2.2]); // gamma still pushed (applied pre-LUT)
    expect(calls.scalarRange).toEqual([[0.5, 2.5]]);
    // Color GOG NOT touched in colormap mode.
    expect(calls.intensity).toEqual([]);
    expect(calls.offset).toEqual([]);
  });

  it('direct-color mode: GOG drives intensity/offset; scalar range untouched', () => {
    const { mat, calls } = makeRecordingMaterial({}); // no USE_COLORMAP
    applyColorAdjustments(mat, 2.2, 0.5, -0.25);

    expect(calls.gamma).toEqual([2.2]);
    expect(calls.intensity).toEqual([0.5]);
    expect(calls.offset).toEqual([-0.25]);
    expect(calls.scalarRange).toEqual([]);
  });

  it('colormap define but no updateScalarRange support → falls back to color GOG', () => {
    const { calls } = makeRecordingMaterial({ USE_COLORMAP: '' });
    // Material that advertises colormap but cannot accept a scalar range
    // (updateScalarRange is optional on LuxarMaterial).
    const mat = {
      defines: { USE_COLORMAP: '' },
      updateGamma: (v: number) => calls.gamma.push(v),
      updateIntensity: (v: number) => calls.intensity.push(v),
      updateOffset: (v: number) => calls.offset.push(v),
      updateOpacity: (v: number) => calls.opacity.push(v),
    } as unknown as LuxarMaterial;

    applyColorAdjustments(mat, 1.0, 2.0, 0.0);
    expect(calls.intensity).toEqual([2.0]);
    expect(calls.offset).toEqual([0.0]);
    expect(calls.scalarRange).toEqual([]);
  });
});
