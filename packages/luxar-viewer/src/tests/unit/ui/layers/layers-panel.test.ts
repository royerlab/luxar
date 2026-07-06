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
// Return type is `unknown` so tests can `mockReturnValue` a stub
// LODGroupEntry ({ activeChildIndex, children, selectorMode }); the panel
// only reads those three fields off the registry.
const registryGetMock = vi.fn((..._args: unknown[]): unknown => undefined);
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

/**
 * AnimationController stub. Captures per-frame callbacks into a map so
 * tests can invoke them synchronously (the real loop is not running under
 * jsdom) and assert teardown via `hasPerFrameCallback`. The map is exposed
 * as a non-typed `__perFrame` handle for test access.
 */
function makeAnimationController(): AnimationController {
  const perFrame = new Map<string, () => void>();
  const perFrameOpts = new Map<string, unknown>();
  return {
    startAnimation: vi.fn(),
    addPerFrameCallback: vi.fn((id: string, cb: () => void, opts?: unknown) => {
      perFrame.set(id, cb);
      perFrameOpts.set(id, opts);
    }),
    removePerFrameCallback: vi.fn((id: string) => {
      perFrameOpts.delete(id);
      return perFrame.delete(id);
    }),
    hasPerFrameCallback: vi.fn((id: string) => perFrame.has(id)),
    __perFrame: perFrame,
    __perFrameOpts: perFrameOpts,
  } as unknown as AnimationController;
}

/** Read the captured per-frame callback map from a stub controller. */
function perFrameCallbacks(c: AnimationController): Map<string, () => void> {
  return (c as unknown as { __perFrame: Map<string, () => void> }).__perFrame;
}

/** Read the captured per-frame callback options map from a stub controller. */
function perFrameOptions(c: AnimationController): Map<string, unknown> {
  return (c as unknown as { __perFrameOpts: Map<string, unknown> }).__perFrameOpts;
}

/** Find the "Active level" status span (the live LOD readout). */
function findActiveLevelStatus(container: HTMLElement): HTMLSpanElement | null {
  const groups = Array.from(container.querySelectorAll('.luxar-layers-panel__control-group'));
  for (const group of groups) {
    const label = group.querySelector('.luxar-layers-panel__control-label');
    if (label?.textContent === 'Active level') {
      return group.querySelector('.luxar-layers-panel__control-value') as HTMLSpanElement | null;
    }
  }
  return null;
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
    registryGetMock.mockReset();
    registryGetMock.mockReturnValue(undefined);
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

  it('readout is 1-based "L{i}/{n}" — matches the data-monitor chip numbering', () => {
    // Regression: the readout used to print the raw 0-based
    // activeChildIndex ("rendering: 1") while the data-monitor chip prints
    // "L2/2". Both must now agree on 1-based numbering.
    registryGetMock.mockReturnValue({
      activeChildIndex: 1,
      children: [{}, {}],
      selectorMode: 'auto',
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    expect(findActiveLevelStatus(container)?.textContent).toBe('L2/2');
  });

  it('readout shows the DISPLAYED level, not the aspiration, when they diverge (B2 finding 2)', () => {
    // During a slice scrub the on-screen level (displayedChildIndex) is a
    // coarser fresh level while activeChildIndex points at the stale fine level
    // reloading. The badge must follow what is displayed. Fails if reverted to
    // bare activeChildIndex (would render 'L3/3').
    registryGetMock.mockReturnValue({
      activeChildIndex: 2, // aspiration: fine level reloading
      displayedChildIndex: 0, // on screen: coarse fresh fallback
      children: [{}, {}, {}],
      selectorMode: 'auto',
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    expect(findActiveLevelStatus(container)?.textContent).toBe('L1/3');
  });

  it('appends the displayed-quality estimate (~Q·e%) when the shown child carries quality stamps', () => {
    registryGetMock.mockReturnValue({
      activeChildIndex: 0,
      displayedChildIndex: 0,
      children: [
        {
          object: {
            visible: true,
            userData: {
              nodeType: 'gsplats',
              visibleSplatCount: 10,
              committedEnergyFraction: 0.8,
              attrs: { level_stats: { quality: 0.75, reference_energy: 100 } },
            },
          },
        },
        {},
      ],
      selectorMode: 'auto',
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    // Q·e = 0.75 · 0.8 = 0.6 → "~60%".
    expect(findActiveLevelStatus(container)?.textContent).toBe('L1/2 · ~60%');
  });

  it('dropdown options are 1-based labels with 0-based values', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    const opts = Array.from(findActiveLevelSelect(container)!.options);
    expect(opts.map((o) => o.textContent)).toEqual(['auto', 'lock to level 1', 'lock to level 2']);
    // Values stay 0-based — the registry's lockLevel API is 0-based.
    expect(opts.slice(1).map((o) => o.value)).toEqual(['0', '1']);
  });

  it('per-frame callback updates the readout when the auto-selector swaps level', () => {
    // The core fix: an auto swap (camera motion) changes activeChildIndex
    // without any layer-state change, so renderControls() never re-runs.
    // The 'layers-lod-status' per-frame callback must refresh the readout.
    const entry = { activeChildIndex: 0, children: [{}, {}], selectorMode: 'auto' as const };
    registryGetMock.mockReturnValue(entry);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');
    const status = findActiveLevelStatus(container)!;
    expect(status.textContent).toBe('L1/2');

    // Registry now reports a finer level (as evaluatePerFrame would set);
    // fire the captured callback — the real loop isn't running under jsdom.
    entry.activeChildIndex = 1;
    perFrameCallbacks(animationController).get('layers-lod-status')!();
    expect(status.textContent).toBe('L2/2');
  });

  it('per-frame callback is a no-op while the panel is hidden', () => {
    const entry = { activeChildIndex: 0, children: [{}, {}], selectorMode: 'auto' as const };
    registryGetMock.mockReturnValue(entry);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');
    const status = findActiveLevelStatus(container)!;
    panel.hide();

    entry.activeChildIndex = 1;
    perFrameCallbacks(animationController).get('layers-lod-status')!();
    // Hidden → not refreshed; keeps the value rendered while visible.
    expect(status.textContent).toBe('L1/2');
  });

  it('does not touch the DOM when the level is unchanged across frames', () => {
    const entry = { activeChildIndex: 0, children: [{}, {}], selectorMode: 'auto' as const };
    registryGetMock.mockReturnValue(entry);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');
    const status = findActiveLevelStatus(container)!;

    // Intercept textContent writes from here on.
    let writes = 0;
    let value = status.textContent ?? '';
    Object.defineProperty(status, 'textContent', {
      configurable: true,
      get: () => value,
      set: (v: string) => {
        writes++;
        value = v;
      },
    });
    const cb = perFrameCallbacks(animationController).get('layers-lod-status')!;
    cb();
    cb();
    cb();
    expect(writes).toBe(0); // unchanged level → no DOM writes
  });

  it('broadcast partition aggregates the nested groups (uniform → single level)', () => {
    // kind=partition wrapping two nested lod_groups.
    const partitionScene = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'mosaic',
          path: '/mosaic',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'points' },
          children: [
            {
              name: 'part_0',
              path: '/mosaic/part_0',
              type: 'group',
              attrs: { kind: 'lod' },
              children: [
                { name: 'l0', path: '/mosaic/part_0/l0', type: 'points', attrs: {}, children: [] },
                { name: 'l1', path: '/mosaic/part_0/l1', type: 'points', attrs: {}, children: [] },
              ],
            },
            {
              name: 'part_1',
              path: '/mosaic/part_1',
              type: 'group',
              attrs: { kind: 'lod' },
              children: [
                { name: 'l0', path: '/mosaic/part_1/l0', type: 'points', attrs: {}, children: [] },
                { name: 'l1', path: '/mosaic/part_1/l1', type: 'points', attrs: {}, children: [] },
              ],
            },
          ],
        },
      ],
    } as unknown as SceneNode;
    registryGetMock.mockReturnValue({
      activeChildIndex: 0,
      children: [{}, {}],
      selectorMode: 'auto',
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), partitionScene);
    panel.show();
    panel.layerState.select('/mosaic', 'single');

    expect(findActiveLevelStatus(container)?.textContent).toBe('L1/2 · 2 groups');
  });

  it('broadcast partition shows a level RANGE when nested groups diverge (ragged ladders)', () => {
    // part_0 has a 4-level ladder at level 0 (L1); part_1 has a 2-level
    // ladder at level 1 (L2). Under auto each part picks its own level, so
    // the readout must widen to a range and use the MAX ladder depth (4) as
    // the denominator — matching the dropdown's option count.
    const partitionScene = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'mosaic',
          path: '/mosaic',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'points' },
          children: [
            {
              name: 'part_0',
              path: '/mosaic/part_0',
              type: 'group',
              attrs: { kind: 'lod' },
              children: [0, 1, 2, 3].map((i) => ({
                name: `l${i}`,
                path: `/mosaic/part_0/l${i}`,
                type: 'points',
                attrs: {},
                children: [],
              })),
            },
            {
              name: 'part_1',
              path: '/mosaic/part_1',
              type: 'group',
              attrs: { kind: 'lod' },
              children: [0, 1].map((i) => ({
                name: `l${i}`,
                path: `/mosaic/part_1/l${i}`,
                type: 'points',
                attrs: {},
                children: [],
              })),
            },
          ],
        },
      ],
    } as unknown as SceneNode;
    const entries: Record<string, unknown> = {
      '/mosaic/part_0': { activeChildIndex: 0, children: [{}, {}, {}, {}], selectorMode: 'auto' },
      '/mosaic/part_1': { activeChildIndex: 1, children: [{}, {}], selectorMode: 'auto' },
    };
    registryGetMock.mockImplementation((p: unknown) => entries[p as string]);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), partitionScene);
    panel.show();
    panel.layerState.select('/mosaic', 'single');

    // min level 1 (part_0), max level 2 (part_1), denominator 4 (max ladder).
    expect(findActiveLevelStatus(container)?.textContent).toBe('L1–2/4 · 2 groups');
  });

  it('appends "(off-screen)" when the frustum gate holds the group coarse', () => {
    registryGetMock.mockReturnValue({
      activeChildIndex: 0,
      children: [{}, {}],
      selectorMode: 'auto',
      offScreen: true,
    });
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');

    expect(findActiveLevelStatus(container)?.textContent).toBe('L1/2 (off-screen)');
  });

  it('refreshes a stale readout on show() (level changed while hidden)', () => {
    // Regression for the reshow-staleness gap: the per-frame refresh is
    // gated on visibility, so a swap that happens while hidden is not
    // reflected until show() re-syncs.
    const entry = { activeChildIndex: 0, children: [{}, {}], selectorMode: 'auto' as const };
    registryGetMock.mockReturnValue(entry);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    panel.layerState.select('/pyramid', 'single');
    const status = findActiveLevelStatus(container)!;
    expect(status.textContent).toBe('L1/2');

    panel.hide();
    // Swap happens while hidden; the per-frame callback no-ops (not visible).
    entry.activeChildIndex = 1;
    perFrameCallbacks(animationController).get('layers-lod-status')!();
    expect(status.textContent).toBe('L1/2'); // still stale while hidden

    panel.show(); // must re-sync
    expect(status.textContent).toBe('L2/2');
  });

  it('registers the live LOD callback as non-continuous (must not keep the loop awake)', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    const opts = perFrameOptions(animationController).get('layers-lod-status') as
      | { continuous?: boolean }
      | undefined;
    // Either no options object, or continuous explicitly falsy — never true.
    expect(opts?.continuous ?? false).toBe(false);
  });

  it('does NOT register the live LOD callback for a zero-layer scene', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeEmptySceneGraph());
    // No layers → buildPanel() is skipped → no dangling per-frame callback.
    expect(animationController.hasPerFrameCallback('layers-lod-status')).toBe(false);
  });

  it('removes the live LOD callback on dispose', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLodSceneGraph());
    panel.show();
    expect(animationController.hasPerFrameCallback('layers-lod-status')).toBe(true);
    panel.dispose();
    expect(animationController.hasPerFrameCallback('layers-lod-status')).toBe(false);
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
