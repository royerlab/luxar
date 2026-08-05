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
 * - per-row load-failure badge (`setFailedLoadsProvider`): exact-path and
 *   descendant-prefix mapping, deterministic reason tooltip, recovery /
 *   null-clear, row-rebuild persistence, and reason/count refresh
 *
 * `materialManager` is mocked because it would touch shader compilation
 * (WebGL); `showToast` is mocked so we can observe the empty-scene path.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { AnimationController } from '../../../../scene/animation/animation-controller';
import type { FailedLoadsProviderPort } from '../../../../data/scene-loader-monitor-port';

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

// `getColormapTexture` builds a real DataTexture; the panel only ever hands
// the result to `material.updateColormapTexture`, so an opaque stub is enough.
// It must be TRUTHY: `applyColormap` treats a null texture as "no colormap
// could be applied", which is the fail-closed path — a null-returning mock
// silently made every colormap-ON test exercise the suppressed branch.
vi.mock('../../../../rendering/colormap-textures', () => ({
  getColormapTexture: vi.fn((name?: string) => (name ? { isTexture: true, name } : null)),
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
import {
  ABSORPTION_DEFAULT_MAX,
  absorptionSliderRange,
} from '../../../../ui/layers/absorption-range';

/**
 * Normalised thumb position for a κ value on a log track — the inverse of
 * `LabeledSlider`'s own mapping, so tests can drive the real input. Position
 * 0 is the dedicated zero stop, so the geometric span starts one step in.
 */
const LOG_TRACK_GAP = 0.001;
function logPosition(value: number, range: { min: number; max: number }): number {
  const t = Math.log(value / range.min) / Math.log(range.max / range.min);
  return LOG_TRACK_GAP + t * (1 - LOG_TRACK_GAP);
}

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

function makeLayeredSceneGraph(
  leafType: 'points' | 'gsplats' | 'lines' | 'mesh' = 'points',
  extraLeafAttrs: Record<string, unknown> = {}
): SceneNode {
  // Single layered data node — mirrors what the Python API emits
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
        type: leafType,
        attrs: { layer: true, type: leafType, ...extraLeafAttrs },
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
      { continuous?: boolean } | undefined;
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

describe('LayersPanel — blend select drives the leaf material', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
  });

  /** Find the blend <select> by its 'Blend' control-group label. */
  function findBlendSelect(root: HTMLElement): HTMLSelectElement | null {
    const groups = Array.from(root.querySelectorAll('.luxar-layers-panel__control-group'));
    for (const group of groups) {
      const label = group.querySelector('.luxar-layers-panel__control-label');
      if (label?.textContent === 'Blend') {
        return group.querySelector('.luxar-layers-panel__select') as HTMLSelectElement | null;
      }
    }
    return null;
  }

  it('change event applies the new mode to the leaf material via applyBlendingMode', () => {
    // Minimal LuxarMaterial-shaped stub: isLuxarMaterial checks
    // updateIntensity + updateGamma; applyComposed also calls
    // updateOpacity/updateOffset; getLeafMaterial clones on first use
    // (clone returns the same stub so the spy survives).
    const applyBlendingMode = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud'; // getMesh resolves the layer path by object name
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph());
    panel.show();
    panel.layerState.select('/cloud', 'single');
    applyBlendingMode.mockClear(); // drop init-time composed applies

    const select = findBlendSelect(container);
    expect(select).not.toBeNull();
    select!.value = 'max';
    select!.dispatchEvent(new Event('change', { bubbles: true }));

    // The real UI path (select change → applyToSelected → applyComposed)
    // must reach the material's own applyBlendingMode with the new mode.
    expect(applyBlendingMode).toHaveBeenCalledWith('max');
    expect(panel.layerState.getLayer('/cloud')!.blendingMode).toBe('max');
  });

  it('colormap toggle re-syncs the range slider, so the first drag cannot revert the window', () => {
    // The colormap select re-defaults the display window (the window means a
    // different thing on each side of the toggle), but it runs with
    // `controlsInteracting = true`, which suppresses the state-change
    // re-render. RangeSlider emits values parsed from its own <input>
    // elements, so if the handler doesn't re-render, the thumbs keep the OLD
    // window and the first drag writes it back — silently reverting the
    // re-default (colormap ON → a [0, 1] window on amplitudes = the #522
    // near-black; OFF → the scalar range re-applied as a colour gain).
    const stubMat: Record<string, unknown> = {
      userData: {},
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateColormapTexture: vi.fn(),
      updateScalarRange: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);
    const geometry = new THREE.BufferGeometry();
    // The C1 fail-closed guard needs real scalar data behind a points leaf,
    // else the colormap is suppressed and the window correctly stays identity
    // (covered by the next test).
    geometry.userData.hasScalars = true;
    const mesh = new THREE.Mesh(geometry, stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'points';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'cloud',
          path: '/cloud',
          type: 'points',
          attrs: {
            layer: true,
            type: 'points',
            has_scalars: true,
            scalar_data_range: [0.0001, 0.02],
            color_data_range: [0.2, 0.6],
          },
          children: [],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const readSliderInputs = () =>
      Array.from(container.querySelectorAll('.luxar-range-slider__input')).map((el) =>
        Number((el as HTMLInputElement).value)
      );

    // Direct colour to start: identity window, and the widget agrees.
    expect(panel.layerState.getLayer('/cloud')!.displayMax).toBeCloseTo(1, 5);

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    );
    expect(cmSelect).toBeDefined();
    cmSelect!.value = 'viridis';
    cmSelect!.dispatchEvent(new Event('change', { bubbles: true }));

    // State moved to the scalar window…
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.displayMin).toBeCloseTo(0.0001, 6);
    expect(layer.displayMax).toBeCloseTo(0.02, 6);
    // …and so did the widget — otherwise the next drag emits the stale [0, 1].
    const [low, high] = readSliderInputs();
    expect(low).toBeCloseTo(layer.displayMin, 6);
    expect(high).toBeCloseTo(layer.displayMax, 6);
  });

  it('a suppressed colormap keeps the direct-colour window (no scalar gain on RGB)', () => {
    // The dropdown is offered whenever a layer *might* take a colormap (any
    // group layer does), but the C1 fail-closed guard suppresses it on a leaf
    // with no scalar data bound. That layer keeps rendering DIRECT COLOUR, so
    // moving its window to the scalar range would apply e.g. a 50× gain to
    // authored RGB — the exact contrast stretch this branch removes.
    const stubMat: Record<string, unknown> = {
      userData: {},
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateColormapTexture: vi.fn(),
      updateScalarRange: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);
    // NO `geometry.userData.hasScalars` stamp → the guard suppresses.
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'points';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'cloud',
          path: '/cloud',
          type: 'points',
          attrs: {
            layer: true,
            type: 'points',
            has_scalars: true,
            scalar_data_range: [0.0001, 0.02],
          },
          children: [],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    );
    cmSelect!.value = 'viridis';
    cmSelect!.dispatchEvent(new Event('change', { bubbles: true }));

    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.displayMin).toBeCloseTo(0, 6);
    expect(layer.displayMax).toBeCloseTo(1, 6);
    // The rejected palette is dropped from the layer state too — keeping it
    // would leave contradictory state (`colormap` set, `scalarWindow` false)
    // that lies to the dropdown + legend and mis-keys the next toggle.
    expect(layer.colormap).toBeUndefined();
    expect(layer.scalarWindow).toBe(false);
    // …and the material never got a scalar LUT.
    expect(stubMat.updateColormapTexture).not.toHaveBeenCalledWith(expect.anything());
  });

  it('keeps a colormap pick when the layer meshes have not streamed in yet', () => {
    // Partition parts and LOD levels stream in over time, so a layer can have
    // NO reachable leaf material when the user picks a palette. That is not a
    // C1 guard suppression: reverting the pick would fight the user mid-load
    // and desync the panel from the LUT an authored-colormap part renders with
    // once it arrives (a later slider drag would then push the identity window
    // into a colormap-active material — the #522 near-black).
    const rootGroup = new THREE.Group(); // deliberately empty: nothing loaded

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'cloud',
          path: '/cloud',
          type: 'points',
          attrs: {
            layer: true,
            type: 'points',
            has_scalars: true,
            scalar_data_range: [0.0001, 0.02],
          },
          children: [],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    )!;
    cmSelect.value = 'viridis';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));

    // The pick sticks — state moves to the scalar window instead of snapping
    // back to "(direct colors)".
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.colormap).toBe('viridis');
    expect(layer.scalarWindow).toBe(true);
    expect(layer.displayMin).toBeCloseTo(0.0001, 6);
    expect(layer.displayMax).toBeCloseTo(0.02, 6);
  });

  /**
   * Stub material whose `updateColormapTexture` emulates the real one: a LUT
   * toggles the `USE_COLORMAP` define, which `applyColorAdjustments` routes
   * on (scalar window vs colour GOG).
   */
  function makeColormapRoutingStub() {
    const stubMat: Record<string, unknown> = {
      userData: {},
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {} as Record<string, unknown>,
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateScalarRange: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.updateColormapTexture = vi.fn((tex: unknown) => {
      const defines = stubMat.defines as Record<string, unknown>;
      if (tex) defines.USE_COLORMAP = '';
      else delete defines.USE_COLORMAP;
    });
    stubMat.clone = vi.fn(() => stubMat);
    return stubMat;
  }

  it('a MIXED group layer keeps the scalar window off its direct-colour leaves', () => {
    // A group layer over one scalar-backed leaf and one scalar-less leaf:
    // picking a colormap moves the LAYER window to the scalar range because
    // the scalar leaf accepted the LUT — but the C1 guard keeps the other
    // leaf on direct colour. That leaf must get the IDENTITY, not the scalar
    // window applied as a ~50× colour gain (the contrast stretch this branch
    // removes).
    const scalarMat = makeColormapRoutingStub();
    const rgbMat = makeColormapRoutingStub();

    const scalarGeom = new THREE.BufferGeometry();
    scalarGeom.userData.hasScalars = true;
    const scalarMesh = new THREE.Mesh(scalarGeom, scalarMat as unknown as THREE.Material);
    scalarMesh.name = '/g/scalar';
    scalarMesh.userData.nodeType = 'points';
    // NO hasScalars stamp → the C1 guard suppresses the LUT on this leaf.
    const rgbMesh = new THREE.Mesh(new THREE.BufferGeometry(), rgbMat as unknown as THREE.Material);
    rgbMesh.name = '/g/rgb';
    rgbMesh.userData.nodeType = 'points';
    const rootGroup = new THREE.Group();
    rootGroup.add(scalarMesh);
    rootGroup.add(rgbMesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'g',
          path: '/g',
          type: 'group',
          attrs: { layer: true },
          children: [
            {
              name: 'scalar',
              path: '/g/scalar',
              type: 'points',
              attrs: { type: 'points', has_scalars: true, scalar_data_range: [0.0001, 0.02] },
              children: [],
            },
            {
              name: 'rgb',
              path: '/g/rgb',
              type: 'points',
              attrs: { type: 'points', color_data_range: [0.2, 0.6] },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/g', 'single');

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    );
    cmSelect!.value = 'viridis';
    cmSelect!.dispatchEvent(new Event('change', { bubbles: true }));

    // The layer window moved to the scalar range (the scalar leaf accepted)…
    const layer = panel.layerState.getLayer('/g')!;
    expect(layer.displayMax).toBeCloseTo(0.02, 6);
    // …the scalar leaf renders through the LUT windowed on that range…
    const scalarRangeCalls = (scalarMat.updateScalarRange as ReturnType<typeof vi.fn>).mock.calls;
    expect(scalarRangeCalls.length).toBeGreaterThan(0);
    const [sMin, sMax] = scalarRangeCalls[scalarRangeCalls.length - 1];
    expect(sMin).toBeCloseTo(0.0001, 6);
    expect(sMax).toBeCloseTo(0.02, 6);
    // …while the direct-colour leaf got the identity, not gain ≈ 50.
    expect(rgbMat.updateIntensity).toHaveBeenLastCalledWith(1);
    expect(rgbMat.updateOffset).toHaveBeenLastCalledWith(0);
  });

  it('switching between two active palettes keeps a user-adjusted scalar window', () => {
    // Re-defaulting the window is for the off↔on MODE flip only — the
    // rendered value is the same scalar on both sides of viridis → plasma,
    // so a window the user dialled in must survive the palette change.
    const stubMat = makeColormapRoutingStub();
    const geometry = new THREE.BufferGeometry();
    geometry.userData.hasScalars = true;
    const mesh = new THREE.Mesh(geometry, stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'points';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'cloud',
          path: '/cloud',
          type: 'points',
          attrs: {
            layer: true,
            type: 'points',
            has_scalars: true,
            scalar_data_range: [0.0001, 0.02],
          },
          children: [],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    )!;
    cmSelect.value = 'viridis';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    expect(panel.layerState.getLayer('/cloud')!.displayMax).toBeCloseTo(0.02, 6);

    // The user narrows the window…
    panel.layerState.setDisplayRange('/cloud', 0.001, 0.01);

    // …and a palette swap keeps it.
    cmSelect.value = 'plasma';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    const after = panel.layerState.getLayer('/cloud')!;
    expect(after.displayMin).toBeCloseTo(0.001, 6);
    expect(after.displayMax).toBeCloseTo(0.01, 6);

    // Switching OFF is a mode flip and re-defaults to the identity.
    cmSelect.value = '';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    expect(panel.layerState.getLayer('/cloud')!.displayMin).toBeCloseTo(0, 6);
    expect(panel.layerState.getLayer('/cloud')!.displayMax).toBeCloseTo(1, 6);
  });

  it('a group layer whose colormap lives on a DESCENDANT toggles by effective mode', () => {
    // The wrapper has no `colormap` attr of its own, but the layer already
    // windows a scalar (`scalarWindow` true from the descendant LUT). The
    // toggle handler must key its off↔on detection on that effective mode:
    // keying on the wrapper's attr misreads a palette pick as off→on (wiping
    // the user's window) and misses the on→off flip entirely, stranding the
    // layer on a scalar window its (now direct-colour) leaves route to the
    // identity — an inert display slider.
    const stubMat = makeColormapRoutingStub();
    const geometry = new THREE.BufferGeometry();
    geometry.userData.hasScalars = true;
    const mesh = new THREE.Mesh(geometry, stubMat as unknown as THREE.Material);
    mesh.name = '/g/p0';
    mesh.userData.nodeType = 'points';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'g',
          path: '/g',
          type: 'group',
          attrs: { layer: true },
          children: [
            {
              name: 'p0',
              path: '/g/p0',
              type: 'points',
              attrs: {
                type: 'points',
                has_scalars: true,
                colormap: 'gray',
                scalar_data_range: [0.0001, 0.02],
              },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/g', 'single');

    const before = panel.layerState.getLayer('/g')!;
    // The wrapper carries no `colormap` attr, but the layer state surfaces the
    // descendant palette so the dropdown and legend reflect the rendered mode
    // (and "(direct colors)" is selectable as an off-switch).
    expect(before.colormap).toBe('gray');
    expect(before.scalarWindow).toBe(true);
    expect(before.displayMax).toBeCloseTo(0.02, 6);

    // The user narrows the window…
    panel.layerState.setDisplayRange('/g', 0.001, 0.01);

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    )!;

    // …and picking a palette is scalar→scalar, NOT off→on: the window survives.
    cmSelect.value = 'plasma';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    expect(panel.layerState.getLayer('/g')!.displayMin).toBeCloseTo(0.001, 6);
    expect(panel.layerState.getLayer('/g')!.displayMax).toBeCloseTo(0.01, 6);

    // Selecting "(direct colors)" IS the on→off flip: identity window,
    // direct-colour mode.
    cmSelect.value = '';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    const after = panel.layerState.getLayer('/g')!;
    expect(after.scalarWindow).toBe(false);
    expect(after.displayMin).toBeCloseTo(0, 6);
    expect(after.displayMax).toBeCloseTo(1, 6);
  });

  it('a kind=partition layer overrides a blending mode stamped on its own parts', () => {
    // Regression (gallery demo): `graft_gsplat_node` used to re-stamp
    // blending_mode on every grafted part. blending_mode is
    // nearest-setter-wins, so each part SHADOWED the wrapper and the layer's
    // single Blend control did nothing — flat/stream/levels layers switched,
    // tiles/overview/adaptive did not. Inside a layer's subtree the LAYER owns
    // the mode, so a part's authored copy must not win.
    const mats = ['/tiles/part_0', '/tiles/part_1'].map((name) => {
      const applyBlendingMode = vi.fn();
      const stubMat: Record<string, unknown> = {
        userData: { blendingMode: 'volumetric' },
        uniforms: { uOpacity: { value: 1.0 } },
        defines: {},
        updateIntensity: vi.fn(),
        updateOffset: vi.fn(),
        updateGamma: vi.fn(),
        updateOpacity: vi.fn(),
        applyBlendingMode,
      };
      stubMat.clone = vi.fn(() => stubMat);
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
      mesh.name = name;
      mesh.userData.nodeType = 'gsplats';
      return { mesh, applyBlendingMode };
    });
    const rootGroup = new THREE.Group();
    for (const { mesh } of mats) rootGroup.add(mesh);

    // The layer is the kind=partition WRAPPER; the parts are plain nodes that
    // (on legacy scenes) carry their own stamped blending_mode.
    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'tiles',
          path: '/tiles',
          type: 'group',
          attrs: {
            layer: true,
            kind: 'partition',
            display_type: 'gsplats',
            blending_mode: 'volumetric',
          },
          children: mats.map(({ mesh }, i) => ({
            name: `part_${i}`,
            path: mesh.name,
            type: 'gsplats',
            attrs: { type: 'gsplats', blending_mode: 'volumetric' },
            children: [],
          })),
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/tiles', 'single');
    for (const { applyBlendingMode } of mats) applyBlendingMode.mockClear();

    const select = findBlendSelect(container);
    expect(select).not.toBeNull();
    select!.value = 'max';
    select!.dispatchEvent(new Event('change', { bubbles: true }));

    // EVERY part follows the layer — not its own stamped 'volumetric'.
    for (const { applyBlendingMode } of mats) {
      expect(applyBlendingMode).toHaveBeenCalledWith('max');
    }
    expect(panel.layerState.getLayer('/tiles')!.blendingMode).toBe('max');
  });

  /**
   * Find the Absorption slider control group by its label. LabeledSlider
   * labels concatenate the name span with the value readout
   * ("Absorption1.00"), so match the first span, not the whole label.
   */
  function findAbsorptionGroup(root: HTMLElement): HTMLElement | null {
    return findControlGroup(root, 'Absorption');
  }

  /** Find a control group by its label text. */
  function findControlGroup(root: HTMLElement, labelText: string): HTMLElement | null {
    const groups = Array.from(root.querySelectorAll('.luxar-layers-panel__control-group'));
    for (const group of groups) {
      const label = group.querySelector('.luxar-layers-panel__control-label span');
      if (label?.textContent === labelText) return group as HTMLElement;
    }
    return null;
  }

  /** A mesh leaf whose material records the three shading setters. */
  function mountMeshLayer(
    container: HTMLElement,
    animationController: AnimationController,
    blendingMode = 'opaque'
  ) {
    const calls = {
      ambient: vi.fn(),
      shadeExponent: vi.fn(),
      alphaCutoff: vi.fn(),
      pickAlphaCutoff: vi.fn(),
    };
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
      updateAmbient: calls.ambient,
      updateShadeExponent: calls.shadeExponent,
      updateAlphaCutoff: calls.alphaCutoff,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, {
      setPickMode: vi.fn(),
      setPickSide: vi.fn(),
      updateOpacityUniform: vi.fn(),
      updateAlphaCutoff: calls.pickAlphaCutoff,
    } as unknown as THREE.Material);
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('mesh', { blending_mode: blendingMode }));
    panel.show();
    panel.layerState.select('/cloud', 'single');
    return { panel, calls };
  }

  it('mesh shading sliders: shown for a mesh layer and hidden for every other type', () => {
    // TYPE-gated, which is new for this panel — every other control here is universal
    // or mode-gated. Mesh is the only SHADED geometry type, so on a points layer these
    // three have no uniform to write and would be controls that visibly do nothing.
    mountMeshLayer(container, animationController);
    for (const label of ['Ambient', 'Shade falloff', 'Alpha cutoff']) {
      const group = findControlGroup(container, label);
      expect(group, `${label} control should exist`).not.toBeNull();
      expect(group!.style.display, `${label} should be visible on a mesh layer`).not.toBe('none');
    }

    // The converse, on a fresh panel over a POINTS layer.
    document.body.innerHTML = '';
    const other = document.createElement('div');
    document.body.appendChild(other);
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);
    const points = new THREE.Points(
      new THREE.BufferGeometry(),
      stubMat as unknown as THREE.Material
    );
    points.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(points);
    const panel2 = new LayersPanel(other, animationController);
    panel2.initFromScene(rootGroup, makeLayeredSceneGraph('points'));
    panel2.show();
    panel2.layerState.select('/cloud', 'single');
    for (const label of ['Ambient', 'Shade falloff', 'Alpha cutoff']) {
      expect(findControlGroup(other, label)!.style.display, `${label} on points`).toBe('none');
    }
  });

  it('mesh Alpha cutoff is gated on the MODE as well as the type', () => {
    // Narrower than the other two: the cutout only exists in `opaque`, so in any other
    // mesh mode the threshold is read by no branch of the fragment shader. And the gate
    // must move on the dropdown CLICK, not on the next selection refresh.
    mountMeshLayer(container, animationController);
    const cutoff = findControlGroup(container, 'Alpha cutoff')!;
    const ambient = findControlGroup(container, 'Ambient')!;
    expect(cutoff.style.display).not.toBe('none');

    const select = findBlendSelect(container)!;
    select.value = 'additive';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(cutoff.style.display).toBe('none');
    // ...while the two that apply in every mesh mode stay put.
    expect(ambient.style.display).not.toBe('none');

    select.value = 'opaque';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(cutoff.style.display).not.toBe('none');
  });

  it('dragging each mesh slider reaches its material setter with the slider value', () => {
    const { panel, calls } = mountMeshLayer(container, animationController);

    const drag = (label: string, value: number): void => {
      const group = findControlGroup(container, label)!;
      const input = group.querySelector('input[type="range"]') as HTMLInputElement;
      input.value = String(value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    };

    drag('Ambient', 0.7);
    expect(calls.ambient).toHaveBeenCalledWith(expect.closeTo(0.7, 6));
    expect(panel.layerState.getLayer('/cloud')!.ambient).toBeCloseTo(0.7, 6);

    drag('Shade falloff', 2.5);
    expect(calls.shadeExponent).toHaveBeenCalledWith(expect.closeTo(2.5, 6));
    expect(panel.layerState.getLayer('/cloud')!.shadeExponent).toBeCloseTo(2.5, 6);

    drag('Alpha cutoff', 0.8);
    expect(calls.alphaCutoff).toHaveBeenCalledWith(expect.closeTo(0.8, 6));
    expect(panel.layerState.getLayer('/cloud')!.alphaCutoff).toBeCloseTo(0.8, 6);
  });

  it('the cutoff drag also reaches the PICK material, so a dissolved region stops being hoverable', () => {
    // The pick pass applies the IDENTICAL cutout (§6.5). A threshold that moved on
    // screen but not in the pick buffer would leave a freshly-dissolved region still
    // hoverable — the exact defect the shared cutout exists to prevent, arriving
    // through the panel instead of through the loader.
    const { calls } = mountMeshLayer(container, animationController);
    const group = findControlGroup(container, 'Alpha cutoff')!;
    const input = group.querySelector('input[type="range"]') as HTMLInputElement;
    input.value = '0.9';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(calls.pickAlphaCutoff).toHaveBeenCalledWith(expect.closeTo(0.9, 6));
  });

  it('a mesh inheriting `volumetric` reports the RESOLVED mode, so the panel matches the render', () => {
    // The UI/render mismatch this guards, raised in review of #1271. `volumetric` has no
    // meaning for a zero-thickness surface, so the mesh material maps it to `opaque` and
    // stamps the RESOLVED mode. With the unresolved value in `LayerInfo` the panel
    // disagreed with the shader in two visible ways AT ONCE: it showed Absorption (which
    // no mesh shader reads) and HID Alpha cutoff exactly when the cutout was active.
    // The pick pass reads the material's resolved mode, so it was right and the UI wasn't.
    const { panel } = mountMeshLayer(container, animationController, 'volumetric');

    // Stored resolved, so every consumer — the Blend dropdown's own value included —
    // sees what renders.
    expect(panel.layerState.getLayer('/cloud')!.blendingMode).toBe('opaque');
    expect(findBlendSelect(container)!.value).toBe('opaque');
    // Alpha cutoff visible (the cutout IS active); Absorption hidden (nothing reads it).
    expect(findControlGroup(container, 'Alpha cutoff')!.style.display).not.toBe('none');
    expect(findAbsorptionGroup(container)!.style.display).toBe('none');
  });

  it('explicitly picking `volumetric` on a mesh snaps back to the mode that renders', () => {
    // Same invariant through the dropdown rather than through inheritance. Resolving on
    // WRITE is what makes the control honest: the surface cannot do volumetric, so the
    // panel must not claim it does.
    mountMeshLayer(container, animationController);
    const select = findBlendSelect(container)!;
    select.value = 'volumetric';
    select.dispatchEvent(new Event('change', { bubbles: true }));

    expect(findControlGroup(container, 'Alpha cutoff')!.style.display).not.toBe('none');
    expect(findAbsorptionGroup(container)!.style.display).toBe('none');
  });

  it('leaves `volumetric` alone on a non-mesh layer', () => {
    // The converse — only mesh maps the mode away. A gsplat layer must keep it, or the
    // resolution would silently disable volumetric rendering for the three types that
    // do implement it.
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('gsplats', { blending_mode: 'volumetric' })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    expect(panel.layerState.getLayer('/cloud')!.blendingMode).toBe('volumetric');
    expect(findAbsorptionGroup(container)!.style.display).not.toBe('none');
  });

  it('absorption slider: hidden outside volumetric, revealed by the mode switch, drives updateAbsorption', () => {
    const updateAbsorption = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption,
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('gsplats'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    // Non-volumetric mode: the κ slider group is hidden.
    const group = findAbsorptionGroup(container);
    expect(group).not.toBeNull();
    expect(group!.style.display).toBe('none');

    // Switching the blend dropdown to volumetric reveals it immediately.
    const select = findBlendSelect(container);
    select!.value = 'volumetric';
    select!.dispatchEvent(new Event('change', { bubbles: true }));
    expect(group!.style.display).not.toBe('none');

    // Dragging the slider reaches the material's updateAbsorption with
    // the COMPOSED κ via the real applyComposed path. The track is LOG
    // (κ spans decades — absorption-range.ts), so the input carries a
    // NORMALISED position: κ = min·(max/min)^t.
    updateAbsorption.mockClear();
    const input = group!.querySelector('input[type="range"]') as HTMLInputElement;
    input.value = String(logPosition(2.5, absorptionSliderRange(1)));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(updateAbsorption).toHaveBeenCalledWith(expect.closeTo(2.5, 6));
    expect(panel.layerState.getLayer('/cloud')!.absorption).toBeCloseTo(2.5, 6);

    // Switching away hides it again.
    select!.value = 'additive';
    select!.dispatchEvent(new Event('change', { bubbles: true }));
    expect(group!.style.display).toBe('none');
  });

  it('absorption slider: the κ track does NOT depend on geometry thickness', () => {
    // τ = κ · rayMass in every geometry family now, from the same normalised
    // ray mass the additive branch emits, so κ ≈ 1 is the anchor everywhere
    // and one fixed track serves every scene.
    //
    // What this guards: the track used to be derived per layer as
    // ABSORPTION_TAU_TARGET/(thickness·chord), because the point and line
    // shaders multiplied τ by a world thickness the gsplat shader had no
    // counterpart for. That is a units conversion, and it could never serve a
    // mixed points→gsplat LOD ladder composing ONE κ over both families. A
    // demo-realistic 1.5e-3-wide line must now get the SAME track as anything
    // else — if it doesn't, the shader convention has drifted apart again.
    const width = 0.0015;
    const updateAbsorption = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption,
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('lines', { max_width: width, blending_mode: 'volumetric' })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    // Drive the track to its far end through the real UI path.
    updateAbsorption.mockClear();
    const group = findAbsorptionGroup(container)!;
    const input = group.querySelector('input[type="range"]') as HTMLInputElement;
    input.value = '1';
    input.dispatchEvent(new Event('input', { bubbles: true }));

    // The far end is the nominal maximum — NOT a width-derived bound
    // (which for 1.5e-3 would have been ~4×10³).
    const pushed = updateAbsorption.mock.calls.at(-1)![0] as number;
    expect(pushed).toBeCloseTo(ABSORPTION_DEFAULT_MAX, 6);
    expect(pushed).toBe(absorptionSliderRange(panel.layerState.getLayer('/cloud')!.absorption).max);
  });

  it('absorption slider: an authored κ BELOW the nominal floor stays on the track (no silent jump on first touch)', () => {
    // The track's floor sits ABSORPTION_LOG_DECADES below its top, so an
    // authored κ beneath that would show its true value in the readout while
    // the thumb could not represent it — and the first input event would
    // silently jump κ up to the floor. absorptionSliderRange lowers the floor
    // onto the live κ to prevent that; this pins the behaviour end to end.
    const authoredKappa = 1e-5; // two decades below the nominal 1e-3 floor
    const width = 6e-4;
    const updateAbsorption = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption,
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('lines', {
        max_width: width,
        blending_mode: 'volumetric',
        absorption: authoredKappa,
      })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    expect(panel.layerState.getLayer('/cloud')!.absorption).toBe(authoredKappa);

    const group = findAbsorptionGroup(container)!;
    const input = group.querySelector('input[type="range"]') as HTMLInputElement;
    // The thumb must be at or above the geometric span's first stop — never
    // pinned to the zero stop or clipped below the floor.
    expect(parseFloat(input.value)).toBeGreaterThanOrEqual(LOG_TRACK_GAP - 1e-12);

    // Touching the slider without moving it must leave κ where it was.
    updateAbsorption.mockClear();
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(updateAbsorption).toHaveBeenLastCalledWith(expect.closeTo(authoredKappa, 12));
    expect(panel.layerState.getLayer('/cloud')!.absorption).toBeCloseTo(authoredKappa, 12);
  });

  it('composeEffective preserves an authored κ on a NON-layer leaf under a layer group', () => {
    // Regression (fail-first vs the pre-fix composeEffective): the
    // non-layer fallback branch omitted `absorption` from its
    // ComposableAttrs, so a leaf-authored κ was silently recomposed to
    // the 1.0 identity — and pushed to the material — by ANY panel
    // interaction (including the applyDisplayRange fan-out at init).
    const updateAbsorption = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption,
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/grp/splats';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    // layer=true GROUP wrapping a non-layer gsplat leaf that authors κ —
    // the standard kind=partition/lod layer shape.
    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'grp',
          path: '/grp',
          type: 'group',
          attrs: { layer: true, blending_mode: 'volumetric' },
          children: [
            {
              name: 'splats',
              path: '/grp/splats',
              type: 'gsplats',
              attrs: { absorption: 0.5 },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/grp', 'single');

    // Drive a panel interaction through the group layer — the composed
    // κ pushed to the leaf material must keep the leaf's authored 0.5
    // (group layer identity 1.0 × leaf 0.5), not wipe it to 1.0.
    updateAbsorption.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.9;
    });
    const grpLayer = panel.layerState.getLayer('/grp')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(grpLayer);
    expect(updateAbsorption).toHaveBeenCalledWith(0.5);
  });

  it('the blending-mode subtree rule does NOT swallow a leaf-authored opacity/gamma/intensity/offset', () => {
    // The subtree rule drops a `blending_mode` authored on a non-layer
    // descendant so the layer's single Blend control wins. It must stay scoped
    // to that ONE attr: the multiplicative attrs still compose and `offset`
    // still sums, so a part's authored values survive. (Mutation-tested: each
    // of these attrs added to the drop list must fail this test.)
    const updateOpacity = vi.fn();
    const updateGamma = vi.fn();
    const updateIntensity = vi.fn();
    const updateOffset = vi.fn();
    const applyBlendingMode = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity,
      updateOffset,
      updateGamma,
      updateOpacity,
      updateAbsorption: vi.fn(),
      applyBlendingMode,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/grp/part_0';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'grp',
          path: '/grp',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          children: [
            {
              name: 'part_0',
              path: '/grp/part_0',
              type: 'gsplats',
              // A part authoring its own everything, including a mode that
              // would otherwise shadow the layer.
              attrs: {
                opacity: 0.5,
                gamma: 2.0,
                intensity: 3.0,
                offset: 0.25,
                blending_mode: 'volumetric',
              },
              children: [],
            },
          ],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/grp', 'single');

    updateOpacity.mockClear();
    updateGamma.mockClear();
    updateIntensity.mockClear();
    updateOffset.mockClear();
    applyBlendingMode.mockClear();

    const grpLayer = panel.layerState.getLayer('/grp')!;
    grpLayer.opacity = 0.4;
    grpLayer.gamma = 1.5;
    grpLayer.blendingMode = 'max';
    // A user pick is explicit — same as the real Blend-select handler — so the
    // wrapper's mode overrides the part's authored one (#1272).
    grpLayer.blendingModeExplicit = true;
    (
      panel as unknown as { applyEngine: { applyBlendingMode(l: unknown): void } }
    ).applyEngine.applyBlendingMode(grpLayer);

    // Multiplicative attrs compose (layer × part); offset sums (0 + 0.25).
    expect(updateOpacity).toHaveBeenCalledWith(0.4 * 0.5);
    expect(updateGamma).toHaveBeenCalledWith(1.5 * 2.0);
    // The layer's display window is the identity (direct colour), so the
    // composed intensity is the part's own 3.0 and the offset its own 0.25.
    expect(updateIntensity).toHaveBeenCalledWith(3.0);
    expect(updateOffset).toHaveBeenCalledWith(0.25);
    // …while the part's own blending_mode is the ONE thing overridden.
    expect(applyBlendingMode).toHaveBeenCalledWith('max');
  });

  it('a panel opacity edit during an in-flight LOD fade rebases the fade snapshot instead of the live uniform', () => {
    // Regression (fail-first vs the pre-fix applyComposed): the LOD fade
    // owns the live opacity uniform while a cross-fade / energy comp is in
    // flight — it re-renders `_lodFadeBase × product` every frame
    // (scene/lod-fade.ts) — so a direct updateOpacity here was clobbered on
    // the next fade frame and the panel edit lost until the fade ended.
    const updateOpacity = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'volumetric' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity,
      updateAbsorption: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    // Mid-fade state as the registry leaves it: authored-opacity snapshot
    // taken, per-node material already owned.
    mesh.userData._lodFadeBase = 1.0;
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('gsplats'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    updateOpacity.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.6;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(layer);
    // The fade snapshot got the new composed value; the live uniform was
    // left to the fade's next frame.
    expect(mesh.userData._lodFadeBase).toBeCloseTo(0.6, 6);
    expect(updateOpacity).not.toHaveBeenCalled();
  });

  it('an opacity edit on a MESH also moves its pick material, so a dissolved surface stops being pickable', () => {
    // The wiring test, not a unit test of either half. A mesh's pick pass computes
    // the SAME coverage as its visual shader — node opacity times per-vertex alpha
    // (spec §6.5) — and in the default `opaque` mode compares it against the cutout
    // threshold. So an opacity edit that dissolves the surface on screen must dissolve
    // it in the pick buffer too.
    //
    // Fail-first check: deleting the `syncMeshPickAppearance` call from
    // `applyComposed` leaves this red while every material-level test stays green,
    // which is exactly the gap that let an unwired `applyMeshShading` pass 865 tests.
    const updateOpacityUniform = vi.fn();
    const pickMat: Record<string, unknown> = {
      setPickMode: vi.fn(),
      setPickSide: vi.fn(),
      updateOpacityUniform,
      updateAlphaCutoff: vi.fn(),
    };
    const visualMat: Record<string, unknown> = {
      userData: { blendingMode: 'opaque' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    visualMat.clone = vi.fn(() => visualMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), visualMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData._layerMaterialCloned = true;
    mesh.userData.nodeType = 'mesh';
    // How `registerNode` leaves it: the pick node hangs off the main node's userData,
    // which is the only handle the panel has to reach the pick material.
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, pickMat as unknown as THREE.Material);
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('mesh'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    updateOpacityUniform.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.3;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(layer);

    expect(updateOpacityUniform).toHaveBeenCalledWith(0.3);
  });

  it('leaves the other three types alone — they have no mesh pick surface to sync', () => {
    // The converse, so the call above cannot be "fixed" by widening it to every type:
    // a points/lines/gsplat pick material derives coverage from its own element data,
    // and pushing a node opacity into it would double-apply.
    const pickMat: Record<string, unknown> = { updateOpacityUniform: vi.fn() };
    const visualMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    visualMat.clone = vi.fn(() => visualMat);

    const points = new THREE.Points(
      new THREE.BufferGeometry(),
      visualMat as unknown as THREE.Material
    );
    points.name = '/cloud';
    points.userData._layerMaterialCloned = true;
    points.userData.nodeType = 'points';
    points.userData.pickNode = new THREE.Mesh(
      points.geometry,
      pickMat as unknown as THREE.Material
    );
    const rootGroup = new THREE.Group();
    rootGroup.add(points);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('points'));
    panel.show();
    panel.layerState.select('/cloud', 'single');
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.3;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(layer);

    expect(pickMat.updateOpacityUniform).not.toHaveBeenCalled();
  });

  it('a mesh opacity edit invalidates the cached pick buffer (stationary-camera hover would otherwise keep stale ids)', () => {
    // The layers panel triggers none of the camera/resize/commit paths that mark
    // the cached pick buffer dirty, so a stationary-camera opacity edit on a mesh
    // would leave the buffer showing the pre-edit coverage. Whenever the sync
    // actually touched a mesh pick material, the panel must mark it dirty.
    const invalidate = vi.fn();
    const pickMat: Record<string, unknown> = {
      setPickMode: vi.fn(),
      setPickSide: vi.fn(),
      updateOpacityUniform: vi.fn(),
      updateAlphaCutoff: vi.fn(),
    };
    const visualMat: Record<string, unknown> = {
      userData: { blendingMode: 'opaque' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    visualMat.clone = vi.fn(() => visualMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), visualMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData._layerMaterialCloned = true;
    mesh.userData.nodeType = 'mesh';
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, pickMat as unknown as THREE.Material);
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.setPickBufferInvalidator(invalidate);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('mesh'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    invalidate.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.3;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(layer);

    expect(invalidate).toHaveBeenCalled();
  });

  it('a mesh alpha-cutoff edit invalidates the cached pick buffer (§6.5 cutout must move in the pick pass too)', () => {
    // The cutout threshold rides to the pick material via `applyMeshAppearance`
    // (not `applyComposed`), so it needs its own invalidation wiring: a cutoff that
    // moved on screen but not in the cached pick buffer would leave a freshly-cut
    // region still hoverable.
    const invalidate = vi.fn();
    const pickMat: Record<string, unknown> = {
      setPickMode: vi.fn(),
      setPickSide: vi.fn(),
      updateOpacityUniform: vi.fn(),
      updateAlphaCutoff: vi.fn(),
    };
    const visualMat: Record<string, unknown> = {
      userData: { blendingMode: 'opaque' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAmbient: vi.fn(),
      updateShadeExponent: vi.fn(),
      updateAlphaCutoff: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    visualMat.clone = vi.fn(() => visualMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), visualMat as unknown as THREE.Material);
    mesh.name = '/cloud';
    mesh.userData._layerMaterialCloned = true;
    mesh.userData.nodeType = 'mesh';
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, pickMat as unknown as THREE.Material);
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.setPickBufferInvalidator(invalidate);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('mesh'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    invalidate.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.alphaCutoff = 0.5;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyMeshAppearance(l: unknown): void } }
    ).applyEngine.applyMeshAppearance(layer);

    expect(invalidate).toHaveBeenCalled();
  });

  it('a non-mesh opacity edit does not touch the pick buffer', () => {
    // The converse: a points/lines/gsplat pick material is not mesh-pick-aware, so
    // the sync is a no-op and there is nothing new to render. Invalidating the
    // buffer anyway would churn an offscreen render on every non-mesh slider drag.
    const invalidate = vi.fn();
    const pickMat: Record<string, unknown> = { updateOpacityUniform: vi.fn() };
    const visualMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
    };
    visualMat.clone = vi.fn(() => visualMat);

    const points = new THREE.Points(
      new THREE.BufferGeometry(),
      visualMat as unknown as THREE.Material
    );
    points.name = '/cloud';
    points.userData._layerMaterialCloned = true;
    points.userData.nodeType = 'points';
    points.userData.pickNode = new THREE.Mesh(
      points.geometry,
      pickMat as unknown as THREE.Material
    );
    const rootGroup = new THREE.Group();
    rootGroup.add(points);

    const panel = new LayersPanel(container, animationController);
    panel.setPickBufferInvalidator(invalidate);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('points'));
    panel.show();
    panel.layerState.select('/cloud', 'single');

    invalidate.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.3;
    });
    const layer = panel.layerState.getLayer('/cloud')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(layer);

    expect(invalidate).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Display-range / gamma routing: colormap (LUT) vs direct-color.
//
// Regression guard for the gamma-on-value fix. When a leaf renders through
// a colormap LUT, the display range (from the authored intensity/offset)
// drives the scalar window (`updateScalarRange`), and the color GOG
// (`updateIntensity`/`updateOffset`) is actively RESET to identity so the
// authored gain does not double-apply — shaping the LUT window AND tinting
// the post-LUT color (#936). Gamma is applied to the value pre-LUT in the
// shader. In direct-color mode the GOG drives intensity/offset as before.
// Gamma is pushed in both modes.
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

describe('LayersPanel — per-row load-failure badge', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
    showToastMock.mockClear();
  });

  /**
   * Minimal stub of the shared failed-loads provider. `set()` mutates the live
   * failed set and `setReason()` mutates a single path's reason, so tests can
   * drive recovery / reason-change through the runtime refresh path.
   */
  function makeFailedProvider(paths: string[], reasons: Record<string, string> = {}) {
    let current = [...paths];
    const provider: FailedLoadsProviderPort = {
      getFailedPaths: () => current,
      retryAll: async () => ({ succeeded: [], failed: [] }),
      getFailedReason: (p: string) => reasons[p],
    };
    return {
      provider,
      set: (next: string[]) => (current = next),
      setReason: (path: string, reason: string) => (reasons[path] = reason),
    };
  }

  /** kind=lod group at /pyramid plus an unrelated leaf sibling at /cloud. */
  function makeGroupPlusSiblingScene(): SceneNode {
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

  function errorBadge(row: HTMLElement | undefined): HTMLElement | null {
    return row?.querySelector('.luxar-layer-row__error') as HTMLElement | null;
  }

  function rowFor(container: HTMLElement, name: string): HTMLElement | undefined {
    return Array.from(container.querySelectorAll<HTMLElement>('.luxar-layer-row')).find(
      (r) => r.querySelector('.luxar-layer-row__name')?.textContent === name
    );
  }

  it('an exact-path failure lights up its row with a reason in the badge label', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const { provider } = makeFailedProvider(['/cloud'], {
      '/cloud': 'Vertex index 3 not found in loaded data',
    });

    panel.setFailedLoadsProvider(provider);

    const row = rowFor(container, 'cloud');
    expect(row?.classList.contains('luxar-layer-row--error')).toBe(true);
    const badge = errorBadge(row);
    expect(badge).not.toBeNull();
    expect(badge?.getAttribute('aria-label')).toContain('Vertex index 3 not found in loaded data');
    expect(badge?.title).toContain('Vertex index 3 not found in loaded data');
  });

  it('falls back to a generic reason when the provider exposes no per-path detail', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    // Provider without getFailedReason (structural DataMonitor-style port).
    const provider: FailedLoadsProviderPort = {
      getFailedPaths: () => ['/cloud'],
      retryAll: async () => ({ succeeded: [], failed: [] }),
    };

    panel.setFailedLoadsProvider(provider);

    const badge = errorBadge(rowFor(container, 'cloud'));
    expect(badge?.getAttribute('aria-label')).toBe(
      'Failed to load — see the data monitor for details'
    );
  });

  it('a group row lights up for a DESCENDANT failure but a string-prefix sibling stays clean', () => {
    // The boundary test: layers `/pyramid` and `/pyramid_hi` share a string
    // prefix, so `startsWith(path)` (no trailing slash) would wrongly light up
    // `/pyramid` for a `/pyramid_hi/...` failure. The `path + '/'` rule must not.
    const scene = {
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
          ],
        },
        {
          name: 'pyramid_hi',
          path: '/pyramid_hi',
          type: 'group',
          attrs: { layer: true, kind: 'lod', display_type: 'points' },
          children: [
            { name: 'leaf', path: '/pyramid_hi/leaf', type: 'points', attrs: {}, children: [] },
          ],
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), scene);
    const { provider } = makeFailedProvider(['/pyramid_hi/leaf'], {
      '/pyramid_hi/leaf': 'network 503',
    });

    panel.setFailedLoadsProvider(provider);

    const hiRow = rowFor(container, 'pyramid_hi');
    const pyramidRow = rowFor(container, 'pyramid');
    // The actual owner lights up…
    expect(hiRow?.classList.contains('luxar-layer-row--error')).toBe(true);
    expect(errorBadge(hiRow)).not.toBeNull();
    // …but the string-prefix sibling stays clean (boundary is `path + '/'`).
    expect(pyramidRow?.classList.contains('luxar-layer-row--error')).toBe(false);
    expect(errorBadge(pyramidRow)).toBeNull();
  });

  it('counts multiple failed descendants of a group in the badge reason', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeGroupPlusSiblingScene());
    const { provider } = makeFailedProvider(['/pyramid/lod_0', '/pyramid/lod_1']);

    panel.setFailedLoadsProvider(provider);

    expect(errorBadge(rowFor(container, 'pyramid'))?.getAttribute('aria-label')).toContain(
      '2 parts failed'
    );
  });

  it('removes the badge once the path leaves the failed set (recovery via per-frame refresh)', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    panel.show();
    const { provider, set } = makeFailedProvider(['/cloud'], { '/cloud': 'boom' });

    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();

    // The path recovers; drive the same per-frame refresh the runtime uses.
    set([]);
    perFrameCallbacks(animationController).get('layers-lod-status')!();

    const row = rowFor(container, 'cloud');
    expect(errorBadge(row)).toBeNull();
    expect(row?.classList.contains('luxar-layer-row--error')).toBe(false);
  });

  it('does not break the row base aria-label while in error', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const { provider } = makeFailedProvider(['/cloud'], { '/cloud': 'boom' });

    panel.setFailedLoadsProvider(provider);

    // The badge is actually present (else this test would pass even if
    // applyRowError did nothing)…
    const row = rowFor(container, 'cloud');
    expect(errorBadge(row)).not.toBeNull();
    // …and the row keeps its base "name (type)" label; the reason lives on
    // the badge, not the row.
    expect(row?.getAttribute('aria-label')).toBe('cloud (points)');
  });

  it('keeps the badge across a row rebuild (resetAllLayers) while the failure persists', () => {
    // renderList() rebuilds rows badge-less; without re-applying, the
    // signature gate would strand the row clean while the failure persists.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const { provider } = makeFailedProvider(['/cloud'], { '/cloud': 'boom' });

    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();

    panel.resetAllLayers(); // rebuilds the row list

    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();
    expect(rowFor(container, 'cloud')?.classList.contains('luxar-layer-row--error')).toBe(true);
  });

  it('setFailedLoadsProvider(null) clears an existing badge (empty-set sentinel)', () => {
    // An empty failed set hashes to the same signature as "no provider"; the
    // null reset sentinel must still force the clear.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const { provider } = makeFailedProvider(['/cloud'], { '/cloud': 'boom' });

    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();

    panel.setFailedLoadsProvider(null);

    const row = rowFor(container, 'cloud');
    expect(errorBadge(row)).toBeNull();
    expect(row?.classList.contains('luxar-layer-row--error')).toBe(false);
  });

  it('refreshes the badge when a still-failing row gains a descendant (count + reason)', () => {
    // Signature folds in each path's reason, so a changed/added reason on a
    // still-failing row re-triggers the refresh — no stale tooltip, no
    // duplicate badge.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeGroupPlusSiblingScene());
    panel.show();
    const { provider, set } = makeFailedProvider(['/pyramid/lod_0'], {
      '/pyramid/lod_0': 'first',
    });

    panel.setFailedLoadsProvider(provider);
    const groupRow = rowFor(container, 'pyramid')!;
    expect(groupRow.querySelectorAll('.luxar-layer-row__error')).toHaveLength(1);
    expect(errorBadge(groupRow)?.getAttribute('aria-label')).not.toContain('parts failed');

    // A second descendant fails; drive the runtime per-frame refresh.
    set(['/pyramid/lod_0', '/pyramid/lod_1']);
    perFrameCallbacks(animationController).get('layers-lod-status')!();

    // Exactly ONE badge (updated in place, not duplicated) with the new count.
    expect(groupRow.querySelectorAll('.luxar-layer-row__error')).toHaveLength(1);
    expect(errorBadge(groupRow)?.getAttribute('aria-label')).toContain('2 parts failed');
  });

  it('refreshes the tooltip when only a REASON changes (path set unchanged)', () => {
    // The signature folds in each path's reason. This pins that behavior for its
    // actual purpose: the failed PATH SET is identical across the two frames, so
    // a paths-only signature would early-return and strand the stale tooltip —
    // only the reason-folded signature re-applies here.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    panel.show();
    const { provider, setReason } = makeFailedProvider(['/cloud'], { '/cloud': 'network 503' });

    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))?.title).toContain('network 503');

    // Same failing path, new reason (e.g. a retry reclassified the cause).
    setReason('/cloud', 'decode error');
    perFrameCallbacks(animationController).get('layers-lod-status')!();

    const badge = errorBadge(rowFor(container, 'cloud'));
    expect(badge?.title).toContain('decode error');
    expect(badge?.getAttribute('aria-label')).toContain('decode error');
  });

  it('the per-frame refresh is gated on visibility and show() catches up', () => {
    // Injected while HIDDEN: setFailedLoadsProvider applies the initial set
    // directly, but the per-frame callback must NOT touch the DOM while hidden,
    // and show() must re-sync to the current set.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    // Panel is hidden (no show()). No initial failures.
    const { provider, set } = makeFailedProvider([]);
    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))).toBeNull();

    // A failure appears while hidden; the per-frame callback must be a no-op.
    set(['/cloud']);
    perFrameCallbacks(animationController).get('layers-lod-status')!();
    expect(errorBadge(rowFor(container, 'cloud'))).toBeNull();

    // Opening the panel catches up to the current failed set.
    panel.show();
    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();
  });

  it('reports the reason of the lexicographically-first descendant (matches.sort())', () => {
    // Two descendants fail, supplied to the stub in REVERSE path order with
    // distinct reasons. The reported reason must be from the lexicographically
    // first path (/pyramid/lod_0), so dropping matches.sort() — which would pick
    // provider-insertion order (/pyramid/lod_1) — fails this test.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeGroupPlusSiblingScene());
    const { provider } = makeFailedProvider(['/pyramid/lod_1', '/pyramid/lod_0'], {
      '/pyramid/lod_0': 'reason-A-first',
      '/pyramid/lod_1': 'reason-B-second',
    });

    panel.setFailedLoadsProvider(provider);

    const label = errorBadge(rowFor(container, 'pyramid'))?.getAttribute('aria-label');
    expect(label).toContain('reason-A-first');
    expect(label).not.toContain('reason-B-second');
  });

  it('detects a change between failure sets that collide under naive path:reason joining', () => {
    // Reasons are arbitrary error text, so a signature built by concatenating
    // `path:reason` and joining with `|` is ambiguous: {'/cloud': 'boom|/pyramid/lod_0:x'}
    // encodes to the same string as {'/cloud': 'boom', '/pyramid/lod_0': 'x'}.
    // The JSON tuple signature must tell them apart, or the transition below
    // early-returns and the pyramid row never gets its badge.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeGroupPlusSiblingScene());
    panel.show();
    const { provider, set, setReason } = makeFailedProvider(['/cloud'], {
      '/cloud': 'boom|/pyramid/lod_0:x',
    });
    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'pyramid'))).toBeNull();

    set(['/cloud', '/pyramid/lod_0']);
    setReason('/cloud', 'boom');
    setReason('/pyramid/lod_0', 'x');
    perFrameCallbacks(animationController).get('layers-lod-status')!();

    expect(errorBadge(rowFor(container, 'pyramid'))).not.toBeNull();
  });

  it('initFromScene drops the previous provider — fresh rows inherit no stale badges', () => {
    // The app injects the provider AFTER initFromScene because initFromScene's
    // clear() resets any prior one. Pin that: re-initializing with a new scene
    // (before the next provider arrives) must not resurrect the old scene's
    // failures, neither immediately nor via the per-frame refresh.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const { provider } = makeFailedProvider(['/cloud'], { '/cloud': 'boom' });
    panel.setFailedLoadsProvider(provider);
    expect(errorBadge(rowFor(container, 'cloud'))).not.toBeNull();

    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    panel.show();
    perFrameCallbacks(animationController).get('layers-lod-status')!();

    const row = rowFor(container, 'cloud');
    expect(errorBadge(row)).toBeNull();
    expect(row?.classList.contains('luxar-layer-row--error')).toBe(false);
  });
});

describe('isColormapActive', () => {
  it('is true only when the USE_COLORMAP define is present', () => {
    expect(isColormapActive(makeRecordingMaterial({ USE_COLORMAP: '' }).mat)).toBe(true);
    expect(isColormapActive(makeRecordingMaterial({}).mat)).toBe(false);
    expect(isColormapActive(makeRecordingMaterial(null).mat)).toBe(false);
  });
});

describe('applyColorAdjustments — colormap vs direct routing', () => {
  it('colormap mode: display range drives the scalar window; color GOG reset to identity', () => {
    const { mat, calls } = makeRecordingMaterial({ USE_COLORMAP: '' });
    // intensity/offset encode display range [0.5, 2.5]:
    //   computeDisplayRange(0.5, -0.25) → { min: 0.5, max: 2.5 }
    applyColorAdjustments(mat, 2.2, 0.5, -0.25);

    expect(calls.gamma).toEqual([2.2]); // gamma still pushed (applied pre-LUT)
    expect(calls.scalarRange).toEqual([[0.5, 2.5]]);
    // The shader always multiplies vColor by uIntensity post-LUT, so the
    // color GOG is actively reset to identity (NOT left stale) — otherwise
    // the authored gain would double-apply, shaping the LUT window AND
    // tinting the mapped color (#936).
    expect(calls.intensity).toEqual([1]);
    expect(calls.offset).toEqual([0]);
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
