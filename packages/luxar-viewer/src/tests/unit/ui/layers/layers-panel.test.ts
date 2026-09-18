// @vitest-environment jsdom
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
import { MESH_DEFAULTS } from '../../../../rendering/materials/mesh/appearance';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';

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
import { getColormapTexture } from '../../../../rendering/colormap-textures';

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

function makeSoundSceneGraph(soundLayer = true, storyVisible = true): SceneNode {
  return {
    name: 'root',
    path: '/',
    type: 'group',
    attrs: {},
    children: [
      {
        name: 'story',
        path: '/story',
        type: 'group',
        attrs: { layer: true, visible: storyVisible },
        children: [
          {
            name: 'hum',
            path: '/story/hum',
            type: 'sound',
            attrs: {
              ...(soundLayer ? { layer: true } : {}),
              gain: 0.5,
              license: 'CC0',
              attribution: 'someone',
              source_url: 'https://example.org/hum',
            },
            children: [],
          },
        ],
      },
    ],
  } as unknown as SceneNode;
}

describe('LayersPanel — sound rows', () => {
  let container: HTMLElement;
  let panel: LayersPanel;
  let port: {
    setNodeMuted: ReturnType<typeof vi.fn<(path: string, muted: boolean) => void>>;
    setNodeGain: ReturnType<typeof vi.fn<(path: string, gain: number) => void>>;
  };

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    panel = new LayersPanel(container, makeAnimationController());
    port = {
      setNodeMuted: vi.fn<(path: string, muted: boolean) => void>(),
      setNodeGain: vi.fn<(path: string, gain: number) => void>(),
    };
    panel.setAudioPort(port);
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    const story = new THREE.Group();
    story.name = '/story';
    const hum = new THREE.Group();
    hum.name = '/story/hum';
    story.add(hum);
    root.add(story);
    panel.initFromScene(root, makeSoundSceneGraph());
  });

  it('renders a sound badge, a gain slider and the provenance tooltip', () => {
    const row = container.querySelector('[data-layer-path="/story/hum"]') as HTMLElement;
    expect(row).not.toBeNull();
    expect(row.querySelector('.luxar-layer-row__badge')?.textContent).toBe('sound');
    const gain = row.querySelector('.luxar-layer-row__gain') as HTMLInputElement;
    expect(gain.value).toBe('0.5');
    expect(row.querySelector('.luxar-layer-row__name')?.getAttribute('title')).toContain(
      'CC0 — someone'
    );
    expect(row.querySelector('.luxar-layer-row__eye')?.getAttribute('title')).toBe('Mute sound');
    // Geometry rows keep their slider-free shape.
    const groupRow = container.querySelector('[data-layer-path="/story"]') as HTMLElement;
    expect(groupRow.querySelector('.luxar-layer-row__gain')).toBeNull();
  });

  it('the eye mutes the node through the port; a parent group eye mutes the subtree', () => {
    const row = container.querySelector('[data-layer-path="/story/hum"]') as HTMLElement;
    (row.querySelector('.luxar-layer-row__eye') as HTMLButtonElement).click();
    expect(port.setNodeMuted).toHaveBeenLastCalledWith('/story/hum', true);
    expect(row.querySelector('.luxar-layer-row__eye')?.getAttribute('title')).toBe('Unmute sound');
    const groupRow = container.querySelector('[data-layer-path="/story"]') as HTMLElement;
    (groupRow.querySelector('.luxar-layer-row__eye') as HTMLButtonElement).click();
    expect(port.setNodeMuted).toHaveBeenLastCalledWith('/story', true);
  });

  it('a parent group eye mutes sound descendants that are not layer rows', () => {
    panel.dispose();
    container.replaceChildren();
    panel = new LayersPanel(container, makeAnimationController());
    panel.setAudioPort(port);
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    const story = new THREE.Group();
    story.name = '/story';
    root.add(story);
    panel.initFromScene(root, makeSoundSceneGraph(false));
    port.setNodeMuted.mockClear();

    const groupRow = container.querySelector('[data-layer-path="/story"]') as HTMLElement;
    (groupRow.querySelector('.luxar-layer-row__eye') as HTMLButtonElement).click();

    expect(port.setNodeMuted).toHaveBeenCalledOnce();
    expect(port.setNodeMuted).toHaveBeenCalledWith('/story', true);
  });

  it('replays authored visibility to audio after sound nodes attach', () => {
    panel.dispose();
    container.replaceChildren();
    panel = new LayersPanel(container, makeAnimationController());
    panel.setAudioPort(port);
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    panel.initFromScene(root, makeSoundSceneGraph(true, false));
    port.setNodeMuted.mockClear();

    panel.pushAudioMutes();

    expect(port.setNodeMuted.mock.calls).toEqual([
      ['/story', true],
      ['/story/hum', false],
    ]);
  });

  it('the gain slider and setLayer({gain}) drive the port and the summary', () => {
    const row = container.querySelector('[data-layer-path="/story/hum"]') as HTMLElement;
    const gain = row.querySelector('.luxar-layer-row__gain') as HTMLInputElement;
    gain.value = '1.25';
    gain.dispatchEvent(new Event('input', { bubbles: true }));
    expect(port.setNodeGain).toHaveBeenLastCalledWith('/story/hum', 1.25);
    port.setNodeGain.mockClear();
    panel.setLayer('/story/hum', { gain: 9 });
    expect(port.setNodeGain).toHaveBeenCalledOnce();
    expect(port.setNodeGain).toHaveBeenCalledWith('/story/hum', 2);
    expect(gain.value).toBe('2');
    const summary = panel.getLayerSummaries().find((l) => l.path === '/story/hum')!;
    expect(summary.type).toBe('sound');
    expect(summary.gain).toBe(2);
    expect(panel.getLayerSummaries().find((l) => l.path === '/story')!.gain).toBeUndefined();
  });

  it('a slider click does not select the row', () => {
    const row = container.querySelector('[data-layer-path="/story/hum"]') as HTMLElement;
    const gain = row.querySelector('.luxar-layer-row__gain') as HTMLInputElement;
    panel.layerState.select('/story', 'single');
    gain.click();
    expect(panel.layerState.getPrimarySelected()?.path).toBe('/story');
  });

  it('right-clicking the gain slider does not open the row menu', () => {
    const gain = container.querySelector<HTMLInputElement>('.luxar-layer-row__gain')!;
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });

    gain.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    expect(document.querySelector('.luxar-context-menu')).toBeNull();
  });
});

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

  it('keeps the rendering controls below the panel, then resets below the safe area', () => {
    const gui = document.createElement('div');
    gui.className = 'luxar-gui';
    document.body.appendChild(gui);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph());
    const panelElement = container.querySelector('.luxar-layers-panel') as HTMLElement;
    panelElement.getBoundingClientRect = vi.fn(() => ({
      bottom: 180,
    })) as unknown as typeof panelElement.getBoundingClientRect;

    panel.show();
    expect(gui.style.top).toBe('188px');

    panel.hide();
    expect(gui.style.top).toContain('20px');
    expect(gui.style.top).toContain('safe-area-inset-top');
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

  it('appends the separate normalized mesh error without energy stamps', () => {
    registryGetMock.mockReturnValue({
      activeChildIndex: 0,
      children: [
        {
          object: {
            visible: true,
            userData: {
              nodeType: 'mesh',
              visibleTriangleCount: 10,
              attrs: { level_stats: { geometric_error: 0.125 } },
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

    expect(findActiveLevelStatus(container)?.textContent).toBe('L1/2 · ε≤13%');
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
      updateLabelStyle: vi.fn(),
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

  it('labels direct-colour and colormapped ranges with their mapping semantics', () => {
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
            intensity: 2.4,
            offset: 0,
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

    const rangeLabel = container.querySelector('.luxar-range-slider__label') as HTMLElement;
    expect(panel.layerState.getLayer('/cloud')!.displayMax).toBeCloseTo(1 / 2.4, 6);
    expect(rangeLabel.textContent).toBe('Colour range');
    expect(rangeLabel.title).toBe(
      "Input RGB values in this range are mapped to the full output range. This controls colour gain and offset, not the layer's data extents."
    );

    const cmSelect = Array.from(container.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'viridis')
    )!;
    cmSelect.value = 'viridis';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    expect(panel.layerState.getLayer('/cloud')!.displayMax).toBeCloseTo(0.02, 6);
    expect(rangeLabel.textContent).toBe('Display range');
    expect(rangeLabel.title).toBe(
      'Scalar data values in this range are mapped across the colormap.'
    );

    cmSelect.value = '';
    cmSelect.dispatchEvent(new Event('change', { bubbles: true }));
    expect(rangeLabel.textContent).toBe('Colour range');
    expect(rangeLabel.title).toContain("not the layer's data extents");
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

  /** A mesh leaf whose material records the shading setters. */
  function mountMeshLayer(
    container: HTMLElement,
    animationController: AnimationController,
    blendingMode = 'opaque',
    attrs: Record<string, unknown> = {}
  ) {
    const calls = {
      ambient: vi.fn(),
      shadeExponent: vi.fn(),
      specular: vi.fn(),
      shininess: vi.fn(),
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
      updateSpecular: calls.specular,
      updateShininess: calls.shininess,
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
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', { blending_mode: blendingMode, ...attrs })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');
    return { panel, calls };
  }

  function mountPartitionMeshLayer(shadings: Array<'flat' | 'none'>): LayersPanel {
    const children = shadings.map((shading, index) => ({
      name: `part_${index}`,
      path: `/surface/part_${index}`,
      type: 'mesh',
      attrs: { type: 'mesh', shading, has_normals: true },
      children: [],
    }));
    const graph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'surface',
          path: '/surface',
          type: 'group',
          attrs: { layer: true, kind: 'partition', display_type: 'mesh' },
          children,
        },
      ],
    } as unknown as SceneNode;
    const rootGroup = new THREE.Group();
    for (const child of children) {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      mesh.name = child.path;
      rootGroup.add(mesh);
    }

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.show();
    panel.layerState.select('/surface', 'single');
    return panel;
  }

  it('mesh shading sliders: shown for a mesh layer and hidden for every other type', () => {
    // TYPE-gated, which is new for this panel — every other control here is universal
    // or mode-gated. Mesh is the only SHADED geometry type, so on a points layer these
    // five have no uniform to write and would be controls that visibly do nothing.
    mountMeshLayer(container, animationController);
    for (const label of ['Ambient', 'Shade falloff', 'Specular', 'Shininess', 'Alpha cutoff']) {
      const group = findControlGroup(container, label);
      expect(group, `${label} control should exist`).not.toBeNull();
      expect(group!.style.display, `${label} should be visible on a mesh layer`).not.toBe('none');
    }
    expect(findControlGroup(container, 'Shininess')!.querySelector('input')!.step).toBe('0.5');

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
    for (const label of ['Ambient', 'Shade falloff', 'Specular', 'Shininess', 'Alpha cutoff']) {
      expect(findControlGroup(other, label)!.style.display, `${label} on points`).toBe('none');
    }
  });

  it('mesh Alpha cutoff is gated on the MODE as well as the type', () => {
    // Narrower than the other four: the cutout only exists in `opaque`, so in any other
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

  it('unlit mesh hides the four inert lighting controls but keeps Alpha cutoff', () => {
    // `none` wins even when stored normals exist. The panel must use the same
    // resolved shading rule as the material rather than treating normals as proof
    // that the four lighting uniforms are active.
    mountMeshLayer(container, animationController, 'opaque', {
      shading: 'none',
      has_normals: true,
    });

    for (const label of ['Ambient', 'Shade falloff', 'Specular', 'Shininess']) {
      expect(findControlGroup(container, label)!.style.display, `${label} on unlit mesh`).toBe(
        'none'
      );
    }
    expect(findControlGroup(container, 'Alpha cutoff')!.style.display).not.toBe('none');
  });

  it("a material='physical' mesh swaps the house sliders for the live physical knob sliders", () => {
    // The physical family runs none of the house shader: the four lighting sliders
    // have no uniform to write, the cutoff and blend mode are the material's own
    // business, and gamma has no term. So all of them hide, and the knob group
    // shows one live slider per numeric knob — seated on what was AUTHORED and, for
    // what was not, on three's default — plus read-only rows for the colours.
    mountMeshLayer(container, animationController, 'opaque', {
      material: 'physical',
      roughness: 0.4,
      clearcoat: 1,
      sheen: 0.5,
      sheen_color: '#ff4d6d',
      transmission: 1,
      attenuation_color: '#f6d148',
    });
    for (const label of [
      'Ambient',
      'Shade falloff',
      'Specular',
      'Shininess',
      'Alpha cutoff',
      'Gamma',
    ]) {
      expect(findControlGroup(container, label)!.style.display, `${label} on physical`).toBe(
        'none'
      );
    }
    expect(findBlendSelect(container)!.parentElement!.style.display).toBe('none');

    const group = findControlGroup(container, 'Physical material')!;
    expect(group).not.toBeNull();
    expect(group.style.display).not.toBe('none');
    // One live slider per numeric knob, in table order, seated on the authored value
    // or the knob default (readout: two decimals; IOR three; "∞" for no attenuation).
    const sliders = Array.from(group.querySelectorAll('.luxar-layers-panel__control-group'))
      .filter((g) => g.querySelector('input[type="range"]') !== null)
      .map((g) => [
        g.querySelector('.luxar-layers-panel__control-label span')!.textContent,
        g.querySelector('.luxar-layers-panel__control-value')!.textContent,
        (g.querySelector('input[type="range"]') as HTMLInputElement).value,
      ]);
    // Plus the one boolean control, after the sliders: live (transmission is on) and off.
    const refract = findControlGroup(group, 'Refract data')!;
    const refractBox = refract.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(refractBox.checked).toBe(false);
    expect(refractBox.disabled).toBe(false);
    expect(group.lastElementChild!.previousElementSibling).toBe(refract); // before the rows
    expect(sliders).toEqual([
      ['Roughness', '0.40', '0.4'],
      ['Metalness', '0.00', '0'],
      ['Clearcoat', '1.00', '1'],
      ['Clearcoat roughness', '0.00', '0'],
      ['Iridescence', '0.00', '0'],
      ['Sheen', '0.50', '0.5'],
      ['Transmission', '1.00', '1'],
      ['IOR', '1.500', '1.5'],
      ['Thickness', '0.00', '0'],
      ['Attenuation distance', '∞', '1'], // log track: position 1 = top stop = ∞
      ['Dispersion', '0.00', '0'],
    ]);
    // The colour knobs stay read-only rows (a colour picker is not a slider).
    const rows = Array.from(group.querySelectorAll('.luxar-layers-panel__physical-row')).map(
      (row) => row.textContent
    );
    expect(rows).toEqual(['Sheen colour#ff4d6d', 'Attenuation colour#f6d148']);
    expect(
      group.querySelector('.luxar-layers-panel__control-label')!.getAttribute('title')
    ).toContain('transmission');

    // Opacity is still live — it is the one generic control the family honours.
    expect(findControlGroup(container, 'Opacity')!.style.display).not.toBe('none');
  });

  it('dragging a physical slider writes the REAL material live, and reset restores the authored value', () => {
    const material = new PhysicalMeshMaterial({ roughness: 0.4, metalness: 1.0 });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', { material: 'physical', roughness: 0.4, metalness: 1 })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const group = findControlGroup(container, 'Physical material')!;
    const inputOf = (label: string): HTMLInputElement =>
      findControlGroup(group, label)!.querySelector('input[type="range"]') as HTMLInputElement;

    // Roughness: a plain-state knob.
    const roughness = inputOf('Roughness');
    roughness.value = '0.05';
    roughness.dispatchEvent(new Event('input'));
    expect(material.roughness).toBeCloseTo(0.05, 6);
    expect(panel.layerState.getLayer('/cloud')!.physicalKnobs!.roughness).toBeCloseTo(0.05, 6);

    // Transmission: crosses zero → glass, and the material's compositing follows.
    expect(material.transparent).toBe(false);
    const transmission = inputOf('Transmission');
    transmission.value = '1';
    transmission.dispatchEvent(new Event('input'));
    expect(material.transmission).toBe(1);
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(material.userData.drawBeforeEmissive).toBe(true);

    // Attenuation distance: the log track's top stop is three's "no attenuation".
    const attenuation = inputOf('Attenuation distance');
    attenuation.value = '0.5';
    attenuation.dispatchEvent(new Event('input'));
    expect(Number.isFinite(material.attenuationDistance)).toBe(true);
    attenuation.value = '1';
    attenuation.dispatchEvent(new Event('input'));
    expect(material.attenuationDistance).toBe(Number.POSITIVE_INFINITY);

    // Reset puts every knob back to what add_mesh(...) authored (or the default).
    panel.resetAllLayers();
    expect(material.roughness).toBe(0.4);
    expect(material.metalness).toBe(1.0);
    expect(material.transmission).toBe(0);
    expect(material.transparent).toBe(false);
    expect(material.userData.drawBeforeEmissive).toBeUndefined();
    expect(inputOf('Roughness').value).toBe('0.4');
  });

  it('the Refract data switch is inert without transmission, flips the material live, and Reset restores it', () => {
    const material = new PhysicalMeshMaterial({ roughness: 0.4 });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', { material: 'physical', roughness: 0.4 })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const group = findControlGroup(container, 'Physical material')!;
    const toggleGroup = findControlGroup(group, 'Refract data')!;
    const checkbox = toggleGroup.querySelector('input[type="checkbox"]') as HTMLInputElement;
    const transmission = findControlGroup(group, 'Transmission')!.querySelector(
      'input[type="range"]'
    ) as HTMLInputElement;

    // No transmission authored: the switch is off and inert, and says why.
    expect(checkbox.checked).toBe(false);
    expect(checkbox.disabled).toBe(true);
    expect(toggleGroup.title).toContain('Transmission');

    // Transmission wakes it; flipping it moves the glass from draw-first to draw-after
    // on the REAL material, with nothing else about its compositing changing.
    transmission.value = '1';
    transmission.dispatchEvent(new Event('input'));
    expect(checkbox.disabled).toBe(false);
    expect(material.userData.drawBeforeEmissive).toBe(true);
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change'));
    expect(material.userData.drawAfterEmissive).toBe(true);
    expect(material.userData.drawBeforeEmissive).toBeUndefined();
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(panel.layerState.getLayer('/cloud')!.physicalKnobs!.refract_data).toBe(true);

    // Reset: authored = no transmission, no refraction — the switch goes back off and inert.
    panel.resetAllLayers();
    expect(material.transmission).toBe(0);
    expect(material.userData.drawAfterEmissive).toBeUndefined();
    expect(material.userData.drawBeforeEmissive).toBeUndefined();
    const checkboxAfter = findControlGroup(
      findControlGroup(container, 'Physical material')!,
      'Refract data'
    )!.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(checkboxAfter.checked).toBe(false);
    expect(checkboxAfter.disabled).toBe(true);
  });

  it('an authored refract_data seats the switch on, and the per-row reset restores knobs AND the switch', () => {
    const material = new PhysicalMeshMaterial({
      transmission: 1.0,
      refractData: true,
      roughness: 0.4,
    });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', {
        material: 'physical',
        transmission: 1.0,
        refract_data: true,
        roughness: 0.4,
      })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const group = findControlGroup(container, 'Physical material')!;
    const checkbox = findControlGroup(group, 'Refract data')!.querySelector(
      'input[type="checkbox"]'
    ) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    expect(checkbox.disabled).toBe(false);
    expect(material.userData.drawAfterEmissive).toBe(true);

    // Drag the layer away from its authored state: switch off, roughness down.
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event('change'));
    expect(material.userData.drawBeforeEmissive).toBe(true);
    const roughness = findControlGroup(group, 'Roughness')!.querySelector(
      'input[type="range"]'
    ) as HTMLInputElement;
    roughness.value = '0.05';
    roughness.dispatchEvent(new Event('input'));
    expect(material.roughness).toBeCloseTo(0.05, 6);

    // The per-row reset (the real user path) restores BOTH onto the material — this
    // path used to reset the readouts but leave the dragged knobs on the surface.
    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    const reset = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-context-menu__item')
    ).find((el) => el.textContent === 'Reset this layer')!;
    reset.click();
    expect(material.roughness).toBe(0.4);
    expect(material.userData.drawAfterEmissive).toBe(true);
    expect(material.userData.drawBeforeEmissive).toBeUndefined();
    panel.dispose();
  });

  it('editing one physical slider preserves authored values beyond other slider tracks', () => {
    const material = new PhysicalMeshMaterial({
      thickness: 25,
      attenuationDistance: 500,
      dispersion: 3,
    });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', {
        material: 'physical',
        thickness: 25,
        attenuation_distance: 500,
        dispersion: 3,
      })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const group = findControlGroup(container, 'Physical material')!;
    const roughness = findControlGroup(group, 'Roughness')!.querySelector(
      'input[type="range"]'
    ) as HTMLInputElement;
    roughness.value = '0.25';
    roughness.dispatchEvent(new Event('input'));

    expect(material.thickness).toBe(25);
    expect(material.attenuationDistance).toBe(500);
    expect(material.dispersion).toBe(3);
    expect(panel.layerState.getLayer('/cloud')!.physicalKnobs).toMatchObject({
      thickness: 25,
      attenuation_distance: 500,
      dispersion: 3,
    });

    const thickness = findControlGroup(group, 'Thickness')!.querySelector(
      'input[type="range"]'
    ) as HTMLInputElement;
    thickness.value = '5';
    thickness.dispatchEvent(new Event('input'));
    expect(material.thickness).toBe(5);

    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    const resetLayer = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-context-menu__item')
    ).find((el) => el.textContent === 'Reset this layer')!;
    resetLayer.click();
    expect(material.thickness).toBe(25);
    expect(material.attenuationDistance).toBe(500);
    expect(material.dispersion).toBe(3);

    thickness.value = '5';
    thickness.dispatchEvent(new Event('input'));
    panel.resetAllLayers();
    expect(material.thickness).toBe(25);
    expect(material.attenuationDistance).toBe(500);
    expect(material.dispersion).toBe(3);
  });

  it('greys out the knobs that change nothing in the current state, with the reason as hover text', () => {
    // A metal (metalness 1, the authored value): the whole glass family is inert
    // and clearcoat roughness waits for a clearcoat. Measured on the reflections
    // demo — a live slider that does nothing reads as broken.
    const material = new PhysicalMeshMaterial({ metalness: 1.0 });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', { material: 'physical', metalness: 1 })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    const group = findControlGroup(container, 'Physical material')!;
    const knob = (label: string): HTMLElement => findControlGroup(group, label)!;
    const inputOf = (label: string): HTMLInputElement =>
      knob(label).querySelector('input[type="range"]') as HTMLInputElement;
    for (const label of [
      'Transmission',
      'IOR',
      'Thickness',
      'Attenuation distance',
      'Dispersion',
      'Clearcoat roughness',
    ]) {
      expect(inputOf(label).disabled, label).toBe(true);
      expect(knob(label).title.length, label).toBeGreaterThan(0);
      expect(
        knob(label).classList.contains('luxar-layers-panel__control-group--inert'),
        label
      ).toBe(true);
    }
    expect(knob('Transmission').title).toMatch(/metal/);
    for (const label of ['Roughness', 'Metalness', 'Clearcoat', 'Iridescence', 'Sheen']) {
      expect(inputOf(label).disabled, label).toBe(false);
    }

    // Lowering metalness wakes the glass family (transmission still 0, so the
    // transmission-dependent knobs stay inert with THAT reason) …
    const metalness = inputOf('Metalness');
    metalness.value = '0';
    metalness.dispatchEvent(new Event('input'));
    expect(inputOf('Transmission').disabled).toBe(false);
    expect(inputOf('IOR').disabled).toBe(false);
    expect(knob('Thickness').title).toMatch(/Transmission/);
    // … and turning transmission up wakes thickness and dispersion, while the
    // attenuation distance still waits for a non-white attenuation colour.
    const transmission = inputOf('Transmission');
    transmission.value = '1';
    transmission.dispatchEvent(new Event('input'));
    expect(inputOf('Thickness').disabled).toBe(false);
    expect(inputOf('Dispersion').disabled).toBe(false);
    expect(knob('Attenuation distance').title).toMatch(/white/);
    // Clearcoat wakes clearcoat roughness.
    const clearcoat = inputOf('Clearcoat');
    clearcoat.value = '0.5';
    clearcoat.dispatchEvent(new Event('input'));
    expect(inputOf('Clearcoat roughness').disabled).toBe(false);
  });

  it('resetAllLayers leaves a REAL physical material exactly where the author put it', () => {
    // The reset path calls `applyBlendingMode` and `applyMeshAppearance` on every
    // layer. On a material WITHOUT `applyBlendingMode`, `layer-apply.ts` falls back to
    // writing the HOUSE shader's blend state for the mesh default `opaque` — which
    // would turn a translucent physical shell opaque and depth-writing on a reset.
    // The wrapper's no-op `applyBlendingMode` and the optional-chained shade setters
    // are what make this a no-op; pinned against the real class, not a stub.
    const material = new PhysicalMeshMaterial({
      roughness: 0.4,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
      opacity: 0.3,
    });
    const before = {
      roughness: material.roughness,
      clearcoat: material.clearcoat,
      clearcoatRoughness: material.clearcoatRoughness,
      transparent: material.transparent,
      depthWrite: material.depthWrite,
      blending: material.blending,
      stamp: material.userData.blendingMode,
    };
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'mesh';
    mesh.userData._layerMaterialCloned = true;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(
      rootGroup,
      makeLayeredSceneGraph('mesh', {
        material: 'physical',
        roughness: 0.4,
        clearcoat: 1,
        clearcoat_roughness: 0.05,
        opacity: 0.3,
      })
    );
    panel.show();
    panel.layerState.select('/cloud', 'single');

    expect(() => panel.resetAllLayers()).not.toThrow();

    expect(material.roughness).toBe(before.roughness);
    expect(material.clearcoat).toBe(before.clearcoat);
    expect(material.clearcoatRoughness).toBe(before.clearcoatRoughness);
    expect(material.transparent).toBe(before.transparent);
    expect(material.depthWrite).toBe(before.depthWrite);
    expect(material.blending).toBe(before.blending);
    expect(material.userData.blendingMode).toBe(before.stamp);
    // Opacity is the one generic control the family honours, and reset restores the
    // AUTHORED value, which is what the material already had.
    expect(material.getOpacity()).toBeCloseTo(0.3, 6);
  });

  it('a HOUSE mesh layer shows no physical listing and keeps every slider', () => {
    mountMeshLayer(container, animationController);
    expect(findControlGroup(container, 'Physical material')!.style.display).toBe('none');
    expect(findControlGroup(container, 'Gamma')!.style.display).not.toBe('none');
    expect(findBlendSelect(container)!.parentElement!.style.display).not.toBe('none');
    expect(findControlGroup(container, 'Ambient')!.style.display).not.toBe('none');
  });

  it('partitioned unlit mesh derives shading from its leaves', () => {
    const panel = mountPartitionMeshLayer(['none']);

    expect(panel.layerState.getLayer('/surface')!.shading).toBe('none');
    for (const label of ['Ambient', 'Shade falloff', 'Specular', 'Shininess']) {
      expect(findControlGroup(container, label)!.style.display, `${label} on partition`).toBe(
        'none'
      );
    }
  });

  it.each([
    ['none', 'flat'],
    ['flat', 'none'],
  ] as const)('mixed partition stays lit for child order %s, %s', (...shadings) => {
    const panel = mountPartitionMeshLayer([...shadings]);

    expect(panel.layerState.getLayer('/surface')!.shading).toBe('flat');
    for (const label of ['Ambient', 'Shade falloff', 'Specular', 'Shininess']) {
      expect(
        findControlGroup(container, label)!.style.display,
        `${label} on mixed partition`
      ).not.toBe('none');
    }
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

    drag('Specular', 0.3);
    expect(calls.specular).toHaveBeenCalledWith(expect.closeTo(0.3, 6));
    expect(panel.layerState.getLayer('/cloud')!.specular).toBeCloseTo(0.3, 6);

    drag('Shininess', 48);
    expect(calls.shininess).toHaveBeenCalledWith(expect.closeTo(48, 6));
    expect(panel.layerState.getLayer('/cloud')!.shininess).toBeCloseTo(48, 6);

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

  it('reset restores the mesh shading values on the surface AND the pick material', () => {
    // The mesh shading uniforms are applied through applyMeshAppearance, which is NOT
    // routed through applyComposed (they have no composition rule). So `resetAllLayers`
    // must call applyMeshAppearance explicitly — otherwise the surface keeps the dragged
    // uAmbient/uShadeExponent/uSpecular/uShininess/uAlphaCutoff on the surface, and
    // uAlphaCutoff on the pick material, while the readouts show reset defaults (#1283).
    const { panel, calls } = mountMeshLayer(container, animationController);

    const drag = (label: string, value: number): void => {
      const group = findControlGroup(container, label)!;
      const input = group.querySelector('input[type="range"]') as HTMLInputElement;
      input.value = String(value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    };

    // Move all five away from their authored defaults.
    drag('Ambient', 0.7);
    drag('Shade falloff', 2.5);
    drag('Specular', 0.3);
    drag('Shininess', 48);
    drag('Alpha cutoff', 0.8);
    // Sanity: the layer state actually moved before we reset.
    expect(panel.layerState.getLayer('/cloud')!.ambient).toBeCloseTo(0.7, 6);
    expect(panel.layerState.getLayer('/cloud')!.shadeExponent).toBeCloseTo(2.5, 6);
    expect(panel.layerState.getLayer('/cloud')!.specular).toBeCloseTo(0.3, 6);
    expect(panel.layerState.getLayer('/cloud')!.shininess).toBeCloseTo(48, 6);
    expect(panel.layerState.getLayer('/cloud')!.alphaCutoff).toBeCloseTo(0.8, 6);

    // Only the reset-driven setter calls should be observed below.
    calls.ambient.mockClear();
    calls.shadeExponent.mockClear();
    calls.specular.mockClear();
    calls.shininess.mockClear();
    calls.alphaCutoff.mockClear();
    calls.pickAlphaCutoff.mockClear();

    panel.resetAllLayers();

    // The scene graph has no mesh appearance attrs, so reset falls back to
    // MESH_DEFAULTS — and those must reach the material, not just the row.
    expect(calls.ambient).toHaveBeenCalledWith(expect.closeTo(MESH_DEFAULTS.ambient, 6));
    expect(calls.shadeExponent).toHaveBeenCalledWith(
      expect.closeTo(MESH_DEFAULTS.shadeExponent, 6)
    );
    expect(calls.specular).toHaveBeenCalledWith(expect.closeTo(MESH_DEFAULTS.specular, 6));
    expect(calls.shininess).toHaveBeenCalledWith(expect.closeTo(MESH_DEFAULTS.shininess, 6));
    expect(calls.alphaCutoff).toHaveBeenCalledWith(expect.closeTo(MESH_DEFAULTS.alphaCutoff, 6));
    // The pick material applies the identical cutout, so it must reset too.
    expect(calls.pickAlphaCutoff).toHaveBeenCalledWith(
      expect.closeTo(MESH_DEFAULTS.alphaCutoff, 6)
    );
  });

  it('reset restores the authored layer order on the render object', () => {
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.Material());
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'gsplats';
    mesh.userData.layerOrder = 99;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('gsplats', { layer_order: 7 }));

    panel.resetAllLayers();

    expect(mesh.userData.layerOrder).toBe(7);
  });

  it('reset reapplies an inherited custom palette with its composed LUT bytes', () => {
    const lut = new Uint8Array(768);
    lut[767] = 255;
    const material = makeColormapRoutingStub();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material as unknown as THREE.Material);
    mesh.name = '/palette/gs';
    mesh.userData.nodeType = 'gsplats';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const graph: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/palette',
          type: 'group',
          attrs: { colormap: 'custom', customLutBytes: lut },
          hasSpatialIndex: false,
          children: [
            {
              path: '/palette/gs',
              type: 'gsplats',
              attrs: { layer: true, amplitude_data_range: [2, 8] },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    vi.mocked(getColormapTexture).mockClear();

    panel.resetAllLayers();

    expect(getColormapTexture).toHaveBeenCalledWith('custom', lut);
  });

  it('resetAllLayers clears label colouring and filtering on the material', () => {
    const material = makeColormapRoutingStub();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material as unknown as THREE.Material);
    mesh.name = '/cloud';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('gsplats'));
    const layer = panel.layerState.getLayer('/cloud')!;
    layer.colorByLabel = true;
    layer.labelFilterId = '7';
    const updateLabelStyle = material.updateLabelStyle as ReturnType<typeof vi.fn>;
    updateLabelStyle.mockClear();

    panel.resetAllLayers();

    expect(updateLabelStyle).toHaveBeenCalledWith(false, 0);
    panel.dispose();
  });

  it('resolves a selected exact label id independently for each selected layer', () => {
    const firstMaterial = makeColormapRoutingStub();
    const secondMaterial = makeColormapRoutingStub();
    const rootGroup = new THREE.Group();
    for (const [path, material] of [
      ['/first', firstMaterial],
      ['/second', secondMaterial],
    ] as const) {
      const mesh = new THREE.Mesh(
        new THREE.BufferGeometry(),
        material as unknown as THREE.Material
      );
      mesh.name = path;
      mesh.userData.nodeType = 'gsplats';
      rootGroup.add(mesh);
    }
    const graph: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/first',
          type: 'gsplats',
          attrs: {
            layer: true,
            label_vocabulary: { '7': 'cell', '9007199254740993': 'artifact' },
          },
          hasSpatialIndex: true,
        },
        {
          path: '/second',
          type: 'gsplats',
          attrs: {
            layer: true,
            label_vocabulary: {
              '3': 'background',
              '7': 'cell',
              '9007199254740993': 'artifact',
            },
          },
          hasSpatialIndex: true,
        },
      ],
    };
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, graph);
    panel.layerState.select('/first', 'single');
    panel.layerState.select('/second', 'add');
    const classesGroup = Array.from(
      container.querySelectorAll<HTMLElement>('.luxar-layers-panel__control-group')
    ).find(
      (group) =>
        group.querySelector('.luxar-layers-panel__control-label')?.textContent === 'Classes'
    )!;
    const filter = classesGroup.querySelectorAll<HTMLSelectElement>('select')[1];
    filter.value = '9007199254740993';
    filter.dispatchEvent(new Event('change', { bubbles: true }));

    expect(firstMaterial.updateLabelStyle).toHaveBeenLastCalledWith(false, 2);
    expect(secondMaterial.updateLabelStyle).toHaveBeenLastCalledWith(false, 3);
    expect(panel.layerState.getLayer('/first')!.labelFilterId).toBe('9007199254740993');
    expect(panel.layerState.getLayer('/second')!.labelFilterId).toBe('9007199254740993');
    panel.dispose();
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

  it('a plain group over a mesh keeps the mesh OPAQUE after a non-blend edit', () => {
    // Defect (#1275), fail-first vs the pre-fix liveLayerAttrs: a PLAIN group
    // (no authored blending_mode) used to emit its DISPLAYED-but-defaulted
    // `additive` mode into the composition chain, so ANY non-blend edit on the
    // group (opacity/gamma/range) flipped a contained mesh from its own `opaque`
    // default to `additive` (glow, depth-write off). With ownership gating
    // (`blendingModeExplicit`) the group emits NO mode, so composeEffective falls back
    // to the mesh leaf's own type-default `opaque`.
    const applyBlendingMode = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'opaque' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption: vi.fn(),
      applyBlendingMode,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/grp/mesh';
    mesh.userData.nodeType = 'mesh';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    // A PLAIN group (layer=true) that authored NO blending_mode, over a
    // non-layer mesh leaf that authored none either.
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
          attrs: { layer: true },
          children: [
            {
              name: 'mesh',
              path: '/grp/mesh',
              type: 'mesh',
              attrs: {},
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

    // A NON-blend edit on the group, then push it through the apply engine.
    applyBlendingMode.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.8;
    });
    const grpLayer = panel.layerState.getLayer('/grp')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(grpLayer);

    expect(applyBlendingMode).toHaveBeenCalledWith('opaque');
    expect(applyBlendingMode).not.toHaveBeenCalledWith('additive');
  });

  it("a plain group preserves a non-layer mesh's AUTHORED blending_mode on a non-blend edit", () => {
    // Regression guard (fail-first vs the unconditional subtree-drop): a plain
    // group (no authored/owned mode) must NOT suppress a descendant's authored
    // mode. A mesh authored `additive` under such a group used to snap to its
    // `opaque` type-default on any non-blend edit, because composeEffective
    // dropped the descendant mode (i>layerDepth, non-layer) AND the group
    // emitted none. The drop now fires only when the edited layer OWNS a mode.
    const applyBlendingMode = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption: vi.fn(),
      applyBlendingMode,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/grp/mesh';
    mesh.userData.nodeType = 'mesh';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    // PLAIN group (owns no mode) over a non-layer mesh that AUTHORED `additive`.
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
          attrs: { layer: true },
          children: [
            {
              name: 'mesh',
              path: '/grp/mesh',
              type: 'mesh',
              attrs: { blending_mode: 'additive' },
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

    applyBlendingMode.mockClear();
    panel.layerState.applyToSelected((l) => {
      l.opacity = 0.8;
    });
    const grpLayer = panel.layerState.getLayer('/grp')!;
    (
      panel as unknown as { applyEngine: { applyOpacity(l: unknown): void } }
    ).applyEngine.applyOpacity(grpLayer);

    expect(applyBlendingMode).toHaveBeenCalledWith('additive');
    expect(applyBlendingMode).not.toHaveBeenCalledWith('opaque');
  });

  it('picking a mode on a PLAIN group broadcasts it to the leaf (group now OWNS it)', () => {
    // Locks the dropdown handler's `l.blendingModeExplicit = true` (mutant-kill: delete
    // that line and this fails). A plain group owns no mode at init, so
    // liveLayerAttrs would emit nothing and the pick would be dropped — the leaf
    // would fall back to its `additive` points default instead of the picked mode.
    const applyBlendingMode = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: { blendingMode: 'additive' },
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
      updateGamma: vi.fn(),
      updateOpacity: vi.fn(),
      updateAbsorption: vi.fn(),
      applyBlendingMode,
    };
    stubMat.clone = vi.fn(() => stubMat);

    const points = new THREE.Points(
      new THREE.BufferGeometry(),
      stubMat as unknown as THREE.Material
    );
    points.name = '/grp/pts';
    const rootGroup = new THREE.Group();
    rootGroup.add(points);

    // PLAIN group (owns no mode) over a non-layer points leaf that authored none.
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
          attrs: { layer: true },
          children: [
            {
              name: 'pts',
              path: '/grp/pts',
              type: 'points',
              attrs: {},
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
    // Precondition: a plain group owns no mode until the user picks one.
    expect(panel.layerState.getLayer('/grp')!.blendingModeExplicit).toBe(false);

    applyBlendingMode.mockClear();
    const select = findBlendSelect(container)!;
    select.value = 'max';
    select.dispatchEvent(new Event('change', { bubbles: true }));

    expect(panel.layerState.getLayer('/grp')!.blendingModeExplicit).toBe(true);
    expect(applyBlendingMode).toHaveBeenCalledWith('max');
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
    // A user picking a mode on the group OWNS it (the dropdown handler /
    // setBlendingMode set this in production) — so the wrapper's mode overrides
    // the part's authored one. A plain group that never authored/picked a mode
    // emits none (see the ownership tests in layer-state.test.ts).
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

  it('mesh shading reaches every part of a kind=partition mesh layer', () => {
    // `add_mesh(partition=…)` writes a kind=partition wrapper and the panel shows
    // that wrapper as one `mesh` layer — so `layer.path` resolves to a THREE.Group
    // with no material. Writing only there left Ambient / Shade exponent / Alpha
    // cutoff visible and completely inert on a partitioned surface. The panel must
    // fan out to the parts, the way every composing control already does.
    const makePartMaterial = (): Record<string, unknown> => {
      const mat: Record<string, unknown> = {
        userData: { blendingMode: 'opaque' },
        uniforms: { uOpacity: { value: 1.0 } },
        defines: {},
        updateIntensity: vi.fn(),
        updateOffset: vi.fn(),
        updateGamma: vi.fn(),
        updateOpacity: vi.fn(),
        updateAmbient: vi.fn(),
        updateShadeExponent: vi.fn(),
        updateSpecular: vi.fn(),
        updateShininess: vi.fn(),
        updateAlphaCutoff: vi.fn(),
        applyBlendingMode: vi.fn(),
      };
      mat.clone = vi.fn(() => mat);
      return mat;
    };
    const materials = [makePartMaterial(), makePartMaterial()];
    const rootGroup = new THREE.Group();
    const wrapper = new THREE.Group();
    wrapper.name = '/surface';
    wrapper.userData.kind = 'partition';
    rootGroup.add(wrapper);
    materials.forEach((mat, i) => {
      const part = new THREE.Mesh(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
      part.name = `/surface/part_${i}`;
      part.userData._layerMaterialCloned = true;
      part.userData.nodeType = 'mesh';
      wrapper.add(part);
    });

    const sceneGraph = {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: [
        {
          name: 'surface',
          path: '/surface',
          type: 'group',
          // `layer` is a compositing attr, so the writer puts it on the wrapper;
          // the shading attrs are not, so they land on each part.
          attrs: { layer: true, kind: 'partition', display_type: 'mesh', max_elements: 10 },
          children: [0, 1].map((i) => ({
            name: `part_${i}`,
            path: `/surface/part_${i}`,
            type: 'mesh',
            attrs: {
              type: 'mesh',
              ambient: 0.4,
              shade_exponent: 3,
              specular: 0.2,
              shininess: 32,
              alpha_cutoff: 0.25,
            },
            children: [],
          })),
        },
      ],
    } as unknown as SceneNode;

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, sceneGraph);
    panel.show();

    const layer = panel.layerState.getLayer('/surface')!;
    expect(layer.type).toBe('mesh');
    // Nothing authored a mode, so the layer opens on the MESH default — keyed on the
    // resolved layer type, not the wrapper's raw `group` type (which would read
    // `additive` and hide the Alpha-cutoff slider on an opaque surface).
    expect(layer.blendingMode).toBe('opaque');
    // ...and the defaulted mode must NOT make the wrapper a composition setter, or it
    // would push `opaque` onto every descendant as if the author had asked for it.
    expect(layer.blendingModeExplicit).toBe(false);
    // The consequence the default exists for: the cutout is active, so its slider shows.
    panel.layerState.select('/surface', 'single');
    expect(findControlGroup(container, 'Alpha cutoff')!.style.display).not.toBe('none');
    // The wrapper carries no shading attrs of its own, so the sliders must open on
    // what the parts are actually rendering with — not on the material defaults.
    expect(layer.ambient).toBe(0.4);
    expect(layer.shadeExponent).toBe(3);
    expect(layer.specular).toBe(0.2);
    expect(layer.shininess).toBe(32);
    expect(layer.alphaCutoff).toBe(0.25);

    layer.ambient = 0.1;
    (
      panel as unknown as { applyEngine: { applyMeshAppearance(l: unknown): void } }
    ).applyEngine.applyMeshAppearance(layer);

    for (const mat of materials) {
      expect(mat.updateAmbient).toHaveBeenCalledWith(0.1);
      expect(mat.updateShadeExponent).toHaveBeenCalledWith(3);
      expect(mat.updateSpecular).toHaveBeenCalledWith(0.2);
      expect(mat.updateShininess).toHaveBeenCalledWith(32);
      expect(mat.updateAlphaCutoff).toHaveBeenCalledWith(0.25);
    }
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

describe('LayersPanel — filter + context-menu lifecycle across dataset reloads', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  /** A scene with enough layers (10 > FILTER_THRESHOLD 8) to show the filter row. */
  function makeManyLayerSceneGraph(prefix = 'layer'): SceneNode {
    return {
      name: 'root',
      path: '/',
      type: 'group',
      attrs: {},
      children: Array.from({ length: 10 }, (_, i) => ({
        name: `${prefix}${i}`,
        path: `/${prefix}${i}`,
        type: 'points',
        attrs: { layer: true, type: 'points' },
        children: [],
      })),
    } as unknown as SceneNode;
  }

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
  });

  it('a dataset reload clears the filter — stale queries must not hide the new scene', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('alpha'));

    const input = container.querySelector<HTMLInputElement>('.luxar-panel-filter__input')!;
    expect(input).not.toBeNull();
    input.value = 'alpha3';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    const filtered = container.querySelectorAll('.luxar-layer-row--filtered').length;
    expect(filtered).toBe(9); // everything but alpha3

    // Reload with a scene whose names never match the old query. Without
    // the clear()-side reset, the carried-over filterText re-applies
    // against a BLANK input and silently hides every new row.
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('beta'));
    expect(container.querySelectorAll('.luxar-layer-row--filtered').length).toBe(0);
    const freshInput = container.querySelector<HTMLInputElement>('.luxar-panel-filter__input')!;
    expect(freshInput.value).toBe('');
    panel.dispose();
  });

  it('a dataset reload closes an open context menu (stale captured actions)', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());

    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(document.querySelector('.luxar-context-menu')).not.toBeNull();

    // The menu is mounted OUTSIDE the panel (viewer container), so removing
    // the panel alone would strand it with actions bound to the old scene.
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('next'));
    expect(document.querySelector('.luxar-context-menu')).toBeNull();
    panel.dispose();
  });

  it('resetLayer pushes the authored gamma/window back onto the MATERIAL, not just the state', () => {
    // Reset's material coverage rides on applyColormap's trailing
    // applyComposed (the same dependency resetAllLayers documents). If that
    // tail is ever refactored away, reset would leave the material rendering
    // the pre-reset gamma/window while the controls show authored values —
    // this test pins the dependency at the material boundary.
    const updateGamma = vi.fn();
    const updateIntensity = vi.fn();
    const updateLabelStyle = vi.fn();
    const stubMat: Record<string, unknown> = {
      userData: {},
      uniforms: { uOpacity: { value: 1.0 } },
      defines: {},
      updateIntensity,
      updateOffset: vi.fn(),
      updateGamma,
      updateOpacity: vi.fn(),
      applyBlendingMode: vi.fn(),
      updateLabelStyle,
    };
    stubMat.clone = vi.fn(() => stubMat);
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), stubMat as unknown as THREE.Material);
    mesh.name = '/layer0';
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeManyLayerSceneGraph());

    // Drag the layer away from its authored state.
    panel.layerState.setGamma('/layer0', 2.5);
    panel.layerState.getLayer('/layer0')!.colorByLabel = true;
    panel.layerState.getLayer('/layer0')!.labelFilterId = '7';
    updateGamma.mockClear();
    updateIntensity.mockClear();

    // Reset via the row context menu item (the real user path).
    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    const reset = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-context-menu__item')
    ).find((el) => el.textContent === 'Reset this layer')!;
    reset.click();

    // State restored AND the material saw the authored values again.
    expect(panel.layerState.getLayer('/layer0')!.gamma).toBe(1.0);
    expect(updateGamma).toHaveBeenCalledWith(1.0);
    expect(updateIntensity).toHaveBeenCalled(); // composed window re-pushed
    expect(updateLabelStyle).toHaveBeenCalledWith(false, 0);
    panel.dispose();
  });

  it('resetLayer restores the authored layer order on the render object', () => {
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.Material());
    mesh.name = '/cloud';
    mesh.userData.nodeType = 'gsplats';
    mesh.userData.layerOrder = 99;
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(rootGroup, makeLayeredSceneGraph('gsplats', { layer_order: 7 }));

    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    const reset = Array.from(
      document.querySelectorAll<HTMLElement>('.luxar-context-menu__item')
    ).find((el) => el.textContent === 'Reset this layer')!;
    reset.click();

    expect(mesh.userData.layerOrder).toBe(7);
    panel.dispose();
  });

  it('Escape from the HEADER menu returns focus to the focused header child, not <body>', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());

    // The native ContextMenu key fires `contextmenu` at the focused element —
    // here the header's close button. The header wrapper itself is a plain
    // <div>; forcing restoreFocus onto it would silently fail and strand
    // focus on <body>.
    const closeBtn = container.querySelector<HTMLElement>('.luxar-layers-panel__close')!;
    closeBtn.focus();
    closeBtn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    const menu = document.querySelector('.luxar-context-menu')!;
    expect(menu).not.toBeNull();

    menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(document.activeElement).toBe(closeBtn);
    panel.dispose();
  });

  it('the pinned row aria-label survives a menu open/close (only aria-expanded moves)', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());

    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    const labelBefore = row.getAttribute('aria-label');
    expect(labelBefore).toBe('layer0 (points)'); // the E2E/AT-pinned "name (type)" shape

    row.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(row.getAttribute('aria-expanded')).toBe('true');
    expect(row.getAttribute('aria-label')).toBe(labelBefore);

    document
      .querySelector('.luxar-context-menu')!
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(row.getAttribute('aria-expanded')).toBe('false');
    expect(row.getAttribute('aria-label')).toBe(labelBefore);
    panel.dispose();
  });

  it('the roving tab stop moves off a filtered row (listbox stays Tab-reachable)', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('alpha'));

    // The panel's initial tab stop sits on the first row (alpha0). Filter it
    // out: a display:none row is unfocusable, so leaving tabIndex=0 there
    // would make the whole listbox unreachable by Tab.
    const input = container.querySelector<HTMLInputElement>('.luxar-panel-filter__input')!;
    input.value = 'alpha1';
    input.dispatchEvent(new Event('input', { bubbles: true }));

    const rows = Array.from(container.querySelectorAll<HTMLElement>('.luxar-layer-row'));
    const tabStops = rows.filter((r) => r.tabIndex === 0);
    expect(tabStops.length).toBe(1);
    expect(tabStops[0].classList.contains('luxar-layer-row--filtered')).toBe(false);
    expect(tabStops[0].textContent).toContain('alpha1');
    panel.dispose();
  });

  it.each([['Home'], ['ArrowUp']])(
    'contains consumed %s row navigation before it reaches window',
    (key) => {
      const panel = new LayersPanel(container, animationController);
      panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());
      panel.show();

      const rows = Array.from(container.querySelectorAll<HTMLElement>('.luxar-layer-row'));
      const row = rows[1];
      row.focus();
      const globalHandler = vi.fn();
      window.addEventListener('keydown', globalHandler);

      row.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));

      expect(document.activeElement).toBe(rows[0]);
      expect(globalHandler).not.toHaveBeenCalled();
      window.removeEventListener('keydown', globalHandler);
      panel.dispose();
    }
  );

  // Named, because ' ' renders as an invisible test title.
  it.each([
    ['Enter', 'Enter'],
    ['Space', ' '],
  ])(
    'contains a consumed %s row selection before it reaches window',
    (_label: string, key: string) => {
      const panel = new LayersPanel(container, animationController);
      panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());
      panel.show();

      const rows = Array.from(container.querySelectorAll<HTMLElement>('.luxar-layer-row'));
      const row = rows[1];
      row.focus();
      const globalHandler = vi.fn();
      window.addEventListener('keydown', globalHandler);

      row.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));

      expect(row.getAttribute('aria-selected')).toBe('true');
      expect(rows[0].getAttribute('aria-selected')).toBe('false');
      // Space especially: uncontained, it also toggles window-level fullscreen.
      expect(globalHandler).not.toHaveBeenCalled();
      window.removeEventListener('keydown', globalHandler);
      panel.dispose();
    }
  );

  it.each([
    ['F10', { shiftKey: true }],
    ['ContextMenu', {}],
  ])('contains the consumed %s menu key before it reaches window', (key, init) => {
    // The narrative case: these are the same two chords the global
    // element-menu shortcut binds, so a leaked event would close the row menu
    // we just opened and open the canvas one instead.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());
    panel.show();

    const row = container.querySelectorAll<HTMLElement>('.luxar-layer-row')[1];
    row.focus();
    const globalHandler = vi.fn();
    window.addEventListener('keydown', globalHandler);

    row.dispatchEvent(
      new KeyboardEvent('keydown', { key, ...init, bubbles: true, cancelable: true })
    );

    const labels = Array.from(document.querySelectorAll('.luxar-context-menu__label')).map(
      (el) => el.textContent
    );
    expect(labels).toContain('Copy layer path'); // the ROW menu, not the eye's
    expect(globalHandler).not.toHaveBeenCalled();
    window.removeEventListener('keydown', globalHandler);
    panel.dispose();
  });

  it('contains panel-control keys while unrelated viewer shortcuts still bubble', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());
    panel.show();

    const opacitySlider = container.querySelector<HTMLInputElement>('.luxar-layers-panel__slider');
    const rangeSlider = container.querySelector<HTMLInputElement>('.luxar-range-slider__input');
    const closeButton = container.querySelector<HTMLButtonElement>('.luxar-layers-panel__close');
    const eyeButton = container.querySelector<HTMLButtonElement>('.luxar-layer-row__eye');
    expect(opacitySlider).not.toBeNull();
    expect(rangeSlider).not.toBeNull();
    expect(closeButton).not.toBeNull();
    expect(eyeButton).not.toBeNull();

    const controls = [opacitySlider!, rangeSlider!, closeButton!, eyeButton!];

    const globalHandler = vi.fn();
    window.addEventListener('keydown', globalHandler);
    try {
      for (const slider of [opacitySlider!, rangeSlider!]) {
        for (const event of [
          new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }),
          new KeyboardEvent('keydown', {
            key: 'ArrowDown',
            shiftKey: true,
            bubbles: true,
            cancelable: true,
          }),
        ]) {
          slider!.dispatchEvent(event);
          expect(event.defaultPrevented).toBe(false);
        }
      }

      for (const control of controls) {
        control.dispatchEvent(new KeyboardEvent('keydown', { key: 'PageDown', bubbles: true }));
      }
      expect(globalHandler).not.toHaveBeenCalled();

      for (const control of controls) {
        control.dispatchEvent(new KeyboardEvent('keydown', { key: ']', bubbles: true }));
      }
      opacitySlider!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      expect(globalHandler.mock.calls.map(([event]) => (event as KeyboardEvent).key)).toEqual([
        ']',
        ']',
        ']',
        ']',
        'Escape',
      ]);
    } finally {
      window.removeEventListener('keydown', globalHandler);
      panel.dispose();
    }
  });

  it('right-clicking a text field inside the panel leaves the native menu alone', () => {
    // The delegated handler suppresses the native menu everywhere on the
    // glass surface, but a text field has no replacement verbs of ours —
    // swallowing it there costs the user right-click paste.
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('alpha'));

    const input = container.querySelector<HTMLInputElement>('.luxar-panel-filter__input')!;
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    input.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(false);
    expect(document.querySelector('.luxar-context-menu')).toBeNull();

    // A row still gets ours, suppression included.
    const row = container.querySelector<HTMLElement>('.luxar-layer-row')!;
    const rowEv = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    row.dispatchEvent(rowEv);
    expect(rowEv.defaultPrevented).toBe(true);
    expect(document.querySelector('.luxar-context-menu')).not.toBeNull();
    panel.dispose();
  });

  it('right-clicking the layer-order number field leaves the native menu alone', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph('alpha'));

    const input = container.querySelector<HTMLInputElement>('.luxar-layers-panel__number')!;
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    input.dispatchEvent(ev);

    expect(ev.defaultPrevented).toBe(false);
    expect(document.querySelector('.luxar-context-menu')).toBeNull();
    panel.dispose();
  });

  it('Shift+F10 on the focused EYE opens the eye menu, not the row menu', () => {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeManyLayerSceneGraph());

    const eye = container.querySelector<HTMLElement>('.luxar-layer-row__eye')!;
    eye.focus();
    eye.dispatchEvent(new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true }));
    const menu = document.querySelector('.luxar-context-menu')!;
    expect(menu).not.toBeNull();
    const labels = Array.from(menu.querySelectorAll('.luxar-context-menu__label')).map(
      (el) => el.textContent
    );
    // Eye menu = visibility verbs only; the row menu's items must be absent.
    expect(labels).toContain('Invert visibility');
    expect(labels).not.toContain('Copy layer path');
    panel.dispose();
  });
});

/**
 * The SHIPPED Layer order control.
 *
 * `LayerStateManager.setLayerOrder` is covered in `layer-state.test.ts`, but
 * those unit tests do not cover the DOM event path a user actually drives.
 * These do: they render the real panel, select the row, and dispatch on the
 * real field that routes through the state setter.
 */
describe('LayersPanel — Layer order control (the shipped path)', () => {
  let container: HTMLElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    showToastMock.mockClear();
  });

  function openPanel(extraLeafAttrs: Record<string, unknown> = {}) {
    const panel = new LayersPanel(container, makeAnimationController());
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph('points', extraLeafAttrs));
    panel.show();
    // Select the row so the controls act on it.
    // 'single' rather than 'add': add TOGGLES, and the panel already selects a
    // row on init, so add would deselect it.
    const first = panel.layerState.getLayers()[0];
    if (first) panel.layerState.select(first.path, 'single');
    return panel;
  }

  const field = (): HTMLInputElement | null =>
    container.querySelector<HTMLInputElement>('.luxar-layers-panel__number');

  function setField(value: string): void {
    const input = field()!;
    input.value = value;
    input.dispatchEvent(new Event('change'));
  }

  it('renders the field, blank when the layer authored no order', () => {
    const panel = openPanel();
    expect(field()).toBeTruthy();
    expect(field()!.value).toBe('');
    expect(field()!.placeholder).toBe('auto');
    panel.dispose();
  });

  it('shows an authored order', () => {
    const panel = openPanel({ layer_order: 7 });
    expect(field()!.value).toBe('7');
    panel.dispose();
  });

  it('shows an inherited order in the auto placeholder', () => {
    const graph: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: {},
      hasSpatialIndex: false,
      children: [
        {
          path: '/ordered',
          type: 'group',
          attrs: { layer_order: 5 },
          hasSpatialIndex: false,
          children: [
            {
              path: '/ordered/cloud',
              type: 'points',
              attrs: { layer: true },
              hasSpatialIndex: true,
            },
          ],
        },
      ],
    };
    const panel = new LayersPanel(container, makeAnimationController());
    panel.initFromScene(new THREE.Group(), graph);
    panel.show();
    panel.layerState.select('/ordered/cloud', 'single');

    expect(panel.layerState.getLayer('/ordered/cloud')!.layerOrder).toBe(5);
    expect(panel.layerState.getLayer('/ordered/cloud')!.layerOrderExplicit).toBe(false);
    expect(field()!.value).toBe('');
    expect(field()!.placeholder).toBe('auto (5)');

    setField('9');
    setField('');
    panel.layerState.select('/ordered/cloud', 'single');
    expect(field()!.value).toBe('');
    expect(field()!.placeholder).toBe('auto (5)');
    panel.dispose();
  });

  it('typing a value marks the layer explicit', () => {
    const panel = openPanel();
    setField('4');
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.layerOrder).toBe(4);
    expect(layer.layerOrderExplicit).toBe(true);
    panel.dispose();
  });

  it('an authored 0 is kept as a real band, not read as absent', () => {
    const panel = openPanel();
    setField('0');
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.layerOrder).toBe(0);
    expect(layer.layerOrderExplicit).toBe(true);
    panel.dispose();
  });

  it('blanking the field clears BOTH the value and the explicit flag', () => {
    const panel = openPanel({ layer_order: 7 });
    setField('');
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.layerOrder).toBeUndefined();
    expect(layer.layerOrderExplicit).toBe(false);
    panel.dispose();
  });

  it('accepts a negative order', () => {
    const panel = openPanel();
    setField('-3');
    expect(panel.layerState.getLayer('/cloud')!.layerOrder).toBe(-3);
    panel.dispose();
  });

  // Junk must clear rather than become 0: 0 is a real band, and inventing it
  // from unparseable input would state an order the user did not choose. The
  // field then echoes back what was actually stored.
  it('junk input clears, and the field echoes the stored state', () => {
    const panel = openPanel({ layer_order: 7 });
    setField('front');
    const layer = panel.layerState.getLayer('/cloud')!;
    expect(layer.layerOrder).toBeUndefined();
    expect(layer.layerOrderExplicit).toBe(false);
    expect(field()!.value).toBe('');
    panel.dispose();
  });

  it('rejects a fractional entry instead of silently changing it', () => {
    const panel = openPanel();
    setField('3.7');
    expect(panel.layerState.getLayer('/cloud')!.layerOrder).toBeUndefined();
    expect(field()!.value).toBe('');
    panel.dispose();
  });

  it('accepts exponent notation as the exact integer it denotes', () => {
    const panel = openPanel();
    setField('1e3');
    expect(panel.layerState.getLayer('/cloud')!.layerOrder).toBe(1000);
    panel.dispose();
  });

  it('rejects integers outside the JavaScript safe range', () => {
    const panel = openPanel();
    setField('9007199254740992');
    expect(panel.layerState.getLayer('/cloud')!.layerOrder).toBeUndefined();
    expect(field()!.value).toBe('');
    panel.dispose();
  });
});

describe('LayersPanel programmatic API (getLayerSummaries / setLayer)', () => {
  let container: HTMLElement;
  let animationController: AnimationController;

  beforeEach(() => {
    document.body.innerHTML = '';
    container = document.createElement('div');
    document.body.appendChild(container);
    animationController = makeAnimationController();
  });

  function openPanel(extraLeafAttrs: Record<string, unknown> = {}): LayersPanel {
    const panel = new LayersPanel(container, animationController);
    panel.initFromScene(new THREE.Group(), makeLayeredSceneGraph('points', extraLeafAttrs));
    return panel;
  }

  it('summarises each layer as copies in panel order', () => {
    const panel = openPanel({ layer_order: 2 });
    const summaries = panel.getLayerSummaries();
    expect(summaries).toHaveLength(1);
    const s = summaries[0];
    expect(s.path).toBe('/cloud');
    expect(s.name).toBe('cloud');
    expect(s.type).toBe('points');
    expect(s.visible).toBe(true);
    expect(s.opacity).toBe(1);
    expect(s.layerOrder).toBe(2);
    expect(s.colormap).toBeNull();
    expect(Array.isArray(s.displayRange)).toBe(true);
    expect(Array.isArray(s.dataRange)).toBe(true);
    // A copy: mutating it must not reach the live state.
    s.opacity = 0.1;
    expect(panel.layerState.getLayer('/cloud')!.opacity).toBe(1);
    panel.dispose();
  });

  it('reports an inherited/automatic order as null', () => {
    const panel = openPanel();
    expect(panel.getLayerSummaries()[0].layerOrder).toBeNull();
    panel.dispose();
  });

  it('applies only the fields present, through the state manager', () => {
    const panel = openPanel();
    const live = panel.layerState.getLayer('/cloud')!;
    const before = { gamma: live.gamma, blending: live.blendingMode };

    panel.setLayer('/cloud', { opacity: 0.4, visible: false });

    expect(live.opacity).toBe(0.4);
    expect(live.visible).toBe(false);
    expect(live.gamma).toBe(before.gamma);
    expect(live.blendingMode).toBe(before.blending);
    expect(panel.getLayerSummaries()[0]).toMatchObject({ opacity: 0.4, visible: false });
    // A material change is only visible when the loop runs.
    expect(animationController.startAnimation).toHaveBeenCalled();
    panel.dispose();
  });

  it('sets display range, gamma, blending and explicit order', () => {
    const panel = openPanel();
    const live = panel.layerState.getLayer('/cloud')!;
    const mid = (live.dataMin + live.dataMax) / 2;

    panel.setLayer('/cloud', {
      displayRange: [live.dataMin, mid],
      gamma: 2,
      blendingMode: 'max',
      layerOrder: 5,
    });

    expect(live.displayMin).toBe(live.dataMin);
    expect(live.displayMax).toBe(mid);
    expect(live.gamma).toBe(2);
    expect(live.blendingMode).toBe('max');
    expect(live.blendingModeExplicit).toBe(true);
    expect(live.layerOrder).toBe(5);
    expect(live.layerOrderExplicit).toBe(true);

    panel.setLayer('/cloud', { layerOrder: null });
    expect(live.layerOrderExplicit).toBe(false);
    expect(panel.getLayerSummaries()[0].layerOrder).toBeNull();
    panel.dispose();
  });

  it('applies a requested display range after switching to a colormap', () => {
    const panel = openPanel({ has_scalars: true, scalar_data_range: [10, 20] });
    const live = panel.layerState.getLayer('/cloud')!;

    panel.setLayer('/cloud', { colormap: 'viridis', displayRange: [12, 18] });

    expect(live.colormap).toBe('viridis');
    expect(live.scalarWindow).toBe(true);
    expect(live.displayMin).toBe(12);
    expect(live.displayMax).toBe(18);
    panel.dispose();
  });

  it('applies a requested display range after releasing a colormap', () => {
    const panel = openPanel({
      has_scalars: true,
      scalar_data_range: [10, 20],
      colormap: 'viridis',
    });
    const live = panel.layerState.getLayer('/cloud')!;

    panel.setLayer('/cloud', { colormap: null, displayRange: [0.2, 0.8] });

    expect(live.colormap).toBeUndefined();
    expect(live.scalarWindow).toBe(false);
    expect(live.displayMin).toBe(0.2);
    expect(live.displayMax).toBe(0.8);
    panel.dispose();
  });

  it('throws on an unknown path instead of failing silently', () => {
    const panel = openPanel();
    expect(() => panel.setLayer('/nope', { opacity: 0.5 })).toThrow(/unknown layer '\/nope'/);
    panel.dispose();
  });
});
