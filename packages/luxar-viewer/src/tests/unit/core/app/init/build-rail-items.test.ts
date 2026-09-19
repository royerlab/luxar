// @vitest-environment jsdom
/**
 * Unit tests for core/app/init/build-rail-items.ts.
 *
 * `buildRailItems` is pure assembly of object literals — the action closures
 * only run on interaction. These tests stub the deps and verify the item set
 * + order, the Home button (momentary fit-scene + reset popover wiring), the
 * Navigation render hook (icon/tooltip mirror the live mode), and the Layers
 * disabled predicate (grayed when the scene has no layers).
 */

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { buildRailItems, type RailItemsDeps } from '../../../../../core/app/init/build-rail-items';
import { RAIL_ICONS, type ControlRailItem } from '../../../../../ui/control-rail';
import { KeyAction } from '../../../../../input/input-handler/key-bindings/actions';
import {
  resetInputProfileForTests,
  setInputProfileOverride,
} from '../../../../../utils/input-capabilities';
import { notifier } from '../../../../../utils/cross-layer/notifier';
import { eventBus } from '../../../../../utils/cross-layer/event-bus';

function makeDeps(
  overrides: {
    controlType?: 'orbit' | 'fly' | 'ortho';
    layerCount?: number;
    hasDims?: boolean;
    /** Docked panels that are already open (drives the exclusivity tests). */
    renderVisible?: boolean;
    layersVisible?: boolean;
    recordingVisible?: boolean;
    /** Whether the loaded scene has sound nodes (drives the Sound button's hidden predicate). */
    hasSoundNodes?: boolean;
    muted?: boolean;
  } = {}
) {
  const controlType = overrides.controlType ?? 'orbit';
  const layerCount = overrides.layerCount ?? 0;
  const hasDims = overrides.hasDims ?? true;
  // The rail closes the OTHER docked panels through the shared panel accessors,
  // so those must hand back a real object with a toggle() to observe.
  const layersToggle = vi.fn();
  const recordingToggle = vi.fn();
  const deps = {
    ui: {
      commands: {
        toggleHelp: vi.fn(),
        toggleControlMode: vi.fn(),
        setControlMode: vi.fn(),
        toggleDimensionSliders: vi.fn(),
        toggleRenderingControls: vi.fn(),
        cycleDataMonitor: vi.fn(),
        toggleCinematicMode: vi.fn(),
        toggleFullscreen: vi.fn(),
        closeAllPanels: vi.fn(),
        handleEscape: vi.fn(),
        togglePerformanceStats: vi.fn(),
        recenterCamera: vi.fn(),
        toggleDatasetBrowser: vi.fn(),
      },
      panels: {
        getLayersPanel: vi.fn().mockReturnValue({ toggle: layersToggle }),
        getRecordingPanel: vi.fn().mockReturnValue({ toggle: recordingToggle }),
        getScaleBar: vi.fn(),
        getColormapLegend: vi.fn(),
        getOverlayManager: vi.fn(),
      },
    },
    shortcutForAction: vi.fn(() => undefined),
    sceneManager: {
      getControlType: vi.fn().mockReturnValue(controlType),
      centerOnOrigin: vi.fn(),
    },
    sceneDims: {
      resetPositions: vi.fn(),
      hasNonDisplayedDimensions: vi.fn().mockReturnValue(hasDims),
    },
    renderingControls: {
      isVisible: vi.fn().mockReturnValue(overrides.renderVisible ?? false),
      saveSettings: vi.fn(),
      resetToDefaults: vi.fn(),
      settings: { cinematicMode: false },
    },
    animationController: { startAnimation: vi.fn() },
    adaptiveDPRManager: {},
    densityGuard: {
      isEnabled: () => true,
      sessionDisabled: false,
      setEnabled: () => {},
      thinning: () => ({ nodes: 0, minKeep: 1 }),
      capElementsPerPixel: () => 4,
    },
    performanceMonitor: { visible: false },
    layersPanel: {
      isVisible: vi.fn().mockReturnValue(overrides.layersVisible ?? false),
      layerState: { count: layerCount },
      resetAllLayers: vi.fn(),
    },
    debugConsole: { toggle: vi.fn(), getIsVisible: vi.fn().mockReturnValue(false) },
    recordingPanel: { isVisible: vi.fn().mockReturnValue(overrides.recordingVisible ?? false) },
    audioEngine: {
      hasSoundNodes: vi.fn().mockReturnValue(overrides.hasSoundNodes ?? false),
      isMuted: vi.fn().mockReturnValue(overrides.muted ?? false),
      setMuted: vi.fn(),
      getState: vi.fn().mockReturnValue({
        state: 'running',
        muted: false,
        masterGain: 0.8,
        panningModel: 'equalpower',
        buses: { ambient: 0.6, voice: 1, effects: 0.8 },
        playing: [],
        hasSoundNodes: overrides.hasSoundNodes ?? false,
      }),
      setAudio: vi.fn(),
    },
  };
  return deps as unknown as RailItemsDeps;
}

/** Typed view of the mocks the exclusivity tests assert on. */
interface DepMocks {
  ui: {
    commands: { toggleRenderingControls: ReturnType<typeof vi.fn> };
    panels: {
      getLayersPanel: () => { toggle: ReturnType<typeof vi.fn> };
      getRecordingPanel: () => { toggle: ReturnType<typeof vi.fn> };
    };
  };
}

const mocks = (deps: RailItemsDeps): DepMocks => deps as unknown as DepMocks;

/** A rail button stub carrying an <svg> + tooltip span, as buildButton produces. */
function makeButtonEl(): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.innerHTML = '<svg></svg><span class="luxar-control-rail__tip">Navigation<kbd>V</kbd></span>';
  return btn;
}

describe('buildRailItems', () => {
  it('reads shortcut labels from the action registry', () => {
    const deps = makeDeps();
    deps.shortcutForAction = vi.fn((actionId) =>
      actionId === KeyAction.toggleHelp ? '?' : undefined
    );

    const help = buildRailItems(deps).find((item) => item.id === 'help');

    expect(help?.shortcut).toBe('?');
    expect(deps.shortcutForAction).toHaveBeenCalledWith(KeyAction.toggleHelp);
  });

  it('produces the expected item ids in order (no screenshot button)', () => {
    const items = buildRailItems(makeDeps());
    expect(items.map((i: ControlRailItem) => i.id)).toEqual([
      'help',
      'home',
      'nav',
      'dims',
      'render',
      'layers',
      'monitor',
      'data',
      'audio',
      'recording',
      'logs',
      'view',
      'settings',
      'perf',
    ]);
    expect(items.some((i: ControlRailItem) => i.id === 'screenshot')).toBe(false);
  });

  it('wires the popover triggers: home/nav/perf are context, settings is click', () => {
    const items = buildRailItems(makeDeps());
    const find = (id: string): ControlRailItem | undefined =>
      items.find((i: ControlRailItem) => i.id === id);
    expect(find('home')?.popover?.trigger).toBe('context');
    expect(find('nav')?.popover?.trigger).toBe('context');
    expect(find('perf')?.popover?.trigger).toBe('context');
    expect(find('settings')?.popover?.trigger).toBe('click');
  });

  describe('Home button', () => {
    const findHome = (deps = makeDeps()): ControlRailItem =>
      buildRailItems(deps).find((i: ControlRailItem) => i.id === 'home')!;

    it('is momentary (a one-shot action, never shows an active state)', () => {
      expect(findHome().momentary).toBe(true);
    });

    it('left-click fires the F-key recenter command', () => {
      const deps = makeDeps();
      const home = buildRailItems(deps).find((i: ControlRailItem) => i.id === 'home')!;
      home.activate();
      expect(
        (deps as unknown as { ui: { commands: { recenterCamera: ReturnType<typeof vi.fn> } } }).ui
          .commands.recenterCamera
      ).toHaveBeenCalledTimes(1);
    });

    it('popover wires the five reset actions to the right deps', () => {
      const deps = makeDeps({ layerCount: 2 }); // layers present → Reset layers enabled
      const home = buildRailItems(deps).find((i: ControlRailItem) => i.id === 'home')!;
      const host = document.createElement('div');
      home.popover!.build(host);

      const chips = Array.from(
        host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__home-chip')
      );
      expect(chips.map((c) => c.getAttribute('aria-label')?.split(' — ')[0])).toEqual([
        'Fit scene',
        'Center on origin',
        'Reset dimensions',
        'Reset rendering',
        'Reset layers',
      ]);

      const d = deps as unknown as {
        ui: { commands: { recenterCamera: ReturnType<typeof vi.fn> } };
        sceneManager: { centerOnOrigin: ReturnType<typeof vi.fn> };
        sceneDims: { resetPositions: ReturnType<typeof vi.fn> };
        renderingControls: { resetToDefaults: ReturnType<typeof vi.fn> };
        layersPanel: { resetAllLayers: ReturnType<typeof vi.fn> };
        animationController: { startAnimation: ReturnType<typeof vi.fn> };
      };
      chips[0].click();
      expect(d.ui.commands.recenterCamera).toHaveBeenCalledTimes(1);
      chips[1].click();
      expect(d.sceneManager.centerOnOrigin).toHaveBeenCalledTimes(1);
      chips[2].click();
      expect(d.sceneDims.resetPositions).toHaveBeenCalledTimes(1);
      chips[3].click();
      expect(d.renderingControls.resetToDefaults).toHaveBeenCalledTimes(1);
      chips[4].click();
      expect(d.layersPanel.resetAllLayers).toHaveBeenCalledTimes(1);
      expect(d.animationController.startAnimation).toHaveBeenCalledTimes(5);
    });

    it('grays the Reset dimensions chip when the scene has no slider dimensions', () => {
      const home = findHome(makeDeps({ hasDims: false }));
      const host = document.createElement('div');
      home.popover!.build(host);
      const chips = host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__home-chip');
      expect(chips[2].disabled).toBe(true);
      expect(chips[0].disabled).toBe(false);
    });

    it('grays the Reset layers chip when the scene has no layers', () => {
      const home = findHome(makeDeps({ layerCount: 0 }));
      const host = document.createElement('div');
      home.popover!.build(host);
      const chips = host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__home-chip');
      expect(chips[4].disabled).toBe(true);
    });
  });

  describe('Navigation render hook', () => {
    it('keeps the registry-derived shortcut in the rendered tooltip and aria-label', () => {
      const deps = makeDeps({ controlType: 'orbit' });
      deps.shortcutForAction = vi.fn((actionId) =>
        actionId === KeyAction.toggleControlMode ? 'Z' : undefined
      );
      const nav = buildRailItems(deps).find((item: ControlRailItem) => item.id === 'nav')!;
      const btn = makeButtonEl();

      nav.render!(btn);

      expect(btn.querySelector('.luxar-control-rail__tip kbd')?.textContent).toBe('Z');
      expect(btn.getAttribute('aria-label')).toContain('(Z)');
    });

    it('omits shortcut markup when the registry has no control-mode label', () => {
      const nav = buildRailItems(makeDeps({ controlType: 'orbit' })).find(
        (item: ControlRailItem) => item.id === 'nav'
      )!;
      const btn = makeButtonEl();

      nav.render!(btn);

      expect(btn.querySelector('.luxar-control-rail__tip kbd')).toBeNull();
      expect(btn.querySelector('.luxar-control-rail__tip')?.textContent).toBe('Navigation · Orbit');
      expect(btn.getAttribute('aria-label')).toBe(
        'Navigation: Orbit — click for fly, right-click or hold for options'
      );
    });

    it('shows the ortho 2×2 grid icon + "Ortho" tooltip when the mode is ortho', () => {
      const nav = buildRailItems(makeDeps({ controlType: 'ortho' })).find(
        (i: ControlRailItem) => i.id === 'nav'
      )!;
      const btn = makeButtonEl();
      nav.render!(btn);
      expect(btn.querySelectorAll('svg rect')).toHaveLength(4); // 2×2 grid, not a cube
      expect(btn.querySelector('.luxar-control-rail__tip')?.textContent).toContain('Ortho');
      expect(btn.getAttribute('aria-label')).toContain('Ortho');
      expect(btn.dataset.navMode).toBe('ortho');
    });

    it('reflects fly mode (paper-plane icon path + "Fly" tooltip)', () => {
      const nav = buildRailItems(makeDeps({ controlType: 'fly' })).find(
        (i: ControlRailItem) => i.id === 'nav'
      )!;
      const btn = makeButtonEl();
      nav.render!(btn);
      expect(btn.querySelectorAll('svg rect')).toHaveLength(0);
      expect(btn.querySelector('svg path')).not.toBeNull();
      expect(btn.querySelector('.luxar-control-rail__tip')?.textContent).toContain('Fly');
    });

    it('is a no-op when the mode has not changed — keeps the SAME svg node (dataset guard)', () => {
      const deps = makeDeps({ controlType: 'orbit' });
      const nav = buildRailItems(deps).find((i: ControlRailItem) => i.id === 'nav')!;
      const btn = makeButtonEl();
      nav.render!(btn);
      const svgAfterFirst = btn.querySelector('svg');
      nav.render!(btn); // second call, same mode → guard should early-return
      // Assert NODE IDENTITY, not outerHTML string: the `if (dataset.navMode ===
      // type) return` guard skips the `svg.outerHTML = icon` reassignment, so the
      // exact same element persists. Without the guard the second call replaces
      // it with a fresh node (identical HTML) — a string compare wouldn't notice,
      // node identity does. This test fails if the guard is removed.
      expect(btn.querySelector('svg')).toBe(svgAfterFirst);
    });
  });

  describe('View options flyout', () => {
    const findView = (deps = makeDeps()): ControlRailItem =>
      buildRailItems(deps).find((i: ControlRailItem) => i.id === 'view')!;

    /** Stub the standard fullscreen element (jsdom has no real fullscreen). */
    const setFullscreen = (on: boolean): void => {
      Object.defineProperty(document, 'fullscreenElement', {
        configurable: true,
        get: () => (on ? document.body : null),
      });
    };
    /** jsdom has no Fullscreen API at all: declare it available like a desktop browser. */
    const setFullscreenEnabled = (enabled: boolean | undefined): void => {
      Object.defineProperty(document, 'fullscreenEnabled', {
        configurable: true,
        get: () => enabled,
      });
    };
    beforeEach(() => setFullscreenEnabled(true));
    afterEach(() => {
      setFullscreen(false);
      setFullscreenEnabled(undefined);
    });

    it('omits the fullscreen chip where the Fullscreen API is absent (iPhone Safari)', () => {
      setFullscreenEnabled(undefined);
      expect(findView().flyout!.map((t) => t.id)).toEqual([
        'scalebar',
        'legend',
        'overlays',
        'cinematic',
      ]);
      setFullscreenEnabled(false);
      expect(findView().flyout!.some((t) => t.id === 'fullscreen')).toBe(false);
    });

    it('carries the five view toggles in order, fullscreen last', () => {
      expect(findView().flyout!.map((t) => t.id)).toEqual([
        'scalebar',
        'legend',
        'overlays',
        'cinematic',
        'fullscreen',
      ]);
    });

    it('fullscreen chip fires the same command as the Space shortcut', () => {
      const deps = makeDeps();
      const fs = findView(deps).flyout!.find((t) => t.id === 'fullscreen')!;
      fs.activate();
      expect(
        (deps as unknown as { ui: { commands: { toggleFullscreen: ReturnType<typeof vi.fn> } } }).ui
          .commands.toggleFullscreen
      ).toHaveBeenCalledTimes(1);
    });

    it('fullscreen chip active-state tracks the live fullscreen element', () => {
      const fs = findView().flyout!.find((t) => t.id === 'fullscreen')!;
      setFullscreen(false);
      expect(fs.isActive!()).toBe(false);
      setFullscreen(true);
      expect(fs.isActive!()).toBe(true);
    });

    it('ONLY the fullscreen chip is excluded from the parent View-button glow', () => {
      // Fullscreen is a session-long ambient state — it must not light the
      // View button all session. The other toggles (cinematic etc.) keep
      // the parent-glow contract.
      const toggles = findView().flyout!;
      for (const t of toggles) {
        expect(!!t.excludeFromParentActive).toBe(t.id === 'fullscreen');
      }
    });
  });

  describe('One docked panel at a time', () => {
    const find = (deps: RailItemsDeps, id: string): ControlRailItem =>
      buildRailItems(deps).find((i: ControlRailItem) => i.id === id)!;

    it('Rendering closes the other two docked panels before opening', () => {
      const deps = makeDeps({ layersVisible: true, recordingVisible: true, layerCount: 3 });
      find(deps, 'render').activate();
      const m = mocks(deps);
      expect(m.ui.panels.getLayersPanel().toggle).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getRecordingPanel().toggle).toHaveBeenCalledTimes(1);
      // Exactly once: the close pass skips 'render', so only the open remains.
      expect(m.ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
    });

    it('Layers closes Rendering + Recording before opening', () => {
      const deps = makeDeps({ renderVisible: true, recordingVisible: true, layerCount: 3 });
      find(deps, 'layers').activate();
      const m = mocks(deps);
      expect(m.ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getRecordingPanel().toggle).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getLayersPanel().toggle).toHaveBeenCalledTimes(1); // the open itself
    });

    it('Recording closes Rendering + Layers before opening', () => {
      const deps = makeDeps({ renderVisible: true, layersVisible: true, layerCount: 3 });
      find(deps, 'recording').activate();
      const m = mocks(deps);
      expect(m.ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getLayersPanel().toggle).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getRecordingPanel().toggle).toHaveBeenCalledTimes(1); // the open itself
    });

    it('CLOSING an already-open panel leaves the others alone', () => {
      // Rendering is the visible one — clicking it must just close it, not
      // churn the (hidden) siblings. Guards the `if (!isVisible())` gate.
      const deps = makeDeps({ renderVisible: true, layersVisible: true, layerCount: 3 });
      find(deps, 'render').activate();
      const m = mocks(deps);
      expect(m.ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getLayersPanel().toggle).not.toHaveBeenCalled();
    });

    // Settings (click-trigger) and Home (context-trigger) cover both popover
    // flavours; Navigation/Performance call the identical helper but need the
    // whole rendering-settings object stubbed to build at all.
    it.each(['settings', 'home'])('building the %s popover closes every docked panel', (id) => {
      const deps = makeDeps({
        renderVisible: true,
        layersVisible: true,
        recordingVisible: true,
        layerCount: 3,
      });
      find(deps, id).popover!.build(document.createElement('div'));
      const m = mocks(deps);
      expect(m.ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getLayersPanel().toggle).toHaveBeenCalledTimes(1);
      expect(m.ui.panels.getRecordingPanel().toggle).toHaveBeenCalledTimes(1);
    });

    it('closes nothing when no docked panel is open', () => {
      const deps = makeDeps({ layerCount: 3 });
      find(deps, 'settings').popover!.build(document.createElement('div'));
      const m = mocks(deps);
      expect(m.ui.commands.toggleRenderingControls).not.toHaveBeenCalled();
      expect(m.ui.panels.getLayersPanel().toggle).not.toHaveBeenCalled();
      expect(m.ui.panels.getRecordingPanel().toggle).not.toHaveBeenCalled();
    });
  });

  describe('Layers disabled predicate', () => {
    it('is disabled when the scene has no layers', () => {
      const layers = buildRailItems(makeDeps({ layerCount: 0 })).find(
        (i: ControlRailItem) => i.id === 'layers'
      )!;
      expect(layers.disabled!()).toBe(true);
    });

    it('is enabled once the scene has layers', () => {
      const layers = buildRailItems(makeDeps({ layerCount: 5 })).find(
        (i: ControlRailItem) => i.id === 'layers'
      )!;
      expect(layers.disabled!()).toBe(false);
    });
  });
  describe('Sound button', () => {
    const findAudio = (deps: RailItemsDeps) => buildRailItems(deps).find((i) => i.id === 'audio')!;
    const engine = (deps: RailItemsDeps) =>
      (deps as unknown as { audioEngine: { setMuted: ReturnType<typeof vi.fn> } }).audioEngine;

    it('is hidden (not merely disabled) when the scene has no sound nodes', () => {
      expect(findAudio(makeDeps({ hasSoundNodes: false })).hidden!()).toBe(true);
      expect(findAudio(makeDeps({ hasSoundNodes: true })).hidden!()).toBe(false);
      expect(findAudio(makeDeps()).disabled).toBeUndefined();
    });

    it('left-click toggles the mute and the active state mirrors it', () => {
      const deps = makeDeps({ hasSoundNodes: true, muted: false });
      const item = findAudio(deps);
      item.activate();
      expect(engine(deps).setMuted).toHaveBeenCalledWith(true);
      expect(item.isActive!()).toBe(false);
      expect(findAudio(makeDeps({ hasSoundNodes: true, muted: true })).isActive!()).toBe(true);
    });

    it('render swaps the icon and tooltip with the mute state', () => {
      const btn = makeButtonEl();
      findAudio(makeDeps({ hasSoundNodes: true, muted: true })).render!(btn);
      expect(btn.dataset.audioState).toBe('muted');
      expect(btn.querySelector('.luxar-control-rail__tip')?.textContent).toBe('Sound · Muted');
      expect(btn.getAttribute('aria-label')).toBe(
        'Sound: muted — click to unmute, right-click or hold for the mixer'
      );
      expect(btn.innerHTML).toContain(RAIL_ICONS.audioMuted.slice(0, 40));
      findAudio(makeDeps({ hasSoundNodes: true, muted: false })).render!(btn);
      expect(btn.dataset.audioState).toBe('on');
      expect(btn.querySelector('.luxar-control-rail__tip')?.textContent).toBe('Sound · On');
      expect(btn.getAttribute('aria-label')).toBe(
        'Sound: on — click to mute, right-click or hold for the mixer'
      );
    });

    it('right-click opens the Sound popover', () => {
      const item = findAudio(makeDeps({ hasSoundNodes: true }));
      expect(item.popover?.trigger).toBe('context');
      expect(item.popover?.title).toBe('Sound');
    });
  });

  describe('coarse pointer (touch-first device)', () => {
    afterEach(() => {
      resetInputProfileForTests();
      vi.restoreAllMocks();
    });

    it('adds a momentary Hide panels item that closes panels without the fullscreen Escape path', () => {
      setInputProfileOverride('touch');
      const deps = makeDeps();
      const items = buildRailItems(deps);
      const hide = items.find((i: ControlRailItem) => i.id === 'hide-panels')!;
      expect(hide).toBeDefined();
      expect(hide.momentary).toBe(true);
      expect(RAIL_ICONS.hidePanels).toContain('<svg');
      expect(hide.icon).toBe(RAIL_ICONS.hidePanels);
      expect(hide.icon).not.toBe(RAIL_ICONS.view);
      hide.activate();
      expect(
        (deps as unknown as { ui: { commands: { closeAllPanels: ReturnType<typeof vi.fn> } } }).ui
          .commands.closeAllPanels
      ).toHaveBeenCalledTimes(1);
      expect(
        (deps as unknown as { ui: { commands: { handleEscape: ReturnType<typeof vi.fn> } } }).ui
          .commands.handleEscape
      ).not.toHaveBeenCalled();
    });

    it('does not add Hide panels on a mouse-and-keyboard machine', () => {
      setInputProfileOverride('mouse');
      expect(buildRailItems(makeDeps()).some((i: ControlRailItem) => i.id === 'hide-panels')).toBe(
        false
      );
    });

    it('help activation closes the docked panels under a coarse pointer, not under a mouse', () => {
      const hideHelp = vi.spyOn(notifier, 'hideHelp').mockImplementation(() => {});
      setInputProfileOverride('touch');
      let deps = makeDeps({ renderVisible: true });
      buildRailItems(deps)
        .find((i: ControlRailItem) => i.id === 'help')!
        .activate();
      const commands = (
        deps as unknown as {
          ui: {
            commands: {
              toggleRenderingControls: ReturnType<typeof vi.fn>;
              toggleHelp: ReturnType<typeof vi.fn>;
            };
          };
        }
      ).ui.commands;
      expect(commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      expect(commands.toggleHelp).toHaveBeenCalledTimes(1);
      expect(hideHelp).not.toHaveBeenCalled();

      setInputProfileOverride('mouse');
      deps = makeDeps({ renderVisible: true });
      buildRailItems(deps)
        .find((i: ControlRailItem) => i.id === 'help')!
        .activate();
      expect(
        (
          deps as unknown as {
            ui: { commands: { toggleRenderingControls: ReturnType<typeof vi.fn> } };
          }
        ).ui.commands.toggleRenderingControls
      ).not.toHaveBeenCalled();
      hideHelp.mockRestore();
    });

    it('monitor activation closes other coarse surfaces without hiding the monitor', () => {
      const hideHelp = vi.spyOn(notifier, 'hideHelp').mockImplementation(() => {});
      const hiddenPanels: string[] = [];
      const off = eventBus.on('panel-hide', ({ panelId }) => hiddenPanels.push(panelId));
      setInputProfileOverride('touch');

      const deps = makeDeps({ renderVisible: true });
      buildRailItems(deps)
        .find((item) => item.id === 'monitor')!
        .activate();

      expect(hideHelp).toHaveBeenCalledTimes(1);
      expect(hiddenPanels).not.toContain('data-monitor');
      expect(mocks(deps).ui.commands.toggleRenderingControls).toHaveBeenCalledTimes(1);
      off();
      hideHelp.mockRestore();
    });

    it('docked panels close Help and the data monitor only under a coarse pointer', () => {
      const hideHelp = vi.spyOn(notifier, 'hideHelp').mockImplementation(() => {});
      const hiddenPanels: string[] = [];
      const off = eventBus.on('panel-hide', ({ panelId }) => hiddenPanels.push(panelId));

      setInputProfileOverride('touch');
      for (const id of ['render', 'layers', 'recording']) {
        const deps = makeDeps({ layerCount: 3 });
        buildRailItems(deps)
          .find((item) => item.id === id)!
          .activate();
      }
      expect(hideHelp).toHaveBeenCalledTimes(3);
      expect(hiddenPanels).toEqual(['data-monitor', 'data-monitor', 'data-monitor']);

      hideHelp.mockClear();
      hiddenPanels.length = 0;
      setInputProfileOverride('mouse');
      for (const id of ['render', 'layers', 'recording']) {
        const deps = makeDeps({ layerCount: 3 });
        buildRailItems(deps)
          .find((item) => item.id === id)!
          .activate();
      }
      expect(hideHelp).not.toHaveBeenCalled();
      expect(hiddenPanels).toEqual([]);
      off();
    });
  });
});
