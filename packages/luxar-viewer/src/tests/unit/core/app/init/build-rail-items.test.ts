/**
 * Unit tests for core/app/init/build-rail-items.ts.
 *
 * `buildRailItems` is pure assembly of object literals — the action closures
 * only run on interaction. These tests stub the deps and verify the item set
 * + order, the Home button (momentary fit-scene + reset popover wiring), the
 * Navigation render hook (icon/tooltip mirror the live mode), and the Layers
 * disabled predicate (grayed when the scene has no layers).
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { buildRailItems, type RailItemsDeps } from '../../../../../core/app/init/build-rail-items';
import type { ControlRailItem } from '../../../../../ui/control-rail';

function makeDeps(
  overrides: {
    controlType?: 'orbit' | 'fly' | 'ortho';
    layerCount?: number;
    hasDims?: boolean;
  } = {}
) {
  const controlType = overrides.controlType ?? 'orbit';
  const layerCount = overrides.layerCount ?? 0;
  const hasDims = overrides.hasDims ?? true;
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
        togglePerformanceStats: vi.fn(),
        recenterCamera: vi.fn(),
      },
      panels: {
        getLayersPanel: vi.fn(),
        getRecordingPanel: vi.fn(),
        getScaleBar: vi.fn(),
        getColormapLegend: vi.fn(),
        getOverlayManager: vi.fn(),
      },
    },
    sceneManager: {
      getControlType: vi.fn().mockReturnValue(controlType),
      centerOnOrigin: vi.fn(),
    },
    sceneDims: {
      resetPositions: vi.fn(),
      hasNonDisplayedDimensions: vi.fn().mockReturnValue(hasDims),
    },
    renderingControls: {
      isVisible: vi.fn().mockReturnValue(false),
      saveSettings: vi.fn(),
      resetToDefaults: vi.fn(),
      settings: { cinematicMode: false },
    },
    animationController: { startAnimation: vi.fn() },
    adaptiveDPRManager: {},
    performanceMonitor: { visible: false },
    layersPanel: {
      isVisible: vi.fn().mockReturnValue(false),
      layerState: { count: layerCount },
      resetAllLayers: vi.fn(),
    },
    debugConsole: { toggle: vi.fn(), getIsVisible: vi.fn().mockReturnValue(false) },
    recordingPanel: { isVisible: vi.fn().mockReturnValue(false) },
  };
  return deps as unknown as RailItemsDeps;
}

/** A rail button stub carrying an <svg> + tooltip span, as buildButton produces. */
function makeButtonEl(): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.innerHTML = '<svg></svg><span class="luxar-control-rail__tip">Navigation<kbd>V</kbd></span>';
  return btn;
}

describe('buildRailItems', () => {
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
    afterEach(() => setFullscreen(false));

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
});
