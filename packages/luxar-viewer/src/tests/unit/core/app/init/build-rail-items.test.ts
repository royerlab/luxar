/**
 * Unit tests for core/app/init/build-rail-items.ts.
 *
 * `buildRailItems` is pure assembly of object literals — the action closures
 * only run on interaction. These tests stub the deps and verify the item set
 * + order, the Home button (momentary fit-scene + reset popover wiring), the
 * Navigation render hook (icon/tooltip mirror the live mode), and the Layers
 * disabled predicate (grayed when the scene has no layers).
 */

import { describe, it, expect, vi } from 'vitest';
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
    layersPanel: { isVisible: vi.fn().mockReturnValue(false), layerState: { count: layerCount } },
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

    it('popover wires the four reset actions to the right deps', () => {
      const deps = makeDeps();
      const home = buildRailItems(deps).find((i: ControlRailItem) => i.id === 'home')!;
      const host = document.createElement('div');
      home.popover!.build(host);

      const rows = Array.from(
        host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__action')
      );
      expect(
        rows.map((r) => r.querySelector('.luxar-control-rail__action-label')?.textContent)
      ).toEqual(['Fit scene', 'Center on origin', 'Reset dimensions', 'Reset rendering']);

      const d = deps as unknown as {
        ui: { commands: { recenterCamera: ReturnType<typeof vi.fn> } };
        sceneManager: { centerOnOrigin: ReturnType<typeof vi.fn> };
        sceneDims: { resetPositions: ReturnType<typeof vi.fn> };
        renderingControls: { resetToDefaults: ReturnType<typeof vi.fn> };
        animationController: { startAnimation: ReturnType<typeof vi.fn> };
      };
      rows[0].click();
      expect(d.ui.commands.recenterCamera).toHaveBeenCalledTimes(1);
      rows[1].click();
      expect(d.sceneManager.centerOnOrigin).toHaveBeenCalledTimes(1);
      rows[2].click();
      expect(d.sceneDims.resetPositions).toHaveBeenCalledTimes(1);
      rows[3].click();
      expect(d.renderingControls.resetToDefaults).toHaveBeenCalledTimes(1);
      expect(d.animationController.startAnimation).toHaveBeenCalledTimes(4);
    });

    it('grays the Reset dimensions row when the scene has no slider dimensions', () => {
      const home = findHome(makeDeps({ hasDims: false }));
      const host = document.createElement('div');
      home.popover!.build(host);
      const rows = host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__action');
      expect(rows[2].disabled).toBe(true);
      expect(rows[0].disabled).toBe(false);
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
