/**
 * Unit tests for core/app/init/build-rail-items.ts.
 *
 * `buildRailItems` is pure assembly of object literals — the action closures
 * only run on interaction. These tests stub the 9 deps and verify the item set
 * + order, the Navigation render hook (icon/tooltip mirror the live mode), and
 * the Layers disabled predicate (grayed when the scene has no layers).
 */

import { describe, it, expect, vi } from 'vitest';
import { buildRailItems, type RailItemsDeps } from '../../../../../core/app/init/build-rail-items';
import type { ControlRailItem } from '../../../../../ui/control-rail';

function makeDeps(
  overrides: { controlType?: 'orbit' | 'fly' | 'ortho'; layerCount?: number } = {}
) {
  const controlType = overrides.controlType ?? 'orbit';
  const layerCount = overrides.layerCount ?? 0;
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
      },
      panels: {
        getLayersPanel: vi.fn(),
        getRecordingPanel: vi.fn(),
        getScaleBar: vi.fn(),
        getColormapLegend: vi.fn(),
        getOverlayManager: vi.fn(),
      },
    },
    sceneManager: { getControlType: vi.fn().mockReturnValue(controlType) },
    renderingControls: {
      isVisible: vi.fn().mockReturnValue(false),
      saveSettings: vi.fn(),
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

  it('wires the popover triggers: nav/perf are context, settings is click', () => {
    const items = buildRailItems(makeDeps());
    const find = (id: string): ControlRailItem | undefined =>
      items.find((i: ControlRailItem) => i.id === id);
    expect(find('nav')?.popover?.trigger).toBe('context');
    expect(find('perf')?.popover?.trigger).toBe('context');
    expect(find('settings')?.popover?.trigger).toBe('click');
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
