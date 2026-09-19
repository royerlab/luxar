/**
 * Unit tests for core/viewer-config-applier.ts.
 *
 * The applier is a pure dispatch function — it takes a ZarrViewerConfig
 * and a typed ports object, then routes individual fields to the right
 * port methods. Tests pass a stub ports object and assert the right
 * methods are called for each field combination.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  applyViewerConfigState,
  type ViewerConfigPorts,
} from '../../../../../core/app/viewer-config/apply-state';
import type { ZarrViewerConfig } from '../../../../../types/zarr';

// vitest's Mock type doesn't structurally satisfy `() => void` so we keep the
// stub shape inline + cast to ViewerConfigPorts at the call site rather than
// inheriting from the interface.
interface PortStubs {
  showHelp: ReturnType<typeof vi.fn>;
  renderingControls: { show: ReturnType<typeof vi.fn>; hide: ReturnType<typeof vi.fn> };
  performanceMonitor: { show: ReturnType<typeof vi.fn> };
  inputHandler: { showDimensionSliders: ReturnType<typeof vi.fn> };
  scaleBar?: { show: ReturnType<typeof vi.fn>; hide: ReturnType<typeof vi.fn> };
  layersPanel?: { show: ReturnType<typeof vi.fn>; hide: ReturnType<typeof vi.fn> };
  overlayManager?: { show: ReturnType<typeof vi.fn>; hide: ReturnType<typeof vi.fn> };
  setTheme: ReturnType<typeof vi.fn>;
  setDimensionValue: ReturnType<typeof vi.fn>;
  setDocumentTitle: ReturnType<typeof vi.fn>;
  startDimensionAnimation?: ReturnType<typeof vi.fn>;
  setDefaultLadderDepth?: ReturnType<typeof vi.fn>;
}

function makePorts(
  overrides: Partial<{
    scaleBar: boolean;
    layersPanel: boolean;
    overlayManager: boolean;
    startDimensionAnimation: boolean;
  }> = {}
): PortStubs {
  const showSlash = () => ({ show: vi.fn(), hide: vi.fn() });
  return {
    showHelp: vi.fn(),
    renderingControls: showSlash(),
    performanceMonitor: { show: vi.fn() },
    inputHandler: { showDimensionSliders: vi.fn() },
    scaleBar: overrides.scaleBar === false ? undefined : showSlash(),
    layersPanel: overrides.layersPanel === false ? undefined : showSlash(),
    overlayManager: overrides.overlayManager === false ? undefined : showSlash(),
    setTheme: vi.fn(),
    setDimensionValue: vi.fn(),
    setDocumentTitle: vi.fn(),
    startDimensionAnimation: overrides.startDimensionAnimation === false ? undefined : vi.fn(),
    setDefaultLadderDepth: vi.fn(),
  };
}

const asPorts = (p: PortStubs) => p as unknown as ViewerConfigPorts;

describe('applyViewerConfigState', () => {
  let ports: PortStubs;

  beforeEach(() => {
    ports = makePorts();
  });

  // core.md W12 strengthening: previously each "returns silently" test
  // spot-checked 3 of the 8 port methods. A mutation that called
  // `inputHandler.showDimensionSliders()` unconditionally would pass the
  // 3-method spot check. Helper below assert NO port method on the
  // entire ViewerConfigPorts surface fires.
  function expectNoPortMethodsCalled(p: PortStubs): void {
    expect(p.showHelp).not.toHaveBeenCalled();
    expect(p.renderingControls.show).not.toHaveBeenCalled();
    expect(p.renderingControls.hide).not.toHaveBeenCalled();
    expect(p.performanceMonitor.show).not.toHaveBeenCalled();
    expect(p.inputHandler.showDimensionSliders).not.toHaveBeenCalled();
    expect(p.scaleBar?.show).not.toHaveBeenCalled();
    expect(p.scaleBar?.hide).not.toHaveBeenCalled();
    expect(p.layersPanel?.show).not.toHaveBeenCalled();
    expect(p.layersPanel?.hide).not.toHaveBeenCalled();
    expect(p.overlayManager?.show).not.toHaveBeenCalled();
    expect(p.overlayManager?.hide).not.toHaveBeenCalled();
    expect(p.setTheme).not.toHaveBeenCalled();
    expect(p.setDimensionValue).not.toHaveBeenCalled();
  }

  it('returns silently when viewerConfig is undefined (NO port method fires)', () => {
    applyViewerConfigState(undefined, asPorts(ports));
    expectNoPortMethodsCalled(ports);
  });

  it('returns silently when viewerConfig is empty (NO port method fires)', () => {
    applyViewerConfigState({}, asPorts(ports));
    expectNoPortMethodsCalled(ports);
  });

  it('returns silently when viewerConfig.ui is an empty object', () => {
    // Boundary: `{ ui: {} }` is NOT the same as `undefined`. The
    // `if (ui)` branch is taken but every nested `=== true` check
    // short-circuits because every property is missing.
    applyViewerConfigState({ ui: {} }, asPorts(ports));
    expectNoPortMethodsCalled(ports);
  });

  describe('UI panel visibility', () => {
    it('show_help: true → showHelp()', () => {
      applyViewerConfigState({ ui: { show_help: true } }, asPorts(ports));
      expect(ports.showHelp).toHaveBeenCalled();
    });

    it('show_help: false is a no-op (only true triggers)', () => {
      applyViewerConfigState({ ui: { show_help: false } }, asPorts(ports));
      expect(ports.showHelp).not.toHaveBeenCalled();
    });

    it('show_rendering_controls: true → renderingControls.show()', () => {
      applyViewerConfigState({ ui: { show_rendering_controls: true } }, asPorts(ports));
      expect(ports.renderingControls.show).toHaveBeenCalled();
      expect(ports.renderingControls.hide).not.toHaveBeenCalled();
    });

    it('show_rendering_controls: false → renderingControls.hide()', () => {
      applyViewerConfigState({ ui: { show_rendering_controls: false } }, asPorts(ports));
      expect(ports.renderingControls.hide).toHaveBeenCalled();
      expect(ports.renderingControls.show).not.toHaveBeenCalled();
    });

    it('show_performance_monitor: true → performanceMonitor.show()', () => {
      applyViewerConfigState({ ui: { show_performance_monitor: true } }, asPorts(ports));
      expect(ports.performanceMonitor.show).toHaveBeenCalled();
    });

    it('show_dimensions: true → inputHandler.showDimensionSliders()', () => {
      applyViewerConfigState({ ui: { show_dimensions: true } }, asPorts(ports));
      expect(ports.inputHandler.showDimensionSliders).toHaveBeenCalled();
    });

    it('show_scale_bar: true → scaleBar.show() when present', () => {
      applyViewerConfigState({ ui: { show_scale_bar: true } }, asPorts(ports));
      expect(ports.scaleBar?.show).toHaveBeenCalled();
    });

    it('show_scale_bar: false → scaleBar.hide() when present', () => {
      applyViewerConfigState({ ui: { show_scale_bar: false } }, asPorts(ports));
      expect(ports.scaleBar?.hide).toHaveBeenCalled();
    });

    it('show_scale_bar is a no-op when scaleBar is absent', () => {
      // core.md W13 strengthening: previously this test only asserted
      // that ports.scaleBar (configured to be absent in the setup)
      // was undefined — which is trivially true. The real contract is
      // that:
      //   (a) the call doesn't throw on the absent port, AND
      //   (b) OTHER ports (renderingControls, layersPanel, etc.) are
      //       unaffected by the absent-scaleBar branch.
      ports = makePorts({ scaleBar: false });
      expect(() =>
        applyViewerConfigState(
          {
            ui: { show_scale_bar: true, show_rendering_controls: true, show_layers: true },
          },
          asPorts(ports)
        )
      ).not.toThrow();
      expect(ports.scaleBar).toBeUndefined();
      // Adjacent ports were still called — the absent-scaleBar branch
      // didn't short-circuit the whole UI block.
      expect(ports.renderingControls.show).toHaveBeenCalled();
      expect(ports.layersPanel?.show).toHaveBeenCalled();
    });

    it('show_layers: true → layersPanel.show() when present', () => {
      applyViewerConfigState({ ui: { show_layers: true } }, asPorts(ports));
      expect(ports.layersPanel?.show).toHaveBeenCalled();
    });

    it('show_layers: false → layersPanel.hide() when present', () => {
      applyViewerConfigState({ ui: { show_layers: false } }, asPorts(ports));
      expect(ports.layersPanel?.hide).toHaveBeenCalled();
    });

    it('show_layers is a no-op when layersPanel is absent', () => {
      // W13 strengthening: same pattern as show_scale_bar — assert
      // adjacent ports are still reached.
      ports = makePorts({ layersPanel: false });
      expect(() =>
        applyViewerConfigState(
          {
            ui: { show_layers: true, show_rendering_controls: true, show_scale_bar: true },
          },
          asPorts(ports)
        )
      ).not.toThrow();
      expect(ports.layersPanel).toBeUndefined();
      expect(ports.renderingControls.show).toHaveBeenCalled();
      expect(ports.scaleBar?.show).toHaveBeenCalled();
    });

    it('show_overlays: true → overlayManager.show() when present', () => {
      applyViewerConfigState({ ui: { show_overlays: true } }, asPorts(ports));
      expect(ports.overlayManager?.show).toHaveBeenCalled();
    });

    it('show_overlays: false → overlayManager.hide() when present', () => {
      applyViewerConfigState({ ui: { show_overlays: false } }, asPorts(ports));
      expect(ports.overlayManager?.hide).toHaveBeenCalled();
    });

    it('show_overlays is a no-op when overlayManager is absent', () => {
      // W13 strengthening: same pattern.
      ports = makePorts({ overlayManager: false });
      expect(() =>
        applyViewerConfigState(
          {
            ui: { show_overlays: true, show_rendering_controls: true, show_layers: true },
          },
          asPorts(ports)
        )
      ).not.toThrow();
      expect(ports.overlayManager).toBeUndefined();
      expect(ports.renderingControls.show).toHaveBeenCalled();
      expect(ports.layersPanel?.show).toHaveBeenCalled();
    });

    it('applies all UI fields in one shot', () => {
      const config: ZarrViewerConfig = {
        ui: {
          show_help: true,
          show_rendering_controls: true,
          show_performance_monitor: true,
          show_dimensions: true,
          show_scale_bar: true,
          show_layers: true,
          show_overlays: true,
        },
      };
      applyViewerConfigState(config, asPorts(ports));
      expect(ports.showHelp).toHaveBeenCalled();
      expect(ports.renderingControls.show).toHaveBeenCalled();
      expect(ports.performanceMonitor.show).toHaveBeenCalled();
      expect(ports.inputHandler.showDimensionSliders).toHaveBeenCalled();
      expect(ports.scaleBar?.show).toHaveBeenCalled();
      expect(ports.layersPanel?.show).toHaveBeenCalled();
      expect(ports.overlayManager?.show).toHaveBeenCalled();
    });
  });

  describe('theme', () => {
    it('forwards theme id to setTheme', () => {
      applyViewerConfigState({ theme: 'dark' }, asPorts(ports));
      expect(ports.setTheme).toHaveBeenCalledWith('dark');
    });

    it('skips setTheme when theme is missing', () => {
      applyViewerConfigState({ ui: { show_help: true } }, asPorts(ports));
      expect(ports.setTheme).not.toHaveBeenCalled();
    });
  });

  describe('dimensions / current_step', () => {
    it('forwards each step value via setDimensionValue', () => {
      applyViewerConfigState({ dimensions: { current_step: [10, 20, 30] } }, asPorts(ports));
      expect(ports.setDimensionValue).toHaveBeenCalledTimes(3);
      expect(ports.setDimensionValue).toHaveBeenNthCalledWith(1, 0, 10);
      expect(ports.setDimensionValue).toHaveBeenNthCalledWith(2, 1, 20);
      expect(ports.setDimensionValue).toHaveBeenNthCalledWith(3, 2, 30);
    });

    it('skips setDimensionValue when current_step is missing', () => {
      applyViewerConfigState({ dimensions: {} }, asPorts(ports));
      expect(ports.setDimensionValue).not.toHaveBeenCalled();
    });

    it('skips setDimensionValue when dimensions is missing entirely', () => {
      applyViewerConfigState({ ui: {} }, asPorts(ports));
      expect(ports.setDimensionValue).not.toHaveBeenCalled();
    });

    it('handles empty current_step array gracefully', () => {
      applyViewerConfigState({ dimensions: { current_step: [] } }, asPorts(ports));
      expect(ports.setDimensionValue).not.toHaveBeenCalled();
    });
  });

  describe('title', () => {
    it('sets the document title from viewer_config.title (trimmed)', () => {
      applyViewerConfigState({ title: '  Rivers of Earth  ' }, asPorts(ports));
      expect(ports.setDocumentTitle).toHaveBeenCalledWith('Rivers of Earth');
    });

    it('ignores an absent or blank title (URL fallback stays in force)', () => {
      applyViewerConfigState({}, asPorts(ports));
      applyViewerConfigState({ title: '   ' }, asPorts(ports));
      expect(ports.setDocumentTitle).not.toHaveBeenCalled();
    });

    it('ignores a non-string title without aborting the rest of the pass', () => {
      // viewer_config is untyped JSON from the scene's zarr attributes, so a
      // number here is reachable from any hand-authored or third-party store.
      // Calling .trim() on it would throw and strand the load before the
      // theme, the dimension state, and the render loop that follow.
      const config = { title: 123, theme: 'dark' } as unknown as ZarrViewerConfig;
      expect(() => applyViewerConfigState(config, asPorts(ports))).not.toThrow();
      expect(ports.setDocumentTitle).not.toHaveBeenCalled();
      expect(ports.setTheme).toHaveBeenCalledWith('dark');
    });
  });

  describe('playback_lod_depth (authored playback detail)', () => {
    it.each([
      [4, 4],
      [2.9, 2],
      ['all', Infinity],
      ['auto', 'auto'],
      ['fast', null],
    ] as Array<[unknown, number | 'auto' | null]>)('maps %j to %j', (raw, expected) => {
      const ports = makePorts();
      applyViewerConfigState({ playback_lod_depth: raw } as never, asPorts(ports));
      expect(ports.setDefaultLadderDepth).toHaveBeenCalledWith(expected);
    });

    it.each([[undefined], [0], [-1], ['deep'], [{}]])(
      'leaves the viewer default alone for %j',
      (raw) => {
        const ports = makePorts();
        applyViewerConfigState({ playback_lod_depth: raw } as never, asPorts(ports));
        expect(ports.setDefaultLadderDepth).not.toHaveBeenCalled();
      }
    );

    it('is applied before the animation block so an opening play runs at the authored detail', () => {
      const ports = makePorts();
      const order: string[] = [];
      ports.setDefaultLadderDepth!.mockImplementation(() => order.push('detail'));
      ports.startDimensionAnimation!.mockImplementation(() => order.push('play'));
      applyViewerConfigState(
        { playback_lod_depth: 3, animation: [{}, {}, {}, { playing: true }] },
        asPorts(ports)
      );
      expect(order).toEqual(['detail', 'play']);
    });
  });

  describe('animation', () => {
    // This block round-tripped through the scene file for a long time with
    // nothing reading it back: the capture path wrote it, the Python
    // ViewerConfig exposed it, the viewer guide described it as restored, and
    // on load it was dropped. These tests are what stop that recurring.
    it('starts playback on the dimension that asks for it', () => {
      const config: ZarrViewerConfig = {
        animation: [{}, {}, {}, { playing: true, target_fps: 24, loop: 'bounce' }],
      };
      applyViewerConfigState(config, asPorts(ports));

      expect(ports.startDimensionAnimation).toHaveBeenCalledTimes(1);
      expect(ports.startDimensionAnimation).toHaveBeenCalledWith(3, {
        targetFPS: 24,
        loopMode: 'bounce',
        direction: undefined,
      });
    });

    it('forwards a per-dimension step override', () => {
      applyViewerConfigState(
        { animation: [{}, { playing: true, step_size: 2.5 }] },
        asPorts(ports)
      );
      expect(ports.startDimensionAnimation).toHaveBeenCalledWith(
        1,
        expect.objectContaining({ stepSize: 2.5 })
      );
    });

    it('sanitizes persisted animation options before starting playback', () => {
      const config = {
        animation: [
          { playing: true, target_fps: 0, loop: 'banana', direction: 'sideways' },
          { playing: true, target_fps: 500, loop: 'once', direction: 'backward' },
          { playing: true, target_fps: Number.NaN },
        ],
      } as unknown as ZarrViewerConfig;

      applyViewerConfigState(config, asPorts(ports));

      expect(ports.startDimensionAnimation).toHaveBeenNthCalledWith(1, 0, {
        targetFPS: 0.1,
        loopMode: undefined,
        direction: undefined,
        stepSize: undefined,
      });
      expect(ports.startDimensionAnimation).toHaveBeenNthCalledWith(2, 1, {
        targetFPS: 120,
        loopMode: 'once',
        direction: 'backward',
        stepSize: undefined,
      });
      expect(ports.startDimensionAnimation).toHaveBeenNthCalledWith(3, 2, {
        targetFPS: undefined,
        loopMode: undefined,
        direction: undefined,
        stepSize: undefined,
      });
    });

    it('leaves stepSize undefined when the scene does not set one', () => {
      // Undefined means Auto — the viewer derives a step from the dimension.
      // Forwarding a 0 or a null here would be an explicit override of it.
      applyViewerConfigState({ animation: [{ playing: true }] }, asPorts(ports));
      expect(ports.startDimensionAnimation).toHaveBeenCalledWith(
        0,
        expect.objectContaining({ stepSize: undefined })
      );
    });

    it('leaves a dimension alone unless playing is exactly true', () => {
      const config: ZarrViewerConfig = {
        animation: [{ playing: false }, { target_fps: 30 }, {}],
      };
      applyViewerConfigState(config, asPorts(ports));

      // `false` is the viewer's own default; re-asserting it would stop a
      // paused scene from simply inheriting whatever the viewer does next.
      expect(ports.startDimensionAnimation).not.toHaveBeenCalled();
    });

    it('starts playback AFTER the opening timepoint is set', () => {
      // Order matters: playback runs on from wherever the dimension was left,
      // so applying it first would make an authored `current_step` look like
      // it had been ignored.
      const calls: string[] = [];
      ports.setDimensionValue.mockImplementation(() => calls.push('step'));
      ports.startDimensionAnimation?.mockImplementation(() => calls.push('play'));

      applyViewerConfigState(
        {
          dimensions: { current_step: [0, 0, 0, 65] },
          animation: [{}, {}, {}, { playing: true }],
        },
        asPorts(ports)
      );

      expect(calls).toEqual(['step', 'step', 'step', 'step', 'play']);
    });

    it('is a no-op when the scene has no animation manager', () => {
      // The helper's lightweight test ports may omit animation support.
      const bare = makePorts({ startDimensionAnimation: false });
      expect(() =>
        applyViewerConfigState({ animation: [{ playing: true }] }, asPorts(bare))
      ).not.toThrow();
    });
  });

  describe('combined config', () => {
    it('applies UI + theme + dimensions in a single pass', () => {
      const config: ZarrViewerConfig = {
        ui: { show_layers: true, show_overlays: false },
        theme: 'frosted-glass',
        dimensions: { current_step: [5, 10] },
      };
      applyViewerConfigState(config, asPorts(ports));

      expect(ports.layersPanel?.show).toHaveBeenCalled();
      expect(ports.overlayManager?.hide).toHaveBeenCalled();
      expect(ports.setTheme).toHaveBeenCalledWith('frosted-glass');
      expect(ports.setDimensionValue).toHaveBeenCalledTimes(2);
    });
  });
});
