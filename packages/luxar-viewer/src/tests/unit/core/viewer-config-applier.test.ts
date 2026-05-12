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
} from '../../../core/viewer-config-applier';
import type { ZarrViewerConfig } from '../../../types/zarr';

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
}

function makePorts(
  overrides: Partial<{
    scaleBar: boolean;
    layersPanel: boolean;
    overlayManager: boolean;
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
  };
}

const asPorts = (p: PortStubs) => p as unknown as ViewerConfigPorts;

describe('applyViewerConfigState', () => {
  let ports: PortStubs;

  beforeEach(() => {
    ports = makePorts();
  });

  it('returns silently when viewerConfig is undefined', () => {
    applyViewerConfigState(undefined, asPorts(ports));
    expect(ports.showHelp).not.toHaveBeenCalled();
    expect(ports.renderingControls.show).not.toHaveBeenCalled();
    expect(ports.setTheme).not.toHaveBeenCalled();
  });

  it('returns silently when viewerConfig is empty', () => {
    applyViewerConfigState({}, asPorts(ports));
    expect(ports.showHelp).not.toHaveBeenCalled();
    expect(ports.renderingControls.show).not.toHaveBeenCalled();
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
      ports = makePorts({ scaleBar: false });
      applyViewerConfigState({ ui: { show_scale_bar: true } }, asPorts(ports));
      // No assertion needed beyond "does not throw" — null guard inside.
      expect(ports.scaleBar).toBeUndefined();
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
      ports = makePorts({ layersPanel: false });
      applyViewerConfigState({ ui: { show_layers: true } }, asPorts(ports));
      expect(ports.layersPanel).toBeUndefined();
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
      ports = makePorts({ overlayManager: false });
      applyViewerConfigState({ ui: { show_overlays: true } }, asPorts(ports));
      expect(ports.overlayManager).toBeUndefined();
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
