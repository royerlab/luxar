/**
 * Unit tests for core/panel-visibility.ts.
 *
 * Pure capture / restore helpers — tests pass a stub ports object
 * and assert the right show/hide calls fire for each saved-state
 * combination.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getPanelVisibilityStates,
  restorePanelVisibilityStates,
  type PanelVisibilityPorts,
  type VisibilityPanel,
} from '../../../../../core/app/viewer-config/panel-visibility';

// vitest's Mock type doesn't structurally satisfy `() => void` so we keep
// the stub shape inline + cast at the call site rather than inheriting.
interface PanelStub {
  show: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  isVisible: ReturnType<typeof vi.fn>;
}

function makePanel(visible = false): PanelStub {
  return {
    show: vi.fn(),
    hide: vi.fn(),
    isVisible: vi.fn().mockReturnValue(visible),
  };
}

const asPanel = (p: PanelStub) => p as unknown as VisibilityPanel;

describe('getPanelVisibilityStates', () => {
  it('captures isVisible() for both panels', () => {
    const renderingControls = makePanel(true);
    const recordingPanel = makePanel(false);

    const states = getPanelVisibilityStates({
      renderingControls: asPanel(renderingControls),
      recordingPanel: asPanel(recordingPanel),
    });

    expect(states.get('renderingControls')).toBe(true);
    expect(states.get('recordingPanel')).toBe(false);
  });

  it('records false for missing panels (no throw on undefined ports)', () => {
    const states = getPanelVisibilityStates({});
    expect(states.get('renderingControls')).toBe(false);
    expect(states.get('recordingPanel')).toBe(false);
  });

  it('captures both true', () => {
    const states = getPanelVisibilityStates({
      renderingControls: asPanel(makePanel(true)),
      recordingPanel: asPanel(makePanel(true)),
    });
    expect(states.get('renderingControls')).toBe(true);
    expect(states.get('recordingPanel')).toBe(true);
  });
});

describe('restorePanelVisibilityStates', () => {
  let renderingControls: PanelStub;
  let recordingPanel: PanelStub;
  let ports: PanelVisibilityPorts;

  beforeEach(() => {
    renderingControls = makePanel(false);
    recordingPanel = makePanel(false);
    ports = {
      renderingControls: asPanel(renderingControls),
      recordingPanel: asPanel(recordingPanel),
    };
  });

  describe('renderingControls', () => {
    it('saved=true → show() (regardless of current visibility)', () => {
      const states = new Map([['renderingControls', true]]);
      restorePanelVisibilityStates(states, ports);
      expect(renderingControls.show).toHaveBeenCalled();
      expect(renderingControls.hide).not.toHaveBeenCalled();
    });

    it('saved=false + currently visible → hide()', () => {
      renderingControls.isVisible.mockReturnValue(true);
      const states = new Map([['renderingControls', false]]);
      restorePanelVisibilityStates(states, ports);
      expect(renderingControls.hide).toHaveBeenCalled();
      expect(renderingControls.show).not.toHaveBeenCalled();
    });

    it('saved=false + already hidden → no-op (no hide call)', () => {
      renderingControls.isVisible.mockReturnValue(false);
      const states = new Map([['renderingControls', false]]);
      restorePanelVisibilityStates(states, ports);
      expect(renderingControls.hide).not.toHaveBeenCalled();
      expect(renderingControls.show).not.toHaveBeenCalled();
    });
  });

  describe('recordingPanel', () => {
    it('saved=true → show()', () => {
      const states = new Map([['recordingPanel', true]]);
      restorePanelVisibilityStates(states, ports);
      expect(recordingPanel.show).toHaveBeenCalled();
    });

    it('saved=false + currently visible → hide()', () => {
      recordingPanel.isVisible.mockReturnValue(true);
      const states = new Map([['recordingPanel', false]]);
      restorePanelVisibilityStates(states, ports);
      expect(recordingPanel.hide).toHaveBeenCalled();
    });

    it('saved=false + already hidden → no-op', () => {
      recordingPanel.isVisible.mockReturnValue(false);
      const states = new Map([['recordingPanel', false]]);
      restorePanelVisibilityStates(states, ports);
      expect(recordingPanel.hide).not.toHaveBeenCalled();
    });
  });

  describe('missing panel safety', () => {
    it('missing renderingControls → restore is a no-op for that panel', () => {
      const portsMissing: PanelVisibilityPorts = {
        recordingPanel: asPanel(recordingPanel),
      };
      const states = new Map([
        ['renderingControls', true],
        ['recordingPanel', false],
      ]);
      // Must not throw despite renderingControls being undefined.
      expect(() => restorePanelVisibilityStates(states, portsMissing)).not.toThrow();
    });

    it('missing recordingPanel → restore is a no-op for that panel', () => {
      const portsMissing: PanelVisibilityPorts = {
        renderingControls: asPanel(renderingControls),
      };
      const states = new Map([
        ['renderingControls', false],
        ['recordingPanel', true],
      ]);
      expect(() => restorePanelVisibilityStates(states, portsMissing)).not.toThrow();
    });
  });

  describe('round-trip', () => {
    it('capture then restore preserves visibility', () => {
      renderingControls.isVisible.mockReturnValue(true);
      recordingPanel.isVisible.mockReturnValue(false);

      const states = getPanelVisibilityStates(ports);
      restorePanelVisibilityStates(states, ports);

      expect(renderingControls.show).toHaveBeenCalled();
      // recordingPanel was hidden → no hide call (already hidden).
      expect(recordingPanel.hide).not.toHaveBeenCalled();
    });
  });
});
