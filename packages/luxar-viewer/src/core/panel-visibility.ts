/**
 * Capture / restore the visibility state of UI panels around a
 * recording or other panel-visibility-sensitive operation.
 *
 * Extracted from `core/app.ts` so the per-panel visibility logic is
 * unit-testable without instantiating the full LuxarApp. The helper
 * is pass-through: it knows about the typed `PanelVisibilityPorts`
 * shape but does not import any panel implementation.
 *
 * @module core/panel-visibility
 */

/** A panel's minimal show/hide/visibility surface. */
export interface VisibilityPanel {
  show(): void;
  hide(): void;
  isVisible(): boolean;
}

/** Set of panels whose visibility we track. */
export interface PanelVisibilityPorts {
  renderingControls?: VisibilityPanel;
  recordingPanel?: VisibilityPanel;
}

/**
 * Snapshot the current visibility of each known panel into a map.
 * Missing panels record `false` so the restore path doesn't surprise
 * a caller after a panel is later torn down.
 */
export function getPanelVisibilityStates(
  ports: PanelVisibilityPorts
): Map<string, boolean> {
  const states = new Map<string, boolean>();
  states.set('renderingControls', ports.renderingControls?.isVisible() ?? false);
  states.set('recordingPanel', ports.recordingPanel?.isVisible() ?? false);
  return states;
}

/**
 * Drive each panel back to the visibility recorded in `states`.
 *
 * For a saved-true entry we call `show()` unconditionally — if the
 * panel is already shown, `show()` should be idempotent. For a
 * saved-false entry we only call `hide()` when the panel is currently
 * visible, which avoids writing visibility state on already-hidden
 * panels (some panels record telemetry on visibility transitions).
 */
export function restorePanelVisibilityStates(
  states: Map<string, boolean>,
  ports: PanelVisibilityPorts
): void {
  if (states.get('renderingControls')) {
    ports.renderingControls?.show();
  } else {
    if (ports.renderingControls?.isVisible()) ports.renderingControls.hide();
  }

  if (states.get('recordingPanel')) {
    ports.recordingPanel?.show();
  } else {
    if (ports.recordingPanel?.isVisible()) ports.recordingPanel.hide();
  }
}
