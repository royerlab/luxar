/**
 * Debug console drag-resize limits (its default size, position and look live
 * in debug-console.css).
 */
export interface DebugConsoleConfig {
  panel: {
    minWidth: number;
    maxWidth: number;
    minHeight: number;
    maxHeight: number;
  };
  resize: {
    borderWidth: number;
  };
}

/**
 * UI configuration for overlays and visual elements
 */
export interface UIConfig {
  /** Inline z-index of the panels that set one (everything else is CSS). */
  zIndex: {
    renderingControls: number;
    recordingPanel: number;
    layersPanel: number;
    statsMonitor: number;
  };
  timings: {
    errorAutoDismissMs: number;
    helpClickDelayMs: number;
  };
  // Styling lives in CSS variables/classes rather than config objects.
  debugConsole: DebugConsoleConfig;
  scaleBar: {
    targetWidthPx: number;
    position: 'bottom-left' | 'bottom-right';
  };
}
