/**
 * Debug console configuration
 */
export interface DebugConsoleConfig {
  panel: {
    defaultWidth: number;
    defaultHeight: number;
    minWidth: number;
    maxWidth: number;
    minHeight: number;
    maxHeight: number;
    bottomOffset: number;
    leftOffset: number;
  };
  resize: {
    borderWidth: number;
  };
  style: {
    backgroundColor: string;
    borderColor: string;
    borderRadius: number;
    backdropBlur: number;
    boxShadow: string;
  };
}

/**
 * UI component configurations
 */
export interface UIComponentsConfig {
  datasetBrowser: {
    borderRadius: {
      panel: number;
      section: number;
      element: number;
    };
    padding: {
      panel: number;
      section: number;
      element: number;
    };
  };
  debugConsole: {
    borderRadius: {
      header: number;
      content: number;
      button: number;
    };
  };
  renderingControls: {
    borderRadius: {
      checkbox: number;
      section: number;
      header: number;
    };
  };
  dataMonitor: {
    borderRadius: {
      card: number;
      section: number;
    };
    padding: {
      default: number;
      compact: number;
    };
  };
}

/**
 * UI configuration for overlays and visual elements
 */
export interface UIConfig {
  zIndex: {
    // Base layer components
    dimensionSliders: number;
    performanceMonitor: number;
    debugConsole: number;
    // Mid-layer overlays
    datasetBrowser: number;
    loading: number;
    error: number;
    help: number;
    renderingControls: number;
    recordingPanel: number;
    layersPanel: number;
    // Top layer
    statsMonitor: number;
  };
  timings: {
    errorAutoDismissMs: number;
    helpClickDelayMs: number;
  };
  spinner: {
    size: number;
    borderWidth: number;
  };
  // Styling lives in CSS variables/classes rather than config objects.
  debugConsole: DebugConsoleConfig;
  components: UIComponentsConfig;
  scaleBar: {
    targetWidthPx: number;
    position: 'bottom-left' | 'bottom-right';
  };
}
