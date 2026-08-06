import type { UIConfig } from './types';

/**
 * UI configuration for overlays and visual elements
 */
export const uiConfig: UIConfig = {
  zIndex: {
    // Base layer components (100-199)
    dimensionSliders: 100, // Dimension sliders at bottom
    performanceMonitor: 100, // Performance stats panel
    debugConsole: 150, // Debug console (slightly above base)

    // Mid-layer overlays (1000-1999)
    datasetBrowser: 1000, // Dataset browser modal
    loading: 1000, // Loading indicator
    error: 1000, // Error messages
    help: 1001, // Help overlay (above errors)
    renderingControls: 1999, // Rendering controls panel (top of mid-layer)
    recordingPanel: 1500, // Recording panel (screenshot/video capture)
    layersPanel: 1500, // Layers panel (per-node controls)

    // Top layer (2000+)
    statsMonitor: 2000, // Three.js stats monitor (always on top)
  },
  timings: {
    errorAutoDismissMs: 10000, // Auto-dismiss error messages after 10s
    helpClickDelayMs: 100, // Delay before help can be closed by click
  },
  spinner: {
    size: 24, // Loading spinner size in pixels
    borderWidth: 3, // Spinner border width
  },
  // Styling is provided by CSS variables and classes in src/styles/
  // (see the theming system in src/themes/).

  // Debug console configuration
  debugConsole: {
    panel: {
      defaultWidth: 600,
      defaultHeight: 400,
      minWidth: 400,
      maxWidth: 1200,
      minHeight: 200,
      maxHeight: 800,
      bottomOffset: 20,
      leftOffset: 20,
    },
    resize: {
      borderWidth: 4,
    },
    style: {
      backgroundColor: 'rgba(30, 30, 30, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      borderRadius: 8,
      backdropBlur: 10,
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    },
  },
  scaleBar: {
    targetWidthPx: 150,
    position: 'bottom-left' as const,
  },
  // UI component-specific configuration for consistent styling
  components: {
    datasetBrowser: {
      borderRadius: {
        panel: 12,
        section: 6,
        element: 4,
      },
      padding: {
        panel: 20,
        section: 15,
        element: 10,
      },
    },
    debugConsole: {
      borderRadius: {
        header: 8,
        content: 4,
        button: 3,
      },
    },
    renderingControls: {
      borderRadius: {
        checkbox: 4,
        section: 4,
        header: 4,
      },
    },
    dataMonitor: {
      borderRadius: {
        card: 4,
        section: 6,
      },
      padding: {
        default: 10,
        compact: 8,
      },
    },
  },
};
