import type { UIConfig } from './types';

/**
 * UI configuration for overlays and visual elements
 */
export const uiConfig: UIConfig = {
  // Only the panels that set an inline z-index read these; every other layer
  // is stacked by CSS (the theme's --luxar-z-* scale).
  zIndex: {
    // The three left-DOCKED panels live below the control rail (modal - 2 in
    // control-rail.css): the rail's hover tooltips extend rightward OVER the
    // docked panel, and a panel above the rail buries them. The dock is
    // exclusive (one panel at a time from the rail), so only the deliberate
    // keyboard-stacking order matters between them.
    renderingControls: 1600, // Rendering controls panel (top of the dock)
    recordingPanel: 1500, // Recording panel (screenshot/video capture)
    layersPanel: 1500, // Layers panel (per-node controls)

    // Top layer (2000+)
    statsMonitor: 2000, // Three.js stats monitor (always on top)
  },
  timings: {
    errorAutoDismissMs: 10000, // Auto-dismiss error messages after 10s
    helpClickDelayMs: 100, // Delay before help can be closed by click
  },
  // Styling is provided by CSS variables and classes in src/styles/
  // (see the theming system in src/themes/).

  // Debug console drag-resize limits (its default size, position and look
  // live in debug-console.css).
  debugConsole: {
    panel: {
      minWidth: 400,
      maxWidth: 1200,
      minHeight: 200,
      maxHeight: 800,
    },
    resize: {
      borderWidth: 4,
    },
  },
  scaleBar: {
    targetWidthPx: 150,
    position: 'bottom-left' as const,
  },
};
