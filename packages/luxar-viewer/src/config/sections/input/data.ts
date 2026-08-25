import type { InputConfig } from './types';

/** Input handling configuration. */
export const inputConfig: InputConfig = {
  defaultSensitivity: 0.1, // Default input sensitivity for adjustments
  keyboard: {
    shortcuts: {
      toggleFullscreen: ' ',
      toggleHelp: 'h',
      toggleDimensions: 'n',
      toggleDatasetBrowser: 'o',
      togglePerformance: 'p',
      toggleRendering: 'r',
      toggleScaleBar: 'b',
      toggleColormapLegend: 'j',
      // Bare key; the Ctrl modifier is applied structurally at the binding site.
      toggleDebugConsole: 'l',
      toggleLayers: 'l',
      recenterCamera: 'f',
      toggleControlMode: 'v',
      toggleInertialMode: 'i',
      toggleCinematicMode: 'c',
      toggleOverlays: 'u',
    },
  },
};
