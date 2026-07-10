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
      toggleDebugConsole: 'ctrl+l',
      toggleLayers: 'l',
      recenterCamera: 'f',
      toggleControlMode: 'v',
      toggleInertialMode: 'i',
      toggleCinematicMode: 'c',
      toggleOverlays: 'u',
    },
    flyModeKeys: [
      'w',
      'a',
      's',
      'd',
      'q',
      'e',
      'W',
      'A',
      'S',
      'D',
      'Q',
      'E',
      'Shift',
      'ArrowUp',
      'ArrowDown',
      'ArrowLeft',
      'ArrowRight',
    ],
    dimensionKeys: ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
  },
};
