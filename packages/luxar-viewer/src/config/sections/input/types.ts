/**
 * Input handling configuration
 */
export interface InputConfig {
  defaultSensitivity: number;
  keyboard: {
    shortcuts: {
      toggleFullscreen: string;
      toggleHelp: string;
      toggleDimensions: string;
      toggleDatasetBrowser: string;
      togglePerformance: string;
      toggleRendering: string;
      toggleScaleBar: string;
      toggleColormapLegend: string;
      toggleDebugConsole: string;
      toggleLayers: string;
      recenterCamera: string;
      toggleControlMode: string;
      toggleInertialMode: string;
      toggleCinematicMode: string;
      toggleOverlays: string;
    };
    flyModeKeys: string[];
    dimensionKeys: string[];
  };
}
