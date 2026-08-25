/** Stable action identifiers shared by bindings and shortcut-label consumers. */
/** Stable action identifiers shared by bindings and shortcut-label consumers. */
export const KeyAction = {
  navigateDimension: 'dimension.navigate',
  selectDimension: 'dimension.select',
  toggleHelp: 'help.toggle',
  toggleDimensions: 'dimensions.toggle',
  toggleDatasetBrowser: 'dataset-browser.toggle',
  openElementMenu: 'element-menu.open',
  togglePerformance: 'performance.toggle',
  toggleRendering: 'rendering.toggle',
  toggleScaleBar: 'scale-bar.toggle',
  toggleColormapLegend: 'colormap-legend.toggle',
  toggleOverlays: 'overlays.toggle',
  toggleRecording: 'recording.toggle',
  captureScreenshot: 'screenshot.capture',
  toggleLayers: 'layers.toggle',
  toggleDebugConsole: 'debug-console.toggle',
  cycleDataMonitor: 'data-monitor.cycle',
  recenterCamera: 'camera.recenter',
  toggleControlMode: 'control-mode.toggle',
  toggleInertialMode: 'inertial-mode.toggle',
  toggleCinematicMode: 'cinematic-mode.toggle',
  toggleFullscreen: 'fullscreen.toggle',
  handleEscape: 'panels.escape',
  exportViewerState: 'viewer-state.export',
  toggleAnimation: 'animation.toggle',
  jumpAnimation: 'animation.jump',
  adjustAnimationSpeed: 'animation.speed',
  flyMove: 'fly.move',
  flyLook: 'fly.look',
  flySpeedBoost: 'fly.speed-boost',
} as const;

/** Union of built-in action identifiers accepted by the viewer façade. */
export type KeyActionId = (typeof KeyAction)[keyof typeof KeyAction];
