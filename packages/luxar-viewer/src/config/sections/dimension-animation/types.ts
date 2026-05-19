/**
 * Dimension animation configuration
 * Controls FPS-based animation through dimension ranges with various loop modes
 */
export interface DimensionAnimationConfig {
  defaults: {
    targetFPS: number;
    loop: 'once' | 'loop' | 'bounce';
    direction: 'forward' | 'backward';
  };
  presets: {
    fps: number[];
    customMin: number;
    customMax: number;
  };
  timing: {
    minFrameTimeMs: number;
    continuousTraverseSeconds: number;
  };
  ui: {
    showFPSFeedback: boolean;
    feedbackThreshold: number;
  };
}
