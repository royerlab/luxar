import type { AdaptiveDPRConfig } from './types';

/**
 * Adaptive pixel ratio configuration for dynamic performance optimization
 */
export const adaptiveDPRConfig: AdaptiveDPRConfig = {
  enabled: true, // Construction-time default; runtime toggle is renderingControls.defaults.adaptiveDPREnabled
  minFPS: 50, // FPS threshold for scaling down resolution
  maxFPS: 58, // FPS threshold for scaling up resolution
  minDPR: 0.5, // Minimum DPR - lower bound before image becomes too pixelated
  scaleDownFactor: 0.9, // Reduce DPR by 10% when scaling down
  scaleUpFactor: 1.05, // Increase DPR by 5% when scaling up
  hysteresisSeconds: 3, // Wait 3 seconds of stable high FPS before scaling up
  evaluationIntervalMs: 500, // Evaluate FPS every 500ms
};
