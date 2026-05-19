/**
 * Adaptive pixel ratio configuration for dynamic performance optimization
 *
 * This system dynamically adjusts the device pixel ratio based on real-time FPS
 * to maintain smooth frame rates during heavy rendering. Uses hysteresis to
 * prevent rapid toggling between quality levels.
 */
export interface AdaptiveDPRConfig {
  /** Enable adaptive DPR system at construction time (default: true).
   *  Runtime toggle is via renderingControls.defaults.adaptiveDPREnabled. */
  enabled: boolean;
  /** FPS threshold for scaling down resolution (default: 50) */
  minFPS: number;
  /** FPS threshold for scaling up resolution (default: 58) */
  maxFPS: number;
  /** Minimum allowed DPR - lower bound before image becomes too pixelated (default: 0.5) */
  minDPR: number;
  /** Factor to multiply DPR when scaling down (default: 0.9 = 10% reduction) */
  scaleDownFactor: number;
  /** Factor to multiply DPR when scaling up (default: 1.05 = 5% increase) */
  scaleUpFactor: number;
  /** Seconds FPS must stay above maxFPS before scaling up (default: 3) */
  hysteresisSeconds: number;
  /** How often to evaluate FPS and adjust DPR in milliseconds (default: 500) */
  evaluationIntervalMs: number;
}
