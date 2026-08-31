import type { ControlsConfig } from './types';

/** Control system configuration. */
export const controlsConfig: ControlsConfig = {
  scaleMultipliers: {
    minDistanceFactor: 0.001,
    maxDistanceFactor: 10000,
    flySpeedFactor: 0.05,
  },
  fly: {
    inertialMode: {
      default: true,
    },
    movement: {
      speed: { min: 0.01, max: 5.0, default: 0.5, step: 0.01 },
      acceleration: { min: 0.1, max: 2.0, default: 0.5, step: 0.1 },
      damping: { min: 0.9, max: 0.99999, default: 0.999, step: 0.0001 },
    },
    rotation: {
      speed: { min: 0.1, max: 5.0, default: 1.5, step: 0.1 },
      damping: { min: 0.9, max: 0.9999, default: 0.99, step: 0.0001 },
    },
    look: {
      mouseSpeed: { min: 0.0005, max: 0.01, default: 0.002, step: 0.0005 },
    },
    physics: {
      velocityThreshold: 1e-4,
      dampingPower: 60,
      angularVelocityThreshold: 1e-4,
    },
  },
  orbit: {
    autoRotate: {
      speed: { min: 0.1, max: 5.0, default: 0.25, step: 0.1 },
    },
    // Auto-dolly. The 50% ceiling is not arbitrary: screen area goes as 1/d²,
    // so a swing of A moves projected area by (1+A)⁴ — 1.75x at 15%, but 5x at
    // 50%, which walks up and down the LOD ladder every cycle and re-fetches
    // chunks on a hosted scene. Defaults mirror DEFAULT_AUTO_DOLLY_AMPLITUDE /
    // DEFAULT_AUTO_DOLLY_PERIOD in controls/types.ts (pinned by a unit test).
    autoDolly: {
      amplitudePercent: { min: 1, max: 50, default: 15, step: 1 },
      period: { min: 1, max: 60, default: 10, step: 0.5 },
    },
    zoom: {
      minDistance: 0.1,
      maxDistance: 1000,
      speed: { min: 0.2, max: 3.0, default: 1.0, step: 0.1 },
    },
    damping: {
      enabled: true,
      // Keep aligned with LuxarOrbitControls' constructor fallback — the two
      // defaults must not drift apart.
      factor: { min: 0.01, max: 0.5, default: 0.25, step: 0.01 },
    },
  },
};
