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
    zoom: {
      minDistance: 0.1,
      maxDistance: 1000,
      speed: { min: 0.2, max: 3.0, default: 1.0, step: 0.1 },
    },
    damping: {
      enabled: true,
      // Default aligned to the value LuxarOrbitControls has always actually
      // used (its constructor fallback): the old 0.05 here was never applied.
      factor: { min: 0.01, max: 0.5, default: 0.25, step: 0.01 },
    },
  },
};
