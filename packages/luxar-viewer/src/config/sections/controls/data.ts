import type { ControlsConfig } from './types';

/** Control system configuration. */
export const controlsConfig: ControlsConfig = {
  // 1.0 = unscaled wheel deltas (the historical feel). Settings > Input.
  wheelZoomSensitivity: 1.0,
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
    // Auto-dolly. The ceiling is a COST boundary, not a safety one — the math
    // is exact at any amplitude and the distance clamps sit orders of magnitude
    // away (a scene framed at 176k units clamps at 327), so a big swing is
    // free to be a deliberate choice. What grows is work: screen area goes as
    // 1/d², so a swing of A moves projected area by (1+A)⁴ and the LOD ladder
    // answers by loading finer levels at the near extreme. Measured on the
    // 100-group embryo demo over one 3 s cycle: 15% → 1.75x area, 118k
    // elements resident, 161 level transitions; 50% → 5.06x, 526k, 392;
    // 95% → 14.46x, 2.29M, 520. That is a 19x resident set for a 6x bigger
    // swing, which a local warm cache absorbs (144 → 129 fps here) and a
    // hosted scene pays for in requests. Hence a high ceiling with a modest
    // default. Defaults mirror DEFAULT_AUTO_DOLLY_AMPLITUDE /
    // DEFAULT_AUTO_DOLLY_PERIOD in controls/types.ts (pinned by a unit test).
    autoDolly: {
      amplitudePercent: { min: 1, max: 95, default: 15, step: 1 },
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
