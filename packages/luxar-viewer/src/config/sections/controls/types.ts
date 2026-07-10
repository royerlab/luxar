/**
 * Configuration range with min/max/default/step
 */
export interface ConfigRange {
  min: number;
  max: number;
  default: number;
  step?: number;
}

/**
 * Fly controls configuration
 */
export interface FlyControlsConfig {
  inertialMode: {
    default: boolean;
  };
  movement: {
    speed: ConfigRange;
    acceleration: ConfigRange;
    damping: ConfigRange;
  };
  rotation: {
    speed: ConfigRange;
    damping: ConfigRange;
  };
  look: {
    mouseSpeed: ConfigRange;
  };
  physics: {
    velocityThreshold: number;
    dampingPower: number;
    angularVelocityThreshold: number;
  };
}

/**
 * Orbit controls configuration
 */
export interface OrbitControlsConfig {
  autoRotate: {
    speed: ConfigRange;
  };
  zoom: {
    minDistance: number;
    maxDistance: number;
    speed: ConfigRange;
  };
  damping: {
    enabled: boolean;
    factor: ConfigRange;
  };
}

/**
 * Scale multipliers for adapting camera controls to scene size.
 * All factors are multiplied by the bounding box diagonal to produce
 * the actual control parameter value.
 */
export interface ScaleMultipliers {
  /** orbit minDistance = diagonal * factor (default: 0.01) */
  minDistanceFactor: number;
  /** orbit maxDistance = diagonal * factor (default: 100) */
  maxDistanceFactor: number;
  /** fly movementSpeed = diagonal * factor (default: 0.05) */
  flySpeedFactor: number;
}

/**
 * Control system configuration
 */
export interface ControlsConfig {
  fly: FlyControlsConfig;
  orbit: OrbitControlsConfig;
  scaleMultipliers: ScaleMultipliers;
}
