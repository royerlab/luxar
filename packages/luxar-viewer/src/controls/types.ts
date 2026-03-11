/**
 * Type definitions for the Luxar control system
 *
 * This file provides strong typing for all control-related interfaces,
 * eliminating the need for 'any' types throughout the codebase.
 */

import type { Controller } from '../ui/gui';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
// Note: ArcballControls doesn't have TypeScript definitions, so we use 'any'
import type { LuxarFlyControls } from './luxar-fly-controls';

/**
 * Available control types
 */
export type ControlType = 'orbit' | 'arcball' | 'fly';

/**
 * Complete control state for both orbit and fly modes
 */
export interface ControlState {
  type: ControlType;
  orbit: OrbitState;
  fly: FlyState;
}

/**
 * Orbit control specific state
 */
export interface OrbitState {
  autoRotate: boolean;
  autoRotateSpeed: number;
  enableDamping: boolean;
  dampingFactor: number;
  minDistance: number;
  maxDistance: number;
  enableZoom: boolean;
  zoomSpeed: number;
}

/**
 * Fly control specific state
 */
export interface FlyState {
  movementSpeed: number;
  rotationSpeed: number;
  lookSpeed: number;
  inertialMode: boolean;
  damping: number;
  rotationDamping: number;
}

/**
 * Type-safe references to GUI controllers
 */
export interface GuiControllers {
  navigation: RenderingControllers;
  bloom?: BloomControllers;
  hdr?: HDRControllers;
}

/**
 * Rendering controls GUI controllers for the entire rendering panel
 */
export interface RenderingControllers {
  controlType?: Controller;
  // Camera controls
  fov?: Controller;
  fovPreset?: Controller;
  nearPlane?: Controller;
  farPlane?: Controller;
  // Dynamic clipping controls
  dynamicClippingEnabled?: Controller;
  clippingAdaptSpeed?: Controller;
  // HDR controls (global EOG)
  exposure?: Controller;
  globalOffset?: Controller;
  globalGamma?: Controller;
  // Chromatic lens distortion controls (replaces old separate lens + chromatic effects)
  chromaticLensDistortionX?: Controller;
  chromaticLensDistortionY?: Controller;
  chromaticLensDispersion?: Controller;
  chromaticLensPrincipalPointX?: Controller;
  chromaticLensPrincipalPointY?: Controller;
  chromaticLensFocalLengthX?: Controller;
  chromaticLensFocalLengthY?: Controller;
  chromaticLensSkew?: Controller;
  // Orbit controls
  autoRotate?: Controller;
  autoRotateSpeed?: Controller;
  // Fly controls
  flyMovementSpeed?: Controller;
  flyRotationSpeed?: Controller;
  flyInertialMode?: Controller;
  flyDamping?: Controller;
  flyRotationDamping?: Controller;
  // Performance controls
  adaptiveDPREnabled?: Controller;
  // Cinematic mode
  cinematicMode?: Controller;
}

/**
 * Bloom effect GUI controllers
 */
export interface BloomControllers {
  threshold?: Controller;
  strength?: Controller;
  radius?: Controller;
}

/**
 * HDR GUI controllers
 */
export interface HDRControllers {
  hdrMultiplier?: Controller;
  toneMapping?: Controller;
}

/**
 * Union type for control instances
 */
export type ControlInstance = OrbitControls | any | LuxarFlyControls; // 'any' for ArcballControls

/**
 * Type guard for OrbitControls
 */
export function isOrbitControls(control: ControlInstance | null): control is OrbitControls {
  return control !== null && 'target' in control && 'autoRotate' in control;
}

/**
 * Type guard for LuxarFlyControls
 */
export function isFlyControls(control: ControlInstance | null): control is LuxarFlyControls {
  return control !== null && 'inertialMode' in control && 'arrowLookSpeed' in control;
}

/**
 * Control configuration with validation ranges
 */
export interface ControlConfig {
  fly: FlyConfig;
  orbit: OrbitConfig;
}

export interface FlyConfig {
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
    mouseSpeed: ConfigValue;
  };
  physics: {
    velocityThreshold: number;
    dampingPower: number;
    angularVelocityThreshold: number;
  };
}

export interface OrbitConfig {
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

export interface ConfigRange {
  min: number;
  max: number;
  default: number;
  step?: number;
}

export interface ConfigValue {
  default: number;
}
