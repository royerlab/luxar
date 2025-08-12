/**
 * Type definitions for the Luxar control system
 * 
 * This file provides strong typing for all control-related interfaces,
 * eliminating the need for 'any' types throughout the codebase.
 */

import type { Controller } from 'lil-gui';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import type { LuxarFlyControls } from './luxar-fly-controls';

/**
 * Available control types
 */
export type ControlType = 'orbit' | 'fly';

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
  lookSpeed: number;
  arrowLookSpeed: number;
  inertialMode: boolean;
  damping: number;
  acceleration: number;
}

/**
 * Type-safe references to GUI controllers
 */
export interface GuiControllers {
  navigation: NavigationControllers;
  bloom?: BloomControllers;
  hdr?: HDRControllers;
}

/**
 * Navigation-specific GUI controllers
 */
export interface NavigationControllers {
  controlType?: Controller;
  // Orbit controls
  autoRotate?: Controller;
  autoRotateSpeed?: Controller;
  // Fly controls
  flyMovementSpeed?: Controller;
  flyInertialMode?: Controller;
  flyDamping?: Controller;
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
 * HDR/Exposure GUI controllers
 */
export interface HDRControllers {
  exposure?: Controller;
  hdrMultiplier?: Controller;
  toneMapping?: Controller;
}

/**
 * Union type for control instances
 */
export type ControlInstance = OrbitControls | LuxarFlyControls;

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
  movement: {
    speed: ConfigRange;
    acceleration: ConfigRange;
    damping: ConfigRange;
  };
  look: {
    mouseSpeed: ConfigValue;
    keyboardSpeed: ConfigValue;
  };
  physics: {
    velocityThreshold: number;
    dampingPower: number;
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