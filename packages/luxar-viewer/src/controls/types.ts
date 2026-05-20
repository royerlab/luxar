/**
 * Type definitions for the Luxar control system.
 *
 * The GUI-controller reference types (RenderingControllers / BloomControllers /
 * HDRControllers) describe handles to lil-gui controllers for the rendering
 * panel — they live here for historical reasons but are consumed entirely
 * by `ui/rendering-controls/`. Moving them would change external import
 * paths; see the controls package README for the deferred cleanup.
 */

import type { Controller } from '../ui/gui';
import type { LuxarOrbitControls } from './luxar-orbit-controls';
import type { LuxarFlyControls } from './luxar-fly-controls';

/** Available control types. */
export type ControlType = 'orbit' | 'fly' | 'ortho';

/** Rendering controls GUI controllers for the entire rendering panel. */
export interface RenderingControllers {
  controlType?: Controller;
  // Camera controls
  fov?: Controller;
  fovPreset?: Controller;
  nearPlane?: Controller;
  farPlane?: Controller;
  // Dynamic clipping controls
  dynamicClippingEnabled?: Controller;
  // HDR controls (global EOG)
  exposure?: Controller;
  globalOffset?: Controller;
  globalGamma?: Controller;
  // Chromatic lens distortion controls
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
  naturalDrag?: Controller;
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

/** Bloom effect GUI controllers. */
export interface BloomControllers {
  threshold?: Controller;
  strength?: Controller;
  radius?: Controller;
}

/** HDR GUI controllers. */
export interface HDRControllers {
  hdrMultiplier?: Controller;
  toneMapping?: Controller;
}

/** Union type for control instances. */
export type ControlInstance = LuxarOrbitControls | LuxarFlyControls;

/** Type guard for LuxarOrbitControls. */
export function isOrbitControls(control: ControlInstance | null): control is LuxarOrbitControls {
  return control !== null && 'target' in control && 'autoRotate' in control;
}

/** Type guard for LuxarFlyControls. */
export function isFlyControls(control: ControlInstance | null): control is LuxarFlyControls {
  return control !== null && 'inertialMode' in control && 'lookSpeed' in control;
}
