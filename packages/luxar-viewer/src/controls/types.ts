/**
 * Type definitions for the Luxar control system.
 */

import type { LuxarOrbitControls } from './luxar-orbit-controls';
import type { LuxarFlyControls } from './luxar-fly-controls';

/** Available control types. */
export type ControlType = 'orbit' | 'fly' | 'ortho';

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
