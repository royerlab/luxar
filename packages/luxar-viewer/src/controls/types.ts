/**
 * Type definitions for the Luxar control system.
 */

import type { LuxarOrbitControls } from './luxar-orbit-controls';
import type { LuxarFlyControls } from './luxar-fly-controls';

/** Available control types. */
export type ControlType = 'orbit' | 'fly' | 'ortho';

/** Return the next camera control type, resetting unknown values to orbit. */
export function nextControlType(current: ControlType | string): ControlType {
  switch (current) {
    case 'orbit':
      return 'fly';
    case 'fly':
      return 'ortho';
    case 'ortho':
      return 'orbit';
    default:
      return 'orbit';
  }
}

/**
 * Axis the orbit turntable (auto-rotation) revolves around, named in the
 * CAMERA frame so the choice describes what the viewer actually sees:
 *
 * - `vertical`   — the screen-up axis. The historical (and default) behavior:
 *                  the scene spins about a vertical line on screen.
 * - `horizontal` — the screen-right axis. The scene tumbles over the top, like
 *                  a wheel rolling away from the viewer.
 * - `view`       — the view direction. A pure roll: the camera never moves, the
 *                  scene spins in the image plane (the same axis Shift+scroll
 *                  roll uses, and with the same sign).
 *
 * Deliberately NOT named x/y/z: the camera frame has up = +Y while the
 * gallery harness already spells WORLD axes `'x' | 'y' | 'z'`
 * (`tests/screenshots/orbit-axis.ts`), and in an nD scientific viewer a bare
 * letter reads as a DATA axis. Every axis is re-derived from the live
 * orientation each frame, but each is invariant under its own rotation, so
 * all three are stable — no drift, and no pole hazard, because `camera.up`
 * is derived from the orientation quaternion rather than clamped to world up.
 */
export type AutoRotateAxis = 'vertical' | 'horizontal' | 'view';

/**
 * Every valid {@link AutoRotateAxis}, in UI order. The single source of truth
 * for enum validation (persisted settings, scene-authored `auto_rotate_axis`),
 * so the viewer's accepted set cannot drift from the type.
 */
export const AUTO_ROTATE_AXES: readonly AutoRotateAxis[] = ['vertical', 'horizontal', 'view'];

/** The turntable axis in use before the axis was selectable — screen-vertical. */
export const DEFAULT_AUTO_ROTATE_AXIS: AutoRotateAxis = 'vertical';

/** Narrow an arbitrary value to an {@link AutoRotateAxis}. */
export function isAutoRotateAxis(value: unknown): value is AutoRotateAxis {
  return typeof value === 'string' && (AUTO_ROTATE_AXES as readonly string[]).includes(value);
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
