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
 * Axis the orbit turntable (auto-rotation) revolves around. Two families:
 *
 * **Camera frame** — named for what the viewer sees, so the choice means the
 * same thing whatever the scene's orientation:
 *
 * - `vertical`   — the screen-up axis. The historical (and default) behavior:
 *                  the scene spins about a vertical line on screen.
 * - `horizontal` — the screen-right axis. The scene tumbles over the top, like
 *                  a wheel rolling away from the viewer.
 * - `view`       — the view direction. A pure roll: the camera never moves, the
 *                  scene spins in the image plane (the same axis Shift+scroll
 *                  roll uses, and with the same sign).
 *
 * **World frame** — a fixed scene axis, the classic turntable: the subject
 * spins about its OWN axis at any camera elevation, where a camera-frame
 * `vertical` turntable makes that axis precess (a spin plus a wobble):
 *
 * - `world-x` / `world-y` / `world-z` — the scene's ±X / ±Y / ±Z.
 *
 * The naming split is deliberate and matches what each family can promise: a
 * bare letter names a WORLD axis here exactly as it does in the gallery
 * harness (`tests/screenshots/orbit-axis.ts`'s per-demo `orbitUp`), while the
 * camera-frame options get words because the camera frame has up = +Y and a
 * letter would read as a DATA axis in an nD scientific viewer.
 *
 * No axis can destabilize the orbit. A camera-frame axis is re-derived from the
 * live orientation each frame but is invariant under its own rotation; a world
 * axis is a constant with no feedback at all. Neither can hit a pole, because
 * `camera.up` is derived from the orientation quaternion rather than clamped to
 * world up. The one degenerate case is benign: a world axis parallel to the
 * view direction leaves the camera where it is and rolls the image, exactly as
 * `view` does.
 */
export type AutoRotateAxis = 'vertical' | 'horizontal' | 'view' | 'world-x' | 'world-y' | 'world-z';

/**
 * Every valid {@link AutoRotateAxis}, in UI order (camera frame first, then
 * world). The single source of truth for enum validation (persisted settings,
 * scene-authored `auto_rotate_axis`), so the viewer's accepted set cannot
 * drift from the type.
 */
export const AUTO_ROTATE_AXES: readonly AutoRotateAxis[] = [
  'vertical',
  'horizontal',
  'view',
  'world-x',
  'world-y',
  'world-z',
];

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
