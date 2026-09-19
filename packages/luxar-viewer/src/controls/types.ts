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
 *                  scene spins in the image plane (the same axis and right-hand
 *                  sign a positive Shift+scroll roll delta uses).
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

/**
 * Default auto-dolly amplitude: peak swing as a fraction of the orbit
 * distance, so ±15% means the camera reaches `d×1.15` and `d÷1.15`.
 *
 * Deliberately modest, for a reason that is not about taste. Screen area goes
 * as `1/d²`, so a swing of `A` moves the projected area of the subject by
 * `(1+A)⁴` between its extremes, and the LOD selector answers by loading finer
 * levels at the near extreme. Measured over one cycle on a 100-group demo:
 * 15% → 1.75× area and 118k elements resident; 50% → 5.06× and 526k; 95% →
 * 14.46× and 2.29M. The slider allows all of it (the motion is sometimes the
 * point, and the distance clamps are orders of magnitude away), but the
 * DEFAULT spans less than one screen-area halving, so it can cross at most one
 * LOD boundary per half-cycle — useful for hosted scenes where cost is requests.
 */
export const DEFAULT_AUTO_DOLLY_AMPLITUDE = 0.15;

/** Default auto-dolly period: seconds per full in-and-out oscillation. */
export const DEFAULT_AUTO_DOLLY_PERIOD = 10;

/**
 * Convert a user-facing dolly amplitude in PERCENT to the fraction the control
 * itself holds (15 → 0.15), and back.
 *
 * The unit split is deliberate. The GUI number controller renders a raw value
 * with no unit formatting, so a stored fraction would show as `0.15` under a
 * label promising percent — hence everything user-facing (the settings key, the
 * zarr attribute, the Python field) carries `percent` in its NAME, while the
 * math holds a fraction. These two helpers are the only places the factor of
 * 100 appears, so the conversion cannot be half-applied at one of the three
 * seams that cross the boundary (control construction, the live setter, and
 * the read-back in `syncCurrentState`).
 */
export function dollyAmplitudeFromPercent(percent: number): number {
  return percent / 100;
}

/** Inverse of {@link dollyAmplitudeFromPercent} (0.15 → 15). */
export function dollyAmplitudeToPercent(amplitude: number): number {
  return amplitude * 100;
}

/**
 * Seconds one full turntable revolution takes, from the stored auto-rotation
 * SPEED — and back.
 *
 * `autoRotateSpeed` is a rate in revolutions per minute: the update step turns
 * by `(2π/60)·speed` radians per second, so a full turn takes `60/speed`
 * seconds (measured: 1.0 → 60.00 s, 5.0 → 12.00 s). That unit is inherited
 * from three.js `OrbitControls`, and it is baked into ~15 shipped demo scenes
 * and every published `.luxar.zarr` as `auto_rotate_speed`, so it must keep
 * meaning exactly what it means today on disk.
 *
 * The UI asks the question users actually have — "how long is one turn?" — and
 * converts here, exactly as {@link dollyAmplitudeFromPercent} does for the
 * dolly. Same trick, same reason: one unit stored, a friendlier one shown, and
 * a single named place where the two meet.
 *
 * Both directions are `60/x`, but they are named separately so a call site
 * reads as a conversion rather than as arithmetic. A non-positive or
 * non-finite input would divide to `Infinity`/`NaN` and freeze or explode the
 * turntable, so it falls back to the other unit's identity instead.
 */
export function secondsPerTurnFromRpm(rpm: number): number {
  if (!Number.isFinite(rpm) || rpm <= 0) return 60;
  return 60 / rpm;
}

/** Inverse of {@link secondsPerTurnFromRpm} (20 s → 3 rpm). */
export function rpmFromSecondsPerTurn(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) return 1;
  return 60 / seconds;
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
