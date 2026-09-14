/**
 * Shared view-axis roll convention for touch twist and Shift+wheel gestures.
 *
 * In y-down screen space, `atan2` grows for a visually clockwise twist. A
 * positive rotation about the camera's forward axis turns the camera clockwise
 * as the user sees it, so the scene turns counter-clockwise; negating the input
 * makes the scene follow the fingers. Shift+wheel adopts that same direction.
 * Orbit and fly apply the rotation about the same axis with the same
 * premultiplication order, so the sign is one law.
 *
 * Controller-specific gains intentionally stay at their call sites: orbit's
 * touch gain is a dimensionless angle multiplier, while fly's touch gain feeds
 * angular velocity and therefore has rate units. Wheel gains likewise retain
 * their existing controller-specific magnitudes.
 */
export const VIEW_AXIS_ROLL_SIGN = -1;

/** Wrap an angle difference into (-π, π] so a twist across ±π does not jump. */
export function wrapAngle(delta: number): number {
  if (!Number.isFinite(delta)) return 0;
  let wrapped = delta;
  while (wrapped > Math.PI) wrapped -= 2 * Math.PI;
  while (wrapped <= -Math.PI) wrapped += 2 * Math.PI;
  return wrapped;
}
