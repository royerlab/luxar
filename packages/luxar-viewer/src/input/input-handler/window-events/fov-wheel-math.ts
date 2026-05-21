/**
 * Pure FOV-from-wheel math. Currently unused by the live wheel handler
 * (which calls `sceneManager.updateFOV(deltaY)` directly), but kept
 * adjacent to the wheel handler as a documented reference + because
 * its tests pin the [10°, 170°] clamp behavior we depend on elsewhere.
 *
 * @module input/input-handler/window-events/fov-wheel-math
 */

import { config } from '../../../config';
import { clamp } from '../../../utils/clamp';

/**
 * Calculate field of view (FOV) change from mouse wheel input.
 *
 * Converts mouse wheel delta to FOV adjustment with configurable sensitivity.
 * FOV is clamped to reasonable range (10°-170° from config) to prevent
 * extreme distortion. Used for Ctrl+wheel FOV control, allowing users to
 * adjust perspective from telephoto (narrow FOV) to wide-angle (wide FOV).
 *
 * Positive delta increases FOV (zoom out), negative delta decreases FOV
 * (zoom in). The sensitivity parameter scales the change rate.
 *
 * @param currentFov - Current field of view in degrees (typically 50-75°)
 * @param delta - Mouse wheel delta from WheelEvent.deltaY (typically -100 to 100)
 * @param sensitivity - Sensitivity multiplier (default from config, typically 0.1)
 * @returns New FOV value in degrees, clamped to config.camera fov range
 */
export function calculateFovChange(
  currentFov: number,
  delta: number,
  sensitivity: number = config.input.defaultSensitivity
): number {
  const change = delta * sensitivity;
  const newFov = currentFov + change;

  // Clamp to config-defined FOV range
  return clamp(newFov, config.camera.fovMin, config.camera.fovMax);
}
