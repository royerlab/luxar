/**
 * Pure FOV / focal-length conversion helpers used by the camera section
 * of the rendering-controls panel.
 *
 * Extracted from camera-setup.ts so the optical math is unit-testable
 * without instantiating a SceneManager or a GUI.
 *
 * @module ui/rendering-controls/fov-utils
 */

/**
 * 35mm-film sensor width in mm. Used as the reference sensor for the
 * "approximate focal length" labels shown next to the FOV slider —
 * matching the conventions of full-frame photography (28mm = wide,
 * 50mm ≈ "natural", 135mm = telephoto).
 */
export const FILM_35MM_SENSOR_WIDTH_MM = 36;

/**
 * Approximate 35mm-equivalent focal length from a horizontal FOV.
 *
 * Inverts the pinhole-camera relation
 *   FOV = 2 · atan( sensorWidth / (2 · focalLength) )
 * to give
 *   focalLength = sensorWidth / (2 · tan(FOV/2))
 *
 * The result is rounded to the nearest integer mm — the slider only
 * displays whole-millimetre values, so any sub-mm precision would be
 * cosmetic.
 *
 * @param fovDegrees - Horizontal field of view in degrees, expected
 *   to be in `(0, 180)`. Outside this range the math still runs but
 *   the result is meaningless: tan(±90°) → ±∞ → focalLength → 0.
 */
export function fovToFocalLength(fovDegrees: number): number {
  const fovRadians = (fovDegrees * Math.PI) / 180;
  const focalLength = FILM_35MM_SENSOR_WIDTH_MM / (2 * Math.tan(fovRadians / 2));
  return Math.round(focalLength);
}

/**
 * Inverse of {@link fovToFocalLength} — the horizontal FOV in degrees
 * a given 35mm-equivalent focal length corresponds to. Currently used
 * by the FOV-preset table shape only; exposed here so the table-build
 * round-trips cleanly in tests.
 *
 * @param focalLengthMm - 35mm-equivalent focal length in mm. Must be
 *   strictly positive; values ≤ 0 produce a non-finite or 0 FOV.
 */
export function focalLengthToFov(focalLengthMm: number): number {
  if (focalLengthMm <= 0) return 0;
  const fovRadians = 2 * Math.atan(FILM_35MM_SENSOR_WIDTH_MM / (2 * focalLengthMm));
  return (fovRadians * 180) / Math.PI;
}
