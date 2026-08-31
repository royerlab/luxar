/**
 * Auto-dolly (sinusoidal back-and-forth) math.
 *
 * The turntable's sibling: where auto-rotation sweeps the camera AROUND the
 * target, auto-dolly breathes it TOWARD and away from it, on a sine. Two tiny
 * pure functions, split out for the same reason `auto-rotate.ts` is — the
 * interactive dolly (driven by wall-clock time in the update sequencer) and
 * the recorded one (driven by frame index in the capture strategies) both go
 * through {@link dollyScale}, so an exported video cannot breathe unlike the
 * preview it was set up from. {@link dollyAmplitudeChangeScale} preserves the
 * same baseline when the live amplitude or phase origin changes.
 *
 * @module controls/luxar-orbit-controls/math/auto-dolly
 */

const TWO_PI = Math.PI * 2;

function logAmplitude(amplitude: number): number {
  return Number.isFinite(amplitude) && amplitude > 0 ? Math.log1p(amplitude) : 0;
}

/**
 * Advance the oscillation phase by `deltaTime` seconds, wrapped to `[0, 2π)`.
 *
 * Wrapping keeps the phase small no matter how long the viewer is left
 * running: `sin` is periodic, so the wrap is exact, while an ever-growing
 * radian count would bleed precision out of the sine after a few hours.
 *
 * A non-finite or non-positive `period` is inert (phase unchanged) rather than
 * an error: this runs inside the render loop, and a hand-edited scene attribute
 * should stop the motion, not produce `NaN` and freeze the camera at an
 * unrecoverable position.
 *
 * @param phase - Current phase in radians.
 * @param deltaTime - Seconds elapsed since the previous sample.
 * @param period - Seconds per full oscillation.
 * @returns The new phase in `[0, 2π)`.
 */
export function advanceDollyPhase(phase: number, deltaTime: number, period: number): number {
  if (!Number.isFinite(period) || period <= 0 || !Number.isFinite(deltaTime)) return phase;
  const next = (phase + (TWO_PI * deltaTime) / period) % TWO_PI;
  // A negative deltaTime (a clock that jumped backwards) leaves `%` negative.
  return next < 0 ? next + TWO_PI : next;
}

/**
 * The multiplicative distance change between two phases.
 *
 * The oscillation is defined in LOG distance — `d(φ) = d₀·exp(−A·sin φ)` with
 * `A = ln(1 + amplitude)` — which is what makes it the sinusoidal mousewheel
 * the feature is named for: the wheel scales distance by a constant factor per
 * click, so equal-sized swings in and out are equal in RATIO, not in scene
 * units. An `amplitude` of 0.15 therefore reaches `d₀×1.15` at its far point
 * and `d₀÷1.15` at its near one, on any scene at any scale. The minus sign
 * makes a rising phase move the camera CLOSER, so switching the feature on
 * begins with an approach.
 *
 * Returning the RATIO between two phases rather than an absolute distance is
 * what lets the user keep zooming while this runs. Distance is only ever
 * multiplied — by the wheel, and by this — and multiplication commutes, so a
 * wheel click mid-oscillation just moves the centre the camera is breathing
 * around, exactly as it would if the motion were not running. Nothing has to
 * arbitrate between the two, and no baseline has to be tracked and re-derived.
 *
 * Taking the difference of two sines (rather than integrating the derivative
 * `−A·cos φ · dφ`) is also what keeps it exact: the amplitude is right whatever
 * the frame rate, and the factors over one full period telescope to exactly 1,
 * so the centre cannot drift over a long session.
 *
 * @param fromPhase - Phase at the previous sample, radians.
 * @param toPhase - Phase now, radians.
 * @param amplitude - Peak swing as a fraction of distance (0.15 = ±15%).
 *   Non-finite, zero, or negative is inert.
 * @returns The factor to multiply the orbit distance by (1 = no change).
 */
export function dollyScale(fromPhase: number, toPhase: number, amplitude: number): number {
  if (!Number.isFinite(fromPhase) || !Number.isFinite(toPhase)) return 1;
  return Math.exp(-logAmplitude(amplitude) * (Math.sin(toPhase) - Math.sin(fromPhase)));
}

/** Preserve the same baseline while changing amplitude at the current phase. */
export function dollyAmplitudeChangeScale(
  phase: number,
  fromAmplitude: number,
  toAmplitude: number
): number {
  if (!Number.isFinite(phase)) return 1;
  return Math.exp((logAmplitude(fromAmplitude) - logAmplitude(toAmplitude)) * Math.sin(phase));
}
