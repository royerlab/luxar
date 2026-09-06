/**
 * Gain-ramp helpers shared by the sound nodes and the ducker.
 *
 * Every audible edge goes through a linear ramp of at least {@link MIN_FADE_MS}
 * so a start or stop never lands as a click (`SOUND_SPEC.md` §7).
 *
 * @module audio/fades
 */

import { MIN_FADE_MS } from '../types/audio';

/**
 * Schedule a linear ramp on `param` from its value at `startAt` to `target`.
 *
 * `now` is the context's current time (the point from which pending automation
 * is cancelled); `startAt` may lie in the future (a delayed start) — the param
 * then holds `from` until `startAt` and ramps afterwards. Returns the time the
 * ramp completes.
 */
export function rampGain(
  param: AudioParam,
  now: number,
  startAt: number,
  target: number,
  ms: number,
  from?: number
): number {
  const durationSec = Math.max(ms, MIN_FADE_MS) / 1000;
  const begin = Math.max(startAt, now);
  param.cancelScheduledValues(now);
  const start = from ?? param.value;
  param.setValueAtTime(start, now);
  if (begin > now) param.setValueAtTime(start, begin);
  const end = begin + durationSec;
  param.linearRampToValueAtTime(target, end);
  return end;
}

/** Decibels → linear gain (`-9 dB` ≈ `0.355`). */
export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}
