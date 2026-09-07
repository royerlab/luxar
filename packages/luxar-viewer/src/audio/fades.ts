/**
 * Gain-ramp helpers shared by the sound nodes and the ducker.
 *
 * Every audible edge goes through a linear ramp of at least {@link MIN_FADE_MS}
 * so a start or stop never lands as a click (`SOUND_SPEC.md` §7).
 *
 * @module audio/fades
 */

import { MIN_FADE_MS } from '../types/audio';

/** What a gain ramp needs; see {@link rampGain}. */
export interface RampOptions {
  /** The context's current time — pending automation is cancelled from here. */
  now: number;
  /** When the ramp begins; defaults to `now`. May lie in the future (a delayed start). */
  startAt?: number;
  target: number;
  /** Ramp length; floored to {@link MIN_FADE_MS}. */
  ms: number;
  /** Start value; defaults to the param's current value. */
  from?: number;
}

/**
 * Schedule a linear ramp on `param` to `target`, holding `from` until `startAt`
 * when that lies in the future. Returns the time the ramp completes.
 */
export function rampGain(param: AudioParam, opts: RampOptions): number {
  const { now, target, ms } = opts;
  const durationSec = Math.max(ms, MIN_FADE_MS) / 1000;
  const begin = Math.max(opts.startAt ?? now, now);
  param.cancelScheduledValues(now);
  const start = opts.from ?? param.value;
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
