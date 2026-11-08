/**
 * The Speed section's rate choices, in either of its two views. Pure: the
 * context menu in `ui/dimension-sliders.ts` turns them into chips.
 *
 * - `fps`: steps per second, from `presets.fps`. A sub-1 rate reads as a
 *   fraction (`1/2`, one step every 2 s), never a decimal.
 * - `period`: seconds per step, from `presets.periodSeconds`, labelled
 *   `20s`, `1m`, `10m` — the natural unit for a slideshow or a slow
 *   time-lapse, where an fps (`0.05`) says nothing at a glance.
 *
 * Either way a choice carries the `fps` it sets; the rate itself is always
 * `targetFPS`.
 */

import { config } from '../../config';
import type { SpeedView } from '../../types/animation';

/** One selectable rate. */
export interface SpeedChoice {
  label: string;
  tooltip: string;
  /** The rate this choice sets, in steps per second. */
  fps: number;
  /** Whether a dimension playing at `fps` is on this choice. */
  matches(fps: number): boolean;
}

/** The two views, in toggle order, with their segment labels. */
export const SPEED_VIEWS: ReadonlyArray<{ view: SpeedView; label: string; tooltip: string }> = [
  { view: 'fps', label: 'fps', tooltip: 'Steps per second' },
  { view: 'period', label: 'period', tooltip: 'Seconds per step' },
];

/** `20s`, `1m`, `90s`: whole minutes in minutes, anything else in seconds. */
export function formatPeriod(seconds: number): string {
  return seconds >= 60 && Number.isInteger(seconds / 60) ? `${seconds / 60}m` : `${seconds}s`;
}

/** `every 20 s`, `every 2 min`: the long form of {@link formatPeriod}. */
function describePeriod(seconds: number): string {
  return seconds >= 60 && Number.isInteger(seconds / 60)
    ? `1 step every ${seconds / 60} min`
    : `1 step every ${seconds} s`;
}

/** Relative tolerance for matching a rate against a period (1/p is inexact). */
const RATE_MATCH_TOLERANCE = 1e-6;

function fpsChoice(fps: number): SpeedChoice {
  const sub = fps < 1;
  return {
    label: sub ? `1/${Math.round(1 / fps)}` : String(fps),
    tooltip: sub ? `1 frame every ${Math.round(1 / fps)} s` : `${fps} FPS`,
    fps,
    matches: (current) => current === fps,
  };
}

function periodChoice(seconds: number): SpeedChoice {
  return {
    label: formatPeriod(seconds),
    tooltip: describePeriod(seconds),
    fps: 1 / seconds,
    matches: (current) =>
      current > 0 && Math.abs(1 / current - seconds) <= seconds * RATE_MATCH_TOLERANCE,
  };
}

/** The Speed section's choices for `view`, in preset order. */
export function speedChoices(view: SpeedView): SpeedChoice[] {
  const { fps, periodSeconds } = config.dimensionAnimation.presets;
  return view === 'period' ? periodSeconds.map(periodChoice) : fps.map(fpsChoice);
}
