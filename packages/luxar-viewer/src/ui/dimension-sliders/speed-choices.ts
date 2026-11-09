/**
 * The Speed section's rate choices, in either of its two views. Pure: the
 * context menu in `ui/dimension-sliders.ts` turns them into chips.
 *
 * - `fps`: steps per second, from `presets.fps`. A sub-1 rate reads as a
 *   fraction (`1/2`, one step every 2 s), never a decimal.
 * - `period`: seconds per step, from `presets.periodSeconds`, labelled
 *   `20s`, `1m`, `2m` — the natural unit for a slideshow or a slow
 *   time-lapse, where an fps (`0.05`) says nothing at a glance. Anything
 *   slower is TYPED (`customPeriodSeconds`): a number and a unit from
 *   seconds to days, up to the viewer's slowest rate.
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

/** The units a period is read and typed in, smallest first. */
export const PERIOD_UNITS = [
  { unit: 's', short: 's', long: 's', seconds: 1 },
  { unit: 'min', short: 'm', long: 'min', seconds: 60 },
  { unit: 'h', short: 'h', long: 'h', seconds: 3600 },
  { unit: 'd', short: 'd', long: 'd', seconds: 86400 },
] as const;

/** One of {@link PERIOD_UNITS}' `unit` keys. */
export type PeriodUnit = (typeof PERIOD_UNITS)[number]['unit'];

/** Relative tolerance for matching a rate against a period (1/p is inexact). */
const RATE_MATCH_TOLERANCE = 1e-6;

/**
 * A period in the unit it reads best in: the largest unit it is a whole
 * number of (`90 s` stays seconds, `7200 s` is `2 h`, `5000 s` stays an exact
 * `5000 s`), else the largest unit it is at least one of, rounded to 3
 * significant digits (`5400.5 s` is `1.5 h`). Under a second stays in seconds.
 */
export function periodInBestUnit(seconds: number): { amount: number; unit: PeriodUnit } {
  const larger = [...PERIOD_UNITS].reverse();
  for (const { unit, seconds: size } of larger) {
    const amount = seconds / size;
    if (amount >= 1 && Math.abs(amount - Math.round(amount)) <= amount * RATE_MATCH_TOLERANCE) {
      return { amount: Math.round(amount), unit };
    }
  }
  const fit = larger.find(({ seconds: size }) => seconds >= size) ?? PERIOD_UNITS[0];
  return { amount: Number((seconds / fit.seconds).toPrecision(3)), unit: fit.unit };
}

function unitInfo(unit: PeriodUnit): (typeof PERIOD_UNITS)[number] {
  return PERIOD_UNITS.find((u) => u.unit === unit) ?? PERIOD_UNITS[0];
}

/** `20s`, `1m`, `90s`, `2h`: the chip label for a period. */
export function formatPeriod(seconds: number): string {
  const { amount, unit } = periodInBestUnit(seconds);
  return `${amount}${unitInfo(unit).short}`;
}

/** `1 step every 20 s`, `1 step every 2 min`: the long form of {@link formatPeriod}. */
export function describePeriod(seconds: number): string {
  const { amount, unit } = periodInBestUnit(seconds);
  return `1 step every ${amount} ${unitInfo(unit).long}`;
}

/**
 * Seconds per step for a typed ``amount`` of ``unit``, or null when that is
 * not a period the viewer plays: not a finite positive number, faster than
 * `presets.customMax` steps a second, or slower than one step every
 * `1 / presets.customMin` seconds (30 days).
 */
export function customPeriodSeconds(amount: number, unit: PeriodUnit): number | null {
  const seconds = amount * unitInfo(unit).seconds;
  const { customMin, customMax } = config.dimensionAnimation.presets;
  const tolerance = 1 + RATE_MATCH_TOLERANCE;
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds * customMax < 1 / tolerance || seconds * customMin > tolerance) return null;
  return seconds;
}

/** Whether a dimension playing at ``fps`` is on one of the period presets. */
export function isPresetPeriod(fps: number): boolean {
  return speedChoices('period').some((choice) => choice.matches(fps));
}

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
