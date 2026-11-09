/**
 * The Speed section's rate choices, in either of its two views. Pure: the
 * context menu in `ui/dimension-sliders.ts` turns them into chips.
 *
 * - `fps`: steps per second, from `presets.fps`. A sub-1 rate reads as a
 *   fraction (`1/2`, one step every 2 s), never a decimal.
 * - `period`: seconds per step, from `presets.periodSeconds`, labelled
 *   `20s`, `1m`, `2m` — the natural unit for a slideshow or a slow
 *   time-lapse, where an fps (`0.05`) says nothing at a glance.
 *
 * Either view also takes a TYPED rate (`customRate`): a number and a unit,
 * seconds to days per step, or steps per second to per day, anywhere in the
 * range the viewer plays.
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
 * A rate in the unit it reads best in, as the view types it: seconds-to-days
 * PER STEP in the period view (:func:`periodInBestUnit`), steps PER second,
 * minute, hour or day in the fps view -- the smallest unit that gives a whole
 * number of steps (`0.05` fps is `3 /min`, `10` fps stays `10 /s`), else the
 * smallest that gives at least one, rounded to 3 significant digits.
 */
export function rateInBestUnit(view: SpeedView, fps: number): { amount: number; unit: PeriodUnit } {
  if (view === 'period') return periodInBestUnit(1 / fps);
  for (const { unit, seconds: size } of PERIOD_UNITS) {
    const amount = fps * size;
    if (amount >= 1 && Math.abs(amount - Math.round(amount)) <= amount * RATE_MATCH_TOLERANCE) {
      return { amount: Math.round(amount), unit };
    }
  }
  const fit = PERIOD_UNITS.find(({ seconds: size }) => fps * size >= 1) ?? PERIOD_UNITS[3];
  return { amount: Number((fps * fit.seconds).toPrecision(3)), unit: fit.unit };
}

/** The unit picker's label for ``unit`` in ``view``: `min` (per step) or `/min`. */
export function rateUnitLabel(view: SpeedView, unit: PeriodUnit): string {
  return view === 'period' ? unitInfo(unit).long : `/${unitInfo(unit).long}`;
}

/**
 * The rate (steps per second) for a typed ``amount`` of ``unit`` in ``view`` --
 * that many seconds, minutes, hours or days per step, or that many steps per
 * one of them -- or null when it is not a rate the viewer plays: not a finite
 * positive number, or outside `[presets.customMin, presets.customMax]` (one
 * step every 30 days to 120 steps a second).
 */
export function customRate(view: SpeedView, amount: number, unit: PeriodUnit): number | null {
  const size = unitInfo(unit).seconds;
  const fps = view === 'period' ? 1 / (amount * size) : amount / size;
  const { customMin, customMax } = config.dimensionAnimation.presets;
  const tolerance = 1 + RATE_MATCH_TOLERANCE;
  if (!Number.isFinite(fps) || fps <= 0 || !Number.isFinite(amount) || amount <= 0) return null;
  if (fps > customMax * tolerance || fps * tolerance < customMin) return null;
  return fps;
}

/** Whether a dimension playing at ``fps`` is on one of ``view``'s chips. */
export function isPresetRate(view: SpeedView, fps: number): boolean {
  return speedChoices(view).some((choice) => choice.matches(fps));
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
