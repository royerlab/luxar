import type { AppConfig } from '../../types';
import type { LodConfig } from './types';

/** `[min, max]` bounds of a numeric field; `open` excludes that end. */
interface Range {
  min: number;
  max: number;
  openMin?: boolean;
  openMax?: boolean;
}

/** Fields that only need a finite value inside a fixed range. */
const RANGES: ReadonlyArray<[keyof LodConfig, Range]> = [
  ['fadeMs', { min: 0, max: Infinity }],
  ['preloadExitBandFraction', { min: 0, max: 0.5, openMin: true }],
  ['fineReloadSettleMs', { min: 0, max: Infinity }],
  ['playbackLoadBudgetFraction', { min: 0, max: Infinity, openMin: true }],
  ['loadEwmaAlpha', { min: 0, max: 1, openMin: true }],
  ['playbackProbeIntervalMs', { min: 0, max: Infinity }],
  ['staleHoldMs', { min: 0, max: Infinity }],
  ['staleHoldMinRatio', { min: 0, max: 1 }],
  ['failedRetryMs', { min: 0, max: Infinity }],
  ['lazyActivationRequestTimeoutMs', { min: 0, max: Infinity, openMin: true }],
  ['partitionFrustumMargin', { min: 0, max: Infinity }],
  ['hysteresisRatio', { min: 0, max: 1, openMax: true }],
  ['maxMedianFootprintPx', { min: 0, max: Infinity, openMin: true }],
];

function inRange(value: number, range: Range): boolean {
  if (!Number.isFinite(value)) return false;
  if (range.openMin ? value <= range.min : value < range.min) return false;
  return range.openMax ? value < range.max : value <= range.max;
}

function describe(range: Range): string {
  const lo = `${range.openMin ? '(' : '['}${range.min}`;
  const hi = `${range.max === Infinity ? '∞' : range.max}${range.openMax || range.max === Infinity ? ')' : ']'}`;
  return `${lo}, ${hi}`;
}

/** Append validation errors for the LOD display-policy configuration. */
export function validateLod(config: AppConfig, errors: string[]): void {
  const lod = config.lod;
  for (const [key, range] of RANGES) {
    if (!inRange(lod[key], range)) {
      errors.push(`lod.${key} must be a finite number in ${describe(range)} (got ${lod[key]})`);
    }
  }
  // Strictly inside the exit band: an entry band as wide as the exit band
  // leaves no hysteresis, so a camera at its edge restarts a visit (and a
  // reload) per wobble.
  const { preloadBandFraction, preloadExitBandFraction } = lod;
  if (
    !Number.isFinite(preloadBandFraction) ||
    preloadBandFraction < 0 ||
    !(preloadBandFraction < preloadExitBandFraction)
  ) {
    errors.push(
      'lod.preloadBandFraction must be in [0, lod.preloadExitBandFraction = ' +
        `${preloadExitBandFraction}) (got ${preloadBandFraction})`
    );
  }
  // The keep budget is the hysteresis of the admission budget: never tighter.
  const { playbackKeepBudgetFraction, playbackLoadBudgetFraction } = lod;
  if (
    !Number.isFinite(playbackKeepBudgetFraction) ||
    !(playbackKeepBudgetFraction >= playbackLoadBudgetFraction)
  ) {
    errors.push(
      'lod.playbackKeepBudgetFraction must be ≥ lod.playbackLoadBudgetFraction ' +
        `(got ${playbackKeepBudgetFraction} < ${playbackLoadBudgetFraction})`
    );
  }
}
