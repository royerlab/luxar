import type { LodConfig } from './types';

/**
 * LOD display-policy defaults. See {@link LodConfig} for what each one tunes
 * and why it has its value.
 *
 * `fadeMs` 250: long enough to read as a dissolve rather than a pop, short
 * enough that the two-level overdraw it costs lasts about 15 frames at 60 fps.
 *
 * `preloadBandFraction` 0.4: the band the retired coverage cross-fade drew both
 * levels in, so the neighbouring level is resident whenever it used to be.
 */
export const lodConfig: LodConfig = {
  fadeMs: 250,
  preloadBandFraction: 0.4,
  preloadExitBandFraction: 0.5,
  fineReloadSettleMs: 130,
  playbackLoadBudgetFraction: 0.8,
  playbackKeepBudgetFraction: 1.0,
  loadEwmaAlpha: 0.3,
  playbackProbeIntervalMs: 1000,
  staleHoldMs: 250,
  staleHoldMinRatio: 0.5,
  failedRetryMs: 2000,
  lazyActivationRequestTimeoutMs: 2000,
  partitionFrustumMargin: 0.1,
  hysteresisRatio: 0.1,
  maxMedianFootprintPx: 1.5,
};
