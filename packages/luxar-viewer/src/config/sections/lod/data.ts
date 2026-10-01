import type { LodConfig } from './types';

/**
 * LOD display-policy defaults.
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
};
