import type { LodConfig } from './types';

/**
 * LOD display-policy defaults.
 *
 * `fadeMs` 250: long enough to read as a dissolve rather than a pop, short
 * enough that the two-level overdraw it costs lasts about 15 frames at 60 fps.
 */
export const lodConfig: LodConfig = {
  fadeMs: 250,
};
