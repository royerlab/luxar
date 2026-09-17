import type { DensityGuardConfig } from './types';

/**
 * Projected-density guard defaults.
 *
 * `capElementsPerPixel` 4: a sum-projected node needs a few elements per
 * pixel to keep its integrated brightness smooth; beyond that every further
 * element is overdraw the GPU pays for and the eye cannot see. The audited
 * pathological views sat at 600–100 000 elements per pixel.
 */
export const densityGuardConfig: DensityGuardConfig = {
  enabled: true, // URL escape hatch: ?noDensityGuard
  capElementsPerPixel: 4,
  nonBlendableCapElementsPerPixel: 1,
  minKeepFraction: 1 / 64,
  enterRatio: 1.5,
  leaveRatio: 0.75,
};
