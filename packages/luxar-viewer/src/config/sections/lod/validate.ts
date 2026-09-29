import type { AppConfig } from '../../types';

/** Append validation errors for the LOD display-policy configuration. */
export function validateLod(config: AppConfig, errors: string[]): void {
  const { fadeMs, preloadBandFraction } = config.lod;
  if (!Number.isFinite(fadeMs) || fadeMs < 0) {
    errors.push(`lod.fadeMs must be a finite number ≥ 0 (got ${fadeMs})`);
  }
  if (
    !Number.isFinite(preloadBandFraction) ||
    preloadBandFraction < 0 ||
    preloadBandFraction > 0.5
  ) {
    errors.push(`lod.preloadBandFraction must be in [0, 0.5] (got ${preloadBandFraction})`);
  }
}
