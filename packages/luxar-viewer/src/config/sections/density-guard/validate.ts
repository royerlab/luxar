import type { AppConfig } from '../../types';

function requirePositive(errors: string[], name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    errors.push(`densityGuard.${name} must be > 0 (got ${value})`);
  }
}

/** Append validation errors for the projected-density guard configuration. */
export function validateDensityGuard(config: AppConfig, errors: string[]): void {
  const g = config.densityGuard;
  requirePositive(errors, 'capElementsPerPixel', g.capElementsPerPixel);
  requirePositive(errors, 'nonBlendableCapElementsPerPixel', g.nonBlendableCapElementsPerPixel);
  if (!Number.isFinite(g.minKeepFraction) || g.minKeepFraction <= 0 || g.minKeepFraction > 1) {
    errors.push(`densityGuard.minKeepFraction must be in (0, 1] (got ${g.minKeepFraction})`);
  }
  if (!Number.isFinite(g.enterRatio) || g.enterRatio < 1) {
    errors.push(`densityGuard.enterRatio must be ≥ 1 (got ${g.enterRatio})`);
  }
  if (!Number.isFinite(g.leaveRatio) || g.leaveRatio <= 0 || g.leaveRatio >= g.enterRatio) {
    errors.push(
      `densityGuard.leaveRatio must be in (0, enterRatio) (got ${g.leaveRatio}, enterRatio ${g.enterRatio})`
    );
  }
}
