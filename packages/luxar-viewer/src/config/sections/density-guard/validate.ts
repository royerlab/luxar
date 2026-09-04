import type { AppConfig } from '../../types';

export function validateDensityGuard(config: AppConfig, errors: string[]): void {
  const g = config.densityGuard;
  if (!Number.isFinite(g.capElementsPerPixel) || g.capElementsPerPixel <= 0) {
    errors.push(`densityGuard.capElementsPerPixel must be > 0 (got ${g.capElementsPerPixel})`);
  }
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
