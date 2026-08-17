import type { AppConfig } from '../../types';

/**
 * Validate depth-sort scheduling configuration.
 *
 * Errors flag values that break the scheduler outright (non-finite or
 * non-positive thresholds — a zero threshold dispatches a sort every
 * frame); warnings flag values that defeat the design intent (an angle
 * so large a full orbit never re-sorts).
 */
export function validateDepthSort(config: AppConfig, errors: string[], warnings: string[]): void {
  const ds = config.depthSort;

  if (!Number.isFinite(ds.angleThresholdDeg) || ds.angleThresholdDeg <= 0) {
    errors.push(
      `depthSort.angleThresholdDeg must be a finite number > 0 (got ${ds.angleThresholdDeg})`
    );
  } else if (ds.angleThresholdDeg > 45) {
    warnings.push(
      `depthSort.angleThresholdDeg (${ds.angleThresholdDeg}) is very coarse; ` +
        'orbiting will show visibly stale ordering before a re-sort fires'
    );
  }

  if (!Number.isFinite(ds.translationFraction) || ds.translationFraction <= 0) {
    errors.push(
      `depthSort.translationFraction must be a finite number > 0 (got ${ds.translationFraction})`
    );
  } else if (ds.translationFraction >= 1) {
    warnings.push(
      `depthSort.translationFraction (${ds.translationFraction}) is >= 1 bounding radius; ` +
        'flying through a node will show stale behind-camera culling before a re-sort fires'
    );
  }

  // 0 disables the guard (the shared withTimeout convention), so only a
  // NEGATIVE or non-finite value is an error.
  if (!Number.isFinite(ds.workerInitTimeoutMs) || ds.workerInitTimeoutMs < 0) {
    errors.push(
      `depthSort.workerInitTimeoutMs must be a finite number >= 0 (got ${ds.workerInitTimeoutMs})`
    );
  } else if (ds.workerInitTimeoutMs > 0 && ds.workerInitTimeoutMs < 1000) {
    warnings.push(
      `depthSort.workerInitTimeoutMs (${ds.workerInitTimeoutMs}) is shorter than a cold WASM ` +
        'instantiate; the first sort will be deferred to the post-load retry on most machines'
    );
  }
}
