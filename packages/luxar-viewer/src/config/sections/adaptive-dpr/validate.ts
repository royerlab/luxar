import type { AppConfig } from '../../types';

/**
 * Validate adaptive DPR configuration.
 *
 * Errors flag values that break the control loop outright (inverted
 * thresholds, non-contracting factors, non-positive timings); warnings
 * flag values that are legal but defeat the design intent (a probe
 * window shorter than the 1s FPS sample window, a gap threshold so
 * large it never fires).
 */
export function validateAdaptiveDPR(config: AppConfig, errors: string[], warnings: string[]): void {
  const dpr = config.adaptiveDPR;

  const finite = (name: string, value: number): boolean => {
    if (!Number.isFinite(value)) {
      errors.push(`adaptiveDPR.${name} must be a finite number (got ${value})`);
      return false;
    }
    return true;
  };

  if (finite('minDPR', dpr.minDPR) && (dpr.minDPR <= 0 || dpr.minDPR > 1)) {
    errors.push(`adaptiveDPR.minDPR must be in (0, 1] (got ${dpr.minDPR})`);
  }
  if (
    finite('scaleDownFactor', dpr.scaleDownFactor) &&
    (dpr.scaleDownFactor <= 0 || dpr.scaleDownFactor >= 1)
  ) {
    errors.push(`adaptiveDPR.scaleDownFactor must be in (0, 1) (got ${dpr.scaleDownFactor})`);
  }
  if (finite('scaleUpFactor', dpr.scaleUpFactor) && dpr.scaleUpFactor <= 1) {
    errors.push(`adaptiveDPR.scaleUpFactor must be > 1 (got ${dpr.scaleUpFactor})`);
  }
  if (finite('hysteresisSeconds', dpr.hysteresisSeconds) && dpr.hysteresisSeconds <= 0) {
    errors.push(`adaptiveDPR.hysteresisSeconds must be > 0 (got ${dpr.hysteresisSeconds})`);
  }
  if (finite('evaluationIntervalMs', dpr.evaluationIntervalMs) && dpr.evaluationIntervalMs <= 0) {
    errors.push(`adaptiveDPR.evaluationIntervalMs must be > 0 (got ${dpr.evaluationIntervalMs})`);
  }

  const ratiosFinite =
    finite('scaleDownFpsRatio', dpr.scaleDownFpsRatio) &&
    finite('scaleUpFpsRatio', dpr.scaleUpFpsRatio);
  if (
    ratiosFinite &&
    !(
      dpr.scaleDownFpsRatio > 0 &&
      dpr.scaleDownFpsRatio < dpr.scaleUpFpsRatio &&
      dpr.scaleUpFpsRatio < 1
    )
  ) {
    errors.push(
      'adaptiveDPR FPS ratios must satisfy 0 < scaleDownFpsRatio < scaleUpFpsRatio < 1 ' +
        `(got down=${dpr.scaleDownFpsRatio}, up=${dpr.scaleUpFpsRatio})`
    );
  }
  if (finite('refreshRateFallback', dpr.refreshRateFallback) && dpr.refreshRateFallback <= 0) {
    errors.push(`adaptiveDPR.refreshRateFallback must be > 0 (got ${dpr.refreshRateFallback})`);
  }
  if (finite('refreshRateCeiling', dpr.refreshRateCeiling)) {
    if (dpr.refreshRateCeiling < 0) {
      errors.push(`adaptiveDPR.refreshRateCeiling must be >= 0 (got ${dpr.refreshRateCeiling})`);
    } else if (dpr.refreshRateCeiling > 0 && dpr.refreshRateCeiling < 23.5) {
      warnings.push(
        `adaptiveDPR.refreshRateCeiling (${dpr.refreshRateCeiling}) is below the slowest ` +
          'real display rate and can prevent adaptive scale-down'
      );
    }
  }
  if (
    finite('midbandGraceSamples', dpr.midbandGraceSamples) &&
    (dpr.midbandGraceSamples < 0 || !Number.isInteger(dpr.midbandGraceSamples))
  ) {
    errors.push(
      `adaptiveDPR.midbandGraceSamples must be a non-negative integer (got ${dpr.midbandGraceSamples})`
    );
  }

  if (finite('probeImprovement', dpr.probeImprovement) && dpr.probeImprovement < 1) {
    errors.push(`adaptiveDPR.probeImprovement must be >= 1 (got ${dpr.probeImprovement})`);
  }
  if (finite('probeMinSamples', dpr.probeMinSamples) && dpr.probeMinSamples < 2) {
    errors.push(`adaptiveDPR.probeMinSamples must be >= 2 (got ${dpr.probeMinSamples})`);
  }
  if (finite('probeWindowMs', dpr.probeWindowMs)) {
    if (dpr.probeWindowMs <= 0) {
      errors.push(`adaptiveDPR.probeWindowMs must be > 0 (got ${dpr.probeWindowMs})`);
    } else if (dpr.probeWindowMs <= 1000) {
      warnings.push(
        `adaptiveDPR.probeWindowMs (${dpr.probeWindowMs}) should exceed the 1s FPS sample window ` +
          'so reallocation jank washes out before the probe settles'
      );
    }
  }

  if (finite('floorTtlMs', dpr.floorTtlMs) && dpr.floorTtlMs <= dpr.probeWindowMs) {
    errors.push(
      `adaptiveDPR.floorTtlMs (${dpr.floorTtlMs}) must exceed probeWindowMs (${dpr.probeWindowMs})`
    );
  }
  if (finite('backoffMultiplier', dpr.backoffMultiplier) && dpr.backoffMultiplier < 1) {
    errors.push(`adaptiveDPR.backoffMultiplier must be >= 1 (got ${dpr.backoffMultiplier})`);
  }
  if (finite('backoffMaxTtlMs', dpr.backoffMaxTtlMs) && dpr.backoffMaxTtlMs < dpr.floorTtlMs) {
    errors.push(
      `adaptiveDPR.backoffMaxTtlMs (${dpr.backoffMaxTtlMs}) must be >= floorTtlMs (${dpr.floorTtlMs})`
    );
  }

  if (finite('ceilingTtlMs', dpr.ceilingTtlMs) && dpr.ceilingTtlMs <= 0) {
    errors.push(`adaptiveDPR.ceilingTtlMs must be > 0 (got ${dpr.ceilingTtlMs})`);
  }
  if (finite('backoffMaxTtlMs', dpr.backoffMaxTtlMs) && dpr.backoffMaxTtlMs < dpr.ceilingTtlMs) {
    errors.push(
      `adaptiveDPR.backoffMaxTtlMs (${dpr.backoffMaxTtlMs}) must be >= ceilingTtlMs (${dpr.ceilingTtlMs})`
    );
  }
  if (
    finite('punishedAscentWindowMs', dpr.punishedAscentWindowMs) &&
    dpr.punishedAscentWindowMs <= 0
  ) {
    errors.push(
      `adaptiveDPR.punishedAscentWindowMs must be > 0 (got ${dpr.punishedAscentWindowMs})`
    );
  }
  if (
    finite('punishedAscentThreshold', dpr.punishedAscentThreshold) &&
    (dpr.punishedAscentThreshold < 1 || !Number.isInteger(dpr.punishedAscentThreshold))
  ) {
    errors.push(
      `adaptiveDPR.punishedAscentThreshold must be a positive integer (got ${dpr.punishedAscentThreshold})`
    );
  }

  if (
    finite('contentChangeRecheckMs', dpr.contentChangeRecheckMs) &&
    dpr.contentChangeRecheckMs <= 0
  ) {
    errors.push(
      `adaptiveDPR.contentChangeRecheckMs must be > 0 (got ${dpr.contentChangeRecheckMs})`
    );
  }
  if (finite('gapResetMs', dpr.gapResetMs)) {
    if (dpr.gapResetMs <= 50) {
      errors.push(`adaptiveDPR.gapResetMs must be > 50 (got ${dpr.gapResetMs})`);
    } else if (dpr.gapResetMs >= 1000) {
      warnings.push(
        `adaptiveDPR.gapResetMs (${dpr.gapResetMs}) is >= the 1s FPS window; ` +
          'stall/idle-resume gaps will poison FPS samples before the reset fires'
      );
    }
  }
}
