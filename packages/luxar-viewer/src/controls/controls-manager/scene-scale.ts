/**
 * Derive scale-aware control parameters from the scene bounding-box
 * diagonal. Extracted from `controls-manager.ts` so the orchestrator
 * stays focused on lifecycle and dispatch.
 *
 * `deriveScaleLimits` is the pure math: diagonal × per-parameter
 * multipliers from `config.controls.scaleMultipliers`. The orchestrator
 * still owns the field writes, the active-control mutation, the gating
 * against stored auto-frame limits, and the logging.
 */

import { config } from '../../config';

/** Scale-derived control parameters: orbit distance bounds and fly speed. */
export interface ScaleLimits {
  minDist: number;
  maxDist: number;
  flySpeed: number;
}

/**
 * Compute scale-aware min/max orbit distance and fly speed by multiplying the
 * scene bounding-box `diagonal` by the per-parameter factors in
 * `config.controls.scaleMultipliers`. Pure math — the caller owns the field
 * writes, active-control mutation, and gating against stored auto-frame limits.
 */
export function deriveScaleLimits(diagonal: number): ScaleLimits {
  const m = config.controls.scaleMultipliers;
  return {
    minDist: diagonal * m.minDistanceFactor,
    maxDist: diagonal * m.maxDistanceFactor,
    flySpeed: diagonal * m.flySpeedFactor,
  };
}
