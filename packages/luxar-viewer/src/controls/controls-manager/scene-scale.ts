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

export interface ScaleLimits {
  minDist: number;
  maxDist: number;
  flySpeed: number;
}

export function deriveScaleLimits(diagonal: number): ScaleLimits {
  const m = config.controls.scaleMultipliers;
  return {
    minDist: diagonal * m.minDistanceFactor,
    maxDist: diagonal * m.maxDistanceFactor,
    flySpeed: diagonal * m.flySpeedFactor,
  };
}
