/**
 * Pick the kind-appropriate timeout from the viewer's performance
 * config. Visibility uses `workerVisibilityTimeoutMs`; projection AND
 * decode share `workerProjectionTimeoutMs` (both are long-running
 * CPU-bound calls — a dedicated decode knob can be added later if
 * telemetry shows a need).
 */

import type { TimeoutKind } from '../errors';

export function pickTimeoutMs(
  kind: TimeoutKind,
  perf: { workerVisibilityTimeoutMs: number; workerProjectionTimeoutMs: number }
): number {
  return kind === 'visibility' ? perf.workerVisibilityTimeoutMs : perf.workerProjectionTimeoutMs;
}
