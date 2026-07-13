/**
 * Pick the kind-appropriate timeout from the viewer's performance
 * config. Projection AND decode share `workerProjectionTimeoutMs`
 * (both are long-running CPU-bound calls — a dedicated decode knob can
 * be added later if telemetry shows a need).
 */

import type { TimeoutKind } from '../errors';

export function pickTimeoutMs(
  _kind: TimeoutKind,
  perf: { workerProjectionTimeoutMs: number }
): number {
  return perf.workerProjectionTimeoutMs;
}
