/**
 * Debug-only per-stage timing for lazy and additive LOD level loads.
 *
 * The per-frame LOD selector and its lazy `ensureLoaded` loads run
 * OUTSIDE any `updateView` cycle, so the `UpdateProfiler` /
 * data-loading-monitor never sees them — during pure camera navigation
 * the monitor shows nothing. This standalone accumulator fills that gap:
 * it records the wall-clock cost of each stage of a lazy level load
 * (fetch/decode, projection+pack, GPU commit, release), plus each additive
 * ladder level load keyed by geometry type, level index, and residency.
 *
 * Disabled by default (zero cost). Enabled only under `?debug` by
 * `installDebugInterface`, which also exposes
 * `window.__luxarDebug.getLodLoadStats()` / `resetLodLoadStats()`.
 *
 * @module data/scene-loader/lod-load-stats
 */

interface StageStat {
  count: number;
  totalMs: number;
  maxMs: number;
}

/** Snapshot shape returned to the debug console. */
export interface LodLoadStatSnapshot {
  count: number;
  totalMs: number;
  avgMs: number;
  maxMs: number;
}

let enabled = false;
const stats = new Map<string, StageStat>();

/** Enable/disable recording. Off by default; turned on under `?debug`. */
export function setLodLoadStatsEnabled(on: boolean): void {
  enabled = on;
}

/** Whether recording is active (cheap guard for hot paths). */
export function lodLoadStatsEnabled(): boolean {
  return enabled;
}

/** Record one stage timing (no-op when disabled). */
export function recordLodLoadStage(stage: string, ms: number): void {
  if (!enabled) return;
  let s = stats.get(stage);
  if (!s) {
    s = { count: 0, totalMs: 0, maxMs: 0 };
    stats.set(stage, s);
  }
  s.count++;
  s.totalMs += ms;
  if (ms > s.maxMs) s.maxMs = ms;
}

/** Time an async stage; passthrough (no timing) when disabled. */
export async function timeLodStage<T>(stage: string, fn: () => Promise<T>): Promise<T> {
  if (!enabled) return fn();
  const t0 = performance.now();
  try {
    return await fn();
  } finally {
    recordLodLoadStage(stage, performance.now() - t0);
  }
}

/** Time an async stage whose bounded key depends on the resolved result. */
export async function timeLodStageWithResult<T>(
  stageForResult: (result: T) => string,
  fn: () => Promise<T>
): Promise<T> {
  if (!enabled) return fn();
  const t0 = performance.now();
  const result = await fn();
  recordLodLoadStage(stageForResult(result), performance.now() - t0);
  return result;
}

/** Time a synchronous stage; passthrough (no timing) when disabled. */
export function timeLodStageSync<T>(stage: string, fn: () => T): T {
  if (!enabled) return fn();
  const t0 = performance.now();
  try {
    return fn();
  } finally {
    recordLodLoadStage(stage, performance.now() - t0);
  }
}

/** Snapshot of all recorded stages (avg/max in ms), keyed by stage name. */
export function snapshotLodLoadStats(): Record<string, LodLoadStatSnapshot> {
  const out: Record<string, LodLoadStatSnapshot> = {};
  for (const [name, s] of stats) {
    out[name] = {
      count: s.count,
      totalMs: +s.totalMs.toFixed(2),
      avgMs: +(s.totalMs / Math.max(1, s.count)).toFixed(3),
      maxMs: +s.maxMs.toFixed(2),
    };
  }
  return out;
}

/** Clear all recorded stats (call before driving a measurement scenario). */
export function resetLodLoadStats(): void {
  stats.clear();
}
