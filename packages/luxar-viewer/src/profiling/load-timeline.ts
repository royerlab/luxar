/**
 * Load timeline — wall-clock milestones of one scene load.
 *
 * The viewer had no time-to-first-paint or time-to-full-load stamp anywhere:
 * the only timeline was the console log, and `__luxarDebug.getState()` is
 * installed only after the dataset load AND the blend warm-up settle (up to
 * 5 s after first paint), so probes that wait for it under-observe the load.
 * This module records the milestones as `performance.mark()` entries (visible
 * in the DevTools Performance panel and via `performance.getEntriesByType`)
 * and mirrors them into a small in-memory snapshot that costs nothing to read.
 *
 * Milestones are recorded ONCE per load: `markLoad('loadStart')` resets the
 * timeline, and a repeated milestone name within the same load is ignored so
 * a retry path cannot overwrite the first observation. Per-geometry first
 * commits use `markFirstCommit(kind)`; refinement passes are counted rather
 * than marked (a 4-D scene with 16 nodes produces hundreds of passes).
 *
 * Everything here is best-effort: a runtime without `performance.mark`
 * (jsdom, an old WebView) still gets the in-memory snapshot, and nothing
 * throws into the load path.
 *
 * @module profiling/load-timeline
 */

/** Geometry kinds that commit through `commit-*-geometry.ts`. */
export type LoadGeometryKind = 'points' | 'lines' | 'gsplats' | 'mesh';

/** Named milestones recorded once per load. */
export type LoadMilestone =
  | 'loadStart'
  | 'metadataReady'
  | 'poolReady'
  | 'wasmReady'
  | 'sceneLoaded'
  | 'initUpdateDone'
  | 'refinementComplete';

/** One recorded mark: name plus `performance.now()` at the time. */
export interface LoadMark {
  name: string;
  /** `performance.now()` at the mark (ms since time origin). */
  t: number;
  detail?: Record<string, unknown>;
}

/** Progressive-refinement counters for the current load. */
export interface RefinementCounters {
  /** Refinement passes run (one pass = every loader offered one rung). */
  passes: number;
  /** Rungs committed across all loaders and passes. */
  rungs: number;
  /** Rungs whose payload came from the slice cache rather than a fetch. */
  rungsFromCache: number;
  /** True once the final geometry phase ran its ladders to completion. */
  complete: boolean;
}

/** Snapshot returned by {@link getLoadTimeline}. All times are `performance.now()` ms. */
export interface LoadTimelineSnapshot {
  /** Milestone marks in recording order (`firstCommit:<kind>` included). */
  marks: LoadMark[];
  /** Milestone name → time, for direct lookup. */
  milestones: Partial<Record<LoadMilestone, number>>;
  /** Geometry kind → time of its first committed geometry this load. */
  firstCommit: Partial<Record<LoadGeometryKind, number>>;
  refinement: RefinementCounters;
  /**
   * Derived durations relative to `loadStart`, or `null` when either end is
   * missing. `ttfpMs` is the earliest first commit of any geometry kind.
   */
  measures: {
    ttfpMs: number | null;
    metadataReadyMs: number | null;
    poolReadyMs: number | null;
    sceneLoadedMs: number | null;
    initUpdateDoneMs: number | null;
    refinementCompleteMs: number | null;
  };
}

/** Upper bound on retained marks; milestones are few, this is a safety net. */
const MAX_MARKS = 512;
const MARK_PREFIX = 'luxar:';

let marks: LoadMark[] = [];
let milestones: Partial<Record<LoadMilestone, number>> = {};
let firstCommit: Partial<Record<LoadGeometryKind, number>> = {};
let refinement: RefinementCounters = freshCounters();

function freshCounters(): RefinementCounters {
  return { passes: 0, rungs: 0, rungsFromCache: 0, complete: false };
}

function now(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

/** Mirror a mark into the User Timing API; never throws. */
function performanceMark(name: string, detail?: Record<string, unknown>): void {
  if (typeof performance === 'undefined' || typeof performance.mark !== 'function') return;
  try {
    // `detail` needs the User Timing L3 signature; older engines throw on the
    // options object, so fall back to the bare form.
    if (detail) {
      performance.mark(MARK_PREFIX + name, { detail });
    } else {
      performance.mark(MARK_PREFIX + name);
    }
  } catch {
    try {
      performance.mark(MARK_PREFIX + name);
    } catch {
      /* User Timing unavailable — the in-memory snapshot still records it. */
    }
  }
}

function push(name: string, t: number, detail?: Record<string, unknown>): void {
  if (marks.length >= MAX_MARKS) marks.shift();
  marks.push(detail ? { name, t, detail } : { name, t });
  performanceMark(name, detail);
}

/**
 * Record a load milestone. `loadStart` resets the timeline (a dataset switch
 * starts a new load); any other milestone is recorded only the first time it
 * is seen within the current load.
 */
export function markLoad(name: LoadMilestone, detail?: Record<string, unknown>): void {
  if (name === 'loadStart') {
    resetLoadTimeline();
  } else if (milestones[name] !== undefined) {
    return;
  }
  const t = now();
  milestones[name] = t;
  push(name, t, detail);
}

/**
 * Record the first committed geometry of a kind for this load. Idempotent:
 * every commit path may call it unconditionally.
 */
export function markFirstCommit(kind: LoadGeometryKind): void {
  if (firstCommit[kind] !== undefined) return;
  const t = now();
  firstCommit[kind] = t;
  push(`firstCommit:${kind}`, t);
}

/**
 * Count one refinement pass. `rungs` is how many loaders advanced a rung in
 * the pass; `rungsFromCache` how many of those were slice-cache restores.
 */
export function noteRefinementPass(rungs: number, rungsFromCache = 0): void {
  refinement.passes += 1;
  refinement.rungs += Math.max(0, rungs);
  refinement.rungsFromCache += Math.max(0, rungsFromCache);
}

/** The final geometry phase ran every ladder to completion (not cancelled). */
export function noteRefinementComplete(): void {
  refinement.complete = true;
  markLoad('refinementComplete', { passes: refinement.passes, rungs: refinement.rungs });
}

/** Forget everything recorded so far (called by `markLoad('loadStart')`). */
export function resetLoadTimeline(): void {
  marks = [];
  milestones = {};
  firstCommit = {};
  refinement = freshCounters();
}

function since(start: number | undefined, end: number | undefined): number | null {
  return start === undefined || end === undefined ? null : end - start;
}

/** Read-only snapshot; safe to call at any time, including before any load. */
export function getLoadTimeline(): LoadTimelineSnapshot {
  const start = milestones.loadStart;
  const commits = Object.values(firstCommit).filter((t): t is number => typeof t === 'number');
  const earliestCommit = commits.length > 0 ? Math.min(...commits) : undefined;
  return {
    marks: marks.map((m) => ({ ...m })),
    milestones: { ...milestones },
    firstCommit: { ...firstCommit },
    refinement: { ...refinement },
    measures: {
      ttfpMs: since(start, earliestCommit),
      metadataReadyMs: since(start, milestones.metadataReady),
      poolReadyMs: since(start, milestones.poolReady),
      sceneLoadedMs: since(start, milestones.sceneLoaded),
      initUpdateDoneMs: since(start, milestones.initUpdateDone),
      refinementCompleteMs: since(start, milestones.refinementComplete),
    },
  };
}
