/**
 * Streaming policy for the progressive loaders (GSplats, Points, Lines, and
 * the Mesh reveal ladder).
 *
 * A progressive loader streams an additive LOD ladder coarse→fine. HOW MANY
 * levels a single `updateView` pass loads — and when it stops — depends on why
 * the pass is running. Encoding that decision as three pure functions keeps the
 * geometry loaders' streaming loops identical by construction (they share
 * this module rather than each re-deriving the rule), and makes the policy
 * unit-testable in isolation.
 *
 * The three disciplines:
 *
 *  - `playback` — foreground, under a per-frame budget (a dimension is
 *    playing/scrubbing). Responsiveness dominates, so a cold level is never
 *    worth blocking on. Whether the pass starts EMPTY or restores a cached
 *    prefix, it streams levels that come back CACHE-RESIDENT until the budget
 *    is spent, stopping at the first cold or slow one exactly as `refine` does.
 *
 *    It used to load LOD 0 and nothing else, on the reasoning that even a
 *    resident level costs ~10-20 ms of dequant+project. That is true, and the
 *    budget deadline is what bounds it — but a level-1 gate is the wrong
 *    instrument, because it assumes LOD 0 is a usable picture. On a node the
 *    viewer SLICES it is not: an additive ladder's rungs are sized against the
 *    WHOLE node, so on a 500-timepoint gsplat leaf LOD 0 is ~42 splats of a
 *    166,443-splat frame and playback rendered an empty screen (#2374, #2376).
 *    Streaming the resident levels costs approximately no network — the coarse
 *    rungs of a sliced ladder are small enough to sit in ONE chunk each, so
 *    they are fetched once and serve every slice — and the worst case is
 *    unchanged at one cold load per pass, since the first cold level still
 *    stops the loop.
 *
 *    Quality is still deepened for later loops by the background `prefetch`
 *    pass; foreground playback only spends budget remaining in the current
 *    tick.
 *  - `prefetch` — background shadow pass warming the next timepoint. No
 *    first-paint constraint, so it deepens toward the FULL decoded ladder (the
 *    only thing that yields the single-concat fast revisit); it never stops at
 *    a cache miss and is bounded only by the pass budget + abort.
 *  - `refine` — foreground with no budget (a static view, or the refine-on-
 *    pause pass). Stream cache-resident levels and stop at the first cold/slow
 *    one so the frame renders; a later pass picks up the rest.
 *
 * @module data/loaders/progressive/streaming-policy
 */

import { CACHE_HIT_THRESHOLD_MS } from './constants';

/** Which streaming discipline a progressive-loader pass follows. */
export type StreamingPassKind = 'playback' | 'prefetch' | 'refine';

/**
 * Classify a streaming pass from the two facts a loader knows at pass start:
 * whether a per-frame budget is active, and whether this is a background
 * prefetch (shadow) pass. Prefetch takes precedence — a prefetch pass always
 * carries a budget, but its discipline is "deepen", not "stay responsive".
 */
export function classifyStreamingPass(
  budgetActive: boolean,
  isPrefetch: boolean
): StreamingPassKind {
  if (isPrefetch) return 'prefetch';
  return budgetActive ? 'playback' : 'refine';
}

/**
 * Whether the streaming loop should load `level` this pass (evaluated at loop
 * top; a `false` breaks the loop).
 *
 * All pass kinds may load every level. Playback responsiveness is enforced by
 * the loop's budget deadline and {@link shouldStopAfterLevel}; keeping that
 * decision out of this gate makes an empty ladder and an identical restored
 * prefix spend the same remaining foreground budget (#2379).
 */
export function shouldLoadLevel(
  _kind: StreamingPassKind,
  // Retained (unused) so the four loop-top call sites keep reading as "may I
  // load THIS level?", and so a future per-level rule has a seam.
  _level: number,
  _startLevel: number
): boolean {
  return true;
}

/**
 * Whether to stop the streaming loop AFTER loading `level`. `refine` and
 * `playback` both stop here — at the first cold (cache-miss) or slow level past
 * the >=1-level first-paint floor. That floor applies only when the ladder
 * started empty; a restored prefix is already showable. For `playback` this is
 * the ONLY brake besides the budget deadline, and it is the one that matters:
 * a cold level costs hundreds of ms, a resident one ~10-20 ms, so residency
 * (not level index) is what separates "affordable inside a tick" from "stalls
 * the tick".
 *
 * `prefetch` deepens regardless of residency — warming cold levels is its whole
 * job — and is bounded by the pass budget + abort instead.
 */
export function shouldStopAfterLevel(
  kind: StreamingPassKind,
  level: number,
  startLevel: number,
  allResident: boolean,
  elapsedMs: number
): boolean {
  if (kind === 'prefetch') return false;
  const pastFirstPaintFloor = level > startLevel || (kind === 'playback' && startLevel > 0);
  return pastFirstPaintFloor && (!allResident || elapsedMs > CACHE_HIT_THRESHOLD_MS);
}
