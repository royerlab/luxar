/**
 * Streaming policy for the progressive loaders (GSplats, Points, Lines).
 *
 * A progressive loader streams an additive LOD ladder coarse→fine. HOW MANY
 * levels a single `updateView` pass loads — and when it stops — depends on why
 * the pass is running. Encoding that decision as three pure functions keeps the
 * three geometry loaders' streaming loops identical by construction (they share
 * this module rather than each re-deriving the rule), and makes the policy
 * unit-testable in isolation.
 *
 * The three disciplines:
 *
 *  - `playback` — foreground, under a per-frame budget (a dimension is
 *    playing/scrubbing). Responsiveness dominates: commit the restored cached
 *    prefix as-is and, when nothing is cached, load only LOD 0 as a first-paint
 *    floor. NEVER synchronously decode further levels — even an L0-resident one
 *    still costs ~10-20ms of dequant+project per level, and a cold one costs
 *    hundreds of ms; either stalls the tick. Quality is deepened for later
 *    loops by the background `prefetch` pass, so each loop restores a deeper
 *    cached prefix and looks better while staying fast.
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
 * top; a `false` breaks the loop). Only `playback` restricts loading — to the
 * first-paint floor (LOD 0 when the ladder starts empty). `prefetch`/`refine`
 * load every level and rely on {@link shouldStopAfterLevel} / the budget
 * deadline to bound the pass.
 */
export function shouldLoadLevel(
  kind: StreamingPassKind,
  level: number,
  startLevel: number
): boolean {
  if (kind === 'playback') return level === 0 && startLevel === 0;
  return true;
}

/**
 * Whether to stop the streaming loop AFTER loading `level`. Only `refine`
 * stops here — at the first cold (cache-miss) or slow level past the ≥1-level
 * first-paint floor — so the frame renders and a later pass continues.
 * `prefetch` deepens regardless of residency (bounded by budget + abort);
 * `playback` never reaches here for `level > 0` (it is gated by
 * {@link shouldLoadLevel}).
 */
export function shouldStopAfterLevel(
  kind: StreamingPassKind,
  level: number,
  startLevel: number,
  allResident: boolean,
  elapsedMs: number
): boolean {
  if (kind !== 'refine') return false;
  return level > startLevel && (!allResident || elapsedMs > CACHE_HIT_THRESHOLD_MS);
}
