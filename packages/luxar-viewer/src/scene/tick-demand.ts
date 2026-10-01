/**
 * The liveness contract of the per-frame LOD components.
 *
 * The viewer's render loop is on-demand: once nothing asks for a frame it
 * idles, and a component whose state only advances inside the per-frame
 * evaluation (a dissolve running on wall time, a failed level cooling down
 * before its retry, a resync waiting for a loader pass) would then stall until
 * the user moved. Each such component states, every frame, until when it needs
 * the loop to keep ticking; the registry folds those deadlines and asks for one
 * tick whenever any lies in the future. A component that forgets to say is the
 * bug class this contract exists to make visible (#2944: A1, A6, the sticky
 * partition-activation request).
 *
 * @module scene/tick-demand
 */

/** Nothing pending: the component does not need the loop. */
export const NO_TICK = Number.NEGATIVE_INFINITY;

/** Pending with no known end time: tick until the component says otherwise. */
export const UNTIL_RESOLVED = Number.POSITIVE_INFINITY;

/** A per-frame component's answer to "until when must the loop keep ticking?". */
export interface TickDemand {
  /**
   * The registry clock time (ms) until which the loop must keep ticking:
   * {@link NO_TICK} when nothing is pending, {@link UNTIL_RESOLVED} while
   * pending with no deadline. Read after the frame's evaluation.
   */
  tickUntilMs(): number;
}
