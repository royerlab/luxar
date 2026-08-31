/**
 * Committed-quality probe for the dimension-animation pacing feedback.
 *
 * The animation manager can measure its own CADENCE — ticks per second against
 * the requested rate — but cadence alone cannot tell the two situations apart
 * that matter to a viewer:
 *
 *   - the loaders cannot keep up and the frames on screen are thin;
 *   - the loaders are keeping up and the PLAYHEAD has slowed to wait for them,
 *     which is the pacing gate working exactly as designed.
 *
 * Before #2377 the second case barely existed: a playback pass committed the
 * first ladder rung and stopped, so it nearly hit the requested rate precisely
 * because it did almost no work per tick. Cadence was met with empty frames and
 * the feedback stayed silent (#2374). Now a pass streams every cache-resident
 * rung, so cadence is routinely missed while every frame shown is complete —
 * and a warning keyed on cadence alone fires on healthy playback, which trains
 * a reader to ignore it.
 *
 * This walks the committed geometry stamps and returns the WORST-SERVED laddered
 * node's `committedEnergyFraction`, so the feedback can say which case it is.
 *
 * @module scene/animation/committed-quality
 */

import { countFromUserData, type FreshnessChild } from '../lod-freshness';

type FreshnessUserData = NonNullable<FreshnessChild['object']['userData']>;

/** The subset of `THREE.Object3D` this walk reads. */
export interface QualityNode {
  visible?: boolean;
  children?: readonly QualityNode[];
  userData?: Omit<FreshnessUserData, 'committedEnergyFraction'> & {
    committedEnergyFraction?: unknown;
  };
}

/**
 * Lowest `committedEnergyFraction` among VISIBLE, NON-EMPTY stamped nodes, or
 * `null`.
 *
 * `null` means "cannot tell" and must be treated as such rather than as good or
 * bad news: an unstamped (legacy) dataset carries no energy figures at all, and
 * a scene with no laddered node has nothing to report. Callers fall back to
 * their cadence-only behaviour there.
 *
 * The reduction is a MINIMUM, not a mean: one node painting fully does not
 * excuse another painting thin, and it is the thinnest thing on screen that a
 * viewer notices. Same reasoning as the sliced-ladder gate reducing with a low
 * percentile rather than a maximum (#2389).
 */
export function worstCommittedEnergy(root: QualityNode | null | undefined): number | null {
  let worst: number | null = null;
  const visit = (node: QualityNode): void => {
    if (node.visible === false) return;
    const userData = node.userData;
    const raw = userData?.committedEnergyFraction;
    const count = countFromUserData(userData as FreshnessUserData | undefined);
    if (count !== 0 && typeof raw === 'number' && Number.isFinite(raw)) {
      worst = worst === null ? raw : Math.min(worst, raw);
    }
    for (const child of node.children ?? []) visit(child);
  };
  if (root) visit(root);
  return worst;
}
