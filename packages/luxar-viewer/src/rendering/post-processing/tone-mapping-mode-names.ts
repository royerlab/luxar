/**
 * Pure mapping from `ToneMappingMode` enum values to human-readable names
 * used in UI status displays (e.g. the rendering-controls panel and the
 * `getEffectsStatus()` snapshot).
 *
 * Centralized here so the table is the single source of truth — the
 * status snapshot, the rendering-controls dropdown, and any future
 * consumers can all import from one place.
 *
 * @module rendering/post-processing/tone-mapping-mode-names
 */

import { ToneMappingMode } from 'postprocessing';

/** Human-readable name for each `ToneMappingMode` we surface in the UI. */
const TONE_MAPPING_MODE_NAMES: Record<ToneMappingMode, string> = {
  [ToneMappingMode.LINEAR]: 'Linear',
  [ToneMappingMode.REINHARD]: 'Reinhard',
  [ToneMappingMode.OPTIMIZED_CINEON]: 'Cineon',
  [ToneMappingMode.ACES_FILMIC]: 'ACES Filmic',
  [ToneMappingMode.AGX]: 'AgX',
  [ToneMappingMode.NEUTRAL]: 'Neutral',
};

/**
 * Look up the display name of a {@link ToneMappingMode} value.
 *
 * Returns `'Unknown'` for `null`, `undefined`, or any value outside the
 * table — defensive in case the upstream library adds new modes faster
 * than this table is updated, or an effect is in a transient state where
 * its `mode` property has not been assigned.
 *
 * Callers that need to distinguish "no effect at all" from "effect with
 * unknown mode" should branch *before* calling this helper, e.g.:
 *
 *   effect ? toneMappingModeName(effect.mode) : 'Off'
 */
export function toneMappingModeName(mode: ToneMappingMode | null | undefined): string {
  if (mode === null || mode === undefined) return 'Unknown';
  return TONE_MAPPING_MODE_NAMES[mode] ?? 'Unknown';
}
