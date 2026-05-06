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
 * Returns `'Off'` when `mode` is `null`/`undefined` (used by the status
 * snapshot when no tone-mapping effect is active) and `'Unknown'` when
 * the value falls outside the table — defensive in case the upstream
 * library adds new modes faster than this table is updated.
 */
export function toneMappingModeName(mode: ToneMappingMode | null | undefined): string {
  if (mode === null || mode === undefined) return 'Off';
  return TONE_MAPPING_MODE_NAMES[mode] ?? 'Unknown';
}
