/**
 * Initialize the loader's ViewState from a scene's `scene_dimensions`
 * metadata. Pure function: takes the raw attrs blob, validates it, and
 * returns a fresh `ViewState` (or `null` if validation fails).
 *
 * The orchestrator stores the returned ViewState back into its private
 * field; `null` means dimensions were missing or invalid and the
 * previous ViewState should be left untouched.
 */

import { log, Modules } from '../../../utils/log';
import type { ViewState } from '../../data-loader-types';
import { ViewStateManager } from '../../view-state-manager';
import { isSceneDimensions } from '../view-state/extend-tolerance';

/**
 * Validate scene dimensions and return the initial ViewState built from
 * them. Returns `null` when the attrs blob is malformed or fails
 * dimension validation — in that case the caller should skip the
 * update (logging happens here).
 */
export function initializeSceneDimensions(sceneDims: unknown): ViewState | null {
  // Validate sceneDims structure
  if (!isSceneDimensions(sceneDims)) {
    log.warning(Modules.SCENE_LOADER, 'Invalid scene_dimensions format, skipping');
    return null;
  }

  // Validate dimensions using ViewStateManager
  const validation = ViewStateManager.validateDimensions(sceneDims.dimensions);

  // Log validation results
  const displayedCount = sceneDims.dimensions.filter((d) => d.display === true).length;
  ViewStateManager.logValidationResults(validation, sceneDims.dimensions.length, displayedCount);

  // Stop if validation failed with errors
  if (!validation.isValid) {
    log.error(Modules.SCENE_LOADER, 'Scene dimensions validation failed, cannot initialize');
    return null;
  }

  // Initialize ViewState using ViewStateManager
  return ViewStateManager.initializeFromDimensions(sceneDims);
}
