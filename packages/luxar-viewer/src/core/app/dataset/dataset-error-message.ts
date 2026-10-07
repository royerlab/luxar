/**
 * The user-facing text for a dataset that failed to load.
 *
 * Two failures carry an actionable message of their own: an archive fault
 * (a wrong `?src=` path — its message names the remedy the store authored)
 * and a refused format version (the message names the version, the supported
 * set and the remedy). Both are found through wrapper causes. Anything else is
 * reported with its own message behind a fixed prefix.
 *
 * @module core/app/dataset/dataset-error-message
 */

import { archiveFaultFrom } from '../../../cache/chunk-source';
import { unsupportedFormatVersionFrom } from '../../../data/format-version';
import { getErrorMessage } from '../../../utils/format-error';

/** Message shown in the persistent error dialog for a failed dataset load. */
export function datasetLoadErrorMessage(error: unknown): string {
  const surfaced = archiveFaultFrom(error) ?? unsupportedFormatVersionFrom(error);
  return surfaced ? surfaced.message : `Failed to load dataset: ${getErrorMessage(error)}`;
}
