/**
 * The viewer's error dialog, with its two recovery hints: the dataset-browser
 * and help shortcuts, resolved live through the app's input bindings. One
 * place for the shortcut set that every caller (bootstrap, the dataset
 * browser, the debug surface) used to restate.
 *
 * @module core/app/error-dialog
 */

import { KeyAction } from '../../input';
import { showError } from '../../ui/error-overlay';

/** The shortcuts the error dialog advertises. */
export const ERROR_DIALOG_SHORTCUTS = {
  datasetBrowser: KeyAction.toggleDatasetBrowser,
  help: KeyAction.toggleHelp,
} as const;

/** Show `message` in the error dialog, advertising {@link ERROR_DIALOG_SHORTCUTS}. */
export function showViewerError(
  message: string,
  shortcutForAction: (actionId: string) => string | undefined,
  options?: { autoDismiss?: boolean }
): void {
  if (options) showError(message, shortcutForAction, ERROR_DIALOG_SHORTCUTS, options);
  else showError(message, shortcutForAction, ERROR_DIALOG_SHORTCUTS);
}
