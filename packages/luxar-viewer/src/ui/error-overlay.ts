/**
 * Error dialog — user-friendly error overlay with guidance and auto-dismiss.
 *
 * Replaces any existing error dialog so a new failure doesn't stack on
 * top of an old one. Dismissible by click, Escape, or auto-timeout
 * (configured via `config.ui.timings.errorAutoDismissMs`). Traps focus
 * inside the dialog while it is open (Tab/Shift+Tab can't escape).
 */

import { config } from '../config';
import { trapFocus } from './help-overlay/focus-trap';
import { getViewerContainer } from '../utils/viewer-container';

const UI_CONFIG = config.ui;

// Audit G17 (viewer-ui-config-themes-core-utils) — the auto-dismiss
// `setTimeout` was never cancelled when the overlay was removed via
// click/Escape/replacement/clearError(), leaking a pending timer until
// it fired and ran a no-op branch. Also: the focus-trap returned a
// release function that was only invoked by the click handler; if the
// overlay was torn down via clearError() the trap stayed attached
// (and its own internal focus setTimeout stayed pending). Track both
// at module scope so all teardown paths can clear them.
let autoDismissTimerId: ReturnType<typeof setTimeout> | null = null;
let activeReleaseTrap: (() => void) | null = null;

function clearAutoDismissTimer(): void {
  if (autoDismissTimerId !== null) {
    clearTimeout(autoDismissTimerId);
    autoDismissTimerId = null;
  }
}

function releaseActiveTrap(): void {
  if (activeReleaseTrap !== null) {
    activeReleaseTrap();
    activeReleaseTrap = null;
  }
}

/**
 * Show a user-facing error dialog over the viewport with the given message.
 *
 * Any dialog from a previous call is torn down first (its auto-dismiss timer
 * cancelled and focus trap released) so failures never stack. The new dialog is
 * dismissible by click, Escape, or an auto-timeout of
 * `config.ui.timings.errorAutoDismissMs`, and traps keyboard focus while open.
 *
 * @param message Human-readable error text to display to the user.
 */
export function showError(message: string) {
  // Remove any existing error messages first — and cancel the timer
  // + release the focus trap that the previous showError() scheduled
  // (otherwise it would fire against an already-removed element and
  // stay pending in test environments using fake timers).
  clearAutoDismissTimer();
  releaseActiveTrap();
  const existingError = document.getElementById('luxar-error-message');
  if (existingError) {
    existingError.remove();
  }

  // Create error dialog with CSS classes
  const errorDiv = document.createElement('div');
  errorDiv.id = 'luxar-error-message';
  errorDiv.className = 'luxar-error-dialog luxar-glass-surface error-message'; // luxar-error-dialog for styling, error-message for E2E tests

  // ARIA attributes for accessibility
  errorDiv.setAttribute('role', 'alertdialog');
  errorDiv.setAttribute('aria-modal', 'true');
  errorDiv.setAttribute('aria-labelledby', 'luxar-error-title');
  errorDiv.setAttribute('aria-describedby', 'luxar-error-message-text');

  // Error icon + title
  const header = document.createElement('div');
  header.className = 'luxar-error-dialog__header';

  const icon = document.createElement('div');
  icon.className = 'luxar-error-dialog__icon';
  icon.textContent = '⚠️';

  const title = document.createElement('div');
  title.id = 'luxar-error-title';
  title.className = 'luxar-error-dialog__title';
  title.textContent = 'Unable to Load Dataset';

  header.appendChild(icon);
  header.appendChild(title);

  // Main error message
  const messageText = document.createElement('div');
  messageText.id = 'luxar-error-message-text';
  messageText.className = 'luxar-error-dialog__message';
  messageText.textContent = message;

  // Helpful guidance section
  const guidance = document.createElement('div');
  guidance.className = 'luxar-error-dialog__guidance';

  const guidanceTitle = document.createElement('div');
  guidanceTitle.className = 'luxar-error-dialog__guidance-title';
  guidanceTitle.textContent = '💡 How to Load a Dataset:';

  const guidanceList = document.createElement('div');
  guidanceList.className = 'luxar-error-dialog__guidance-content';
  guidanceList.innerHTML = `
    <div class="luxar-error-dialog__guidance-item">
      <strong>1. Add dataset to URL</strong><br/>
      <code class="luxar-error-dialog__guidance-code">
        ${window.location.origin}/?src=/path/to/dataset.zarr
      </code>
    </div>
    <div class="luxar-error-dialog__guidance-item">
      <strong>2. Or browse available datasets</strong><br/>
      <span class="luxar-error-dialog__guidance-description">Press <kbd class="luxar-error-dialog__guidance-kbd">O</kbd> key to open dataset browser</span>
    </div>
    <div class="luxar-error-dialog__guidance-item">
      <strong>3. Dataset format</strong><br/>
      <span class="luxar-error-dialog__guidance-description">Luxar loads Zarr-format datasets with points, lines, and other primitives</span>
    </div>
    <div class="luxar-error-dialog__guidance-item">
      <strong>4. Need help?</strong><br/>
      <span class="luxar-error-dialog__guidance-description">Press <kbd class="luxar-error-dialog__guidance-kbd">H</kbd> to see all keyboard shortcuts</span>
    </div>
  `;

  guidance.appendChild(guidanceTitle);
  guidance.appendChild(guidanceList);

  // Dismiss button (focusable target for accessibility)
  const dismissBtn = document.createElement('button');
  dismissBtn.className = 'luxar-error-dialog__dismiss';
  dismissBtn.textContent = 'Click anywhere or press Escape to dismiss';
  dismissBtn.style.background = 'none';
  dismissBtn.style.border = 'none';
  dismissBtn.style.color = 'inherit';
  dismissBtn.style.font = 'inherit';
  dismissBtn.style.cursor = 'pointer';
  dismissBtn.style.width = '100%';
  dismissBtn.style.padding = '0';

  errorDiv.appendChild(header);
  errorDiv.appendChild(messageText);
  errorDiv.appendChild(guidance);
  errorDiv.appendChild(dismissBtn);

  const dismissError = () => {
    clearAutoDismissTimer();
    releaseActiveTrap();
    errorDiv.remove();
  };

  // Add click handler to dismiss
  errorDiv.addEventListener('click', () => {
    dismissError();
  });

  // Auto-dismiss after configured timeout. Timer id is stored at module
  // scope so dismissError()/clearError()/a replacement showError() can
  // cancel it (audit G17 — was a real timer leak).
  autoDismissTimerId = setTimeout(() => {
    autoDismissTimerId = null;
    if (errorDiv.parentNode) {
      dismissError();
    }
  }, UI_CONFIG.timings.errorAutoDismissMs);

  getViewerContainer().appendChild(errorDiv);

  // Trap focus within the error dialog. Stored at module scope so the
  // clearError() teardown path can release it without going through
  // dismissError() (which is a closure scoped to this showError call).
  activeReleaseTrap = trapFocus(errorDiv);
}

/**
 * Programmatically dismiss the current error dialog, if any.
 *
 * Mirrors the click/Escape teardown path: cancels the pending auto-dismiss
 * timer, releases the focus trap, and removes the dialog element. A no-op when
 * no error is shown.
 */
export function clearError() {
  clearAutoDismissTimer();
  releaseActiveTrap();
  const errorDiv = document.getElementById('luxar-error-message');
  if (errorDiv) {
    errorDiv.remove();
  }
}
