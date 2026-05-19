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

const UI_CONFIG = config.ui;

export function showError(message: string) {
  // Remove any existing error messages first
  const existingError = document.getElementById('luxar-error-message');
  if (existingError) {
    existingError.remove();
  }

  // Create error dialog with CSS classes
  const errorDiv = document.createElement('div');
  errorDiv.id = 'luxar-error-message';
  errorDiv.className = 'luxar-error-dialog error-message'; // luxar-error-dialog for styling, error-message for E2E tests

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

  // Track focus trap release function
  let releaseTrap: (() => void) | null = null;

  const dismissError = () => {
    if (releaseTrap) {
      releaseTrap();
      releaseTrap = null;
    }
    errorDiv.remove();
  };

  // Add click handler to dismiss
  errorDiv.addEventListener('click', () => {
    dismissError();
  });

  // Auto-dismiss after configured timeout
  setTimeout(() => {
    if (errorDiv.parentNode) {
      dismissError();
    }
  }, UI_CONFIG.timings.errorAutoDismissMs);

  document.body.appendChild(errorDiv);

  // Trap focus within the error dialog
  releaseTrap = trapFocus(errorDiv);
}

export function clearError() {
  const errorDiv = document.getElementById('luxar-error-message');
  if (errorDiv) {
    errorDiv.remove();
  }
}
