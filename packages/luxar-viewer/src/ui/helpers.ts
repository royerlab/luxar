/**
 * UI utility functions for the Luxar viewer.
 *
 * Provides helper functions for common UI patterns:
 * - Loading indicators (spinner during scene load)
 * - Error messages (user-friendly error display with guidance)
 * - Help overlay (keyboard shortcuts and controls)
 * - UI cleanup (resource management)
 *
 * These utilities create temporary UI elements (overlays, dialogs) that
 * exist outside the main component tree and are cleaned up automatically
 * or explicitly via cleanupUI().
 *
 * @module ui/helpers
 */

import { config } from '../config';

// UI Configuration constants
const UI_CONFIG = config.ui;

// Ensure spinner CSS is only injected once
function ensureSpinnerCSS() {
  if (!document.getElementById('spinner-styles')) {
    const style = document.createElement('style');
    style.id = 'spinner-styles';
    style.textContent =
      '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
    document.head.appendChild(style);
  }
}

/**
 * Create and show animated loading indicator.
 *
 * Displays a centered loading spinner with "Loading scene..." text.
 * Shown during initial scene loading or when switching datasets.
 * Automatically positioned over viewport center with semi-transparent
 * dark background.
 *
 * @returns The loading indicator element (for reference, can be removed
 *          via hideLoadingIndicator() or by calling element.remove())
 *
 * @example
 * ```typescript
 * // Show loading spinner before fetch
 * const loader = showLoadingIndicator();
 *
 * try {
 *   await loadScene(url);
 * } finally {
 *   hideLoadingIndicator();  // Remove spinner
 * }
 * ```
 */
export function showLoadingIndicator(): HTMLElement {
  ensureSpinnerCSS();

  const loadingDiv = document.createElement('div');
  loadingDiv.id = 'loading-indicator';
  loadingDiv.style.position = 'fixed';
  loadingDiv.style.top = '50%';
  loadingDiv.style.left = '50%';
  loadingDiv.style.transform = 'translate(-50%, -50%)';
  loadingDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
  loadingDiv.style.color = '#e0e0e0';
  loadingDiv.style.padding = '20px';
  loadingDiv.style.borderRadius = '8px';
  loadingDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  loadingDiv.style.fontSize = '16px';
  loadingDiv.style.zIndex = String(UI_CONFIG.zIndex.loading);
  loadingDiv.style.textAlign = 'center';

  const spinner = document.createElement('div');
  spinner.style.border = '3px solid rgba(255, 255, 255, 0.3)';
  spinner.style.borderTop = '3px solid white';
  spinner.style.borderRadius = '50%';
  spinner.style.width = '24px';
  spinner.style.height = '24px';
  spinner.style.animation = 'spin 1s linear infinite';
  spinner.style.margin = '0 auto 10px auto';

  const text = document.createElement('div');
  text.textContent = 'Loading scene...';
  text.id = 'loading-text';

  loadingDiv.appendChild(spinner);
  loadingDiv.appendChild(text);
  document.body.appendChild(loadingDiv);

  return loadingDiv;
}

/**
 * Hide and remove loading indicator from DOM.
 *
 * Removes the loading spinner created by showLoadingIndicator().
 * Safe to call multiple times or when no indicator exists (no-op).
 *
 * @example
 * ```typescript
 * // After scene loads successfully or fails
 * hideLoadingIndicator();
 * ```
 */
export function hideLoadingIndicator() {
  const loadingDiv = document.getElementById('loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }
}

/**
 * Display user-friendly error message with helpful guidance.
 *
 * Shows a styled error dialog with:
 * - Error icon and title
 * - Specific error message
 * - Helpful instructions for loading datasets
 * - Keyboard shortcuts reminder
 * - Auto-dismiss after timeout
 *
 * The dialog is dismissible by clicking, pressing Escape/Enter/Space,
 * or automatically after configured timeout (default 30 seconds).
 *
 * Replaces any existing error message to avoid cluttering the UI.
 *
 * @param message - Error message to display. Should be user-friendly and
 *                  actionable (e.g., "Dataset not found" rather than "404 Error")
 *
 * @example
 * ```typescript
 * // Show error when scene fails to load
 * try {
 *   await loadScene(url);
 * } catch (error) {
 *   showError(`Failed to load dataset: ${error.message}`);
 * }
 * ```
 *
 * @example
 * ```typescript
 * // Custom error for missing URL parameter
 * if (!url) {
 *   showError('No dataset URL provided. Add ?src=... to the URL.');
 * }
 * ```
 */
export function showError(message: string) {
  // Remove any existing error messages first
  const existingError = document.getElementById('error-message');
  if (existingError) {
    existingError.remove();
  }

  const errorDiv = document.createElement('div');
  errorDiv.id = 'error-message';
  errorDiv.className = 'error-message'; // Add class for E2E tests
  errorDiv.style.outline = 'none';
  errorDiv.style.position = 'fixed';
  errorDiv.style.top = '50%';
  errorDiv.style.left = '50%';
  errorDiv.style.transform = 'translate(-50%, -50%)';
  errorDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.95)';
  errorDiv.style.color = '#e0e0e0';
  errorDiv.style.padding = '24px';
  errorDiv.style.borderRadius = '12px';
  errorDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  errorDiv.style.fontSize = '14px';
  errorDiv.style.zIndex = String(UI_CONFIG.zIndex.error);
  errorDiv.style.maxWidth = '520px';
  errorDiv.style.textAlign = 'left';
  errorDiv.style.cursor = 'pointer';
  errorDiv.style.backdropFilter = 'blur(10px)';
  errorDiv.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.4)';
  errorDiv.style.border = '1px solid rgba(255, 100, 100, 0.3)';

  // ARIA attributes for accessibility
  errorDiv.setAttribute('role', 'alertdialog');
  errorDiv.setAttribute('aria-modal', 'true');
  errorDiv.setAttribute('aria-labelledby', 'error-title');
  errorDiv.setAttribute('aria-describedby', 'error-message-text');

  // Error icon + title
  const header = document.createElement('div');
  header.style.display = 'flex';
  header.style.alignItems = 'center';
  header.style.marginBottom = '16px';
  header.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';
  header.style.paddingBottom = '12px';

  const icon = document.createElement('div');
  icon.textContent = '⚠️';
  icon.style.fontSize = '24px';
  icon.style.marginRight = '12px';

  const title = document.createElement('div');
  title.id = 'error-title';
  title.textContent = 'Unable to Load Dataset';
  title.style.fontSize = '16px';
  title.style.fontWeight = '600';
  title.style.color = '#ff9999';

  header.appendChild(icon);
  header.appendChild(title);

  // Main error message
  const messageText = document.createElement('div');
  messageText.id = 'error-message-text';
  messageText.textContent = message;
  messageText.style.marginBottom = '16px';
  messageText.style.color = '#ffcccc';
  messageText.style.lineHeight = '1.5';

  // Helpful guidance section
  const guidance = document.createElement('div');
  guidance.style.marginTop = '16px';
  guidance.style.padding = '12px';
  guidance.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
  guidance.style.borderRadius = '6px';
  guidance.style.borderLeft = '3px solid rgba(76, 175, 80, 0.5)';

  const guidanceTitle = document.createElement('div');
  guidanceTitle.textContent = '💡 How to Load a Dataset:';
  guidanceTitle.style.fontWeight = '600';
  guidanceTitle.style.marginBottom = '8px';
  guidanceTitle.style.color = '#88cc88';

  const guidanceList = document.createElement('div');
  guidanceList.style.fontSize = '13px';
  guidanceList.style.color = '#cccccc';
  guidanceList.style.lineHeight = '1.6';

  guidanceList.innerHTML = `
    <div style="margin-bottom: 8px;">
      <strong>1. Add dataset to URL</strong><br/>
      <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 3px; font-size: 12px;">
        http://localhost:5173/?src=/path/to/dataset.zarr
      </code>
    </div>
    <div style="margin-bottom: 8px;">
      <strong>2. Or browse available datasets</strong><br/>
      <span style="opacity: 0.8;">Press <kbd style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 3px; font-family: monospace;">O</kbd> key to open dataset browser</span>
    </div>
    <div style="margin-bottom: 8px;">
      <strong>3. Dataset format</strong><br/>
      <span style="opacity: 0.8;">Luxar loads Zarr-format datasets with point cloud data</span>
    </div>
    <div>
      <strong>4. Need help?</strong><br/>
      <span style="opacity: 0.8;">Press <kbd style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 3px; font-family: monospace;">H</kbd> to see all keyboard shortcuts</span>
    </div>
  `;

  guidance.appendChild(guidanceTitle);
  guidance.appendChild(guidanceList);

  // Dismiss instructions
  const dismissText = document.createElement('div');
  dismissText.textContent = 'Click anywhere or press Escape to dismiss';
  dismissText.style.marginTop = '16px';
  dismissText.style.fontSize = '11px';
  dismissText.style.opacity = '0.5';
  dismissText.style.color = 'rgba(255, 255, 255, 0.6)';
  dismissText.style.textAlign = 'center';
  dismissText.style.fontStyle = 'italic';

  errorDiv.appendChild(header);
  errorDiv.appendChild(messageText);
  errorDiv.appendChild(guidance);
  errorDiv.appendChild(dismissText);

  // Add click handler to dismiss
  errorDiv.addEventListener('click', () => {
    errorDiv.remove();
  });

  // Add keyboard navigation support
  errorDiv.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Escape') {
      event.preventDefault();
      errorDiv.remove();
    }
  });

  // Auto-dismiss after configured timeout
  setTimeout(() => {
    if (errorDiv.parentNode) {
      errorDiv.remove();
    }
  }, UI_CONFIG.timings.errorAutoDismissMs);

  document.body.appendChild(errorDiv);
}

/**
 * Clean up all temporary UI elements and resources.
 *
 * Removes all UI elements created by helper functions:
 * - Loading indicators (spinners)
 * - Error messages
 * - Help overlays
 * - Associated CSS styles
 *
 * Useful during application teardown or when resetting UI state.
 * Safe to call even if no UI elements exist (no-op).
 *
 * @example
 * ```typescript
 * // During app teardown
 * cleanupUI();
 * sceneManager.dispose();
 * inputHandler.dispose();
 * ```
 */
export function cleanupUI() {
  // Remove spinner CSS styles
  const spinnerStyles = document.getElementById('spinner-styles');
  if (spinnerStyles) {
    spinnerStyles.remove();
  }

  // Remove any lingering loading indicators
  const loadingDiv = document.getElementById('loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }

  // Remove any lingering error messages
  const errorDiv = document.getElementById('error-message');
  if (errorDiv) {
    errorDiv.remove();
  }

  // Remove any lingering help overlays
  const helpDiv = document.getElementById('help-overlay');
  if (helpDiv) {
    helpDiv.remove();
  }
}

/**
 * Create and show keyboard shortcuts help overlay.
 *
 * Displays comprehensive help panel with all available keyboard shortcuts
 * organized by category (Basic Controls, Fly Mode, nD Navigation, etc.).
 * Categories are collapsible for better organization.
 *
 * Triggered by H key. Dismissible by clicking anywhere, pressing Escape,
 * or clicking the close button.
 *
 * @example
 * ```typescript
 * // User presses H key
 * showHelpOverlay();
 * // Help panel appears in top-right corner
 * ```
 */
export function showHelpOverlay() {
  // Prevent opening multiple overlays - if one exists, do nothing
  const existingHelp = document.getElementById('help-overlay');
  if (existingHelp) {
    return; // Don't create a new one, just return
  }

  const helpDiv = document.createElement('div');
  helpDiv.id = 'help-overlay';
  helpDiv.style.position = 'fixed';
  helpDiv.style.top = '20px';
  helpDiv.style.right = '20px';
  helpDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.95)';
  helpDiv.style.color = '#e0e0e0';
  helpDiv.style.padding = '15px';
  helpDiv.style.borderRadius = '8px';
  helpDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  helpDiv.style.fontSize = '12px';
  helpDiv.style.zIndex = String(UI_CONFIG.zIndex.help);
  helpDiv.style.width = '380px';
  helpDiv.style.maxHeight = '80vh';
  helpDiv.style.overflowY = 'auto';
  helpDiv.style.backdropFilter = 'blur(10px)';
  helpDiv.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
  helpDiv.style.outline = 'none !important';
  helpDiv.setAttribute('role', 'dialog');
  helpDiv.setAttribute('aria-modal', 'true');
  helpDiv.setAttribute('aria-labelledby', 'help-overlay-title');

  // Create header with title and close button
  const header = document.createElement('div');
  header.style.display = 'flex';
  header.style.justifyContent = 'space-between';
  header.style.alignItems = 'center';
  header.style.marginBottom = '10px';
  header.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
  header.style.paddingBottom = '6px';

  const title = document.createElement('h3');
  title.id = 'help-overlay-title';
  title.textContent = 'Luxar Controls & Shortcuts';
  title.style.margin = '0';
  title.style.fontSize = '14px';
  title.style.fontWeight = 'bold';

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.style.cssText = `
    background: none;
    border: none;
    color: #999;
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  closeBtn.onmouseover = () => (closeBtn.style.color = '#fff');
  closeBtn.onmouseout = () => (closeBtn.style.color = '#999');
  closeBtn.title = 'Close (Escape)';

  header.appendChild(title);
  header.appendChild(closeBtn);

  // Define help categories with expandable sections
  const helpCategories = [
    {
      title: '🎮 Basic Controls',
      expanded: true,
      items: [
        '🖱️ Drag: Rotate view',
        '🖱️ Wheel: Zoom in/out',
        '🖱️ Right drag: Pan camera',
        '⎵ Space: Toggle fullscreen',
        'H: Toggle this help',
        'V: Switch view mode (Orbit/Arcball/Fly)',
        'F: Recenter camera on scene',
        'O: Open dataset browser',
        'Esc: Exit fullscreen / Close panels',
      ],
    },
    {
      title: '🚁 Fly Mode Controls',
      expanded: false,
      items: [
        'WASD: Move forward/back/left/right',
        '⌥W/⌥S (Alt+W/S): Move up/down',
        '⇧ Shift: 2x speed boost',
        '↑↓←→: Look up/down/left/right',
        'Q/E: Roll left/right (barrel roll)',
        '🖱️ Drag: Free look (rotate view)',
        'I: Toggle inertial mode',
        'Note: Press V to enter fly mode',
      ],
    },
    {
      title: '📐 nD Navigation',
      expanded: false,
      items: [
        '1-9: Select dimension to control',
        '[ / ]: Navigate selected dimension',
        'N: Dimension sliders panel',
      ],
    },
    {
      title: '⚙️ Advanced Settings',
      expanded: false,
      items: [
        'R: Rendering controls panel',
        'P: Performance monitor',
        'M: Cycle data monitor (mini/expanded/off)',
        'C: Toggle cinematic mode (noise/vignette/CA/lens)',
        '⇧ + Wheel: Adjust field of view',
        'Ctrl+L: Debug console',
      ],
    },
    {
      title: '💡 Tips',
      expanded: false,
      items: [
        '• Try fly mode (V) for exploration',
        '• Use inertial mode (I) for smooth coasting',
        '• Enable auto-rotation in settings',
        '• Use WASD + arrows for precise fly control',
      ],
    },
  ];

  const controlsList = document.createElement('div');

  // Create collapsible categories
  helpCategories.forEach((category, categoryIndex) => {
    // Category header (clickable)
    const categoryHeader = document.createElement('div');
    categoryHeader.style.fontWeight = 'bold';
    categoryHeader.style.marginTop = categoryIndex > 0 ? '12px' : '8px';
    categoryHeader.style.marginBottom = '8px';
    categoryHeader.style.color = '#4CAF50';
    categoryHeader.style.cursor = 'pointer';
    categoryHeader.style.userSelect = 'none';
    categoryHeader.style.display = 'flex';
    categoryHeader.style.alignItems = 'center';

    const categoryArrow = document.createElement('span');
    categoryArrow.textContent = '▶';
    categoryArrow.style.fontSize = '10px';
    categoryArrow.style.marginRight = '5px';
    categoryArrow.style.transition = 'transform 0.2s';
    categoryArrow.style.display = 'inline-block';
    categoryArrow.style.transform = category.expanded ? 'rotate(90deg)' : 'rotate(0deg)';
    categoryArrow.style.color = 'rgba(255, 255, 255, 0.6)';

    const categoryTitle = document.createElement('span');
    categoryTitle.textContent = category.title;

    categoryHeader.appendChild(categoryArrow);
    categoryHeader.appendChild(categoryTitle);

    // Category content container
    const categoryContent = document.createElement('div');
    categoryContent.style.display = category.expanded ? 'block' : 'none';
    categoryContent.style.marginBottom = '4px';
    categoryContent.style.borderLeft = '2px solid rgba(76, 175, 80, 0.2)';
    categoryContent.style.marginLeft = '8px';
    categoryContent.style.paddingLeft = '12px';

    // Add items to category
    category.items.forEach((item) => {
      const itemDiv = document.createElement('div');
      itemDiv.textContent = item;
      itemDiv.style.marginBottom = '6px';
      itemDiv.style.lineHeight = '1.4';

      // Special styling for certain items
      if (item.startsWith('•')) {
        itemDiv.style.color = '#aaa';
        itemDiv.style.fontSize = '11px';
      } else if (item.startsWith('Note:')) {
        itemDiv.style.color = '#ff9800';
        itemDiv.style.fontSize = '11px';
        itemDiv.style.fontStyle = 'italic';
      } else {
        itemDiv.style.color = '#e0e0e0';
      }

      categoryContent.appendChild(itemDiv);
    });

    // Toggle functionality
    categoryHeader.addEventListener('click', (e) => {
      e.stopPropagation();
      const isExpanded = categoryContent.style.display !== 'none';
      categoryContent.style.display = isExpanded ? 'none' : 'block';
      categoryArrow.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(90deg)';
    });

    controlsList.appendChild(categoryHeader);
    controlsList.appendChild(categoryContent);
  });

  // Add footer note
  const footerNote = document.createElement('div');
  footerNote.textContent = 'Click anywhere or press Esc to close';
  footerNote.style.marginTop = '12px';
  footerNote.style.paddingTop = '8px';
  footerNote.style.borderTop = '1px solid rgba(255, 255, 255, 0.1)';
  footerNote.style.fontSize = '11px';
  footerNote.style.color = '#888';
  footerNote.style.textAlign = 'center';

  helpDiv.appendChild(header);
  helpDiv.appendChild(controlsList);
  helpDiv.appendChild(footerNote);

  // Track whether we're in the process of closing to prevent race conditions
  let isClosing = false;

  // Function to close help overlay
  const closeHelp = () => {
    // Guard against multiple simultaneous close calls
    if (isClosing) return;
    isClosing = true;

    const help = document.getElementById('help-overlay');
    if (help) {
      // Remove global click listener first
      document.removeEventListener('click', handleDocumentClick);
      // Then remove the panel
      help.remove();
    }
  };

  // Global click handler to close help when clicking outside
  const handleDocumentClick = (event: MouseEvent) => {
    const target = event.target as Element;
    if (!helpDiv.contains(target)) {
      closeHelp();
    }
  };

  // Wire up close button to use the proper cleanup function
  closeBtn.onclick = () => closeHelp();

  // Add click handler within help panel - but NOT to close
  // (clicking inside should not close, only clicking outside should)
  // So we remove the helpDiv.addEventListener('click', closeHelp) line

  // Add keyboard navigation support
  helpDiv.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Escape') {
      event.preventDefault();
      closeHelp();
    }
  });

  document.body.appendChild(helpDiv);

  // Add global click listener after a short delay to prevent immediate closure
  setTimeout(() => {
    // Only add if the help div still exists and hasn't been closed
    if (!isClosing && document.getElementById('help-overlay')) {
      document.addEventListener('click', handleDocumentClick);
    }
  }, UI_CONFIG.timings.helpClickDelayMs);
}

/**
 * Hide and remove help overlay from DOM.
 *
 * Removes the keyboard shortcuts help panel. Safe to call multiple
 * times or when no overlay exists (no-op).
 *
 * @example
 * ```typescript
 * // Close help programmatically
 * hideHelpOverlay();
 * ```
 */
export function hideHelpOverlay() {
  const helpDiv = document.getElementById('help-overlay');
  if (helpDiv) {
    helpDiv.remove();
  }
}

/**
 * Clear any existing error messages from display.
 *
 * Removes error dialog if present. Useful before showing new error
 * or when dismissing errors programmatically.
 */
export function clearError() {
  const errorDiv = document.getElementById('error-message');
  if (errorDiv) {
    errorDiv.remove();
  }
}
