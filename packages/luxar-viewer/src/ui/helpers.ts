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

/** Module-level reference to the active help overlay click handler, for cleanup */
let activeHelpClickHandler: ((event: MouseEvent) => void) | null = null;

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
  const loadingDiv = document.createElement('div');
  loadingDiv.id = 'loading-indicator';
  loadingDiv.className = 'luxar-loading-indicator';

  const spinner = document.createElement('div');
  spinner.className = 'luxar-loading-indicator__spinner';

  const text = document.createElement('div');
  text.className = 'luxar-loading-indicator__text';
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
 * The dialog is dismissible by clicking, pressing Escape (closes all panels),
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

  // Create error dialog with CSS classes
  const errorDiv = document.createElement('div');
  errorDiv.id = 'error-message';
  errorDiv.className = 'luxar-error-dialog error-message'; // luxar-error-dialog for styling, error-message for E2E tests

  // ARIA attributes for accessibility
  errorDiv.setAttribute('role', 'alertdialog');
  errorDiv.setAttribute('aria-modal', 'true');
  errorDiv.setAttribute('aria-labelledby', 'error-title');
  errorDiv.setAttribute('aria-describedby', 'error-message-text');

  // Error icon + title
  const header = document.createElement('div');
  header.className = 'luxar-error-dialog__header';

  const icon = document.createElement('div');
  icon.className = 'luxar-error-dialog__icon';
  icon.textContent = '⚠️';

  const title = document.createElement('div');
  title.id = 'error-title';
  title.className = 'luxar-error-dialog__title';
  title.textContent = 'Unable to Load Dataset';

  header.appendChild(icon);
  header.appendChild(title);

  // Main error message
  const messageText = document.createElement('div');
  messageText.id = 'error-message-text';
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
        http://localhost:5173/?src=/path/to/dataset.zarr
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

  // Dismiss instructions
  const dismissText = document.createElement('div');
  dismissText.className = 'luxar-error-dialog__dismiss';
  dismissText.textContent = 'Click anywhere or press Escape to dismiss';

  errorDiv.appendChild(header);
  errorDiv.appendChild(messageText);
  errorDiv.appendChild(guidance);
  errorDiv.appendChild(dismissText);

  // Add click handler to dismiss
  errorDiv.addEventListener('click', () => {
    errorDiv.remove();
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

  // Remove any lingering help overlays (use hideHelpOverlay to clean up click listener)
  hideHelpOverlay();
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
  helpDiv.className = 'luxar-help-overlay';
  helpDiv.setAttribute('role', 'dialog');
  helpDiv.setAttribute('aria-modal', 'true');
  helpDiv.setAttribute('aria-labelledby', 'help-overlay-title');

  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'luxar-help-overlay__header';

  const title = document.createElement('h3');
  title.id = 'help-overlay-title';
  title.className = 'luxar-help-overlay__title';
  title.textContent = 'Luxar Controls & Shortcuts';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'luxar-help-overlay__close-btn';
  closeBtn.textContent = '×';
  closeBtn.title = 'Close (Escape)';

  header.appendChild(title);
  header.appendChild(closeBtn);

  // Define help categories with expandable sections
  const helpCategories = [
    {
      title: '🎮 Basic Controls',
      expanded: true,
      items: [
        '🖱️ Drag: Pan camera',
        '🖱️ Right drag: Rotate view',
        '🖱️ ⇧+Drag: Rotate view (alternative)',
        '🖱️ Wheel: Zoom in/out',
        '🖱️ ⇧+Wheel: Roll (rotate around view axis)',
        '⎵ Space: Toggle fullscreen',
        'H: Toggle this help',
        'V: Switch view mode (Orbit/Fly/Ortho)',
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
        '🖱️ Drag: Strafe (pan camera)',
        '🖱️ Right drag: Free look (rotate view)',
        '🖱️ Wheel: Move forward/backward',
        '🖱️ ⇧+Wheel: Roll (rotate around view axis)',
        'I: Toggle inertial mode',
        'Note: Press V to enter fly mode',
      ],
    },
    {
      title: '📐 Ortho Mode Controls',
      expanded: false,
      items: [
        '🖱️ Drag: Pan camera',
        '🖱️ Wheel: Zoom in/out',
        '🖱️ ⇧+Wheel: Roll (rotate around view axis)',
        'No rotation — 2D viewing mode',
        'Note: Press V to cycle to ortho mode',
      ],
    },
    {
      title: '🧭 nD Navigation',
      expanded: false,
      items: [
        '1-9: Select dimension to control',
        '[ / ]: Navigate selected dimension',
        'N: Dimension sliders panel',
        'K: Play/pause dimension animation',
        'Home / End: Jump to dimension start/end',
        '⇧ + ↑/↓: Animation speed up/down',
      ],
    },
    {
      title: '⚙️ Advanced Settings',
      expanded: false,
      items: [
        'R: Rendering controls panel',
        'T: Recording panel (screenshot/video)',
        'G: Quick screenshot',
        'B: Toggle scale bar',
        'J: Toggle colormap legend',
        'P: Performance monitor',
        'M: Cycle data monitor (mini/expanded/off)',
        'C: Toggle cinematic mode (noise/vignette/CA/lens)',
        'Ctrl/⌘ + Wheel: Adjust field of view (perspective only)',
        'L: Layers panel',
        'Ctrl+L: Debug console',
        'Ctrl+⇧+S: Export viewer state to clipboard',
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
  helpCategories.forEach((category) => {
    // Category header (clickable)
    const categoryHeader = document.createElement('div');
    categoryHeader.className = 'luxar-help-overlay__category-header';

    const categoryArrow = document.createElement('span');
    categoryArrow.className = `luxar-help-overlay__category-arrow ${category.expanded ? 'luxar-help-overlay__category-arrow--expanded' : ''}`;
    categoryArrow.textContent = '▶';

    const categoryTitle = document.createElement('span');
    categoryTitle.textContent = category.title;

    categoryHeader.appendChild(categoryArrow);
    categoryHeader.appendChild(categoryTitle);

    // Category content container
    const categoryContent = document.createElement('div');
    categoryContent.className = `luxar-help-overlay__category-content ${category.expanded ? '' : 'luxar-help-overlay__category-content--collapsed'}`;

    // Add items to category
    category.items.forEach((item) => {
      const itemDiv = document.createElement('div');
      itemDiv.textContent = item;

      // Special styling for certain items
      if (item.startsWith('•')) {
        itemDiv.className = 'luxar-help-overlay__item luxar-help-overlay__item--tip';
      } else if (item.startsWith('Note:')) {
        itemDiv.className = 'luxar-help-overlay__item luxar-help-overlay__item--note';
      } else {
        itemDiv.className = 'luxar-help-overlay__item';
      }

      categoryContent.appendChild(itemDiv);
    });

    // Toggle functionality
    categoryHeader.addEventListener('click', (e) => {
      e.stopPropagation();
      const isExpanded = !categoryContent.classList.contains(
        'luxar-help-overlay__category-content--collapsed'
      );

      if (isExpanded) {
        categoryContent.classList.add('luxar-help-overlay__category-content--collapsed');
        categoryArrow.classList.remove('luxar-help-overlay__category-arrow--expanded');
      } else {
        categoryContent.classList.remove('luxar-help-overlay__category-content--collapsed');
        categoryArrow.classList.add('luxar-help-overlay__category-arrow--expanded');
      }
    });

    controlsList.appendChild(categoryHeader);
    controlsList.appendChild(categoryContent);
  });

  // Add footer note
  const footerNote = document.createElement('div');
  footerNote.className = 'luxar-help-overlay__footer';
  footerNote.textContent = 'Click anywhere or press Esc to close';

  // Create scroll wrapper (separates scrolling from glass effect container)
  const scrollWrapper = document.createElement('div');
  scrollWrapper.className = 'luxar-help-overlay__scroll';
  scrollWrapper.appendChild(header);
  scrollWrapper.appendChild(controlsList);
  scrollWrapper.appendChild(footerNote);

  helpDiv.appendChild(scrollWrapper);

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
      activeHelpClickHandler = null;
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

  activeHelpClickHandler = handleDocumentClick;

  // Wire up close button to use the proper cleanup function
  closeBtn.onclick = () => closeHelp();

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
    if (activeHelpClickHandler) {
      document.removeEventListener('click', activeHelpClickHandler);
      activeHelpClickHandler = null;
    }
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

/**
 * Show a brief toast notification that auto-dismisses.
 *
 * @param message - Text to display
 * @param durationMs - How long to show (default 2000ms)
 */
export function showToast(message: string, durationMs: number = 2000): void {
  // Remove existing toast if any
  const existing = document.getElementById('luxar-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'luxar-toast';
  toast.textContent = message;
  Object.assign(toast.style, {
    position: 'fixed',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    padding: '10px 20px',
    borderRadius: '8px',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    color: '#fff',
    fontSize: '14px',
    fontFamily: 'system-ui, sans-serif',
    zIndex: '99999',
    pointerEvents: 'none',
    transition: 'opacity 0.3s ease',
    opacity: '1',
  });

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, durationMs);
}
