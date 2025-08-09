// UI utility functions for the scene player

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

// Function to create and show loading indicator
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

// Function to hide loading indicator
export function hideLoadingIndicator() {
  const loadingDiv = document.getElementById('loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }
}

// Function to display error message to user with dismiss functionality
export function showError(message: string) {
  // Remove any existing error messages first
  const existingError = document.getElementById('error-message');
  if (existingError) {
    existingError.remove();
  }

  const errorDiv = document.createElement('div');
  errorDiv.id = 'error-message';
  errorDiv.style.outline = 'none'; // Remove focus outline
  errorDiv.style.position = 'fixed';
  errorDiv.style.top = '50%';
  errorDiv.style.left = '50%';
  errorDiv.style.transform = 'translate(-50%, -50%)';
  errorDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
  errorDiv.style.color = '#e0e0e0';
  errorDiv.style.padding = '15px';
  errorDiv.style.borderRadius = '8px';
  errorDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  errorDiv.style.fontSize = '14px';
  errorDiv.style.zIndex = String(UI_CONFIG.zIndex.error);
  errorDiv.style.maxWidth = '400px';
  errorDiv.style.textAlign = 'center';
  errorDiv.style.cursor = 'pointer';
  errorDiv.style.backdropFilter = 'blur(10px)';
  errorDiv.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
  errorDiv.style.border = '1px solid rgba(255, 50, 50, 0.5)';

  const messageText = document.createElement('div');
  messageText.textContent = message;
  messageText.style.marginBottom = '10px';
  messageText.style.fontWeight = '500';
  messageText.style.color = '#ff6b6b';

  const dismissText = document.createElement('div');
  dismissText.textContent = 'Click to dismiss';
  dismissText.style.fontSize = '11px';
  dismissText.style.opacity = '0.6';
  dismissText.style.color = 'rgba(255, 255, 255, 0.8)';

  errorDiv.appendChild(messageText);
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

// Function to clean up UI resources
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

// Function to create and show help overlay
export function showHelpOverlay() {
  // Remove any existing help overlay first
  const existingHelp = document.getElementById('help-overlay');
  if (existingHelp) {
    existingHelp.remove();
  }

  const helpDiv = document.createElement('div');
  helpDiv.id = 'help-overlay';
  helpDiv.style.position = 'fixed';
  helpDiv.style.top = '20px';
  helpDiv.style.right = '20px';
  helpDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
  helpDiv.style.color = '#e0e0e0';
  helpDiv.style.padding = '15px';
  helpDiv.style.borderRadius = '8px';
  helpDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  helpDiv.style.fontSize = '12px';
  helpDiv.style.zIndex = '1001';
  helpDiv.style.width = '300px';
  helpDiv.style.cursor = 'pointer';
  helpDiv.style.backdropFilter = 'blur(10px)';
  helpDiv.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
  helpDiv.style.outline = 'none !important'; // Remove focus outline

  const title = document.createElement('div');
  title.textContent = '3D Scene Controls';
  title.style.fontSize = '14px';
  title.style.fontWeight = 'bold';
  title.style.marginBottom = '10px';
  title.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
  title.style.paddingBottom = '6px';

  const controls = [
    '🖱️ Mouse drag: Rotate camera',
    '🖱️ Mouse wheel: Zoom in/out',
    '⇧ + Mouse wheel: Change FOV',
    '🖱️ Right drag: Pan camera',
    '⎵ Press Space: Toggle fullscreen',
    '❓ Press H: Toggle this help',
    'Press O: Open dataset browser',
    'Press P: Toggle performance stats',
    'Press R: Rendering controls',
    'Press C: Toggle center (origin/bounding box)',
    'Ctrl+L: Debug console',
    '',
    '📐 nD Navigation (if applicable):',
    '🎛️ Press D: Toggle dimension sliders',
    '🔢 Press 1-9: Select dimension to control',
    '⬅️➡️ Press [ / ]: Navigate selected dimension',
    '',
    'Click anywhere to close',
  ];

  const controlsList = document.createElement('div');
  controls.forEach((control) => {
    const controlItem = document.createElement('div');
    if (control === '') {
      // Empty line for spacing
      controlItem.style.marginBottom = '4px';
    } else {
      controlItem.textContent = control;
      controlItem.style.marginBottom = '8px';
      controlItem.style.lineHeight = '1.4';
      // Special styling for section headers
      if (control.includes('nD Navigation')) {
        controlItem.style.fontWeight = 'bold';
        controlItem.style.marginTop = '8px';
      }
    }
    controlsList.appendChild(controlItem);
  });

  helpDiv.appendChild(title);
  helpDiv.appendChild(controlsList);

  // Function to close help overlay
  const closeHelp = () => {
    const help = document.getElementById('help-overlay');
    if (help) {
      help.remove();
      document.removeEventListener('click', handleDocumentClick);
    }
  };

  // Global click handler to close help when clicking outside
  const handleDocumentClick = (event: MouseEvent) => {
    const target = event.target as Element;
    if (!helpDiv.contains(target)) {
      closeHelp();
    }
  };

  // Add click handler within help panel to close
  helpDiv.addEventListener('click', closeHelp);

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
    document.addEventListener('click', handleDocumentClick);
  }, UI_CONFIG.timings.helpClickDelayMs);
}

// Function to hide help overlay
export function hideHelpOverlay() {
  const helpDiv = document.getElementById('help-overlay');
  if (helpDiv) {
    helpDiv.remove();
  }
}

// Function to clear any existing error messages
export function clearError() {
  const errorDiv = document.getElementById('error-message');
  if (errorDiv) {
    errorDiv.remove();
  }
}
