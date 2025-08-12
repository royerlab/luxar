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
  helpDiv.style.backgroundColor = 'rgba(30, 30, 30, 0.95)';
  helpDiv.style.color = '#e0e0e0';
  helpDiv.style.padding = '15px';
  helpDiv.style.borderRadius = '8px';
  helpDiv.style.fontFamily =
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
  helpDiv.style.fontSize = '12px';
  helpDiv.style.zIndex = '1001';
  helpDiv.style.width = '380px';
  helpDiv.style.maxHeight = '80vh';
  helpDiv.style.overflowY = 'auto';
  helpDiv.style.backdropFilter = 'blur(10px)';
  helpDiv.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
  helpDiv.style.outline = 'none !important';

  const title = document.createElement('div');
  title.textContent = 'Luxar Controls & Shortcuts';
  title.style.fontSize = '14px';
  title.style.fontWeight = 'bold';
  title.style.marginBottom = '10px';
  title.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
  title.style.paddingBottom = '6px';

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
        'V: Switch view mode (Orbit/Fly)',
        'F: Recenter camera on scene',
        'O: Open dataset browser',
        'Esc: Close panels',
      ]
    },
    {
      title: '🚁 Fly Mode Controls',
      expanded: false,
      items: [
        'WASD: Move forward/back/left/right',
        '⌥W/⌥S (Alt+W/S): Move up/down',
        '↑↓←→: Look up/down/left/right',
        '🖱️ Drag: Free look (rotate view)',
        'I: Toggle inertial mode',
        'Note: Press V to enter fly mode',
      ]
    },
    {
      title: '📐 nD Navigation',
      expanded: false,
      items: [
        '1-9: Select dimension to control',
        '[ / ]: Navigate selected dimension',
        'N: Dimension sliders panel',
      ]
    },
    {
      title: '⚙️ Advanced Settings',
      expanded: false,
      items: [
        'R: Rendering controls panel',
        'P: Performance monitor',
        'C: Toggle center (origin/bbox)',
        '⇧ + Wheel: Adjust field of view',
        'Ctrl+L: Debug console',
      ]
    },
    {
      title: '💡 Tips',
      expanded: false,
      items: [
        '• Try fly mode (V) for exploration',
        '• Use inertial mode (I) for smooth coasting',
        '• Enable auto-rotation in settings',
        '• Use WASD + arrows for precise fly control',
      ]
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
    categoryHeader.style.justifyContent = 'space-between';
    
    const categoryTitle = document.createElement('span');
    categoryTitle.textContent = category.title;
    
    const categoryArrow = document.createElement('span');
    categoryArrow.textContent = category.expanded ? '▼' : '▶';
    categoryArrow.style.fontSize = '10px';
    categoryArrow.style.marginLeft = '10px';
    categoryArrow.style.transition = 'transform 0.2s';
    
    categoryHeader.appendChild(categoryTitle);
    categoryHeader.appendChild(categoryArrow);
    
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
      categoryArrow.textContent = isExpanded ? '▶' : '▼';
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
  
  helpDiv.appendChild(title);
  helpDiv.appendChild(controlsList);
  helpDiv.appendChild(footerNote);

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
