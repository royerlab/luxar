/**
 * Keyboard-shortcuts help overlay (H key).
 *
 * Displays a comprehensive, collapsible-by-category help panel.
 * Dismissible by clicking outside, pressing Escape, or via the close
 * button. Traps focus inside the panel while it is open.
 *
 * Module-level handles for the global click listener and focus-trap
 * release live here so hideHelpOverlay() can tear them down even when
 * it's called from outside (e.g. by ui-cleanup).
 */

import { config } from '../config';
import { trapFocus } from './help-overlay/focus-trap';
import { getViewerContainer } from '../utils/viewer-container';

const UI_CONFIG = config.ui;

let activeHelpClickHandler: ((event: MouseEvent) => void) | null = null;
let activeHelpFocusTrapRelease: (() => void) | null = null;

export function showHelpOverlay() {
  // Prevent opening multiple overlays - if one exists, do nothing
  const existingHelp = document.getElementById('luxar-help-overlay');
  if (existingHelp) {
    return; // Don't create a new one, just return
  }

  const helpDiv = document.createElement('div');
  helpDiv.id = 'luxar-help-overlay';
  helpDiv.className = 'luxar-help-overlay luxar-glass-surface';
  helpDiv.setAttribute('role', 'dialog');
  helpDiv.setAttribute('aria-modal', 'true');
  helpDiv.setAttribute('aria-labelledby', 'luxar-help-overlay-title');

  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'luxar-help-overlay__header';

  const title = document.createElement('h3');
  title.id = 'luxar-help-overlay-title';
  title.className = 'luxar-help-overlay__title';
  title.textContent = 'Luxar Controls & Shortcuts';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'luxar-help-overlay__close-btn';
  closeBtn.textContent = '×';
  closeBtn.title = 'Close (Escape)';
  closeBtn.setAttribute('aria-label', 'Close help overlay');
  closeBtn.setAttribute('aria-keyshortcuts', 'Escape');

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
        'U: Toggle overlays',
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

    // Release focus trap before removing element
    if (activeHelpFocusTrapRelease) {
      activeHelpFocusTrapRelease();
      activeHelpFocusTrapRelease = null;
    }

    const help = document.getElementById('luxar-help-overlay');
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

  getViewerContainer().appendChild(helpDiv);

  // Trap focus within the help overlay
  activeHelpFocusTrapRelease = trapFocus(helpDiv);

  // Add global click listener after a short delay to prevent immediate closure
  setTimeout(() => {
    // Only add if the help div still exists and hasn't been closed
    if (!isClosing && document.getElementById('luxar-help-overlay')) {
      document.addEventListener('click', handleDocumentClick);
    }
  }, UI_CONFIG.timings.helpClickDelayMs);
}

export function hideHelpOverlay() {
  const helpDiv = document.getElementById('luxar-help-overlay');
  if (helpDiv) {
    // Release focus trap before removing element
    if (activeHelpFocusTrapRelease) {
      activeHelpFocusTrapRelease();
      activeHelpFocusTrapRelease = null;
    }
    if (activeHelpClickHandler) {
      document.removeEventListener('click', activeHelpClickHandler);
      activeHelpClickHandler = null;
    }
    helpDiv.remove();
  }
}
