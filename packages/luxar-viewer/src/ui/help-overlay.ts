/**
 * Keyboard-shortcuts help overlay (H key).
 *
 * A data-driven shortcut reference styled after the viewer's "quiet
 * instrument" design language: tick-motif section micro-headers, stroke SVG
 * section icons (no emoji), and two-column rows of <kbd> chips + muted
 * descriptions. All sections are visible (no collapsing) — hierarchy comes
 * from typography, and the panel scrolls.
 *
 * Dismissible by clicking outside, pressing Escape, or via the close
 * button. Traps focus inside the panel while it is open.
 *
 * Module-level handles for the delayed click-listener registration, the
 * listener itself, and the focus-trap release live here so hideHelpOverlay()
 * can tear them down even when it's called from outside (e.g. by ui-cleanup).
 */

import { config } from '../config';
import { trapFocus } from './help-overlay/focus-trap';
import { getViewerContainer } from '../utils/viewer-container';
import { RAIL_ICONS } from './control-rail/icons';

const UI_CONFIG = config.ui;

let activeHelpClickHandler: ((event: MouseEvent) => void) | null = null;
let activeHelpClickTimer: ReturnType<typeof setTimeout> | null = null;
let activeHelpFocusTrapRelease: (() => void) | null = null;

/** One shortcut row: chip text(s) + what they do. */
interface HelpEntry {
  /** Key/gesture chips, rendered as <kbd> (e.g. ['V'] or ['⇧', 'Wheel']). */
  keys: string[];
  label: string;
}

interface HelpSection {
  title: string;
  /** Stroke SVG icon (rail icon set) shown beside the section title. */
  icon: string;
  /** Optional muted context line under the title (e.g. how to enter a mode). */
  note?: string;
  entries: HelpEntry[];
}

/**
 * The shortcut reference. Maintained by hand alongside the key bindings in
 * input/input-handler/key-bindings/ — keep the two in sync when bindings
 * change.
 */
const HELP_SECTIONS: HelpSection[] = [
  {
    title: 'Basics',
    icon: RAIL_ICONS.navOrbit,
    entries: [
      { keys: ['Drag'], label: 'Pan camera' },
      { keys: ['Right drag'], label: 'Rotate view' },
      { keys: ['⇧', 'Drag'], label: 'Rotate view (alternative)' },
      { keys: ['Wheel'], label: 'Zoom in / out' },
      { keys: ['⇧', 'Wheel'], label: 'Roll around the view axis' },
      { keys: ['Space'], label: 'Toggle fullscreen' },
      { keys: ['F'], label: 'Fit scene (recenter camera)' },
      { keys: ['V'], label: 'View mode: orbit / fly / ortho' },
      { keys: ['O'], label: 'Open dataset browser' },
      { keys: ['H'], label: 'Toggle this help' },
      { keys: ['Esc'], label: 'Exit fullscreen / close panels' },
    ],
  },
  {
    title: 'Fly mode',
    icon: RAIL_ICONS.navFly,
    note: 'Press V until the fly icon shows',
    entries: [
      { keys: ['W', 'A', 'S', 'D'], label: 'Move forward / left / back / right' },
      { keys: ['⌥', 'W / S'], label: 'Move up / down' },
      { keys: ['⇧'], label: 'Hold for 2× speed boost' },
      { keys: ['↑ ↓ ← →'], label: 'Look around' },
      { keys: ['Q / E'], label: 'Roll left / right' },
      { keys: ['Drag'], label: 'Strafe (pan camera)' },
      { keys: ['Right drag'], label: 'Free look' },
      { keys: ['Wheel'], label: 'Move forward / backward' },
      { keys: ['I'], label: 'Toggle inertial mode (smooth coasting)' },
    ],
  },
  {
    title: 'Ortho mode',
    icon: RAIL_ICONS.navOrtho,
    note: 'Press V until the grid icon shows — 2D viewing, no rotation',
    entries: [
      { keys: ['Drag'], label: 'Pan camera' },
      { keys: ['Wheel'], label: 'Zoom in / out' },
      { keys: ['⇧', 'Wheel'], label: 'Roll around the view axis' },
    ],
  },
  {
    title: 'nD navigation',
    icon: RAIL_ICONS.dims,
    entries: [
      { keys: ['1 – 9'], label: 'Select dimension to control' },
      { keys: ['[', ']'], label: 'Step along the selected dimension' },
      { keys: ['N'], label: 'Dimension sliders panel' },
      { keys: ['K'], label: 'Play / pause dimension animation' },
      { keys: ['Home', 'End'], label: 'Jump to dimension start / end' },
      { keys: ['⇧', '↑ / ↓'], label: 'Animation speed up / down' },
    ],
  },
  {
    title: 'Panels & tools',
    icon: RAIL_ICONS.settings,
    entries: [
      { keys: ['R'], label: 'Rendering controls' },
      { keys: ['L'], label: 'Layers panel' },
      { keys: ['N'], label: 'Dimension sliders' },
      { keys: ['M'], label: 'Data monitor (mini / expanded / off)' },
      { keys: ['P'], label: 'Performance monitor' },
      { keys: ['T'], label: 'Recording panel (screenshot / video)' },
      { keys: ['G'], label: 'Quick screenshot' },
      { keys: ['B'], label: 'Scale bar' },
      { keys: ['J'], label: 'Colormap legend' },
      { keys: ['U'], label: 'Overlays' },
      { keys: ['C'], label: 'Cinematic mode (noise / vignette / lens)' },
      { keys: ['Ctrl/⌘', 'Wheel'], label: 'Adjust field of view (perspective)' },
      { keys: ['Ctrl', 'L'], label: 'Debug console' },
      { keys: ['Ctrl', '⇧', 'S'], label: 'Export viewer state to clipboard' },
    ],
  },
];

/** Closing prose tips (no key chips). */
const HELP_TIPS: string[] = [
  'Try fly mode (V) for exploration, with inertial mode (I) for smooth coasting.',
  'The left rail mirrors every panel shortcut — hover its buttons for a reminder.',
  'Right-click the Home button for deeper resets (origin, dimensions, rendering, layers).',
];

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

  const controlsList = document.createElement('div');
  controlsList.className = 'luxar-help-overlay__sections';

  for (const section of HELP_SECTIONS) {
    const sectionEl = document.createElement('section');
    sectionEl.className = 'luxar-help-overlay__section';

    const heading = document.createElement('div');
    heading.className = 'luxar-help-overlay__section-title';
    heading.innerHTML = section.icon;
    const headingText = document.createElement('span');
    headingText.textContent = section.title;
    heading.appendChild(headingText);
    sectionEl.appendChild(heading);

    if (section.note) {
      const note = document.createElement('div');
      note.className = 'luxar-help-overlay__section-note';
      note.textContent = section.note;
      sectionEl.appendChild(note);
    }

    for (const entry of section.entries) {
      const row = document.createElement('div');
      row.className = 'luxar-help-overlay__row';

      const keys = document.createElement('span');
      keys.className = 'luxar-help-overlay__keys';
      for (const key of entry.keys) {
        const kbd = document.createElement('kbd');
        kbd.textContent = key;
        keys.appendChild(kbd);
      }

      const desc = document.createElement('span');
      desc.className = 'luxar-help-overlay__desc';
      desc.textContent = entry.label;

      row.appendChild(keys);
      row.appendChild(desc);
      sectionEl.appendChild(row);
    }

    controlsList.appendChild(sectionEl);
  }

  // Tips — prose, no key chips.
  const tipsEl = document.createElement('section');
  tipsEl.className = 'luxar-help-overlay__section';
  const tipsHeading = document.createElement('div');
  tipsHeading.className = 'luxar-help-overlay__section-title';
  tipsHeading.innerHTML = RAIL_ICONS.cinematic;
  const tipsText = document.createElement('span');
  tipsText.textContent = 'Tips';
  tipsHeading.appendChild(tipsText);
  tipsEl.appendChild(tipsHeading);
  for (const tip of HELP_TIPS) {
    const tipEl = document.createElement('div');
    tipEl.className = 'luxar-help-overlay__tip';
    tipEl.textContent = tip;
    tipsEl.appendChild(tipEl);
  }
  controlsList.appendChild(tipsEl);

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

  // Function to close this specific help overlay. The identity check prevents
  // a stale handler from an older overlay from tearing down a newer one.
  const closeHelp = () => {
    if (document.getElementById('luxar-help-overlay') !== helpDiv) {
      document.removeEventListener('click', handleDocumentClick);
      return;
    }
    hideHelpOverlay();
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

  // Add global click listener after a short delay to prevent immediate closure.
  // Capture the timer identity so an obsolete callback cannot clear or attach
  // over the state of a subsequently opened overlay.
  const clickTimer = setTimeout(() => {
    if (activeHelpClickTimer === clickTimer) {
      activeHelpClickTimer = null;
    }
    if (
      activeHelpClickHandler === handleDocumentClick &&
      document.getElementById('luxar-help-overlay') === helpDiv
    ) {
      document.addEventListener('click', handleDocumentClick);
    }
  }, UI_CONFIG.timings.helpClickDelayMs);
  activeHelpClickTimer = clickTimer;
}

export function hideHelpOverlay() {
  if (activeHelpClickTimer !== null) {
    clearTimeout(activeHelpClickTimer);
    activeHelpClickTimer = null;
  }

  // Release focus trap before removing element
  if (activeHelpFocusTrapRelease) {
    activeHelpFocusTrapRelease();
    activeHelpFocusTrapRelease = null;
  }

  if (activeHelpClickHandler) {
    document.removeEventListener('click', activeHelpClickHandler);
    activeHelpClickHandler = null;
  }

  document.getElementById('luxar-help-overlay')?.remove();
}
