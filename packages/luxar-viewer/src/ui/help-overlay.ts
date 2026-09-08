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
 * Initial focus goes to the overlay CONTAINER, not the filter field: a
 * focused text input trips `InputHandler`'s typing guard, which would swallow
 * the second `H` and make the "toggle" one-way (issue #1922). Type-to-filter
 * still starts on the first printable keystroke via `installTypeToFilter`.
 *
 * Module-level handles for the delayed click-listener registration, the
 * listener itself, the focus-trap release and the type-to-filter release live
 * here so hideHelpOverlay() can tear them down even when it's called from
 * outside (e.g. by ui-cleanup).
 */

import { trapFocus } from './help-overlay/focus-trap';
import { ALWAYS_GLOBAL_KEYS, installTypeToFilter } from './help-overlay/type-to-filter';
import { getViewerContainer } from '../utils/viewer-container';
import { RAIL_ICONS } from './control-rail/icons';
import { getInputProfile } from '../utils/input-capabilities';
import type { RegisteredShortcutBindings, ShortcutHelpSectionId } from '../types/shortcut-help';
import { config } from '../config';

const UI_CONFIG = config.ui;

let activeHelpClickHandler: ((event: MouseEvent) => void) | null = null;
let activeHelpClickTimer: ReturnType<typeof setTimeout> | null = null;
let activeHelpFocusTrapRelease: (() => void) | null = null;
let activeHelpTypeToFilterRelease: (() => void) | null = null;

/** One shortcut row: chip text(s) + what they do. */
interface HelpEntry {
  keys: string[];
  label: string;
  order: number;
}

interface HelpSection {
  id?: ShortcutHelpSectionId;
  title: string;
  icon: string;
  note?: string;
  /** Rendered only when the device reports touch points (phones, tablets). */
  touchOnly?: boolean;
  entries: HelpEntry[];
}

const HELP_SECTIONS: HelpSection[] = [
  {
    id: 'basics',
    title: 'Basics',
    icon: RAIL_ICONS.navOrbit,
    entries: [
      { keys: ['Drag'], label: 'Pan camera', order: 1 },
      { keys: ['Right drag'], label: 'Rotate view', order: 2 },
      { keys: ['⇧', 'Drag'], label: 'Rotate view (alternative)', order: 3 },
      { keys: ['Wheel'], label: 'Zoom in / out', order: 4 },
      { keys: ['⇧', 'Wheel'], label: 'Roll around the view axis', order: 5 },
      { keys: ['Click'], label: 'Open the hovered element link', order: 6 },
      { keys: ['Right click'], label: 'Actions for the hovered element', order: 7 },
    ],
  },
  {
    title: 'Touch',
    icon: RAIL_ICONS.navOrbit,
    note: 'Phones and tablets — in ortho mode one finger pans and twist does not roll',
    touchOnly: true,
    entries: [
      { keys: ['1 finger'], label: 'Rotate (fly mode: look around)', order: 1 },
      { keys: ['2 fingers'], label: 'Pan (fly mode: strafe)', order: 2 },
      { keys: ['Pinch'], label: 'Zoom (fly mode: move forward / back)', order: 3 },
      { keys: ['Twist'], label: 'Roll around the view axis', order: 4 },
    ],
  },
  {
    id: 'fly',
    title: 'Fly mode',
    icon: RAIL_ICONS.navFly,
    note: 'Press V until the fly icon shows',
    entries: [
      { keys: ['Drag'], label: 'Strafe (pan camera)', order: 70 },
      { keys: ['Right drag'], label: 'Free look', order: 80 },
      { keys: ['Wheel'], label: 'Move forward / backward', order: 90 },
    ],
  },
  {
    title: 'Ortho mode',
    icon: RAIL_ICONS.navOrtho,
    note: 'Press V until the grid icon shows — 2D viewing, no rotation',
    entries: [
      { keys: ['Drag'], label: 'Pan camera', order: 1 },
      { keys: ['Wheel'], label: 'Zoom in / out', order: 2 },
      { keys: ['⇧', 'Wheel'], label: 'Roll around the view axis', order: 3 },
    ],
  },
  {
    id: 'dimensions',
    title: 'nD navigation',
    icon: RAIL_ICONS.dims,
    entries: [
      {
        keys: ['Wheel'],
        label: 'On a slider: step (⇧ fine, ⌃ coarse, ⌃⇧ extra-fine)',
        order: 30,
      },
    ],
  },
  {
    id: 'panels',
    title: 'Panels & tools',
    icon: RAIL_ICONS.settings,
    entries: [
      { keys: ['Ctrl/⌘', 'Wheel'], label: 'Adjust field of view (perspective)', order: 115 },
    ],
  },
];

/**
 * The sections this device shows. Keyboard-and-mouse machines never see the
 * touch section, so the desktop help text is unchanged by it.
 */
function visibleHelpSections(): HelpSection[] {
  const hasTouch = getInputProfile().touchPoints > 0;
  return HELP_SECTIONS.filter((section) => !section.touchOnly || hasTouch);
}

function getRegisteredHelpEntries(
  bindings: RegisteredShortcutBindings
): Map<ShortcutHelpSectionId, HelpEntry[]> {
  const entries = new Map<ShortcutHelpSectionId, HelpEntry[]>();
  const groups = new Set<string>();
  for (const contextBindings of bindings.values()) {
    for (const binding of contextBindings) {
      if (!binding.help || groups.has(binding.help.group)) continue;
      groups.add(binding.help.group);
      const sectionEntries = entries.get(binding.help.section) ?? [];
      sectionEntries.push({
        keys: binding.help.keys ? [...binding.help.keys] : [binding.shortcutLabel ?? binding.key],
        label: binding.description,
        order: binding.help.order,
      });
      entries.set(binding.help.section, sectionEntries);
    }
  }
  return entries;
}

/** Closing prose tips (no key chips). */
const HELP_TIPS: string[] = [
  'Try fly mode (V) for exploration, with inertial mode (I) for smooth coasting.',
  'The left rail mirrors every panel shortcut — hover its buttons for a reminder.',
  'Right-click the Home button for deeper resets (origin, dimensions, rendering, layers).',
];

/**
 * Build and mount the keyboard-shortcut help overlay, rendering the grouped
 * shortcut sections and closing tips defined in this module.
 *
 * Idempotent: if an overlay is already open it returns immediately rather than
 * stacking a second one. Sets up a focus trap and a delayed
 * click-outside-to-dismiss handler (the delay avoids catching the same click
 * that opened it); {@link hideHelpOverlay} tears both down.
 */
export function showHelpOverlay(bindings: RegisteredShortcutBindings) {
  // Prevent opening multiple overlays - if one exists, do nothing
  const existingHelp = document.getElementById('luxar-help-overlay');
  if (existingHelp) {
    return; // Don't create a new one, just return
  }

  const helpDiv = document.createElement('div');
  helpDiv.id = 'luxar-help-overlay';
  helpDiv.className = 'luxar-help-overlay luxar-glass-surface luxar-panel-pop';
  helpDiv.setAttribute('role', 'dialog');
  helpDiv.setAttribute('aria-modal', 'true');
  helpDiv.setAttribute('aria-labelledby', 'luxar-help-overlay-title');

  // Create header with title and close button
  const header = document.createElement('div');
  header.className = 'luxar-help-overlay__header luxar-panel-header';

  const title = document.createElement('h3');
  title.id = 'luxar-help-overlay-title';
  title.className = 'luxar-help-overlay__title';
  title.textContent = 'Luxar Controls & Shortcuts';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'luxar-help-overlay__close-btn luxar-panel-close';
  // Stroke ✕ in the rail icon contract (was the text glyph '×').
  closeBtn.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12"/><path d="M18 6L6 18"/></svg>';
  closeBtn.title = 'Close (Escape)';
  closeBtn.setAttribute('aria-label', 'Close help overlay');
  closeBtn.setAttribute('aria-keyshortcuts', 'Escape');

  header.appendChild(title);
  header.appendChild(closeBtn);

  // Type-to-filter across shortcut keys + descriptions (pinned under the
  // header, outside the scroll area). Sections with no surviving rows hide.
  const filterWrap = document.createElement('div');
  filterWrap.className = 'luxar-help-overlay__filter luxar-panel-filter';
  const filterIcon = document.createElement('span');
  filterIcon.className = 'luxar-panel-filter__icon';
  filterIcon.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="M15.8 15.8L21 21"/></svg>';
  filterIcon.setAttribute('aria-hidden', 'true');
  const filterInput = document.createElement('input');
  filterInput.type = 'text';
  filterInput.className = 'luxar-panel-filter__input';
  filterInput.placeholder = 'Filter shortcuts…';
  filterInput.setAttribute('aria-label', 'Filter keyboard shortcuts');
  filterInput.autocomplete = 'off';
  filterWrap.appendChild(filterIcon);
  filterWrap.appendChild(filterInput);

  const controlsList = document.createElement('div');
  controlsList.className = 'luxar-help-overlay__sections';

  const registeredEntries = getRegisteredHelpEntries(bindings);

  for (const section of visibleHelpSections()) {
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

    const entries = [
      ...section.entries,
      ...(section.id ? (registeredEntries.get(section.id) ?? []) : []),
    ].sort((a, b) => a.order - b.order);

    for (const entry of entries) {
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

  // Empty-state note for the filter (hidden until a query matches nothing).
  const noMatches = document.createElement('div');
  noMatches.className = 'luxar-help-overlay__no-matches';
  noMatches.textContent = 'No shortcuts match.';
  noMatches.style.display = 'none';
  controlsList.appendChild(noMatches);

  const applyHelpFilter = (): void => {
    const q = filterInput.value.trim().toLowerCase();
    let anyVisible = false;
    for (const section of Array.from(
      controlsList.querySelectorAll<HTMLElement>('.luxar-help-overlay__section')
    )) {
      let sectionVisible = false;
      for (const item of Array.from(
        section.querySelectorAll<HTMLElement>('.luxar-help-overlay__row, .luxar-help-overlay__tip')
      )) {
        const match = !q || (item.textContent ?? '').toLowerCase().includes(q);
        item.style.display = match ? '' : 'none';
        if (match) sectionVisible = true;
      }
      section.style.display = sectionVisible ? '' : 'none';
      if (sectionVisible) anyVisible = true;
    }
    noMatches.style.display = anyVisible ? 'none' : '';
  };
  filterInput.addEventListener('input', applyHelpFilter);
  // Keystrokes stay local (typing 'v' must not switch camera modes). Escape
  // with a query clears it; an empty Escape falls through and closes the
  // overlay as before.
  filterInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && filterInput.value) {
      e.stopPropagation();
      filterInput.value = '';
      applyHelpFilter();
      return;
    }
    if (!ALWAYS_GLOBAL_KEYS.has(e.key)) e.stopPropagation();
  });

  // Add footer note
  const footerNote = document.createElement('div');
  footerNote.className = 'luxar-help-overlay__footer';
  footerNote.textContent = 'Click anywhere or press Esc to close';

  // Scroll wrapper (separates scrolling from the glass-effect container, UI
  // Design Guide §7.4). The header stays OUTSIDE it, pinned at the top of
  // the panel — only the sections and footer scroll.
  const scrollWrapper = document.createElement('div');
  scrollWrapper.className = 'luxar-help-overlay__scroll';
  scrollWrapper.appendChild(controlsList);
  scrollWrapper.appendChild(footerNote);

  helpDiv.appendChild(header);
  helpDiv.appendChild(filterWrap);

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

  // Trap focus within the help overlay. `autoFocusFirst: false` because
  // installTypeToFilter below parks focus on the panel container instead —
  // leaving the trap's own 0ms timer armed would fight it (and land focus on
  // the close button).
  activeHelpFocusTrapRelease = trapFocus(helpDiv, { autoFocusFirst: false });
  // Container focus + first-printable-key forwarding into the filter. No
  // text field holds focus, so a second `H` survives InputHandler's typing
  // guard and closes the overlay; `H` itself is passed through for exactly
  // that reason and so cannot be the first character of a filter query.
  activeHelpTypeToFilterRelease = installTypeToFilter(helpDiv, () => filterInput, {
    passthroughKeys: [config.input.keyboard.shortcuts.toggleHelp],
  });

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

/**
 * Dismiss the help overlay opened by {@link showHelpOverlay}.
 *
 * Cancels the pending click-outside timer, releases the type-to-filter
 * forwarder and the focus trap, and removes the overlay element, restoring
 * focus to where it was before the overlay opened. A no-op when no overlay is
 * present.
 */
export function hideHelpOverlay() {
  if (activeHelpClickTimer !== null) {
    clearTimeout(activeHelpClickTimer);
    activeHelpClickTimer = null;
  }
  // Release the type-to-filter forwarder before the trap: the trap's cleanup
  // restores focus to the pre-open element, and the forwarder must not be
  // listening on a detached container afterwards.
  if (activeHelpTypeToFilterRelease) {
    activeHelpTypeToFilterRelease();
    activeHelpTypeToFilterRelease = null;
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
