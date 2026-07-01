/**
 * ControlRail — a slim, always-visible vertical activity rail docked to the
 * left edge of the viewer.
 *
 * Luxar's UI is otherwise entirely keyboard-triggered (H, N, R, L, P, O, …),
 * which means a first-time visitor sees a bare canvas with no hint that any
 * controls exist. The rail fixes that: one discoverable, recognizable icon per
 * panel, each wired to the *exact same* command the keyboard shortcut fires
 * (via {@link InputHandler.getUiActions}), with a tooltip showing the shortcut.
 *
 * Design notes:
 * - Matches the viewer design tokens (`--luxar-*`), so it themes for all four
 *   themes automatically.
 * - Idle-dims after a few seconds of no pointer movement so it recedes into the
 *   immersive canvas, and wakes on pointer movement / hover.
 * - Reflects live open/closed state per panel (active highlight).
 * - Shows a one-time first-run hint (localStorage-gated) pointing at itself.
 *
 * @module ui/control-rail
 */

import { getViewerContainer } from '../utils/viewer-container';

/** One button in the rail. */
export interface ControlRailItem {
  /** Stable id (also used as a data attribute). */
  id: string;
  /** Human-readable name, shown in the tooltip + aria-label. */
  title: string;
  /** Keyboard shortcut shown in the tooltip (display only). */
  shortcut?: string;
  /** Inline SVG markup for the icon. */
  icon: string;
  /** Invoked on click — should be the same action as the shortcut. */
  activate: () => void;
  /** CSS selector whose visible presence means this item's panel is open. */
  openSelector?: string;
  /** Explicit active check (takes precedence over {@link openSelector}). */
  isActive?: () => boolean;
  /** Momentary action (e.g. screenshot) — never shows an active state. */
  momentary?: boolean;
  /** Insert a separator before this item. */
  separatorBefore?: boolean;
  /**
   * If set, this button opens a horizontal flyout of toggles instead of
   * firing {@link activate}. The parent button shows active when the flyout
   * is open or any toggle inside it is active.
   */
  flyout?: ControlRailToggle[];
}

/** A compact icon toggle shown inside a {@link ControlRailItem.flyout}. */
export interface ControlRailToggle {
  id: string;
  title: string;
  shortcut?: string;
  icon: string;
  activate: () => void;
  openSelector?: string;
  isActive?: () => boolean;
}

const HINT_STORAGE_KEY = 'luxar-control-rail-hint-dismissed';
const COLLAPSED_STORAGE_KEY = 'luxar-control-rail-collapsed';
const IDLE_MS = 2600;
/** Collapsed handle lingers a little longer, then fades to barely-visible. */
const COLLAPSED_IDLE_MS = 5000;

/** Chevron used by the collapse/expand handle (rotated via CSS when collapsed). */
const CHEVRON_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';

export class ControlRail {
  private readonly root: HTMLDivElement;
  private readonly container: HTMLElement;
  private readonly items: ControlRailItem[];
  private readonly buttons = new Map<string, HTMLButtonElement>();
  private hint?: HTMLDivElement;
  private idleTimer?: number;
  private refreshRaf?: number;
  private disposed = false;
  private collapsed = false;
  /** Open flyout descriptor + its DOM, or null when none is open. */
  private flyout?: { item: ControlRailItem; el: HTMLDivElement; btn: HTMLButtonElement };
  // Pointer movement anywhere wakes the (expanded) rail so it brightens while
  // the user is active. When collapsed or in fullscreen the rail is meant to
  // stay out of the way, so it reveals on *hover* only — not on any move.
  private readonly onContainerMove = (): void => {
    if (this.collapsed || this.root.classList.contains('is-fullscreen')) return;
    this.wake();
  };
  private readonly onFullscreenChange = (): void => this.syncFullscreen();
  private readonly onDocPointerDown = (e: PointerEvent): void => this.maybeCloseFlyout(e);
  private readonly onDocKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') this.closeFlyout();
    // Keyboard shortcuts (H, N, R, …) toggle panels — refresh active-state.
    this.scheduleRefresh();
  };
  // Any click may open/close a panel (a rail button, or a panel's own × button),
  // so refresh active-state after the interaction settles.
  private readonly onDocClick = (): void => this.scheduleRefresh();

  constructor(
    items: ControlRailItem[],
    /** Optional element docked at the rail's bottom (e.g. the perf readout). */
    private readonly footer?: HTMLElement
  ) {
    this.items = items;
    this.container = getViewerContainer();

    this.root = document.createElement('div');
    this.root.className = 'luxar-control-rail luxar-glass-surface is-awake';
    this.root.setAttribute('role', 'toolbar');
    this.root.setAttribute('aria-label', 'Viewer controls');
    this.root.setAttribute('aria-orientation', 'vertical');

    for (const item of items) {
      if (item.separatorBefore) {
        const sep = document.createElement('div');
        sep.className = 'luxar-control-rail__sep';
        this.root.appendChild(sep);
      }
      this.root.appendChild(this.buildButton(item));
    }

    // Docked footer (e.g. the performance readout): sits just below the last
    // item; its own visibility is controlled by its owner (the Performance
    // toggle). It stays put when the rail collapses (see the .is-collapsed
    // row layout), appearing to the right of the collapse handle.
    if (this.footer) {
      this.root.appendChild(this.footer);
    }

    // Collapse/expand handle — always visible so a collapsed rail can always
    // be brought back (never re-hides discoverability).
    this.root.appendChild(this.buildCollapseButton());

    // Keep global, canvas/body-focus-gated shortcuts (e.g. Space = fullscreen)
    // working after the rail is used with the mouse. Browsers focus a <button>
    // on pointer press, so after a click the button would hold focus and
    // swallow the next Space. preventDefault on mousedown is unreliable here
    // (Chrome focuses on pointerdown, which fires first), so instead we blur
    // the button after a *pointer* click (event.detail > 0) to return focus to
    // the canvas/body. Keyboard activation (Enter/Space → click with
    // detail === 0) keeps focus, so keyboard navigation is unaffected.
    // Delegated on the root so it also covers the collapse handle and flyout
    // chips (which bubble up here).
    this.root.addEventListener('click', (e) => {
      if (e.detail > 0) {
        (e.target as HTMLElement | null)?.closest('button')?.blur();
      }
    });

    this.container.appendChild(this.root);
    // Marker class lets left-anchored panels offset to clear the rail.
    this.container.classList.add('luxar-has-control-rail');

    // Restore persisted collapsed state.
    let startCollapsed = false;
    try {
      startCollapsed = localStorage.getItem(COLLAPSED_STORAGE_KEY) === '1';
    } catch {
      /* ignore */
    }
    this.setCollapsed(startCollapsed, false);

    // Idle-dim behaviour.
    this.container.addEventListener('pointermove', this.onContainerMove);
    // Hovering the rail (incl. the collapsed handle) always wakes it.
    this.root.addEventListener('pointerenter', () => this.wake());
    // In fullscreen the rail hides and only reveals on hover; restore on exit.
    // webkit* covers Safari < 16.4.
    document.addEventListener('fullscreenchange', this.onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', this.onFullscreenChange);
    // Close the flyout on outside click / Escape.
    document.addEventListener('pointerdown', this.onDocPointerDown, true);
    document.addEventListener('keydown', this.onDocKeyDown);
    // Refresh active-state on interactions that can toggle a panel (see onDoc*).
    document.addEventListener('click', this.onDocClick);
    this.syncFullscreen();
    this.scheduleSleep();

    // Reflect live panel open/closed state (event-driven — see scheduleRefresh).
    this.refresh();

    if (!startCollapsed) this.maybeShowHint();
  }

  private buildCollapseButton(): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'luxar-control-rail__btn luxar-control-rail__collapse';
    btn.setAttribute('aria-label', 'Collapse controls');
    btn.innerHTML = CHEVRON_ICON;
    const tip = document.createElement('span');
    tip.className = 'luxar-control-rail__tip';
    tip.setAttribute('role', 'tooltip');
    tip.textContent = 'Hide controls';
    btn.appendChild(tip);
    btn.addEventListener('click', () => this.setCollapsed(!this.collapsed, true));
    return btn;
  }

  /** Collapse to just the handle, or expand back to the full rail. */
  setCollapsed(collapsed: boolean, persist = true): void {
    this.collapsed = collapsed;
    if (collapsed) this.closeFlyout();
    this.root.classList.toggle('is-collapsed', collapsed);
    const handle = this.root.querySelector<HTMLButtonElement>('.luxar-control-rail__collapse');
    if (handle) {
      handle.setAttribute('aria-label', collapsed ? 'Show controls' : 'Collapse controls');
      const tip = handle.querySelector('.luxar-control-rail__tip');
      if (tip) tip.textContent = collapsed ? 'Show controls' : 'Hide controls';
    }
    if (collapsed) this.dismissHint();
    if (persist) {
      try {
        localStorage.setItem(COLLAPSED_STORAGE_KEY, collapsed ? '1' : '0');
      } catch {
        /* ignore */
      }
    }
    this.wake();
  }

  private buildButton(item: ControlRailItem): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'luxar-control-rail__btn';
    btn.dataset.railId = item.id;
    const label = item.shortcut ? `${item.title} (${item.shortcut})` : item.title;
    btn.setAttribute('aria-label', label);
    btn.innerHTML = item.icon;

    const tip = document.createElement('span');
    tip.className = 'luxar-control-rail__tip';
    tip.setAttribute('role', 'tooltip');
    tip.innerHTML = item.shortcut
      ? `${escapeHtml(item.title)}<kbd>${escapeHtml(item.shortcut)}</kbd>`
      : escapeHtml(item.title);
    btn.appendChild(tip);

    btn.addEventListener('click', () => {
      this.dismissHint();
      if (item.flyout) {
        this.toggleFlyout(item, btn);
        return;
      }
      try {
        item.activate();
      } catch {
        /* a panel toggle throwing must not break the rail */
      }
      // Active-state refresh is handled by the document-click listener.
    });

    this.buttons.set(item.id, btn);
    return btn;
  }

  private wake(): void {
    if (this.disposed) return;
    this.root.classList.add('is-awake');
    this.scheduleSleep();
  }

  private scheduleSleep(): void {
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    const delay = this.collapsed ? COLLAPSED_IDLE_MS : IDLE_MS;
    this.idleTimer = window.setTimeout(() => {
      // Stay awake while the pointer is over the rail itself.
      if (this.root.matches(':hover')) {
        this.scheduleSleep();
        return;
      }
      this.root.classList.remove('is-awake');
    }, delay);
  }

  /** Reflect fullscreen state — the rail hides (hover-to-reveal) in fullscreen. */
  private syncFullscreen(): void {
    if (this.disposed) return;
    const webkitEl = (document as Document & { webkitFullscreenElement?: Element | null })
      .webkitFullscreenElement;
    this.root.classList.toggle('is-fullscreen', !!(document.fullscreenElement || webkitEl));
    this.wake();
  }

  /** Coalesce active-state refreshes to one per frame. */
  private scheduleRefresh(): void {
    if (this.disposed || this.refreshRaf !== undefined) return;
    this.refreshRaf = requestAnimationFrame(() => {
      this.refreshRaf = undefined;
      this.refresh();
    });
  }

  private refresh(): void {
    if (this.disposed) return;
    for (const item of this.items) {
      if (item.momentary) continue;
      const btn = this.buttons.get(item.id);
      if (!btn) continue;
      if (item.flyout) {
        // A flyout button is "active" when its popover is open or any of its
        // toggles are on; also refresh each chip's own state.
        const anyOn = item.flyout.some((t) => this.isToggleActive(t));
        btn.classList.toggle('is-active', anyOn || this.flyout?.item === item);
        this.refreshFlyoutChips(item);
        continue;
      }
      btn.classList.toggle('is-active', this.isItemActive(item));
    }
  }

  private isItemActive(item: ControlRailItem): boolean {
    if (item.isActive) {
      try {
        return !!item.isActive();
      } catch {
        return false;
      }
    }
    return item.openSelector ? isPanelVisible(item.openSelector, this.container) : false;
  }

  private isToggleActive(t: ControlRailToggle): boolean {
    if (t.isActive) {
      try {
        return !!t.isActive();
      } catch {
        return false;
      }
    }
    return t.openSelector ? isPanelVisible(t.openSelector, this.container) : false;
  }

  private refreshFlyoutChips(item: ControlRailItem): void {
    if (this.flyout?.item !== item) return;
    for (const t of item.flyout ?? []) {
      const chip = this.flyout.el.querySelector<HTMLButtonElement>(`[data-toggle-id="${t.id}"]`);
      chip?.classList.toggle('is-active', this.isToggleActive(t));
    }
  }

  private toggleFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (this.flyout?.item === item) {
      this.closeFlyout();
    } else {
      this.openFlyout(item, btn);
    }
  }

  private openFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    this.closeFlyout();
    const el = document.createElement('div');
    el.className = 'luxar-control-rail__flyout luxar-glass-surface';
    el.setAttribute('role', 'menu');
    for (const t of item.flyout ?? []) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'luxar-control-rail__chip';
      chip.dataset.toggleId = t.id;
      chip.setAttribute('aria-label', t.shortcut ? `${t.title} (${t.shortcut})` : t.title);
      chip.innerHTML = t.icon;
      const tip = document.createElement('span');
      tip.className = 'luxar-control-rail__chip-tip';
      tip.setAttribute('role', 'tooltip');
      tip.innerHTML = t.shortcut
        ? `${escapeHtml(t.title)}<kbd>${escapeHtml(t.shortcut)}</kbd>`
        : escapeHtml(t.title);
      chip.appendChild(tip);
      chip.addEventListener('click', () => {
        try {
          t.activate();
        } catch {
          /* ignore */
        }
        // Active-state refresh is handled by the document-click listener.
      });
      el.appendChild(chip);
    }
    this.root.appendChild(el);
    // Align the flyout's vertical centre with the opening button.
    el.style.top = `${btn.offsetTop + btn.offsetHeight / 2}px`;
    // Flip the chip tooltips above when the flyout sits near the viewport bottom
    // (the View button is low in the rail), so they don't clip off-screen.
    if (el.getBoundingClientRect().bottom > window.innerHeight - 48) {
      el.classList.add('luxar-control-rail__flyout--up');
    }
    this.flyout = { item, el, btn };
    this.refresh();
    this.wake();
  }

  private closeFlyout(): void {
    if (!this.flyout) return;
    this.flyout.el.remove();
    this.flyout = undefined;
    this.refresh();
  }

  private maybeCloseFlyout(e: PointerEvent): void {
    if (!this.flyout) return;
    const target = e.target as Node | null;
    if (this.flyout.el.contains(target) || this.flyout.btn.contains(target)) return;
    this.closeFlyout();
  }

  private maybeShowHint(): void {
    let seen = false;
    try {
      seen = localStorage.getItem(HINT_STORAGE_KEY) === '1';
    } catch {
      /* private mode / storage blocked — just show it */
    }
    if (seen) return;

    const hint = document.createElement('div');
    hint.className = 'luxar-control-rail-hint';
    hint.innerHTML =
      '<button class="luxar-control-rail-hint__close" type="button" aria-label="Dismiss">&times;</button>' +
      '<b>New here?</b><br>Hover these controls, or press <kbd>H</kbd> — dimensions, rendering, layers &amp; more.';
    hint
      .querySelector('.luxar-control-rail-hint__close')
      ?.addEventListener('click', () => this.dismissHint());
    this.container.appendChild(hint);
    this.hint = hint;
  }

  private dismissHint(): void {
    if (!this.hint) return;
    try {
      localStorage.setItem(HINT_STORAGE_KEY, '1');
    } catch {
      /* ignore */
    }
    this.hint.remove();
    this.hint = undefined;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    if (this.refreshRaf !== undefined) cancelAnimationFrame(this.refreshRaf);
    this.closeFlyout();
    this.container.removeEventListener('pointermove', this.onContainerMove);
    document.removeEventListener('fullscreenchange', this.onFullscreenChange);
    document.removeEventListener('webkitfullscreenchange', this.onFullscreenChange);
    document.removeEventListener('pointerdown', this.onDocPointerDown, true);
    document.removeEventListener('keydown', this.onDocKeyDown);
    document.removeEventListener('click', this.onDocClick);
    this.container.classList.remove('luxar-has-control-rail');
    this.hint?.remove();
    this.root.remove();
    this.buttons.clear();
  }
}

/** A panel counts as "open" when its root element is present and rendered. */
function isPanelVisible(selector: string, scope: ParentNode): boolean {
  const el = scope.querySelector(selector) ?? document.querySelector(selector);
  if (!(el instanceof HTMLElement)) return false;
  if (el.getClientRects().length === 0) return false;
  const s = getComputedStyle(el);
  return s.visibility !== 'hidden' && Number(s.opacity) > 0.01;
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c
  );
}

/**
 * Default icon set (inline SVG, currentColor stroke) keyed by rail item id,
 * kept here so the pipeline that wires actions stays readable.
 */
export const RAIL_ICONS: Record<string, string> = {
  help: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7"/><line x1="12" y1="17" x2="12" y2="17.01"/></svg>',
  dims: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/></svg>',
  render:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2.2"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2.2"/></svg>',
  layers:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  perf: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18a8 8 0 0 1 16 0"/><line x1="12" y1="18" x2="16" y2="11"/></svg>',
  data: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h5l2 2h9v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7z"/></svg>',
  monitor:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2 6 4-13 2 7h6"/></svg>',
  recording:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
  screenshot:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>',
  logs: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3"/><line x1="13" y1="15" x2="17" y2="15"/></svg>',
  view: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>',
  scalebar:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="9" width="18" height="6" rx="1"/><path d="M7 9v3M11 9v4M15 9v3M19 9v4"/></svg>',
  legend:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="4" width="5" height="16" rx="1"/><line x1="13" y1="6" x2="18" y2="6"/><line x1="13" y1="12" x2="18" y2="12"/><line x1="13" y1="18" x2="18" y2="18"/></svg>',
  overlays:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="12" height="12" rx="1"/><rect x="9" y="9" width="11" height="11" rx="1"/></svg>',
  cinematic:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l7-7"/><path d="M15 3l1.2 3.3L19.5 7.5l-3.3 1.2L15 12l-1.2-3.3L10.5 7.5l3.3-1.2z"/></svg>',
};
