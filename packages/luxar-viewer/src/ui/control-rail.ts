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
import { isDocumentFullscreen } from '../utils/fullscreen';

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
  /**
   * If set, this button can open a vertical panel popover hosting rich
   * controls (sliders/dropdowns), built lazily into a host element. Unlike a
   * {@link flyout} (a row of toggle chips), a popover holds arbitrary content.
   * The trigger decides how it opens relative to {@link activate}:
   * - `'click'`: primary click opens the popover ({@link activate} unused).
   * - `'context'`: right-click opens the popover; primary click still fires
   *   {@link activate} (e.g. Performance: left-click toggles the readout,
   *   right-click opens the DPR controls).
   */
  popover?: ControlRailPopover;
  /**
   * Optional per-refresh hook to sync the button's icon/label/tooltip from live
   * state (e.g. the Navigation button reflecting the current control mode).
   * Called on every {@link ControlRail} refresh with the button element.
   */
  render?: (btn: HTMLButtonElement) => void;
  /**
   * Optional predicate for a disabled (grayed, non-interactive) state — e.g.
   * the Layers button when the scene has no layers. Re-evaluated on every
   * refresh; when true the button gets the native `disabled` attribute so it
   * can't be clicked or focused. Fire a refresh (see {@link ControlRail}
   * event listeners) when the underlying condition changes.
   */
  disabled?: () => boolean;
}

/** A rich, lazily-built panel popover anchored to a {@link ControlRailItem}. */
export interface ControlRailPopover {
  /**
   * Populate the popover body. Called each time the popover opens (rebuilt
   * fresh so it always reflects live state). May return a teardown callback
   * run when the popover closes (e.g. to clear intervals / dispose a GUI).
   */
  build: (host: HTMLElement) => void | (() => void);
  /** How the popover opens: on primary click, or on right-click. */
  trigger: 'click' | 'context';
  /** Accessible label for the popover group (defaults to the item title). */
  title?: string;
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

/**
 * How many rails currently mark `document.body`. The rail-clearance marker
 * class is shared/global, so it's reference-counted: only the last rail to
 * dispose removes it, otherwise tearing down one viewer instance would strip
 * the offset from every other live instance's left-anchored panels.
 */
let bodyMarkerRefs = 0;
const BODY_MARKER_CLASS = 'luxar-has-control-rail';

/** Chevron used by the collapse/expand handle (rotated via CSS when collapsed). */
const CHEVRON_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';

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
  /**
   * The currently-open overlay (chip flyout OR panel popover) + its DOM, or
   * undefined when none is open. Only one overlay is open at a time. `dispose`
   * is the popover builder's teardown (undefined for chip flyouts).
   */
  private overlay?: {
    item: ControlRailItem;
    el: HTMLDivElement;
    btn: HTMLButtonElement;
    kind: 'flyout' | 'popover';
    dispose?: () => void;
  };
  // Pointer movement anywhere wakes the (expanded) rail so it brightens while
  // the user is active. When collapsed or in fullscreen the rail is meant to
  // stay out of the way, so it reveals on *hover* only — not on any move.
  private readonly onContainerMove = (): void => {
    if (this.collapsed || this.root.classList.contains('is-fullscreen')) return;
    this.wake();
  };
  private readonly onFullscreenChange = (): void => this.syncFullscreen();
  // External state changes that must re-sync button state without a user
  // interaction: a scene load / layer change flips the Layers disabled
  // predicate; a control-mode switch (V key, rail cycle, or popover selector)
  // changes the Navigation button's icon/tooltip.
  private readonly onExternalStateChange = (): void => this.scheduleRefresh();
  private readonly onDocPointerDown = (e: PointerEvent): void => this.maybeCloseOverlay(e);
  private readonly onDocKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') this.closeOverlay();
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
    // `button, [tabindex]` — not just `button`: the docked perf readout is a
    // focusable <div tabindex="0">, and leaving focus on it would swallow the
    // next Space (fullscreen) just like a focused button does.
    this.root.addEventListener('click', (e) => {
      if (e.detail > 0) {
        (e.target as HTMLElement | null)?.closest<HTMLElement>('button, [tabindex]')?.blur();
      }
    });

    // Suppress the native browser context menu anywhere on the rail (buttons,
    // separators, background, flyout, popover) — right-click is our own
    // affordance, and a leaked browser menu over the rail looks broken. For a
    // button that owns a 'context'-trigger popover, open it instead. Delegated
    // here (not per-button) so EVERY right-click on the rail is captured.
    this.root.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const btnEl = (e.target as HTMLElement | null)?.closest<HTMLButtonElement>(
        '.luxar-control-rail__btn'
      );
      const item = btnEl ? this.items.find((it) => it.id === btnEl.dataset.railId) : undefined;
      if (item?.popover?.trigger === 'context' && !btnEl?.disabled) {
        this.dismissHint();
        this.togglePopover(item, btnEl!);
      }
    });

    this.container.appendChild(this.root);

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
    // Keyboard focus entering the rail must reveal it too — otherwise a Tab
    // into the rail while it's idle-dimmed, collapsed, or (opacity:0) in
    // fullscreen lands on an invisible control with an invisible focus ring
    // (WCAG 2.4.7). scheduleSleep() keeps it awake while focus stays within.
    this.root.addEventListener('focusin', () => this.wake());
    // In fullscreen the rail hides and only reveals on hover; restore on exit.
    // webkit* covers Safari < 16.4.
    document.addEventListener('fullscreenchange', this.onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', this.onFullscreenChange);
    // Close the flyout on outside click / Escape.
    document.addEventListener('pointerdown', this.onDocPointerDown, true);
    document.addEventListener('keydown', this.onDocKeyDown);
    // Refresh active-state on interactions that can toggle a panel (see onDoc*).
    document.addEventListener('click', this.onDocClick);
    // Refresh on external state changes that don't originate from a click:
    // layer population (Layers disabled state) and control-mode switches
    // (Navigation icon/tooltip).
    window.addEventListener('luxar-layers-changed', this.onExternalStateChange);
    window.addEventListener('luxar-control-mode-changed', this.onExternalStateChange);
    this.syncFullscreen();
    this.scheduleSleep();

    // Reflect live panel open/closed state (event-driven — see scheduleRefresh).
    this.refresh();

    if (!startCollapsed) this.maybeShowHint();

    // Marker class lets left-anchored panels offset to clear the rail. Applied
    // to document.body (the universal ancestor) rather than the viewer container,
    // because some panels mount to body (layers, debug console) while others
    // mount into the container (gui) — an embedder with a scoped (non-body)
    // container would otherwise miss the body-mounted panels. Reference-counted
    // (bodyMarkerRefs) so multiple instances share safely. Done LAST so a throw
    // anywhere above cannot leak a ref (the instance is never returned/disposed).
    bodyMarkerRefs += 1;
    document.body.classList.add(BODY_MARKER_CLASS);
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
    if (collapsed) this.closeOverlay();
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
    if (item.flyout || item.popover) {
      // The button opens a popover (chip group or panel); advertise + track it.
      btn.setAttribute('aria-haspopup', 'true');
      btn.setAttribute('aria-expanded', 'false');
    }
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
      // A 'click'-trigger popover opens on primary click; a 'context'-trigger
      // popover leaves primary click for activate() (opens on right-click).
      if (item.popover?.trigger === 'click') {
        this.togglePopover(item, btn);
        return;
      }
      try {
        item.activate();
      } catch {
        /* a panel toggle throwing must not break the rail */
      }
      // Active-state refresh is handled by the document-click listener.
    });

    // Right-click handling is delegated on the rail root (see constructor) so
    // the native browser menu is suppressed across the WHOLE rail, not just on
    // the two buttons that open a context popover.

    // Reflect initial dynamic icon/label state (e.g. Navigation mode).
    item.render?.(btn);

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
      // Stay awake while the pointer is over the rail, or keyboard focus is
      // within it (so a focused control never dims out from under the user).
      if (this.root.matches(':hover') || this.root.matches(':focus-within')) {
        this.scheduleSleep();
        return;
      }
      this.root.classList.remove('is-awake');
    }, delay);
  }

  /** Reflect fullscreen state — the rail hides (hover-to-reveal) in fullscreen. */
  private syncFullscreen(): void {
    if (this.disposed) return;
    this.root.classList.toggle('is-fullscreen', isDocumentFullscreen());
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
      const btn = this.buttons.get(item.id);
      if (!btn) continue;
      // Sync any dynamic icon/label (e.g. Navigation mode) before active-state.
      item.render?.(btn);
      // Disabled state (e.g. Layers with no layers): native `disabled` so the
      // button is grayed, unfocusable, and can't be clicked.
      if (item.disabled) {
        const isDisabled = item.disabled();
        btn.disabled = isDisabled;
        if (isDisabled) {
          // A disabled control shouldn't also read as active/open.
          btn.classList.remove('is-active');
          continue;
        }
      }
      if (item.momentary) continue;
      if (item.flyout) {
        // A flyout button is "active" when its popover is open or any of its
        // toggles are on; also refresh each chip's own state.
        const anyOn = item.flyout.some((t) => this.isToggleActive(t));
        btn.classList.toggle('is-active', anyOn || this.overlay?.item === item);
        this.refreshFlyoutChips(item);
        continue;
      }
      // A panel-popover button is active when its own isActive() is true (e.g.
      // Performance readout visible) or its popover is currently open.
      btn.classList.toggle('is-active', this.isItemActive(item) || this.overlay?.item === item);
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
    if (this.overlay?.item !== item || this.overlay.kind !== 'flyout') return;
    for (const t of item.flyout ?? []) {
      const chip = this.overlay.el.querySelector<HTMLButtonElement>(`[data-toggle-id="${t.id}"]`);
      if (!chip) continue;
      const on = this.isToggleActive(t);
      chip.classList.toggle('is-active', on);
      chip.setAttribute('aria-pressed', String(on));
    }
  }

  private toggleFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (this.overlay?.item === item) {
      this.closeOverlay();
    } else {
      this.openFlyout(item, btn);
    }
  }

  private openFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    this.closeOverlay();
    const el = document.createElement('div');
    el.className = 'luxar-control-rail__flyout luxar-glass-surface';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', `${item.title} options`);
    // Pointer arrow as a real child (not ::before): the glass themes claim the
    // ::before/::after pseudo-elements of every .luxar-glass-surface, which
    // would otherwise clobber a pseudo-element arrow (liquid-glass regression).
    const arrow = document.createElement('span');
    arrow.className = 'luxar-control-rail__flyout-arrow';
    arrow.setAttribute('aria-hidden', 'true');
    el.appendChild(arrow);
    for (const t of item.flyout ?? []) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'luxar-control-rail__chip';
      chip.dataset.toggleId = t.id;
      chip.setAttribute('aria-label', t.shortcut ? `${t.title} (${t.shortcut})` : t.title);
      chip.setAttribute('aria-pressed', String(this.isToggleActive(t)));
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
    btn.setAttribute('aria-expanded', 'true');
    this.overlay = { item, el, btn, kind: 'flyout' };
    this.refresh();
    this.wake();
  }

  private togglePopover(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (this.overlay?.item === item) {
      this.closeOverlay();
    } else {
      this.openPopover(item, btn);
    }
  }

  /**
   * Open a vertical panel popover anchored to `btn`, hosting the rich controls
   * built by {@link ControlRailPopover.build}. Mirrors the flyout's glass
   * surface + real-child arrow, but flows content vertically and vertically
   * clamps within the viewport (popovers can be tall).
   */
  private openPopover(item: ControlRailItem, btn: HTMLButtonElement): void {
    this.closeOverlay();
    const popover = item.popover;
    if (!popover) return;

    const el = document.createElement('div');
    el.className = 'luxar-control-rail__popover luxar-glass-surface';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', `${popover.title ?? item.title}`);
    // Arrow as a real child (not ::before) — the glass themes claim
    // .luxar-glass-surface::before/::after (see openFlyout).
    const arrow = document.createElement('span');
    arrow.className = 'luxar-control-rail__popover-arrow';
    arrow.setAttribute('aria-hidden', 'true');
    el.appendChild(arrow);

    const body = document.createElement('div');
    body.className = 'luxar-control-rail__popover-body';
    el.appendChild(body);

    this.root.appendChild(el);

    // Build the content; capture any teardown for closeOverlay().
    let dispose: (() => void) | undefined;
    try {
      const teardown = popover.build(body);
      if (typeof teardown === 'function') dispose = teardown;
    } catch {
      /* a popover builder throwing must not break the rail */
    }

    // Anchor near the button, then clamp inside the viewport (measured after
    // content is built so the height is real). Prefer aligning the popover top
    // with the button; if it would overflow the bottom, shift it up.
    el.style.top = `${btn.offsetTop}px`;
    const rect = el.getBoundingClientRect();
    const margin = 8;
    if (rect.bottom > window.innerHeight - margin) {
      const shift = rect.bottom - (window.innerHeight - margin);
      el.style.top = `${Math.max(margin, btn.offsetTop - shift)}px`;
    }
    // Keep the arrow pointing at the button even after a vertical shift.
    const arrowTop = btn.offsetTop + btn.offsetHeight / 2 - el.offsetTop;
    arrow.style.top = `${arrowTop}px`;

    btn.setAttribute('aria-expanded', 'true');
    this.overlay = { item, el, btn, kind: 'popover', dispose };
    this.refresh();
    this.wake();
  }

  private closeOverlay(): void {
    if (!this.overlay) return;
    const { btn, el, dispose } = this.overlay;
    btn.setAttribute('aria-expanded', 'false');
    // If keyboard focus is inside the overlay (e.g. closing via Escape while a
    // chip/control is focused), return it to the opener instead of dropping it
    // to <body> — otherwise the user loses their place in the tab order.
    const focusWasInside = el.contains(document.activeElement);
    try {
      dispose?.();
    } catch {
      /* teardown errors must not leave the overlay half-open */
    }
    el.remove();
    this.overlay = undefined;
    if (focusWasInside) btn.focus();
    this.refresh();
  }

  private maybeCloseOverlay(e: PointerEvent): void {
    if (!this.overlay) return;
    const target = e.target as Node | null;
    if (this.overlay.el.contains(target) || this.overlay.btn.contains(target)) return;
    this.closeOverlay();
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
    // Announce the one-time hint to assistive tech. It's injected once and never
    // updated, so role=status (a polite live region) reads it once without spam.
    hint.setAttribute('role', 'status');
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
    this.closeOverlay();
    this.container.removeEventListener('pointermove', this.onContainerMove);
    document.removeEventListener('fullscreenchange', this.onFullscreenChange);
    document.removeEventListener('webkitfullscreenchange', this.onFullscreenChange);
    document.removeEventListener('pointerdown', this.onDocPointerDown, true);
    document.removeEventListener('keydown', this.onDocKeyDown);
    document.removeEventListener('click', this.onDocClick);
    window.removeEventListener('luxar-layers-changed', this.onExternalStateChange);
    window.removeEventListener('luxar-control-mode-changed', this.onExternalStateChange);
    bodyMarkerRefs = Math.max(0, bodyMarkerRefs - 1);
    if (bodyMarkerRefs === 0) document.body.classList.remove(BODY_MARKER_CLASS);
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
  monitor: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2 6 4-13 2 7h6"/></svg>',
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
  // Navigation modes — the rail button swaps between these to mirror the live
  // camera control type (orbit / fly / ortho). Same 24×24 / currentColor style.
  navOrbit:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/></svg>',
  navFly:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11l18-7-7 18-2.5-8L3 11z"/></svg>',
  // Ortho: a 2×2 quadrant grid — the classic orthographic multi-view glyph.
  // Deliberately NOT a cube (which would collide with the dimensions icon).
  navOrtho:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="6.5" height="6.5" rx="1"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1"/></svg>',
  // Settings (gear) — houses theme + other viewer-wide preferences.
  settings:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
};
