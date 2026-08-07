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
 * - Shows a one-time first-run hint (localStorage-gated) pointing at itself;
 *   it auto-fades after a few seconds and any click/keypress dismisses it.
 *
 * The flyout + panel-popover lifecycle lives in {@link RailOverlay}
 * (control-rail/rail-overlay.ts); this class owns the buttons, idle/collapse/fullscreen
 * behaviour, and the active-state refresh that reads the overlay's open item.
 *
 * @module ui/control-rail
 */

import { getViewerContainer } from '../utils/viewer-container';
import { isDocumentFullscreen } from '../utils/fullscreen';
import { RailOverlay } from './control-rail/rail-overlay';
import { isPanelVisible, escapeHtml } from './control-rail/dom-helpers';
import type { ControlRailItem } from './control-rail/types';

/**
 * Rail item descriptors, re-exported so callers configuring a rail need only
 * import from `ui/control-rail`: {@link ControlRailItem} (one icon button) plus
 * the two optional shapes it composes — `ControlRailPopover` (a rich, lazily
 * built panel anchored to the item) and `ControlRailToggle` (a compact toggle
 * shown inside the item's flyout). `RAIL_ICONS` rides along for the same
 * reason, though `help-overlay.ts` and `rail-panels/home-popover.ts` still
 * reach it directly via `control-rail/icons` — keep this re-export either way.
 */
export type { ControlRailItem, ControlRailPopover, ControlRailToggle } from './control-rail/types';
export { RAIL_ICONS } from './control-rail/icons';

const HINT_STORAGE_KEY = 'luxar-control-rail-hint-dismissed';
/** First-run hint fades away on its own if the user never interacts. */
const HINT_AUTO_HIDE_MS = 10_000;
/** Matches the .is-leaving opacity transition in control-rail.css. */
const HINT_FADE_MS = 400;
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
  /** Flyout + panel-popover lifecycle (only one open at a time). */
  private readonly overlay: RailOverlay;
  private hint?: HTMLDivElement;
  private hintAutoHideTimer?: number;
  private hintFadeTimer?: number;
  private idleTimer?: number;
  private refreshRaf?: number;
  private disposed = false;
  private collapsed = false;
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
  private readonly onDocPointerDown = (e: PointerEvent): void => {
    // Any press anywhere (canvas, a panel, the rail itself) means the user is
    // already interacting — the first-run hint has served its purpose.
    this.dismissHint();
    this.overlay.maybeCloseOnPointer(e);
  };
  private readonly onDocKeyDown = (e: KeyboardEvent): void => {
    // Same for any keypress (e.g. the H the hint itself suggests).
    this.dismissHint();
    if (e.key === 'Escape') this.overlay.close();
    // Keyboard shortcuts (H, N, R, …) toggle panels — refresh active-state.
    this.scheduleRefresh();
  };
  // Any click may open/close a panel (a rail button, or a panel's own × button),
  // so refresh active-state after the interaction settles.
  private readonly onDocClick = (): void => this.scheduleRefresh();

  /**
   * Build the rail DOM, wire the global listeners, and mount it in the viewer
   * container. Items are rendered in the order given; the rail starts collapsed
   * if the user left it that way last session.
   */
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

    // Overlay owns the flyout/popover DOM; it re-syncs the rail on open/close.
    this.overlay = new RailOverlay({
      root: this.root,
      container: this.container,
      onOverlayChange: () => this.refresh(),
      wake: () => this.wake(),
    });

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
        this.overlay.togglePopover(item, btnEl!);
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
    // Close the overlay on outside click / Escape.
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

  /** The chevron handle that collapses the rail to a stub and expands it back. */
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
    if (collapsed) this.overlay.close();
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

  /** Build one rail button, including its tooltip and flyout/popover wiring. */
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
        this.overlay.toggleFlyout(item, btn);
        return;
      }
      // A 'click'-trigger popover opens on primary click; a 'context'-trigger
      // popover leaves primary click for activate() (opens on right-click).
      if (item.popover?.trigger === 'click') {
        this.overlay.togglePopover(item, btn);
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
    this.safeRender(item, btn);

    this.buttons.set(item.id, btn);
    return btn;
  }

  /**
   * Run an item's optional `render` hook fail-soft — a throwing hook must not
   * abort the refresh loop for the other buttons (matches the guarded
   * `activate`/`isActive` callbacks elsewhere in this class).
   */
  private safeRender(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (!item.render) return;
    try {
      item.render(btn);
    } catch {
      /* a render hook throwing must not break the rail */
    }
  }

  /** Bring the rail to full opacity and restart the idle-fade countdown. */
  private wake(): void {
    if (this.disposed) return;
    this.root.classList.add('is-awake');
    this.scheduleSleep();
  }

  /** Arm the idle timer that dims the rail once the pointer stays away. */
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
    // Fullscreen state feeds active-state (the View-options fullscreen chip),
    // and the change can arrive without a click/keydown (Escape, browser UI)
    // — and always after the async fullscreen request resolves — so the
    // click-driven refresh alone would leave the chip stale.
    this.scheduleRefresh();
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

  /** Re-read every item's active/disabled state and repaint the buttons. */
  private refresh(): void {
    if (this.disposed) return;
    for (const item of this.items) {
      const btn = this.buttons.get(item.id);
      if (!btn) continue;
      // Sync any dynamic icon/label (e.g. Navigation mode) before active-state.
      this.safeRender(item, btn);
      // Disabled state (e.g. Layers with no layers): native `disabled` so the
      // button is grayed, unfocusable, and can't be clicked. Fail-soft — a
      // throwing predicate must not abort the loop for the remaining buttons.
      if (item.disabled) {
        let isDisabled = false;
        try {
          isDisabled = item.disabled();
        } catch {
          /* a disabled predicate throwing leaves the button enabled */
        }
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
        const anyOn = this.overlay.anyToggleActive(item);
        btn.classList.toggle('is-active', anyOn || this.overlay.activeItem === item);
        this.overlay.refreshChips(item);
        continue;
      }
      // A panel-popover button is active when its own isActive() is true (e.g.
      // Performance readout visible) or its popover is currently open.
      btn.classList.toggle(
        'is-active',
        this.isItemActive(item) || this.overlay.activeItem === item
      );
    }
  }

  /** Resolve an item's active state from its own predicate, else from its panel. */
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

  /** Show the one-time 'controls live here' hint, unless it was seen before. */
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
    // The hint is a nudge, not a modal: if the user never interacts it fades
    // away on its own (and any pointer/keyboard interaction dismisses it
    // immediately — see onDocPointerDown / onDocKeyDown).
    this.hintAutoHideTimer = window.setTimeout(() => this.fadeOutHint(), HINT_AUTO_HIDE_MS);
  }

  /** Auto-hide path: fade the hint out, then dismiss (and persist) for real. */
  private fadeOutHint(): void {
    if (!this.hint) return;
    this.hint.classList.add('is-leaving');
    this.hintFadeTimer = window.setTimeout(() => this.dismissHint(), HINT_FADE_MS);
  }

  /** Remove the hint for good and remember that the user has seen it. */
  private dismissHint(): void {
    if (this.hintAutoHideTimer) window.clearTimeout(this.hintAutoHideTimer);
    if (this.hintFadeTimer) window.clearTimeout(this.hintFadeTimer);
    this.hintAutoHideTimer = undefined;
    this.hintFadeTimer = undefined;
    if (!this.hint) return;
    try {
      localStorage.setItem(HINT_STORAGE_KEY, '1');
    } catch {
      /* ignore */
    }
    this.hint.remove();
    this.hint = undefined;
  }

  /** Tear down listeners, timers, the overlay and the rail DOM. Idempotent. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    if (this.hintAutoHideTimer) window.clearTimeout(this.hintAutoHideTimer);
    if (this.hintFadeTimer) window.clearTimeout(this.hintFadeTimer);
    if (this.refreshRaf !== undefined) cancelAnimationFrame(this.refreshRaf);
    this.overlay.dispose();
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
