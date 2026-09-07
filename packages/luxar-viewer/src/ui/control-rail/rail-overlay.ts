/**
 * RailOverlay — the flyout + panel-popover lifecycle for the control rail.
 *
 * Only one overlay is open at a time. This owns opening/closing, positioning,
 * outside-click/routed dismissal, focus return, and chip active-state; the
 * {@link ControlRail} owns the buttons and, via the injected callbacks, the
 * button active-state refresh (which reads {@link RailOverlay.activeItem}).
 *
 * Two overlay flavours share this machinery:
 * - **flyout**: a horizontal row of icon toggle chips (View options).
 * - **popover**: a vertical panel hosting rich controls built lazily by the
 *   item's `popover.build(host)` (Navigation / Settings / Performance).
 *
 * @module ui/control-rail/rail-overlay
 */

import { isPanelVisible, escapeHtml } from './dom-helpers';
import type { ControlRailItem, ControlRailToggle } from './types';

/** Dependencies the overlay needs from its owning {@link ControlRail}. */
export interface RailOverlayDeps {
  /** The rail root — overlays are appended here and positioned relative to it. */
  root: HTMLElement;
  /** The viewer container — scope for `isPanelVisible` chip checks. */
  container: HTMLElement;
  /** Called after every open/close so the rail re-syncs button active-state. */
  onOverlayChange: () => void;
  /** Wake the rail (undim) — called on open. */
  wake: () => void;
}

export class RailOverlay {
  /**
   * The currently-open overlay (chip flyout OR panel popover) + its DOM, or
   * undefined when none is open. `teardown` is the popover builder's cleanup
   * (undefined for chip flyouts).
   */
  private current?: {
    item: ControlRailItem;
    el: HTMLDivElement;
    btn: HTMLButtonElement;
    kind: 'flyout' | 'popover';
    teardown?: () => void;
  };

  // The flyout's tooltip-flip (--up) depends on the viewport height, which
  // changes on window resize and on entering/leaving fullscreen while the
  // flyout stays open — recompute instead of trusting the open-time value.
  private readonly onViewportChange = (): void => this.refreshFlyoutFlip();

  constructor(private readonly deps: RailOverlayDeps) {
    window.addEventListener('resize', this.onViewportChange);
    document.addEventListener('fullscreenchange', this.onViewportChange);
    // webkit* covers Safari < 16.4 (see utils/fullscreen.ts).
    document.addEventListener('webkitfullscreenchange', this.onViewportChange);
  }

  /** The item whose overlay is currently open, or undefined. */
  get activeItem(): ControlRailItem | undefined {
    return this.current?.item;
  }

  /** Re-derive the open flyout's tooltip-flip class from the live viewport. */
  private refreshFlyoutFlip(): void {
    if (this.current?.kind !== 'flyout') return;
    const el = this.current.el;
    el.classList.toggle(
      'luxar-control-rail__flyout--up',
      el.getBoundingClientRect().bottom > window.innerHeight - 48
    );
  }

  private isToggleActive(t: ControlRailToggle): boolean {
    if (t.isActive) {
      try {
        return !!t.isActive();
      } catch {
        return false;
      }
    }
    return t.openSelector ? isPanelVisible(t.openSelector, this.deps.container) : false;
  }

  /** Whether any of the item's flyout toggles is currently active. */
  anyToggleActive(item: ControlRailItem): boolean {
    return (item.flyout ?? []).some((t) => !t.excludeFromParentActive && this.isToggleActive(t));
  }

  /** Re-sync each chip's active-state for the open flyout of `item` (no-op otherwise). */
  refreshChips(item: ControlRailItem): void {
    if (this.current?.item !== item || this.current.kind !== 'flyout') return;
    for (const t of item.flyout ?? []) {
      const chip = this.current.el.querySelector<HTMLButtonElement>(`[data-toggle-id="${t.id}"]`);
      if (!chip) continue;
      const on = this.isToggleActive(t);
      chip.classList.toggle('is-active', on);
      chip.setAttribute('aria-pressed', String(on));
    }
  }

  toggleFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (this.current?.item === item) {
      this.close();
    } else {
      this.openFlyout(item, btn);
    }
  }

  openFlyout(item: ControlRailItem, btn: HTMLButtonElement): void {
    this.close();
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
    this.deps.root.appendChild(el);
    // Align the flyout's vertical centre with the opening button.
    el.style.top = `${this.topInRoot(btn) + btn.offsetHeight / 2}px`;
    btn.setAttribute('aria-expanded', 'true');
    this.current = { item, el, btn, kind: 'flyout' };
    // Flip the chip tooltips above when the flyout sits near the viewport
    // bottom (the View button is low in the rail), so they don't clip
    // off-screen. Re-derived on resize/fullscreenchange while open.
    this.refreshFlyoutFlip();
    this.deps.onOverlayChange();
    this.deps.wake();
  }

  togglePopover(item: ControlRailItem, btn: HTMLButtonElement): void {
    if (this.current?.item === item) {
      this.close();
    } else {
      this.openPopover(item, btn);
    }
  }

  /**
   * Open a vertical panel popover anchored to `btn`, hosting the rich controls
   * built by the item's `popover.build`. Mirrors the flyout's glass surface +
   * real-child arrow, but flows content vertically and vertically clamps within
   * the viewport (popovers can be tall).
   */
  openPopover(item: ControlRailItem, btn: HTMLButtonElement): void {
    this.close();
    const popover = item.popover;
    if (!popover) return;

    const el = document.createElement('div');
    el.className = 'luxar-control-rail__popover luxar-glass-surface luxar-panel-pop';
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

    this.deps.root.appendChild(el);

    // Build the content; capture any teardown for close().
    let teardown: (() => void) | undefined;
    try {
      const cleanup = popover.build(body);
      if (typeof cleanup === 'function') teardown = cleanup;
    } catch {
      /* a popover builder throwing must not break the rail */
    }

    // Anchor near the button, then clamp inside the viewport (measured after
    // content is built so the height is real). Prefer aligning the popover top
    // with the button; if it would overflow the bottom, shift it up.
    const btnTop = this.topInRoot(btn);
    el.style.top = `${btnTop}px`;
    const rect = el.getBoundingClientRect();
    const margin = 8;
    if (rect.bottom > window.innerHeight - margin) {
      const shift = rect.bottom - (window.innerHeight - margin);
      el.style.top = `${Math.max(margin, btnTop - shift)}px`;
    }
    // Keep the arrow pointing at the button even after a vertical shift.
    const arrowTop = btnTop + btn.offsetHeight / 2 - el.offsetTop;
    arrow.style.top = `${arrowTop}px`;

    btn.setAttribute('aria-expanded', 'true');
    this.current = { item, el, btn, kind: 'popover', teardown };
    this.deps.onOverlayChange();
    this.deps.wake();
  }

  /** Close the open overlay (if any), running its teardown + returning focus. */
  close(): void {
    if (!this.current) return;
    const { btn, el, teardown } = this.current;
    btn.setAttribute('aria-expanded', 'false');
    // If keyboard focus is inside the overlay (e.g. closing via Escape while a
    // chip/control is focused), return it to the opener instead of dropping it
    // to <body> — otherwise the user loses their place in the tab order.
    const focusWasInside = el.contains(document.activeElement);
    try {
      teardown?.();
    } catch {
      /* teardown errors must not leave the overlay half-open */
    }
    el.remove();
    this.current = undefined;
    if (focusWasInside) btn.focus();
    this.deps.onOverlayChange();
  }

  /** Close the overlay if a pointerdown landed outside it (and its opener). */
  maybeCloseOnPointer(e: PointerEvent): void {
    if (!this.current) return;
    const target = e.target as Node | null;
    if (this.current.el.contains(target) || this.current.btn.contains(target)) return;
    this.close();
  }

  /** Tear down the overlay (rail disposal). */
  /**
   * A button's top edge in the rail root's coordinate space — what an
   * absolutely positioned popover/flyout child of the root needs for
   * `style.top`. Equals `btn.offsetTop` while the items wrapper is
   * layout-transparent (fine pointers), and stays correct when the wrapper is
   * a scroll box (coarse pointers), where `offsetTop` would ignore its
   * `scrollTop`.
   */
  private topInRoot(btn: HTMLElement): number {
    const root = this.deps.root;
    return btn.getBoundingClientRect().top - root.getBoundingClientRect().top - root.clientTop;
  }

  dispose(): void {
    this.close();
    window.removeEventListener('resize', this.onViewportChange);
    document.removeEventListener('fullscreenchange', this.onViewportChange);
    document.removeEventListener('webkitfullscreenchange', this.onViewportChange);
  }
}
