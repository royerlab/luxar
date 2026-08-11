/**
 * Shared right-click context menu (UI Design Guide §7).
 *
 * Generalizes the dimension-slider menu's algorithm (fixed position at the
 * cursor, viewport clamping, click-outside via a next-tick document listener,
 * Escape) and fixes its gaps: full menu ARIA (`role="menu"` / `menuitem` /
 * `menuitemradio` + `aria-checked`), roving focus with
 * ArrowUp/Down/Home/End, one level of side-flyout submenus
 * (ArrowRight/ArrowLeft), and focus return to the opener on close.
 *
 * All geometry is transform-only on entry (glass-safe, §10.2). The menu is
 * mounted on the viewer container so it inherits theming, and positioned
 * `fixed` at popover z-tier. Coordinates of (0,0) are tolerated (synthetic
 * events in tests carry no clientX/Y).
 *
 * @module ui/overlay-widgets/context-menu
 */

import { getViewerContainer } from '../../utils/viewer-container';
import { EventGroup } from '../../utils/cross-layer/event-group';

/** One entry of a context menu (or of a submenu). */
export interface ContextMenuItem {
  label: string;
  /** Inline stroke SVG (rail icon contract); rendered at 13px. */
  icon?: string;
  /** 'radio' renders a check dot + aria-checked; default is a plain action. */
  kind?: 'action' | 'radio';
  checked?: boolean;
  disabled?: boolean;
  /** Hairline separator above this item. */
  separatorBefore?: boolean;
  /** Invoked on activation; the whole menu closes afterwards. */
  action?: () => void;
  /** One level only — a submenu item must not carry its own submenu. */
  submenu?: ContextMenuItem[];
}

/** Options for {@link openContextMenu}. */
export interface ContextMenuOptions {
  x: number;
  y: number;
  ariaLabel: string;
  items: ContextMenuItem[];
  /** Called exactly once, after the menu is torn down (any close path). */
  onClose?: () => void;
  /**
   * Element to return focus to on close. Defaults to the element focused at
   * open time — right for the keyboard path, but a mouse right-click does
   * not reliably focus its target first, so a caller with a known opener
   * should pass it explicitly.
   */
  restoreFocus?: HTMLElement | null;
}

const MARGIN = 8;

/** The single open menu — opening a new one closes the previous. */
let activeClose: (() => void) | null = null;

/**
 * Open a context menu at (x, y). Returns a close handle (idempotent).
 * Only one context menu exists at a time.
 */
export function openContextMenu(opts: ContextMenuOptions): () => void {
  activeClose?.();

  const events = new EventGroup();
  const container = getViewerContainer();
  const previouslyFocused = opts.restoreFocus ?? (document.activeElement as HTMLElement | null);
  let submenuEl: HTMLElement | null = null;
  let closed = false;

  const root = buildMenuEl(opts.ariaLabel);

  const closeSubmenu = (): void => {
    if (!submenuEl) return;
    submenuEl.remove();
    submenuEl = null;
    for (const it of Array.from(root.querySelectorAll('[aria-expanded="true"]'))) {
      it.setAttribute('aria-expanded', 'false');
    }
  };

  const close = (): void => {
    if (closed) return;
    closed = true;
    closeSubmenu();
    events.dispose();
    root.remove();
    if (activeClose === close) activeClose = null;
    // Restore focus to the opener (menus steal focus for keyboard nav).
    previouslyFocused?.focus?.();
    opts.onClose?.();
  };

  const activate = (item: ContextMenuItem): void => {
    if (item.disabled) return;
    item.action?.();
    close();
  };

  const openSubmenu = (item: ContextMenuItem, anchor: HTMLElement): void => {
    closeSubmenu();
    if (!item.submenu?.length) return;
    anchor.setAttribute('aria-expanded', 'true');
    const sub = buildMenuEl(`${item.label} submenu`);
    sub.classList.add('luxar-context-menu--submenu');
    populate(sub, item.submenu, /* allowSubmenus= */ false);
    container.appendChild(sub);
    const a = anchor.getBoundingClientRect();
    placeMenu(sub, a.right + 2, a.top - 4);
    submenuEl = sub;
    focusFirstItem(sub);
  };

  /** Build item/separator elements into a menu element. */
  const populate = (menu: HTMLElement, items: ContextMenuItem[], allowSubmenus: boolean): void => {
    for (const item of items) {
      if (item.separatorBefore) {
        const sep = document.createElement('div');
        sep.className = 'luxar-context-menu__sep';
        sep.setAttribute('role', 'separator');
        menu.appendChild(sep);
      }
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'luxar-context-menu__item';
      el.setAttribute('role', item.kind === 'radio' ? 'menuitemradio' : 'menuitem');
      if (item.kind === 'radio') el.setAttribute('aria-checked', String(!!item.checked));
      if (item.disabled) {
        el.disabled = true;
        el.setAttribute('aria-disabled', 'true');
      }
      el.tabIndex = -1; // roving focus — the menu manages it

      if (item.kind === 'radio') {
        const dot = document.createElement('span');
        dot.className = 'luxar-context-menu__radio';
        dot.setAttribute('aria-hidden', 'true');
        el.appendChild(dot);
      } else if (item.icon) {
        const icon = document.createElement('span');
        icon.className = 'luxar-context-menu__icon';
        icon.innerHTML = item.icon;
        icon.setAttribute('aria-hidden', 'true');
        el.appendChild(icon);
      }

      const label = document.createElement('span');
      label.className = 'luxar-context-menu__label';
      label.textContent = item.label;
      el.appendChild(label);

      const hasSub = allowSubmenus && !!item.submenu?.length;
      if (hasSub) {
        el.setAttribute('aria-haspopup', 'menu');
        el.setAttribute('aria-expanded', 'false');
        const arrow = document.createElement('span');
        arrow.className = 'luxar-context-menu__sub-arrow';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.innerHTML =
          '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg>';
        el.appendChild(arrow);
      }

      events.on(el, 'click', (e) => {
        e.stopPropagation();
        if (hasSub) openSubmenu(item, el);
        else activate(item);
      });
      if (hasSub) {
        events.on(el, 'pointerenter', () => openSubmenu(item, el));
      } else if (allowSubmenus) {
        // Entering a plain ROOT item retracts any open sibling submenu.
        // Only in the root menu: items INSIDE a submenu take this branch
        // too (their own `hasSub` is forced false), and giving them the
        // retract handler closes the submenu under the cursor the moment
        // the pointer reaches it — mouse-clicking any submenu entry
        // becomes impossible.
        events.on(el, 'pointerenter', () => closeSubmenu());
      }

      menu.appendChild(el);
    }
  };

  populate(root, opts.items, true);
  container.appendChild(root);
  placeMenu(root, opts.x, opts.y);
  focusFirstItem(root);

  // Keyboard: roving focus within whichever menu owns focus.
  events.on(root, 'keydown', (e) => handleMenuKeys(e as KeyboardEvent));
  const handleMenuKeys = (e: KeyboardEvent): void => {
    const menu = submenuEl && submenuEl.contains(document.activeElement) ? submenuEl : root;
    const items = enabledItems(menu);
    const idx = items.indexOf(document.activeElement as HTMLElement);
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowUp': {
        e.preventDefault();
        e.stopPropagation();
        if (!items.length) return;
        // With no item focused (idx -1, e.g. right after a pointer-driven
        // submenu retract dropped focus), enter the list at the APG ends:
        // ArrowDown → first, ArrowUp → last. The modular walk alone would
        // land ArrowUp on the second-to-last item.
        const next =
          idx === -1
            ? items[e.key === 'ArrowDown' ? 0 : items.length - 1]
            : e.key === 'ArrowDown'
              ? items[(idx + 1) % items.length]
              : items[(idx - 1 + items.length) % items.length];
        focusItem(next);
        break;
      }
      case 'Home':
      case 'End':
        e.preventDefault();
        e.stopPropagation();
        focusItem(items[e.key === 'Home' ? 0 : items.length - 1]);
        break;
      case 'ArrowRight': {
        e.preventDefault();
        e.stopPropagation();
        const el = document.activeElement as HTMLElement;
        if (el?.getAttribute('aria-haspopup') === 'menu') el.click();
        break;
      }
      case 'ArrowLeft':
        e.preventDefault();
        e.stopPropagation();
        if (menu === submenuEl) {
          const opener = root.querySelector<HTMLElement>('[aria-expanded="true"]');
          closeSubmenu();
          opener?.focus();
        }
        break;
      case 'Escape':
        e.preventDefault();
        e.stopPropagation();
        if (menu === submenuEl) {
          const opener = root.querySelector<HTMLElement>('[aria-expanded="true"]');
          closeSubmenu();
          opener?.focus();
        } else {
          close();
        }
        break;
      case 'Tab':
        // A menu is not a Tab stop container — any Tab dismisses.
        close();
        break;
      default:
        break;
    }
  };
  // Submenus live outside `root`, so they need the same handler.
  events.on(container, 'keydown', (e) => {
    if (submenuEl?.contains(e.target as Node)) handleMenuKeys(e as KeyboardEvent);
  });

  // Outside-click dismissal, attached next tick so the opening right-click
  // itself doesn't immediately close the menu (the dimension-menu trick).
  const attachTimer = setTimeout(() => {
    events.on(
      document,
      'pointerdown',
      (e) => {
        const t = e.target as Node;
        if (!root.contains(t) && !(submenuEl?.contains(t) ?? false)) close();
      },
      { capture: true }
    );
  }, 0);
  events.add(() => clearTimeout(attachTimer));

  activeClose = close;
  return close;
}

function buildMenuEl(ariaLabel: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'luxar-context-menu';
  el.setAttribute('role', 'menu');
  el.setAttribute('aria-label', ariaLabel);
  return el;
}

function enabledItems(menu: HTMLElement): HTMLElement[] {
  return Array.from(menu.querySelectorAll<HTMLElement>('.luxar-context-menu__item:not(:disabled)'));
}

/**
 * Focus an item AND keep it in view — a menu capped to the viewport height
 * scrolls, and jsdom lacks the native scroll-on-focus, hence the explicit
 * (guarded) scrollIntoView.
 */
function focusItem(el: HTMLElement | undefined): void {
  if (!el) return;
  el.focus();
  el.scrollIntoView?.({ block: 'nearest' });
}

function focusFirstItem(menu: HTMLElement): void {
  focusItem(enabledItems(menu)[0]);
}

/** Clamp a fixed-position menu into the viewport (tolerates x/y of 0). */
function placeMenu(el: HTMLElement, x: number, y: number): void {
  // Moving a menu is not enough for one TALLER than the viewport (the full
  // colormap submenu on a short window): cap the height so the surface
  // scrolls (`overflow-y: auto` in its CSS) instead of pinning to the top
  // margin with the lower entries unreachable.
  el.style.maxHeight = `${Math.max(0, window.innerHeight - 2 * MARGIN)}px`;
  el.style.left = `${Math.max(MARGIN, x)}px`;
  el.style.top = `${Math.max(MARGIN, y)}px`;
  const r = el.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (r.right > vw - MARGIN) el.style.left = `${Math.max(MARGIN, vw - MARGIN - r.width)}px`;
  if (r.bottom > vh - MARGIN) el.style.top = `${Math.max(MARGIN, vh - MARGIN - r.height)}px`;
}
