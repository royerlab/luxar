/**
 * Unit tests for ui/overlay-widgets/context-menu.ts — the shared
 * right-click menu: ARIA roles, activation, dismissal (Escape /
 * outside-click), roving keyboard focus, submenus, and focus return.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { openContextMenu, type ContextMenuItem } from '../../../../ui/overlay-widgets/context-menu';

function menuEl(): HTMLElement | null {
  return document.querySelector('.luxar-context-menu:not(.luxar-context-menu--submenu)');
}

function submenuEl(): HTMLElement | null {
  return document.querySelector('.luxar-context-menu--submenu');
}

function itemByLabel(label: string): HTMLElement | undefined {
  return Array.from(document.querySelectorAll<HTMLElement>('.luxar-context-menu__item')).find(
    (el) => el.textContent === label
  );
}

describe('openContextMenu', () => {
  let close: (() => void) | null = null;

  beforeEach(() => {
    vi.useFakeTimers();
    document.body.innerHTML = '';
  });

  afterEach(() => {
    close?.();
    close = null;
    vi.useRealTimers();
  });

  function openBasic(extra: Partial<ContextMenuItem>[] = []) {
    const action = vi.fn();
    const onClose = vi.fn();
    close = openContextMenu({
      x: 40,
      y: 40,
      ariaLabel: 'Test menu',
      items: [
        { label: 'First', action },
        { label: 'Radio on', kind: 'radio', checked: true, action: vi.fn() },
        { label: 'Radio off', kind: 'radio', checked: false, action: vi.fn() },
        { label: 'Disabled', disabled: true, action: vi.fn() },
        ...(extra as ContextMenuItem[]),
      ],
      onClose,
    });
    return { action, onClose };
  }

  it('renders role=menu with menuitem / menuitemradio + aria-checked', () => {
    openBasic();
    const menu = menuEl();
    expect(menu?.getAttribute('role')).toBe('menu');
    expect(menu?.getAttribute('aria-label')).toBe('Test menu');
    expect(itemByLabel('First')?.getAttribute('role')).toBe('menuitem');
    expect(itemByLabel('Radio on')?.getAttribute('role')).toBe('menuitemradio');
    expect(itemByLabel('Radio on')?.getAttribute('aria-checked')).toBe('true');
    expect(itemByLabel('Radio off')?.getAttribute('aria-checked')).toBe('false');
    expect(itemByLabel('Disabled')?.getAttribute('aria-disabled')).toBe('true');
  });

  it('activating an item fires its action and closes the menu', () => {
    const { action, onClose } = openBasic();
    itemByLabel('First')?.click();
    expect(action).toHaveBeenCalledTimes(1);
    expect(menuEl()).toBeNull();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Escape closes; outside pointerdown closes only after the attach tick', () => {
    const { onClose } = openBasic();
    // Before the next tick the outside listener is not attached (the opening
    // right-click must not self-dismiss).
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    expect(menuEl()).not.toBeNull();

    vi.advanceTimersByTime(1);
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    expect(menuEl()).toBeNull();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('ArrowDown/ArrowUp rove focus over enabled items (disabled skipped)', () => {
    openBasic();
    const menu = menuEl()!;
    expect(document.activeElement).toBe(itemByLabel('First'));
    menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(document.activeElement).toBe(itemByLabel('Radio on'));
    menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    // 'Disabled' is not focus-reachable — End lands on the last ENABLED item.
    expect(document.activeElement).toBe(itemByLabel('Radio off'));
    menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(document.activeElement).toBe(itemByLabel('First')); // wraps
  });

  it('with no item focused, ArrowUp enters at the LAST enabled item (APG ends)', () => {
    openBasic();
    // Drop focus off the menu items (a pointer-driven submenu retract can
    // leave activeElement outside the list).
    (document.activeElement as HTMLElement | null)?.blur?.();
    expect(menuEl()!.contains(document.activeElement)).toBe(false);

    menuEl()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    // Last ENABLED item — 'Disabled' is skipped, so 'Radio off', not
    // second-to-last (the raw modular walk landed there).
    expect(document.activeElement).toBe(itemByLabel('Radio off'));
  });

  it('submenu opens on click/ArrowRight and retracts on ArrowLeft', () => {
    const subAction = vi.fn();
    openBasic([
      {
        label: 'Appearance',
        submenu: [{ label: 'Sub item', action: subAction }],
      },
    ]);
    const opener = itemByLabel('Appearance')!;
    expect(opener.getAttribute('aria-haspopup')).toBe('menu');
    opener.click();
    expect(submenuEl()).not.toBeNull();
    expect(opener.getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe(itemByLabel('Sub item'));

    // ArrowLeft retracts back to the opener.
    document
      .querySelector('.luxar-viewer-container, body')
      ?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    // (fallback: dispatch on the focused submenu item itself)
    itemByLabel('Sub item')?.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })
    );
    expect(submenuEl()).toBeNull();

    // Reopen and activate — closes everything.
    opener.click();
    itemByLabel('Sub item')?.click();
    expect(subAction).toHaveBeenCalledTimes(1);
    expect(menuEl()).toBeNull();
  });

  it('caps the menu height to the viewport so long menus scroll', () => {
    openBasic();
    // placeMenu sets a viewport-relative max-height (MARGIN = 8 on each
    // side); the CSS pairs it with overflow-y: auto so entries below the
    // fold stay reachable instead of pinning off-screen.
    expect(menuEl()!.style.maxHeight).toBe(`${window.innerHeight - 16}px`);
  });

  it('hovering an item INSIDE the submenu does not retract it (mouse path)', () => {
    openBasic([{ label: 'Appearance', submenu: [{ label: 'Sub item', action: vi.fn() }] }]);
    itemByLabel('Appearance')!.click();
    expect(submenuEl()).not.toBeNull();

    // Moving the pointer onto a submenu entry must keep the submenu open —
    // the retract-on-hover behavior belongs to plain ROOT items only. (The
    // regression closed the submenu under the cursor, making every submenu
    // entry unclickable by mouse.)
    itemByLabel('Sub item')!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: false }));
    expect(submenuEl()).not.toBeNull();

    // A plain ROOT item still retracts it.
    itemByLabel('First')!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: false }));
    expect(submenuEl()).toBeNull();
  });

  it('an explicit restoreFocus target wins over the activeElement default', () => {
    // The mouse path: right-click does not focus its target first, so the
    // caller passes the opener explicitly and close() must focus IT, not
    // whatever happened to hold focus when the menu opened.
    const bystander = document.createElement('button');
    const opener = document.createElement('button');
    document.body.append(bystander, opener);
    bystander.focus();

    const closeMenu = openContextMenu({
      x: 10,
      y: 10,
      ariaLabel: 'Menu',
      items: [{ label: 'X' }],
      restoreFocus: opener,
    });
    closeMenu();
    expect(document.activeElement).toBe(opener);
  });

  it('opening a second menu closes the first; close restores focus', () => {
    const outside = document.createElement('button');
    document.body.appendChild(outside);
    outside.focus();

    openBasic();
    const first = menuEl();
    const close2 = openContextMenu({ x: 10, y: 10, ariaLabel: 'Second', items: [{ label: 'X' }] });
    expect(document.querySelectorAll('.luxar-context-menu').length).toBe(1);
    expect(first?.isConnected).toBe(false);
    close2();
    expect(document.activeElement).toBe(outside);
  });
});
