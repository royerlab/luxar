// @vitest-environment jsdom
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { ControlRail, RAIL_ICONS, type ControlRailItem } from '../../../ui/control-rail';

function items(overrides: Partial<ControlRailItem>[] = []): ControlRailItem[] {
  const base: ControlRailItem[] = [
    { id: 'help', title: 'Help', shortcut: 'H', icon: RAIL_ICONS.help, activate: vi.fn() },
    {
      id: 'render',
      title: 'Rendering',
      shortcut: 'R',
      icon: RAIL_ICONS.render,
      activate: vi.fn(),
      isActive: () => false,
    },
    {
      id: 'screenshot',
      title: 'Screenshot',
      shortcut: 'G',
      icon: RAIL_ICONS.screenshot,
      activate: vi.fn(),
      momentary: true,
      separatorBefore: true,
    },
  ];
  return base.map((b, i) => ({ ...b, ...(overrides[i] ?? {}) }));
}

let rail: ControlRail | undefined;

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  document.body.innerHTML = '';
});

afterEach(() => {
  rail?.dispose();
  rail = undefined;
  vi.useRealTimers();
});

describe('ControlRail', () => {
  it('renders one button per item with accessible label + shortcut', () => {
    rail = new ControlRail(items());
    const itemBtns = document.querySelectorAll<HTMLButtonElement>('[data-rail-id]');
    expect(itemBtns.length).toBe(3);
    const help = document.querySelector<HTMLButtonElement>('[data-rail-id="help"]');
    expect(help?.getAttribute('aria-label')).toBe('Help (H)');
    // toolbar semantics
    expect(document.querySelector('.luxar-control-rail')?.getAttribute('role')).toBe('toolbar');
    // separator inserted before the screenshot item
    expect(document.querySelectorAll('.luxar-control-rail__sep').length).toBe(1);
    // always-present collapse handle (not an item — no data-rail-id)
    expect(document.querySelector('.luxar-control-rail__collapse')).not.toBeNull();
  });

  it('keeps item buttons + separators in the items wrapper, and everything else on the root', () => {
    rail = new ControlRail(items(), document.createElement('div'));
    const root = document.querySelector('.luxar-control-rail')!;
    const wrapper = root.querySelector('.luxar-control-rail__items')!;
    expect(wrapper.parentElement).toBe(root);
    expect(wrapper.getAttribute('role')).toBe('presentation');
    // Every item button and separator is inside the wrapper (the coarse-pointer
    // scroll box), in item order.
    expect(wrapper.querySelectorAll('[data-rail-id]').length).toBe(3);
    expect(wrapper.querySelectorAll('.luxar-control-rail__sep').length).toBe(1);
    expect(
      Array.from(wrapper.querySelectorAll('[data-rail-id]')).map((b) =>
        b.getAttribute('data-rail-id')
      )
    ).toEqual(['help', 'render', 'screenshot']);
    // The footer and the collapse handle are NOT in it: scrolling the wrapper
    // must never clip them, and popovers/flyouts append to the root too.
    expect(root.querySelector('.luxar-control-rail__collapse')?.parentElement).toBe(root);
    expect(wrapper.querySelector('.luxar-control-rail__collapse')).toBeNull();
    expect(root.children.length).toBe(3); // wrapper, footer, collapse handle
  });

  it('a touch long-press on a context-popover button opens it without activating (touch)', () => {
    const activate = vi.fn();
    const build = vi.fn();
    rail = new ControlRail(items([{}, { activate, popover: { build, trigger: 'context' } }]));
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="render"]')!;
    btn.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        pointerType: 'touch',
        clientX: 20,
        clientY: 100,
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(500);
    expect(build).toHaveBeenCalledTimes(1);
    expect(document.querySelector('.luxar-control-rail__popover')).not.toBeNull();
    // The release click is swallowed: the button's primary action must not run.
    btn.dispatchEvent(
      new PointerEvent('pointerup', { pointerId: 1, pointerType: 'touch', bubbles: true })
    );
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(activate).not.toHaveBeenCalled();
  });

  it('a mouse press never long-presses', () => {
    const build = vi.fn();
    rail = new ControlRail(items([{}, { popover: { build, trigger: 'context' } }]));
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="render"]')!;
    btn.dispatchEvent(
      new PointerEvent('pointerdown', { pointerId: 1, pointerType: 'mouse', bubbles: true })
    );
    vi.advanceTimersByTime(2000);
    expect(build).not.toHaveBeenCalled();
  });

  it('collapses and expands via the handle, persisting the state', () => {
    rail = new ControlRail(items());
    const railEl = document.querySelector('.luxar-control-rail')!;
    const handle = document.querySelector<HTMLButtonElement>('.luxar-control-rail__collapse')!;
    expect(railEl.classList.contains('is-collapsed')).toBe(false);

    handle.click();
    expect(railEl.classList.contains('is-collapsed')).toBe(true);
    expect(localStorage.getItem('luxar-control-rail-collapsed')).toBe('1');
    expect(handle.getAttribute('aria-label')).toBe('Show controls');

    handle.click();
    expect(railEl.classList.contains('is-collapsed')).toBe(false);
    expect(localStorage.getItem('luxar-control-rail-collapsed')).toBe('0');
  });

  it('restores a persisted collapsed state on construction (and skips the hint)', () => {
    localStorage.setItem('luxar-control-rail-collapsed', '1');
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail')?.classList.contains('is-collapsed')).toBe(
      true
    );
    // No first-run hint while starting collapsed.
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
  });

  it('adds a container marker class so panels can reserve the rail gutter', () => {
    rail = new ControlRail(items());
    expect(document.body.classList.contains('luxar-has-control-rail')).toBe(true);
    rail.dispose();
    expect(document.body.classList.contains('luxar-has-control-rail')).toBe(false);
  });

  it('invokes the item activate() on click and reflects it as the same command', () => {
    const its = items();
    rail = new ControlRail(its);
    document.querySelector<HTMLButtonElement>('[data-rail-id="render"]')!.click();
    expect(its[1].activate).toHaveBeenCalledOnce();
  });

  it('marks an item active when isActive() is true', () => {
    // The constructor's initial refresh() reflects state synchronously.
    const its = items([{}, { isActive: () => true }]);
    rail = new ControlRail(its);
    expect(document.querySelector('[data-rail-id="render"]')?.classList.contains('is-active')).toBe(
      true
    );
  });

  it('re-evaluates active-state on document interaction (event-driven refresh)', () => {
    // #3: active-state is refreshed on a document click and on a routed keydown
    // (rAF-debounced), NOT by polling — so it reflects a panel opened or CLOSED
    // by any means, including a panel's own × button. Capture the rAF callback
    // so we can flush it deterministically in the same order the real loop would.
    const rafQueue: FrameRequestCallback[] = [];
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
      cb: FrameRequestCallback
    ) => {
      rafQueue.push(cb);
      return 1;
    }) as typeof requestAnimationFrame);
    const flushRaf = (): void => rafQueue.shift()?.(0);

    let open = false;
    rail = new ControlRail(items([{}, { isActive: () => open }]));
    const active = () =>
      document.querySelector('[data-rail-id="render"]')?.classList.contains('is-active');
    // Initial synchronous refresh: inactive.
    expect(active()).toBe(false);

    // Panel opens by some external means → a document click schedules a refresh.
    open = true;
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushRaf();
    expect(active()).toBe(true);

    // …and clears again when the panel is closed (e.g. via its own × button).
    open = false;
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushRaf();
    expect(active()).toBe(false);

    // A keydown the input router reports as handled reaches the rail through
    // handleRoutedKeyDown(), which must schedule a refresh too — a shortcut
    // (L, R, N, …) opens a panel with no click anywhere.
    open = true;
    rail.handleRoutedKeyDown();
    flushRaf();
    expect(active()).toBe(true);

    // …and the same path clears it when the shortcut closes the panel again.
    open = false;
    rail.handleRoutedKeyDown();
    flushRaf();
    expect(active()).toBe(false);

    raf.mockRestore();
  });

  it('re-syncs active-state on fullscreenchange (state can flip without a click)', () => {
    // The View-options fullscreen chip keys isActive off the live fullscreen
    // element, which changes without a click/keydown (Escape, browser UI) and
    // always after the async fullscreen request resolves — syncFullscreen must
    // schedule a refresh, not just toggle the rail's is-fullscreen class.
    const rafQueue: FrameRequestCallback[] = [];
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
      cb: FrameRequestCallback
    ) => {
      rafQueue.push(cb);
      return 1;
    }) as typeof requestAnimationFrame);
    const flushRaf = (): void => rafQueue.shift()?.(0);

    let fullscreen = false;
    rail = new ControlRail(items([{}, { isActive: () => fullscreen }]));
    const active = () =>
      document.querySelector('[data-rail-id="render"]')?.classList.contains('is-active');
    expect(active()).toBe(false);

    fullscreen = true;
    document.dispatchEvent(new Event('fullscreenchange'));
    flushRaf();
    expect(active()).toBe(true);

    fullscreen = false;
    document.dispatchEvent(new Event('fullscreenchange'));
    flushRaf();
    expect(active()).toBe(false);

    raf.mockRestore();
  });

  it('never marks a momentary item active', () => {
    rail = new ControlRail(items([{}, {}, { momentary: true, isActive: () => true }]));
    expect(
      document.querySelector('[data-rail-id="screenshot"]')?.classList.contains('is-active')
    ).toBe(false);
  });

  it('shows a first-run hint once, then remembers dismissal', () => {
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail-hint')).not.toBeNull();
    document.querySelector<HTMLButtonElement>('.luxar-control-rail-hint__close')!.click();
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBe('1');

    // A fresh rail does not show the hint again.
    rail.dispose();
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
  });

  it('clicking any button dismisses the first-run hint', () => {
    rail = new ControlRail(items());
    document.querySelector<HTMLButtonElement>('[data-rail-id="help"]')!.click();
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
  });

  it('auto-fades the first-run hint after a few seconds and persists dismissal', () => {
    rail = new ControlRail(items());
    const hint = document.querySelector('.luxar-control-rail-hint')!;
    // Auto-hide kicks in at 10s: fade class first, removal after the transition.
    vi.advanceTimersByTime(10_000);
    expect(hint.classList.contains('is-leaving')).toBe(true);
    vi.advanceTimersByTime(400);
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBe('1');
  });

  it('any pointerdown anywhere dismisses the first-run hint', () => {
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail-hint')).not.toBeNull();
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBe('1');
  });

  it('a routed keypress dismisses the first-run hint', () => {
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail-hint')).not.toBeNull();
    rail.handleRoutedKeyDown();
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBe('1');
  });

  it('a routed keypress after dispose does not burn the first-run hint', () => {
    // dispose() detaches the hint without dismissing it (the user never saw it
    // long enough to count). A late routed keydown must not persist the
    // "seen" flag on its way out, or the next session loses the hint for good.
    rail = new ControlRail(items());
    expect(document.querySelector('.luxar-control-rail-hint')).not.toBeNull();
    rail.dispose();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBeNull();

    rail.handleRoutedKeyDown();
    expect(localStorage.getItem('luxar-control-rail-hint-dismissed')).toBeNull();
  });

  it('returns focus to the body after a pointer click (keeps Space/global shortcuts working)', () => {
    // Regression: a rail button that keeps focus after a mouse click swallows
    // the next Space (canvas/body-gated fullscreen). The rail blurs the button
    // after a *pointer* click (event.detail > 0) so focus returns to the body.
    rail = new ControlRail(items());
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="help"]')!;
    btn.focus();
    expect(document.activeElement).toBe(btn);
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
    expect(document.activeElement).not.toBe(btn);
  });

  it('keeps focus on keyboard activation (click with detail 0)', () => {
    // Keyboard users (Enter/Space → click with detail === 0) must keep focus
    // so rail navigation stays usable; only pointer clicks blur.
    rail = new ControlRail(items());
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="help"]')!;
    btn.focus();
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
    expect(document.activeElement).toBe(btn);
  });

  it('opens a flyout of compact toggles (with tooltip-below) and fires a toggle on chip click', () => {
    const toggleFn = vi.fn();
    const its: ControlRailItem[] = [
      {
        id: 'view',
        title: 'View options',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 'scalebar',
            title: 'Scale bar',
            shortcut: 'B',
            icon: RAIL_ICONS.scalebar,
            activate: toggleFn,
            isActive: () => false,
          },
        ],
      },
    ];
    rail = new ControlRail(its);
    // Closed initially.
    expect(document.querySelector('.luxar-control-rail__flyout')).toBeNull();

    // Click the flyout button → popover with a chip + a below-tooltip.
    document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!.click();
    const fly = document.querySelector('.luxar-control-rail__flyout');
    expect(fly).not.toBeNull();
    const chip = fly!.querySelector<HTMLButtonElement>('[data-toggle-id="scalebar"]')!;
    expect(chip).not.toBeNull();
    expect(chip.getAttribute('aria-label')).toBe('Scale bar (B)');
    expect(chip.querySelector('.luxar-control-rail__chip-tip')).not.toBeNull();

    // Clicking a chip fires its toggle and keeps the flyout open.
    chip.click();
    expect(toggleFn).toHaveBeenCalledOnce();
    expect(document.querySelector('.luxar-control-rail__flyout')).not.toBeNull();

    // Re-clicking the flyout button closes it.
    document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!.click();
    expect(document.querySelector('.luxar-control-rail__flyout')).toBeNull();
  });

  it('an excludeFromParentActive toggle never lights the parent flyout button', () => {
    // The fullscreen chip's contract: active INSIDE the flyout, but a
    // session-long ambient state must not glow the View button all session.
    // A normal active toggle (cinematic) still does.
    let ambientOn = true;
    let normalOn = false;
    rail = new ControlRail([
      {
        id: 'view',
        title: 'View options',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 'ambient',
            title: 'Ambient',
            icon: RAIL_ICONS.scalebar,
            activate: vi.fn(),
            isActive: () => ambientOn,
            excludeFromParentActive: true,
          },
          {
            id: 'normal',
            title: 'Normal',
            icon: RAIL_ICONS.cinematic,
            activate: vi.fn(),
            isActive: () => normalOn,
          },
        ],
      },
    ]);
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!;
    // Only the excluded toggle is on → parent must NOT glow.
    expect(btn.classList.contains('is-active')).toBe(false);

    // A normal toggle turning on still lights the parent (constructor-time
    // refresh is synchronous; rebuild to re-evaluate deterministically).
    rail.dispose();
    ambientOn = false;
    normalOn = true;
    rail = new ControlRail([
      {
        id: 'view',
        title: 'View options',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 'normal',
            title: 'Normal',
            icon: RAIL_ICONS.cinematic,
            activate: vi.fn(),
            isActive: () => normalOn,
          },
        ],
      },
    ]);
    const btn2 = document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!;
    expect(btn2.classList.contains('is-active')).toBe(true);
  });

  it('recomputes the flyout tooltip-flip on fullscreenchange/resize while open', () => {
    rail = new ControlRail([
      {
        id: 'view',
        title: 'View',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [{ id: 't', title: 'T', icon: RAIL_ICONS.scalebar, activate: vi.fn() }],
      },
    ]);
    document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!.click();
    const fly = document.querySelector<HTMLElement>('.luxar-control-rail__flyout')!;
    expect(fly).not.toBeNull();

    // Force the flyout to sit "near the viewport bottom" and fire the
    // viewport-change events the overlay now listens for.
    fly.getBoundingClientRect = () => ({ bottom: window.innerHeight + 100 }) as DOMRect;
    document.dispatchEvent(new Event('fullscreenchange'));
    expect(fly.classList.contains('luxar-control-rail__flyout--up')).toBe(true);

    // And back: viewport grows → the flip clears on resize.
    fly.getBoundingClientRect = () => ({ bottom: 0 }) as DOMRect;
    window.dispatchEvent(new Event('resize'));
    expect(fly.classList.contains('luxar-control-rail__flyout--up')).toBe(false);
  });

  it('closes the flyout on an outside pointerdown', () => {
    rail = new ControlRail([
      {
        id: 'view',
        title: 'View',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 't',
            title: 'T',
            icon: RAIL_ICONS.scalebar,
            activate: vi.fn(),
            isActive: () => false,
          },
        ],
      },
    ]);
    document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!.click();
    expect(document.querySelector('.luxar-control-rail__flyout')).not.toBeNull();
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(document.querySelector('.luxar-control-rail__flyout')).toBeNull();
  });

  it('reveals on keyboard focus (focusin wakes the rail — WCAG 2.4.7)', () => {
    // A Tab into the rail while it is idle-dimmed / collapsed / (opacity:0)
    // in fullscreen must reveal it, else focus lands on an invisible control.
    rail = new ControlRail(items());
    const railEl = document.querySelector('.luxar-control-rail')!;
    railEl.classList.remove('is-awake');
    document
      .querySelector<HTMLButtonElement>('[data-rail-id="help"]')!
      .dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    expect(railEl.classList.contains('is-awake')).toBe(true);
  });

  it('flyout exposes a correct ARIA popover pattern + a real arrow element', () => {
    const its: ControlRailItem[] = [
      {
        id: 'view',
        title: 'View options',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 'scalebar',
            title: 'Scale bar',
            shortcut: 'B',
            icon: RAIL_ICONS.scalebar,
            activate: vi.fn(),
            isActive: () => false,
          },
        ],
      },
    ];
    rail = new ControlRail(its);
    const opener = document.querySelector('[data-rail-id="view"]')!;
    // Opener advertises the popover and its collapsed state.
    expect(opener.getAttribute('aria-haspopup')).toBe('true');
    expect(opener.getAttribute('aria-expanded')).toBe('false');

    (opener as HTMLButtonElement).click();
    expect(opener.getAttribute('aria-expanded')).toBe('true');
    const fly = document.querySelector('.luxar-control-rail__flyout')!;
    // role=group (a set of toggles), not an incomplete role=menu.
    expect(fly.getAttribute('role')).toBe('group');
    // Arrow is a real child element (not ::before, which the glass themes claim).
    expect(fly.querySelector('.luxar-control-rail__flyout-arrow')).not.toBeNull();
    // Chips are toggle buttons with aria-pressed reflecting state.
    const chip = fly.querySelector('[data-toggle-id="scalebar"]')!;
    expect(chip.getAttribute('aria-pressed')).toBe('false');

    (opener as HTMLButtonElement).click();
    expect(opener.getAttribute('aria-expanded')).toBe('false');
  });

  it('does not own document Escape while a flyout is open', () => {
    const its: ControlRailItem[] = [
      {
        id: 'view',
        title: 'View',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          {
            id: 'scalebar',
            title: 'Scale bar',
            icon: RAIL_ICONS.scalebar,
            activate: vi.fn(),
            isActive: () => false,
          },
        ],
      },
    ];
    rail = new ControlRail(its);
    const opener = document.querySelector<HTMLButtonElement>('[data-rail-id="view"]')!;
    opener.click();
    const chip = document.querySelector<HTMLButtonElement>('[data-toggle-id="scalebar"]')!;
    chip.focus();
    expect(document.activeElement).toBe(chip);
    // Escape is routed by InputHandler/PanelCoordinator. The rail must not
    // close independently, or it bypasses recording/fullscreen precedence.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(document.querySelector('.luxar-control-rail__flyout')).not.toBeNull();
    expect(document.activeElement).toBe(chip);

    rail.closeOverlay();
    expect(document.querySelector('.luxar-control-rail__flyout')).toBeNull();
    expect(document.activeElement).toBe(opener);
  });

  it('blurs a focusable (non-button) footer on pointer click, keeping Space working', () => {
    // The docked perf readout is a <div tabindex=0>; leaving focus on it would
    // swallow Space=fullscreen exactly like a focused button does.
    const footer = document.createElement('div');
    footer.tabIndex = 0;
    footer.id = 'perf-like';
    rail = new ControlRail(items(), footer);
    footer.focus();
    expect(document.activeElement).toBe(footer);
    footer.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
    expect(document.activeElement).not.toBe(footer);
  });

  it('reference-counts the document.body marker across instances', () => {
    rail = new ControlRail(items()); // instance A (disposed in afterEach)
    const b = new ControlRail(items()); // instance B
    expect(document.body.classList.contains('luxar-has-control-rail')).toBe(true);
    // Disposing B must NOT strip the marker while A is still alive.
    b.dispose();
    expect(document.body.classList.contains('luxar-has-control-rail')).toBe(true);
    // Only when the last instance disposes does the marker go away.
    rail.dispose();
    rail = undefined;
    expect(document.body.classList.contains('luxar-has-control-rail')).toBe(false);
  });

  it('docks a footer element (e.g. the perf readout) inside the rail', () => {
    const footer = document.createElement('div');
    footer.id = 'my-footer';
    rail = new ControlRail(items(), footer);
    const mounted = document.querySelector('.luxar-control-rail #my-footer');
    expect(mounted).toBe(footer);
  });

  it('dispose() removes all DOM and stops timers/listeners', () => {
    rail = new ControlRail(items());
    rail.dispose();
    expect(document.querySelector('.luxar-control-rail')).toBeNull();
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    // no throw / no work after dispose
    expect(() => vi.advanceTimersByTime(1000)).not.toThrow();
  });

  // ── Panel popovers (RailOverlay popover branch), disabled predicate, and
  //    context-menu suppression — all added in the rail-reorg; the flyout tests
  //    above don't exercise these paths.

  it('opens a click-trigger panel popover, builds content, and runs teardown on close', () => {
    const teardown = vi.fn();
    const build = vi.fn((host: HTMLElement) => {
      const content = document.createElement('div');
      content.className = 'popover-content';
      host.appendChild(content);
      return teardown;
    });
    rail = new ControlRail([
      {
        id: 'settings',
        title: 'Settings',
        icon: RAIL_ICONS.settings,
        activate: vi.fn(),
        popover: { trigger: 'click', title: 'Settings', build },
      },
    ]);
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="settings"]')!;
    expect(document.querySelector('.luxar-control-rail__popover')).toBeNull();

    btn.click();
    const pop = document.querySelector('.luxar-control-rail__popover');
    expect(pop).not.toBeNull();
    expect(build).toHaveBeenCalledOnce();
    expect(pop!.querySelector('.popover-content')).not.toBeNull();
    expect(pop!.getAttribute('role')).toBe('group');
    expect(pop!.querySelector('.luxar-control-rail__popover-arrow')).not.toBeNull();
    expect(btn.getAttribute('aria-expanded')).toBe('true');

    // Re-click closes the popover AND runs the builder's teardown.
    btn.click();
    expect(document.querySelector('.luxar-control-rail__popover')).toBeNull();
    expect(teardown).toHaveBeenCalledOnce();
    expect(btn.getAttribute('aria-expanded')).toBe('false');
  });

  it('anchors a panel popover from root-relative rects when the items wrapper scrolls', () => {
    rail = new ControlRail([
      {
        id: 'settings',
        title: 'Settings',
        icon: RAIL_ICONS.settings,
        activate: vi.fn(),
        popover: { trigger: 'click', build: vi.fn() },
      },
    ]);
    const root = document.querySelector<HTMLElement>('.luxar-control-rail')!;
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="settings"]')!;
    root.getBoundingClientRect = () => ({ top: 100 }) as DOMRect;
    btn.getBoundingClientRect = () => ({ top: 140 }) as DOMRect;
    Object.defineProperty(root, 'clientTop', { configurable: true, value: 2 });
    Object.defineProperty(btn, 'offsetHeight', { configurable: true, value: 38 });

    btn.click();

    const pop = document.querySelector<HTMLElement>('.luxar-control-rail__popover')!;
    const arrow = pop.querySelector<HTMLElement>('.luxar-control-rail__popover-arrow')!;
    expect(pop.style.top).toBe('38px');
    expect(arrow.style.top).toBe('57px');
  });

  it('a context-trigger popover opens on right-click; left-click still fires activate()', () => {
    const activate = vi.fn();
    const build = vi.fn(() => vi.fn());
    rail = new ControlRail([
      {
        id: 'perf',
        title: 'Performance',
        icon: RAIL_ICONS.perf,
        activate,
        popover: { trigger: 'context', build },
      },
    ]);
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="perf"]')!;

    // Left-click fires activate, does NOT open the popover.
    btn.click();
    expect(activate).toHaveBeenCalledOnce();
    expect(document.querySelector('.luxar-control-rail__popover')).toBeNull();

    // Right-click opens the popover and does NOT fire activate again.
    btn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(document.querySelector('.luxar-control-rail__popover')).not.toBeNull();
    expect(build).toHaveBeenCalledOnce();
    expect(activate).toHaveBeenCalledOnce();
  });

  it('suppresses the native context menu anywhere on the rail (no leaked browser menu)', () => {
    rail = new ControlRail(items()); // none of help/render/screenshot has a popover
    const e = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    document.querySelector('[data-rail-id="help"]')!.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(true);
    // No popover opened for a non-popover button.
    expect(document.querySelector('.luxar-control-rail__popover')).toBeNull();
  });

  it('applies a disabled() predicate as the native disabled attribute + suppresses active-state', () => {
    // isActive() is true, but disabled() wins: the button must be grayed/inert
    // and NOT marked active. Verified on the constructor's synchronous refresh.
    rail = new ControlRail(items([{}, { disabled: () => true, isActive: () => true }]));
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="render"]')!;
    expect(btn.disabled).toBe(true);
    expect(btn.classList.contains('is-active')).toBe(false);
  });

  it('re-evaluates disabled() when a luxar-layers-changed event fires (no click needed)', () => {
    const rafQueue: FrameRequestCallback[] = [];
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
      cb: FrameRequestCallback
    ) => {
      rafQueue.push(cb);
      return 1;
    }) as typeof requestAnimationFrame);
    const flushRaf = (): void => rafQueue.shift()?.(0);

    let hasLayers = false;
    rail = new ControlRail(items([{}, { disabled: () => !hasLayers }]));
    const btn = document.querySelector<HTMLButtonElement>('[data-rail-id="render"]')!;
    expect(btn.disabled).toBe(true); // no layers at construction

    // Scene loads layers → a layers-changed event schedules a refresh.
    hasLayers = true;
    window.dispatchEvent(new CustomEvent('luxar-layers-changed'));
    flushRaf();
    expect(btn.disabled).toBe(false);

    raf.mockRestore();
  });
});

describe('ControlRail — hidden items', () => {
  it('does not render a button whose hidden predicate is true, and shows it once the predicate flips', () => {
    let hasSound = false;
    rail = new ControlRail(
      items([
        {},
        { id: 'audio', title: 'Sound', icon: RAIL_ICONS.audio, hidden: () => !hasSound },
        {},
      ])
    );
    const btn = document.querySelector('[data-rail-id="audio"]') as HTMLButtonElement;
    expect(btn.hidden).toBe(true);

    hasSound = true;
    window.dispatchEvent(new Event('luxar-audio-changed'));
    vi.runOnlyPendingTimers();
    expect(btn.hidden).toBe(false);
  });

  it('a throwing hidden predicate leaves the button shown', () => {
    rail = new ControlRail(
      items([
        {
          hidden: () => {
            throw new Error('boom');
          },
        },
      ])
    );
    const btn = document.querySelector('[data-rail-id="help"]') as HTMLButtonElement;
    expect(btn.hidden).toBe(false);
  });
});
