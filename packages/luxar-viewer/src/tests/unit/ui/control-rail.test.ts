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
    expect(
      document.querySelector('[data-rail-id="render"]')?.classList.contains('is-active')
    ).toBe(true);
  });

  it('re-evaluates active-state on document interaction (event-driven refresh)', () => {
    // #3: active-state is refreshed on document click/keydown (rAF-debounced),
    // NOT by polling — so it reflects a panel opened or CLOSED by any means,
    // including a panel's own × button. Capture the rAF callback so we can flush
    // it deterministically in the same order the real loop would.
    const rafQueue: FrameRequestCallback[] = [];
    const raf = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation(((cb: FrameRequestCallback) => {
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

  it('closes the flyout on an outside pointerdown', () => {
    rail = new ControlRail([
      {
        id: 'view',
        title: 'View',
        icon: RAIL_ICONS.view,
        activate: vi.fn(),
        flyout: [
          { id: 't', title: 'T', icon: RAIL_ICONS.scalebar, activate: vi.fn(), isActive: () => false },
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
});
