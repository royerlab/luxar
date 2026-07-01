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
    const its = items([{}, { isActive: () => true }]);
    rail = new ControlRail(its);
    vi.advanceTimersByTime(500); // let the refresh interval run
    expect(
      document.querySelector('[data-rail-id="render"]')?.classList.contains('is-active')
    ).toBe(true);
  });

  it('never marks a momentary item active', () => {
    rail = new ControlRail(items([{}, {}, { momentary: true, isActive: () => true }]));
    vi.advanceTimersByTime(500);
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

  it('dispose() removes all DOM and stops the refresh timer', () => {
    rail = new ControlRail(items());
    rail.dispose();
    expect(document.querySelector('.luxar-control-rail')).toBeNull();
    expect(document.querySelector('.luxar-control-rail-hint')).toBeNull();
    // no throw / no work after dispose
    expect(() => vi.advanceTimersByTime(1000)).not.toThrow();
  });
});
