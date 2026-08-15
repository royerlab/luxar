// @vitest-environment jsdom
/**
 * Unit tests for `ui/toast.ts`.
 *
 * The load-bearing pin here is the §5.1.3 half of issue #1483: the toast fades
 * `opacity` on its own root, so that root must NOT be a `luxar-glass-surface`
 * (under liquid-glass the refraction/tint layers would ride the fade). The
 * stylesheet side — the fade itself, and the dark tint the theme now supplies
 * in place of the glass `::after` — is pinned in
 * `tests/unit/styles/glass-surface-constraints.test.ts`.
 *
 * The rest is the badge's plain behaviour contract: one toast at a time,
 * message text, and the fade-then-remove dismiss timeline.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { showToast } from '../../../ui/toast';

function currentToast(): HTMLElement | null {
  return document.getElementById('luxar-toast');
}

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('showToast', () => {
  it('is deliberately NOT a glass surface — it fades its own root (§5.1.3)', () => {
    showToast('hello');
    const toast = currentToast();
    expect(toast).not.toBeNull();
    expect(toast!.classList.contains('luxar-glass-surface')).toBe(false);
    expect(toast!.classList.contains('luxar-toast')).toBe(true);
  });

  it('renders the message and starts fully opaque', () => {
    showToast('caches cleared');
    const toast = currentToast()!;
    expect(toast.textContent).toBe('caches cleared');
    expect(toast.style.opacity).toBe('1');
  });

  it('replaces any existing toast so rapid calls collapse into one', () => {
    showToast('first');
    showToast('second');
    expect(document.querySelectorAll('#luxar-toast').length).toBe(1);
    expect(currentToast()!.textContent).toBe('second');
  });

  it('fades the root to 0 after the duration, then removes it 300ms later', () => {
    vi.useFakeTimers();
    showToast('bye', 1000);
    const toast = currentToast()!;

    vi.advanceTimersByTime(999);
    expect(toast.style.opacity).toBe('1');

    // The fade is driven on the root element itself — legal only because the
    // root is not glassed (see the first test).
    vi.advanceTimersByTime(1);
    expect(toast.style.opacity).toBe('0');
    expect(currentToast()).not.toBeNull();

    vi.advanceTimersByTime(300);
    expect(currentToast()).toBeNull();
  });
});
